#!/usr/bin/env python3
"""
SILICONE TENSILE ANALYSIS (Instron / Bluehill exports)

What it does, for every sample, the same way:
  1. reads the Bluehill file (Time, Displacement, Force, Tensile strain, Tensile stress)
  2. smooths the small force ripple
  3. starts every curve where Bluehill started the test (strain 0 at the test preload, ~0.1 N),
     so all curves start from the same point; the starting force is recorded so you can check it
  4. finds the break point and cuts the curve there
  5. calculates Young's modulus (10-50 % strain), UTS, strain at break, stress at 100 % strain
Then for each material it makes mean +/- SD curves, bar charts and statistics.

FOLDERS (keep your own folder and file names - nothing needs renaming):
  silicone_tensile.py
  raw/
    silicone/
      Silicone 00-20 10% thinner/    <- every sample file for this material
      Silicone 00-30/
      Silicone 00-30 10% thinner/

RUN (in the VS Code terminal, from the folder that holds this script):
  python silicone_tensile.py --check      <- lists what it found and reads one file, analyses nothing
  python silicone_tensile.py              <- full analysis -> results_silicone/
"""

import argparse
import csv
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# SETTINGS - the only part you should need to edit
# =============================================================================
HERE = Path(__file__).resolve().parent
DATA_FOLDER = HERE / "raw" / "silicone"      # one sub-folder per material
RESULTS_FOLDER = HERE / "results_silicone"
EXCLUSIONS_FILE = HERE / "silicone_exclusions.csv"   # created automatically

# Folder name -> label used in tables and figures (also fixes the typo in the folder name).
# Order here = order in figures. Folders not listed are still analysed, using the folder name.
LABELS = {
    "Silicone 00-10":             "00-10",
    "Silicone 00-20":             "00-20",
    "Silicone 00-20 10% thinner": "00-20 + 10% thinner",
    "Silicone 00-30":             "00-30",
    "Silicone 00-30 10% thinner": "00-30 + 10% thinner",
    "Sillicone 00-30 10% thinner": "00-30 + 10% thinner",
}

SMOOTH_POINTS = 25              # moving-average window; 25 points = 0.25 s at 100 Hz. 1 = no smoothing
MODULUS_RANGE_PCT = (10, 50)    # strain window (%) for the Young's modulus fit
TOE_COMPENSATION = True         # remove initial slack (e.g. a curled sample straightening) - see analyse()
TOE_STRESS_WINDOW_KPA = (5, 20) # straight part used to find the slack: 5-20 kPa above the starting stress
BREAK_DROP = 0.5                # break = first point after the peak where the (unsmoothed) stress falls below 50 % of peak
CURVE_POINTS = 500              # resolution of the aligned curves
FIG_DPI = 300

COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]  # colour-blind safe
plt.rcParams.update({"font.size": 10, "font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

# =============================================================================
# Reading files
# =============================================================================


def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(p).name)]


def data_files(folder):
    return sorted([f for f in Path(folder).iterdir()
                   if f.is_file() and f.suffix.lower() in (".csv", ".txt")], key=natural_key)


def read_bluehill(path):
    """Return a DataFrame with columns time_s, disp_mm, force_N, strain (fraction), stress_kPa."""
    text = None
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = Path(path).read_text(encoding=enc)
            if "\x00" not in text:
                break
        except (UnicodeDecodeError, UnicodeError):
            continue
    lines = text.splitlines()
    head = "\n".join(lines[:60])
    delim = max([",", ";", "\t"], key=head.count)
    rows = list(csv.reader(lines, delimiter=delim))

    hdr_i = None
    for i, r in enumerate(rows):
        low = " ".join(c.lower() for c in r)
        if "force" in low or "load" in low:
            hdr_i = i
            break
    if hdr_i is None:
        raise ValueError("no header row with a Force/Load column found")
    header = [c.strip() for c in rows[hdr_i]]
    units = {}
    data = []
    for r in rows[hdr_i + 1:]:
        r = (r + [""] * len(header))[:len(header)]
        try:
            vals = [float(c.strip().strip('"').replace(",", ".") if delim == ";" else c.strip().strip('"'))
                    if c.strip() else np.nan for c in r]
        except ValueError:
            if not data and not units:
                units = {h: c.strip() for h, c in zip(header, r)}
            continue
        if not all(np.isnan(vals)):
            data.append(vals)
    df = pd.DataFrame(data, columns=header).dropna(axis=1, how="all")

    def col(*keys, avoid=()):
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in keys) and not any(a in lc for a in avoid):
                return c
        return None

    c_time = col("time")
    c_disp = col("displacement", "extension", "position", avoid=("strain",))
    c_force = col("force", "load")
    c_strain = col("strain")
    c_stress = col("stress")
    missing = [n for n, c in [("Force", c_force), ("Displacement", c_disp),
                              ("Tensile strain", c_strain), ("Tensile stress", c_stress)] if c is None]
    if missing:
        raise ValueError(f"column(s) not found: {missing}. Columns in file: {list(df.columns)}")

    def unit(c):
        return units.get(c, "").lower()

    out = pd.DataFrame({
        "time_s": df[c_time] if c_time else np.arange(len(df)) * 0.01,
        "disp_mm": df[c_disp],
        "force_N": df[c_force] * (1000.0 if "kn" in unit(c_force) else 1.0),
        "strain": df[c_strain] * (0.01 if ("%" in unit(c_strain) or unit(c_strain) == "") else 1.0),
        "stress_kPa": df[c_stress] * (1000.0 if ("mpa" in unit(c_stress) or unit(c_stress) == "")
                                      else 1.0 if "kpa" in unit(c_stress) else 0.001),
    }).dropna().reset_index(drop=True)
    if out.force_N.max() < abs(out.force_N.min()):        # tension recorded as negative
        out[["force_N", "strain", "stress_kPa"]] *= -1
    info = {"delimiter": repr(delim), "columns": list(df.columns), "units": units,
            "used": {"time": c_time, "displacement": c_disp, "force": c_force,
                     "strain": c_strain, "stress": c_stress}}
    return out, info

# =============================================================================
# Analysis of one sample
# =============================================================================


def smooth(y):
    if SMOOTH_POINTS <= 1:
        return np.asarray(y, float)
    return pd.Series(y).rolling(SMOOTH_POINTS, center=True, min_periods=1).mean().to_numpy()


def analyse(df):
    raw = df.stress_kPa.to_numpy()
    strain = df.strain.to_numpy()

    # break: found on the UNsmoothed signal, so the sudden drop is not smeared by smoothing
    ipk_raw = int(np.argmax(raw))
    if ipk_raw < 10:
        raise ValueError("stress peaks at the very start - check this file")
    after = np.where(raw[ipk_raw:] < BREAK_DROP * raw[ipk_raw])[0]
    fail = ipk_raw + after[0] if len(after) else len(raw)   # first point of the final collapse
    # the failure takes several data points (and may come in steps), so the break is taken as
    # the highest measured stress before the collapse = the point where failure starts
    ib = int(np.argmax(raw[:fail]))

    # start point = start of the Bluehill test (same preload for every sample)
    i0 = 0
    e = strain[i0:ib + 1] - strain[i0]          # aligned strain (fraction), starts at 0
    s = smooth(raw[i0:ib + 1])                   # smoothed stress up to the break only (kPa)
    uts = float(raw[ib])                         # measured (unsmoothed) peak stress

    # TOE COMPENSATION (as in ASTM tensile standards): if the sample was slack at the start
    # (e.g. slightly curled), the first part of the curve is flat. Fit the first straight part
    # of the curve (between 5 and 20 kPa above the starting stress), extend that line back to
    # the starting stress, and count strain from there. A sample with no slack gives ~0 shift.
    toe = 0.0
    if TOE_COMPENSATION:
        s0 = float(np.median(s[:5]))
        lo_s, hi_s = s0 + TOE_STRESS_WINDOW_KPA[0], s0 + TOE_STRESS_WINDOW_KPA[1]
        a_lo = np.where(s >= lo_s)[0]
        a_hi = np.where(s >= hi_s)[0]
        if len(a_lo) and len(a_hi) and a_hi[0] - a_lo[0] >= 5:
            tf = stats.linregress(e[a_lo[0]:a_hi[0]], s[a_lo[0]:a_hi[0]])
            if tf.slope > 0:
                toe = max(0.0, (s0 - tf.intercept) / tf.slope)
        if toe > 0:
            keep = e >= toe
            e, s = e[keep] - toe, s[keep]

    lo, hi = MODULUS_RANGE_PCT[0] / 100, MODULUS_RANGE_PCT[1] / 100
    m = (e >= lo) & (e <= hi)
    if m.sum() >= 5:
        fit = stats.linregress(e[m], s[m])
        E, b, r2 = fit.slope, fit.intercept, fit.rvalue ** 2
    else:
        E = b = r2 = np.nan
    emono = np.maximum.accumulate(e)
    s100 = float(np.interp(1.0, emono, s)) if emono[-1] >= 1.0 else np.nan

    t = df.time_s.to_numpy()
    rate = float(np.polyfit(t, df.disp_mm.to_numpy(), 1)[0] * 60) if len(t) > 2 else np.nan
    mm = np.abs(df.strain.to_numpy()) > 0.01
    area = float(np.median(df.force_N[mm] / (df.stress_kPa[mm] / 1000))) if mm.any() else np.nan
    gauge = float(np.median((df.disp_mm[mm] - df.disp_mm.iloc[0]) / df.strain[mm])) if mm.any() else np.nan

    res = {"youngs_modulus_kPa": E, "modulus_fit_R2": r2, "uts_kPa": uts,
           "strain_at_break_pct": float(e[-1] * 100), "stress_at_100pct_kPa": s100,
           "start_force_N": float(np.median(df.force_N.to_numpy()[:5])),
           "toe_slack_removed_pct": toe * 100,
           "area_mm2_bluehill": area, "gauge_mm_bluehill": gauge, "rate_mm_min": rate}
    curve = {"strain": e, "stress": s, "fit": (E, b), "break": (e[-1], uts),
             "raw_strain": strain - strain[i0] - toe, "raw_stress": df.stress_kPa.to_numpy()}
    return res, curve

# =============================================================================
# Plots
# =============================================================================


def save(fig, path):
    fig.tight_layout()
    fig.savefig(f"{path}.png", dpi=FIG_DPI)
    fig.savefig(f"{path}.pdf")
    plt.close(fig)


def qc_plot(c, r, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(c["raw_strain"] * 100, c["raw_stress"], color="0.75", lw=0.8, label="raw")
    ax.plot(c["strain"] * 100, c["stress"], color=COLOURS[0], lw=1.6, label="analysed (smoothed)")
    lo, hi = MODULUS_RANGE_PCT
    ax.axvspan(lo, hi, color="red", alpha=0.07)
    E, b = c["fit"]
    if np.isfinite(E):
        x = np.array([0, min(hi * 2, c["strain"][-1] * 100)]) / 100
        ax.plot(x * 100, E * x + b, "r--", lw=1, label=f"E = {E:.1f} kPa (R² {r['modulus_fit_R2']:.3f})")
    ax.plot(0, c["stress"][0], "go", ms=6, label="start of test")
    bx, by = c["break"]
    ax.plot(bx * 100, by, "kx", ms=9, mew=2, label="break")
    # zoomed inset around the break so its position can be checked
    ins = ax.inset_axes([0.08, 0.42, 0.38, 0.5])
    rs, rt = c["raw_strain"] * 100, c["raw_stress"]
    zm = (rs >= bx * 100 - 40) & (rs <= bx * 100 + 15)
    ins.plot(rs[zm], rt[zm], ".-", color="0.55", lw=0.8, ms=2)
    ins.plot(bx * 100, by, "kx", ms=9, mew=2)
    ins.set_title("zoom: break", fontsize=7)
    ins.tick_params(labelsize=6)
    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("Stress (kPa)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def common_grid(curves, upto=None):
    smax = upto if upto is not None else min(c["strain"].max() for c in curves)
    grid = np.linspace(0, smax, CURVE_POINTS)
    ys = []
    for c in curves:
        x = np.maximum.accumulate(c["strain"])
        x, idx = np.unique(x, return_index=True)
        y = np.interp(grid, x, c["stress"][idx])
        y[grid > x.max()] = np.nan                   # no data beyond this sample's break
        ys.append(y)
    return grid, np.array(ys)

# =============================================================================
# Main
# =============================================================================


def find_groups():
    if not DATA_FOLDER.is_dir():
        sys.exit(f"Folder not found: {DATA_FOLDER}\n"
                 "Make a folder 'raw', inside it 'silicone', and inside that one folder per material.")
    folders = [d for d in DATA_FOLDER.iterdir() if d.is_dir()]
    if not folders:
        if data_files(DATA_FOLDER):
            sys.exit(f"The files are directly inside {DATA_FOLDER}.\n"
                     "Put each material's files in its own sub-folder (e.g. 'Silicone 00-30').")
        sys.exit(f"No material folders inside {DATA_FOLDER}")
    order = list(dict.fromkeys(LABELS.values()))
    def rank(d):
        lab = LABELS.get(d.name.strip(), d.name.strip())
        return (order.index(lab) if lab in order else len(order), d.name)
    return [(d, LABELS.get(d.name.strip(), d.name.strip())) for d in sorted(folders, key=rank)]


def check():
    groups = find_groups()
    print(f"Looking in: {DATA_FOLDER}\n")
    first = None
    for d, lab in groups:
        fs = data_files(d)
        print(f"  {d.name:<32} -> label '{lab}'   {len(fs)} file(s)")
        for f in fs:
            print(f"      {f.name}")
        first = first or (fs[0] if fs else None)
    if first is None:
        sys.exit("\nNo .csv or .txt files found.")
    print(f"\nTest-reading {first.name} ...")
    df, info = read_bluehill(first)
    print(f"  delimiter {info['delimiter']}, columns: {info['columns']}")
    print(f"  using: {info['used']}")
    print(f"  {len(df)} data rows, max force {df.force_N.max():.3f} N, "
          f"max strain {df.strain.max() * 100:.1f} %")
    res, _ = analyse(df)
    print(f"  quick result: E = {res['youngs_modulus_kPa']:.1f} kPa, UTS = {res['uts_kPa']:.1f} kPa, "
          f"strain at break = {res['strain_at_break_pct']:.0f} %")
    print("\nIf that looks right, run:  python silicone_tensile.py")


def load_exclusions(all_ids):
    if EXCLUSIONS_FILE.exists():
        ex = pd.read_csv(EXCLUSIONS_FILE, dtype=str, keep_default_na=False)
    else:
        ex = pd.DataFrame(columns=["material_folder", "file", "exclude", "reason"])
    known = set(zip(ex.material_folder, ex.file))
    new = [{"material_folder": g, "file": f, "exclude": "N", "reason": ""}
           for g, f in all_ids if (g, f) not in known]
    if new:
        ex = pd.concat([ex, pd.DataFrame(new)], ignore_index=True)
        ex.to_csv(EXCLUSIONS_FILE, index=False)
    return {(r.material_folder, r.file): r.reason for r in ex.itertuples()
            if str(r.exclude).strip().upper().startswith("Y")}


def run():
    groups = find_groups()
    all_ids = [(d.name, f.name) for d, _ in groups for f in data_files(d)]
    excluded = load_exclusions(all_ids)
    out = RESULTS_FOLDER
    (out / "qc").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    rows, curves = [], {}
    print(f"Reading from {DATA_FOLDER}\n")
    for d, lab in groups:
        print(f"[{lab}]  ({d.name})")
        for k, f in enumerate(data_files(d), start=1):
            sid = f"{lab} #{k}"
            row = {"material": lab, "sample": k, "sample_id": sid,
                   "folder": d.name, "file": f.name, "status": "ok", "flags": ""}
            if (d.name, f.name) in excluded:
                row["status"] = f"excluded: {excluded[(d.name, f.name)]}"
                print(f"   #{k:<3} {f.name:<40} EXCLUDED ({excluded[(d.name, f.name)]})")
                rows.append(row)
                continue
            try:
                df, _ = read_bluehill(f)
                res, c = analyse(df)
                row.update(res)
                if np.isfinite(res["modulus_fit_R2"]) and res["modulus_fit_R2"] < 0.98:
                    row["flags"] += "low modulus-fit R2; "
                if res["strain_at_break_pct"] < MODULUS_RANGE_PCT[1]:
                    row["flags"] += "broke before end of modulus window; "
                safe = re.sub(r"[^\w\-]+", "_", f"{lab}_{k}_{f.stem}")
                qc_plot(c, res, f"{sid}  ({f.name})", out / "qc" / f"{safe}.png")
                curves.setdefault(lab, []).append((sid, c))
                print(f"   #{k:<3} {f.name:<40} ok   E={res['youngs_modulus_kPa']:6.1f} kPa  "
                      f"break={res['strain_at_break_pct']:5.0f} %")
            except Exception as e:
                row["status"] = f"error: {e}"
                print(f"   #{k:<3} {f.name:<40} ERROR: {e}")
            rows.append(row)
        print()

    res = pd.DataFrame(rows)
    ok = res[res.status == "ok"].copy()
    if ok.empty:
        sys.exit("No samples analysed successfully - see the errors above.")
    # flag early breaks for checking (not excluded automatically)
    med = ok.groupby("material").strain_at_break_pct.transform("median")
    early = ok.index[ok.strain_at_break_pct < 0.6 * med]
    res.loc[early, "flags"] += "early break - check grips; "
    f0 = ok.start_force_N.median()
    odd = ok.index[(ok.start_force_N - f0).abs() > 0.5 * abs(f0)]
    res.loc[odd, "flags"] += f"starting force differs from the others ({f0:.3f} N typical); "
    res.to_csv(out / "per_sample_results.csv", index=False)

    order = [lab for _, lab in groups if lab in curves]
    col = {lab: COLOURS[i % len(COLOURS)] for i, lab in enumerate(order)}
    metrics = {"youngs_modulus_kPa": "Young's modulus (kPa)",
               "uts_kPa": "Ultimate tensile strength (kPa)",
               "strain_at_break_pct": "Strain at break (%)",
               "stress_at_100pct_kPa": "Stress at 100% strain (kPa)"}

    # summary table
    summ = ok.groupby("material")[list(metrics)].agg(["mean", "std", "count"]).reindex(order)
    summ.columns = [f"{m}_{s}".replace("_std", "_SD").replace("_count", "_n") for m, s in summ.columns]
    summ.round(3).to_csv(out / "summary_by_material.csv")

    # aligned curves exported for Excel / Prism
    gmax = max(c["strain"].max() for v in curves.values() for _, c in v)
    grid = np.linspace(0, gmax, CURVE_POINTS)
    table = {"strain_pct": grid * 100}
    mean_table = {"strain_pct": grid * 100}
    for lab in order:
        _, ys = common_grid([c for _, c in curves[lab]], upto=gmax)
        for (sid, _), y in zip(curves[lab], ys):
            table[f"{sid} stress_kPa"] = y
        # mean only where every sample in the group still has data
        full = np.all(np.isfinite(ys), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_table[f"{lab} mean_kPa"] = np.where(full, np.nanmean(ys, 0), np.nan)
            mean_table[f"{lab} SD_kPa"] = np.where(full, np.nanstd(ys, 0, ddof=1) if len(ys) > 1 else 0, np.nan)
    pd.DataFrame(table).round(4).to_csv(out / "aligned_curves_all_samples.csv", index=False)
    pd.DataFrame(mean_table).round(4).to_csv(out / "aligned_curves_mean_SD.csv", index=False)

    # Figure 1: mean +/- SD per material (up to the shortest break in that material)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for lab in order:
        x, ys = common_grid([c for _, c in curves[lab]])
        m = ys.mean(0)
        sd = ys.std(0, ddof=1) if len(ys) > 1 else np.zeros_like(m)
        ax.plot(x * 100, m, color=col[lab], lw=2, label=f"{lab} (n={len(ys)})")
        ax.fill_between(x * 100, m - sd, m + sd, color=col[lab], alpha=0.2, lw=0)
    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("Stress (kPa)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    save(fig, out / "figures" / "fig_mean_SD_curves")

    # Figure 2: every sample, coloured by material, each to its own break
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for lab in order:
        for i, (_, c) in enumerate(curves[lab]):
            ax.plot(c["strain"] * 100, c["stress"], color=col[lab], lw=1, alpha=0.75,
                    label=lab if i == 0 else None)
            ax.plot(c["break"][0] * 100, c["break"][1], "x", color=col[lab], ms=5)
    ax.set_xlabel("Strain (%)")
    ax.set_ylabel("Stress (kPa)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    save(fig, out / "figures" / "fig_all_samples_overlay")

    # Figure 3: one panel per material, same axes
    n = len(order)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), sharex=True, sharey=True, squeeze=False)
    for ax, lab in zip(axes[0], order):
        for _, c in curves[lab]:
            ax.plot(c["strain"] * 100, c["stress"], color=col[lab], lw=1)
        ax.set_title(f"{lab} (n={len(curves[lab])})", fontsize=9)
        ax.set_xlabel("Strain (%)")
    axes[0][0].set_ylabel("Stress (kPa)")
    save(fig, out / "figures" / "fig_panels_per_material")

    # Bar charts
    rng = np.random.default_rng(0)
    for mkey, label in metrics.items():
        if ok[mkey].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(1.3 * n + 1.6, 3.6))
        for i, lab in enumerate(order):
            v = ok.loc[ok.material == lab, mkey].dropna().to_numpy()
            if len(v) == 0:
                continue
            ax.bar(i, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, color=col[lab],
                   alpha=0.75, width=0.6, capsize=4)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color="k", zorder=3)
        ax.set_xticks(range(n))
        ax.set_xticklabels(order, rotation=20 if n > 3 else 0, ha="right" if n > 3 else "center")
        ax.set_ylabel(label)
        save(fig, out / "figures" / f"bar_{mkey}")

    # statistics
    lines = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for mkey, label in metrics.items():
            data = {lab: ok.loc[ok.material == lab, mkey].dropna().to_numpy() for lab in order}
            data = {k: v for k, v in data.items() if len(v) >= 2}
            lines.append(f"=== {label} ===")
            for k, v in data.items():
                sw = stats.shapiro(v).pvalue if len(v) >= 3 else np.nan
                lines.append(f"  {k:<24} n={len(v)}  mean={v.mean():.4g}  SD={v.std(ddof=1):.4g}  "
                             f"normality (Shapiro) p={sw:.3f}")
            names, vals = list(data), list(data.values())
            if len(vals) == 2:
                t = stats.ttest_ind(*vals, equal_var=False)
                lines.append(f"  Welch t-test: t = {t.statistic:.3f}, p = {t.pvalue:.4g}")
            elif len(vals) >= 3:
                F, p = stats.f_oneway(*vals)
                lines.append(f"  Levene (equal variances) p = {stats.levene(*vals).pvalue:.4f}")
                lines.append(f"  One-way ANOVA: F = {F:.3f}, p = {p:.4g}")
                lines.append(f"  Kruskal-Wallis (non-parametric check): p = {stats.kruskal(*vals).pvalue:.4g}")
                tk = stats.tukey_hsd(*vals)
                lines.append("  Tukey HSD pairwise (* = p < 0.05):")
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        pv = tk.pvalue[i, j]
                        lines.append(f"    {names[i]} vs {names[j]}: p = {pv:.4g}{' *' if pv < 0.05 else ''}")
            else:
                lines.append("  needs at least two materials with n >= 2")
            lines.append("")
    (out / "statistics.txt").write_text("\n".join(lines))

    # settings used - for the methods section
    (out / "settings_used.txt").write_text(
        f"Smoothing: centred moving average, {SMOOTH_POINTS} points\n"
        f"Zero strain: start of the Bluehill test; starting force (median) {ok.start_force_N.median():.3f} N\n"
        f"Toe compensation: {'on' if TOE_COMPENSATION else 'off'} - straight part {TOE_STRESS_WINDOW_KPA[0]}-{TOE_STRESS_WINDOW_KPA[1]} kPa above start stress extrapolated back to start stress; strain counted from there\n"
        f"Young's modulus: linear fit, {MODULUS_RANGE_PCT[0]}-{MODULUS_RANGE_PCT[1]} % strain (after toe compensation)\n"
        f"Break: point of highest measured stress before the stress collapsed below {BREAK_DROP:.0%} of peak; UTS = that measured peak stress\n"
        f"Cross-section used by Bluehill (median): {ok.area_mm2_bluehill.median():.2f} mm2\n"
        f"Gauge length used by Bluehill (median): {ok.gauge_mm_bluehill.median():.2f} mm\n"
        f"Crosshead rate (median): {ok.rate_mm_min.median():.0f} mm/min\n")

    print("Samples analysed per material: " +
          ", ".join(f"{lab} = {len(curves[lab])}" for lab in order))
    flagged = res[res["flags"].astype(str).str.len() > 0]
    if len(flagged):
        print("Check these QC plots:")
        for r in flagged.itertuples():
            print(f"   {r.sample_id} ({r.file}): {r.flags.strip('; ')}")
    print(f"\nDone. Results in: {out}")
    print("Look at the qc/ plots first, then figures/, summary_by_material.csv and statistics.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Silicone tensile analysis")
    ap.add_argument("--check", action="store_true", help="list folders/files and test-read one file")
    a = ap.parse_args()
    check() if a.check else run()
