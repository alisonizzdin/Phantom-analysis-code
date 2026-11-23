#!/usr/bin/env python3
"""
WAX 3-POINT BEND ANALYSIS (Instron / Bluehill exports)

For every wax bar, the same way:
  1. reads the Bluehill file (Time, Displacement, Force, Flexure displacement,
     Flexure strain, Flexure stress)
  2. flexural modulus  = linear fit between 0.05 % and 0.25 % flexural strain (ISO 178 range)
  3. flexural strength = highest measured stress before the bar breaks
  4. finds the fracture point (sudden drop in force) - or reports "no fracture" if the bar
     only bent and never snapped
  5. deflection at fracture and work to fracture (energy needed to snap it)
Then for each wax: mean +/- SD curves, bar charts and statistics.

FOLDERS (keep your own names):
  wax_bend.py
  raw/
    wax_bend/
      Beeswax/            <- every bar tested for this wax
      Paraffin/
      ...

RUN (VS Code terminal, in the folder that holds this script):
  python wax_bend.py --check
  python wax_bend.py
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
# SETTINGS
# =============================================================================
HERE = Path(__file__).resolve().parent
DATA_FOLDER = HERE / "raw" / "wax_bend"
RESULTS_FOLDER = HERE / "results_wax_bend"
EXCLUSIONS_FILE = HERE / "wax_bend_exclusions.csv"

# Optional: folder name -> label for figures. Order here = order in figures.
# Folders not listed keep their own name and are added at the end.
LABELS = {
    # "Beeswax": "Beeswax",
    # "25% paraffin 75% beeswax": "P25:B75",
}

MODULUS_RANGE_PCT = (0.05, 0.25)   # flexural strain window (%) for the modulus (ISO 178)
FRACTURE_DROP = 0.5                # fracture = force falls below 50 % of its maximum
SMOOTH_POINTS = 5                  # light smoothing for plotted curves only (1 = none)
CURVE_POINTS = 400
FIG_DPI = 300

COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
           "#000000", "#999933", "#882255", "#44AA99"]
plt.rcParams.update({"font.size": 10, "font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

# =============================================================================
# Reading
# =============================================================================


def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(p).name)]


def data_files(folder):
    return sorted([f for f in Path(folder).iterdir()
                   if f.is_file() and f.suffix.lower() in (".csv", ".txt")], key=natural_key)


def read_bluehill(path):
    text = None
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = Path(path).read_text(encoding=enc)
            if "\x00" not in text:
                break
        except (UnicodeDecodeError, UnicodeError):
            continue
    lines = text.splitlines()
    delim = max([",", ";", "\t"], key="\n".join(lines[:60]).count)
    rows = list(csv.reader(lines, delimiter=delim))
    hdr_i = next((i for i, r in enumerate(rows)
                  if any("force" in c.lower() or "load" in c.lower() for c in r)), None)
    if hdr_i is None:
        raise ValueError("no header row with a Force/Load column found")
    header = [c.strip() for c in rows[hdr_i]]
    units, data = {}, []
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
            if all(k in lc for k in keys) and not any(a in lc for a in avoid):
                return c
        return None

    c_time = col("time")
    c_force = col("force") or col("load")
    c_defl = col("flexure", "displacement", avoid=("strain",)) or col("displacement", avoid=("strain",))
    c_strain = col("strain")
    c_stress = col("stress")
    missing = [n for n, c in [("Force", c_force), ("Flexure displacement", c_defl),
                              ("Flexure strain", c_strain), ("Flexure stress", c_stress)] if c is None]
    if missing:
        raise ValueError(f"column(s) not found: {missing}. Columns in file: {list(df.columns)}")

    def unit(c):
        return units.get(c, "").lower()

    out = pd.DataFrame({
        "time_s": df[c_time] if c_time else np.arange(len(df)) * 0.02,
        "defl_mm": df[c_defl],
        "force_N": df[c_force] * (1000.0 if "kn" in unit(c_force) else 1.0),
        "strain": df[c_strain] * (0.01 if ("%" in unit(c_strain) or unit(c_strain) == "") else 1.0),
        "stress_MPa": df[c_stress] * (1.0 if ("mpa" in unit(c_stress) or unit(c_stress) == "")
                                      else 0.001 if "kpa" in unit(c_stress) else 1e-6),
    }).dropna().reset_index(drop=True)
    if out.force_N.max() < abs(out.force_N.min()):
        out[["force_N", "strain", "stress_MPa"]] *= -1
    info = {"columns": list(df.columns),
            "used": {"time": c_time, "force": c_force, "deflection": c_defl,
                     "strain": c_strain, "stress": c_stress}}
    return out, info

# =============================================================================
# One bar
# =============================================================================


def smooth(y):
    if SMOOTH_POINTS <= 1:
        return np.asarray(y, float)
    return pd.Series(y).rolling(SMOOTH_POINTS, center=True, min_periods=1).mean().to_numpy()


def analyse(df):
    F = df.force_N.to_numpy()
    s = df.stress_MPa.to_numpy()
    e = df.strain.to_numpy() - df.strain.iloc[0]
    d = df.defl_mm.to_numpy() - df.defl_mm.iloc[0]

    imax = int(np.argmax(F))
    if imax < 5:
        raise ValueError("force peaks at the very start - check this file")
    drop = np.where(F[imax:] < FRACTURE_DROP * F[imax])[0]
    fractured = len(drop) > 0
    collapse = imax + drop[0] if fractured else len(F)
    ib = int(np.argmax(F[:collapse]))           # fracture onset = highest force before collapse

    lo, hi = MODULUS_RANGE_PCT[0] / 100, MODULUS_RANGE_PCT[1] / 100
    w = (e[:ib + 1] >= lo) & (e[:ib + 1] <= hi)
    if w.sum() >= 4:
        fit = stats.linregress(e[:ib + 1][w], s[:ib + 1][w])
        Ef, b0, r2 = fit.slope, fit.intercept, fit.rvalue ** 2
    else:
        Ef = b0 = r2 = np.nan

    work_mJ = float(np.trapezoid(F[:ib + 1], d[:ib + 1])) if hasattr(np, "trapezoid") \
        else float(np.trapz(F[:ib + 1], d[:ib + 1]))
    mm = d > 0.05
    span_info = {}
    if mm.sum() > 5:
        k1 = np.median(s[mm] / F[mm])            # = 3L / (2 b h^2)
        k2 = np.median(e[mm] / d[mm])            # = 6 h / L^2
        span_info = {"stress_per_N": k1, "strain_per_mm": k2}
    t = df.time_s.to_numpy()
    rate = float(np.polyfit(t[ib // 4: ib + 1], df.defl_mm.to_numpy()[ib // 4: ib + 1], 1)[0] * 60) \
        if ib > 8 else np.nan

    res = {"flexural_modulus_MPa": Ef, "modulus_fit_R2": r2,
           "flexural_strength_MPa": float(s[ib]), "max_force_N": float(F[ib]),
           "strain_at_max_pct": float(e[ib] * 100), "deflection_at_max_mm": float(d[ib]),
           "work_to_fracture_mJ": work_mJ, "fractured": "yes" if fractured else "no",
           "rate_mm_min": rate, **span_info}
    curve = {"strain": e[:ib + 1], "stress": smooth(s[:ib + 1]), "fit": (Ef, b0),
             "raw_strain": e, "raw_stress": s, "break": (e[ib], s[ib]), "fractured": fractured}
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
    ax.plot(c["raw_strain"] * 100, c["raw_stress"], ".-", color="0.6", lw=0.8, ms=2, label="raw")
    ax.plot(c["strain"] * 100, c["stress"], color=COLOURS[0], lw=1.6, label="analysed")
    lo, hi = MODULUS_RANGE_PCT
    ax.axvspan(lo, hi, color="red", alpha=0.08)
    E, b = c["fit"]
    if np.isfinite(E):
        x = np.array([0, min(hi * 2.5, c["strain"][-1] * 100)]) / 100
        ax.plot(x * 100, E * x + b, "r--", lw=1, label=f"E = {E:.0f} MPa (R² {r['modulus_fit_R2']:.3f})")
    bx, by = c["break"]
    ax.plot(bx * 100, by, "kx", ms=9, mew=2,
            label="fracture" if c["fractured"] else "max (no fracture)")
    ax.set_xlabel("Flexural strain (%)")
    ax.set_ylabel("Flexural stress (MPa)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def common_grid(curves):
    smax = min(c["strain"].max() for c in curves)
    grid = np.linspace(0, smax, CURVE_POINTS)
    ys = []
    for c in curves:
        x = np.maximum.accumulate(c["strain"])
        x, idx = np.unique(x, return_index=True)
        ys.append(np.interp(grid, x, c["stress"][idx]))
    return grid, np.array(ys)

# =============================================================================
# Main
# =============================================================================


def find_groups():
    if not DATA_FOLDER.is_dir():
        sys.exit(f"Folder not found: {DATA_FOLDER}\n"
                 "Inside 'raw' make a folder 'wax_bend', and inside that one folder per wax.")
    folders = [d for d in DATA_FOLDER.iterdir() if d.is_dir()]
    if not folders:
        sys.exit(f"No wax folders inside {DATA_FOLDER} (each wax needs its own sub-folder).")
    order = list(dict.fromkeys(LABELS.values()))

    def rank(d):
        lab = LABELS.get(d.name.strip(), d.name.strip())
        return (order.index(lab) if lab in order else len(order), natural_key(d.name))
    return [(d, LABELS.get(d.name.strip(), d.name.strip())) for d in sorted(folders, key=rank)]


def check():
    groups = find_groups()
    print(f"Looking in: {DATA_FOLDER}\n")
    first = None
    for d, lab in groups:
        fs = data_files(d)
        print(f"  {d.name:<32} -> label '{lab}'   {len(fs)} file(s)")
        first = first or (fs[0] if fs else None)
    if first is None:
        sys.exit("\nNo .csv or .txt files found.")
    print(f"\nTest-reading {first.name} ...")
    df, info = read_bluehill(first)
    print(f"  using: {info['used']}")
    res, _ = analyse(df)
    print(f"  {len(df)} rows; modulus {res['flexural_modulus_MPa']:.0f} MPa, strength "
          f"{res['flexural_strength_MPa']:.2f} MPa, fractured: {res['fractured']}")
    if "stress_per_N" in res:
        k1, k2 = res["stress_per_N"], res["strain_per_mm"]
        print("  Bluehill geometry consistent with, e.g.:")
        for L in (40, 50, 60, 64, 80):
            h = k2 * L * L / 6
            b = 3 * L / (2 * k1 * h * h)
            print(f"     span {L} mm -> bar {b:.2f} mm wide x {h:.2f} mm thick")
    print("\nIf that looks right, run:  python wax_bend.py")


def load_exclusions(all_ids):
    if EXCLUSIONS_FILE.exists():
        ex = pd.read_csv(EXCLUSIONS_FILE, dtype=str, keep_default_na=False)
    else:
        ex = pd.DataFrame(columns=["wax_folder", "file", "exclude", "reason"])
    known = set(zip(ex.wax_folder, ex.file))
    new = [{"wax_folder": g, "file": f, "exclude": "N", "reason": ""}
           for g, f in all_ids if (g, f) not in known]
    if new:
        ex = pd.concat([ex, pd.DataFrame(new)], ignore_index=True)
        ex.to_csv(EXCLUSIONS_FILE, index=False)
    return {(r.wax_folder, r.file): r.reason for r in ex.itertuples()
            if str(r.exclude).strip().upper().startswith("Y")}


def run():
    groups = find_groups()
    excluded = load_exclusions([(d.name, f.name) for d, _ in groups for f in data_files(d)])
    out = RESULTS_FOLDER
    (out / "qc").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    rows, curves = [], {}
    print(f"Reading from {DATA_FOLDER}\n")
    for d, lab in groups:
        print(f"[{lab}]")
        for k, f in enumerate(data_files(d), start=1):
            row = {"wax": lab, "sample": k, "sample_id": f"{lab} #{k}", "folder": d.name,
                   "file": f.name, "status": "ok", "flags": ""}
            if (d.name, f.name) in excluded:
                row["status"] = f"excluded: {excluded[(d.name, f.name)]}"
                print(f"   #{k:<3} {f.name:<38} EXCLUDED ({excluded[(d.name, f.name)]})")
                rows.append(row)
                continue
            try:
                df, _ = read_bluehill(f)
                res, c = analyse(df)
                row.update(res)
                if res["fractured"] == "no":
                    row["flags"] += "did not fracture (bent only); "
                if np.isfinite(res["modulus_fit_R2"]) and res["modulus_fit_R2"] < 0.98:
                    row["flags"] += "low modulus-fit R2; "
                safe = re.sub(r"[^\w\-]+", "_", f"{lab}_{k}_{f.stem}")
                qc_plot(c, res, f"{row['sample_id']}  ({f.name})", out / "qc" / f"{safe}.png")
                curves.setdefault(lab, []).append(c)
                print(f"   #{k:<3} {f.name:<38} ok   E={res['flexural_modulus_MPa']:7.1f} MPa  "
                      f"strength={res['flexural_strength_MPa']:6.3f} MPa  fractured={res['fractured']}")
            except Exception as e:
                row["status"] = f"error: {e}"
                print(f"   #{k:<3} {f.name:<38} ERROR: {e}")
            rows.append(row)
        print()

    res = pd.DataFrame(rows)
    ok = res[res.status == "ok"].copy()
    if ok.empty:
        sys.exit("No bars analysed successfully - see the errors above.")
    res.to_csv(out / "per_sample_results.csv", index=False)

    order = [lab for _, lab in groups if lab in curves]
    col = {lab: COLOURS[i % len(COLOURS)] for i, lab in enumerate(order)}
    metrics = {"flexural_modulus_MPa": "Flexural modulus (MPa)",
               "flexural_strength_MPa": "Flexural strength (MPa)",
               "max_force_N": "Force at fracture (N)",
               "strain_at_max_pct": "Flexural strain at fracture (%)",
               "deflection_at_max_mm": "Deflection at fracture (mm)",
               "work_to_fracture_mJ": "Work to fracture (mJ)"}

    summ = ok.groupby("wax")[list(metrics)].agg(["mean", "std", "count"]).reindex(order)
    summ.columns = [f"{m}_{s}".replace("_std", "_SD").replace("_count", "_n") for m, s in summ.columns]
    summ.insert(0, "fractured", ok.groupby("wax").fractured.apply(
        lambda v: f"{(v == 'yes').sum()}/{len(v)}").reindex(order))
    summ.round(4).to_csv(out / "summary_by_wax.csv")

    # mean +/- SD curves
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for lab in order:
        x, ys = common_grid(curves[lab])
        m = ys.mean(0)
        sd = ys.std(0, ddof=1) if len(ys) > 1 else np.zeros_like(m)
        ax.plot(x * 100, m, color=col[lab], lw=2, label=f"{lab} (n={len(ys)})")
        ax.fill_between(x * 100, m - sd, m + sd, color=col[lab], alpha=0.2, lw=0)
    ax.set_xlabel("Flexural strain (%)")
    ax.set_ylabel("Flexural stress (MPa)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    save(fig, out / "figures" / "fig_mean_SD_curves")

    # every bar, coloured by wax
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for lab in order:
        for i, c in enumerate(curves[lab]):
            ax.plot(c["strain"] * 100, c["stress"], color=col[lab], lw=1, alpha=0.8,
                    label=lab if i == 0 else None)
            ax.plot(c["break"][0] * 100, c["break"][1], "x", color=col[lab], ms=5)
    ax.set_xlabel("Flexural strain (%)")
    ax.set_ylabel("Flexural stress (MPa)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    save(fig, out / "figures" / "fig_all_samples_overlay")

    # bar charts
    n = len(order)
    rng = np.random.default_rng(0)
    for mkey, label in metrics.items():
        if ok[mkey].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(0.9 * n + 1.8, 3.8))
        for i, lab in enumerate(order):
            v = ok.loc[ok.wax == lab, mkey].dropna().to_numpy()
            if len(v) == 0:
                continue
            ax.bar(i, v.mean(), yerr=v.std(ddof=1) if len(v) > 1 else 0, color=col[lab],
                   alpha=0.75, width=0.65, capsize=4)
            ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=12, color="k", zorder=3)
        ax.set_xticks(range(n))
        ax.set_xticklabels(order, rotation=35 if n > 3 else 0, ha="right" if n > 3 else "center")
        ax.set_ylabel(label)
        save(fig, out / "figures" / f"bar_{mkey}")

    # statistics
    lines = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for mkey, label in metrics.items():
            data = {lab: ok.loc[ok.wax == lab, mkey].dropna().to_numpy() for lab in order}
            data = {k: v for k, v in data.items() if len(v) >= 2}
            lines.append(f"=== {label} ===")
            for k, v in data.items():
                sw = stats.shapiro(v).pvalue if len(v) >= 3 else np.nan
                lines.append(f"  {k:<28} n={len(v)}  mean={v.mean():.4g}  SD={v.std(ddof=1):.4g}  "
                             f"normality (Shapiro) p={sw:.3f}")
            names, vals = list(data), list(data.values())
            if len(vals) == 2:
                t = stats.ttest_ind(*vals, equal_var=False)
                lines.append(f"  Welch t-test: t = {t.statistic:.3f}, p = {t.pvalue:.4g}")
            elif len(vals) >= 3:
                F, p = stats.f_oneway(*vals)
                dfw = sum(len(v) for v in vals) - len(vals)
                lines.append(f"  Levene (equal variances) p = {stats.levene(*vals).pvalue:.4f}")
                lines.append(f"  One-way ANOVA: F({len(vals) - 1}, {dfw}) = {F:.3f}, p = {p:.4g}")
                lines.append(f"  Kruskal-Wallis (non-parametric check): p = {stats.kruskal(*vals).pvalue:.4g}")
                tk = stats.tukey_hsd(*vals)
                lines.append("  Tukey HSD pairwise (* = p < 0.05):")
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        pv = tk.pvalue[i, j]
                        lines.append(f"    {names[i]} vs {names[j]}: p = {pv:.4g}{' *' if pv < 0.05 else ''}")
            else:
                lines.append("  needs at least two waxes with n >= 2")
            lines.append("")
    (out / "statistics.txt").write_text("\n".join(lines))

    k1, k2 = ok.get("stress_per_N"), ok.get("strain_per_mm")
    geo = ""
    if k1 is not None and k1.notna().any():
        k1m, k2m = k1.median(), k2.median()
        geo = "Bluehill geometry consistent with: " + "; ".join(
            f"span {L} mm -> {3 * L / (2 * k1m * (k2m * L * L / 6) ** 2):.2f} x {k2m * L * L / 6:.2f} mm"
            for L in (50, 60, 64)) + "\n"
    (out / "settings_used.txt").write_text(
        f"Flexural modulus: linear fit, {MODULUS_RANGE_PCT[0]}-{MODULUS_RANGE_PCT[1]} % flexural strain (ISO 178 range)\n"
        f"Fracture: highest measured force before force fell below {FRACTURE_DROP:.0%} of maximum\n"
        f"Flexural strength: measured stress at that point\n"
        f"Work to fracture: area under force-deflection curve up to fracture\n"
        f"Crosshead rate (median): {ok.rate_mm_min.median():.1f} mm/min\n" + geo)

    print("Bars analysed per wax: " + ", ".join(f"{lab} = {len(curves[lab])}" for lab in order))
    flagged = res[res["flags"].astype(str).str.len() > 0]
    if len(flagged):
        print("Check these QC plots:")
        for r in flagged.itertuples():
            print(f"   {r.sample_id} ({r.file}): {r.flags.strip('; ')}")
    print(f"\nDone. Results in: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Wax 3-point bend analysis")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    check() if a.check else run()
    