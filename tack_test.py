#!/usr/bin/env python3
#!/usr/bin/env python3
"""
TACK (ADHESION) TEST ANALYSIS - silicone-coated wax samples vs plastic control

Each Bluehill file holds the PULL-OFF part of one test: it starts with the probe pressed
into the sample (compressive force), then the crosshead lifts away. If the surface is
tacky, the force goes briefly POSITIVE (the surface holds on to the probe) before
returning to zero. For every test the script measures:
  - peak adhesion force  = highest pull-off force above the baseline (N)
  - work of adhesion     = area under the positive pull-off force vs displacement (mJ = N.mm)
  - detachment distance  = how far the probe travelled while still held (mm)
  - starting compression = how hard the probe was pressed in (should be similar for all tests)
  - noise level, and whether the adhesion is measurable above it (peak > 3 x noise)
Then: comparison with the plastic control within each batch (Dunnett), two-way ANOVA
(surface x batch), and figures.

FOLDERS (keep your own names):
  tack_test.py
  raw/
    tackiness test/
      A-1st-batch/    A-2nd-batch/    A-3rd-batch/      <- 5 files each
      ...
      H-3rd-batch/
      z-Control-1st-batch/  z-control-2nd-batch/  z-control-3rd-batch/
The wax letter is read from the start of the folder name, the batch from "1st/2nd/3rd",
and any folder containing "control" is the plastic control. Folders without a batch
number (e.g. "test no paper") are skipped and listed.

RUN (VS Code terminal):
  python tack_test.py --check
  python tack_test.py
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
# SETTINGS - edit these
# =============================================================================
HERE = Path(__file__).resolve().parent
DATA_CANDIDATES = [HERE / "raw" / "tackiness test", HERE / "raw" / "tack_test", HERE / "raw" / "tack"]
RESULTS_FOLDER = HERE / "results_tack"
EXCLUSIONS_FILE = HERE / "tack_exclusions.csv"

# >>> FILL IN: which wax each letter is. These names appear in every table and figure. <<<
WAX_NAMES = {
    "A": "Wax A",
    "B": "Wax B",
    "C": "Wax C",
    "D": "Wax D",
    "E": "Wax E",
    "F": "Wax F",
    "G": "Wax G",
    "H": "Wax H",
}
CONTROL_NAME = "Plastic (control)"

# When each batch was coated, in minutes after mixing started (from the lab timings):
#   mixing 2:35, vacuum 2:27, wait 0:36, batch 1 coating 6:51, wait 16:24,
#   batch 2 coating 7:58, wait 7:10.46, batch 3 coating 6:46
BATCH_WINDOW_MIN = {1: (5.63, 12.48), 2: (28.88, 36.85), 3: (44.02, 50.79)}

SMOOTH_POINTS = 5            # light moving average before finding the peak (1 = none)
BASELINE_FRACTION = 0.25     # last 25 % of the record (probe clear of the surface) = zero-force baseline
MEASURABLE_FACTOR = 3.0      # adhesion counts as measurable if peak > 3 x baseline noise
FIG_DPI = 300

BATCH_SHADES = [1.0, 0.65, 0.35]            # batch 1, 2, 3 = dark -> light of one colour
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

    def col(*keys):
        return next((c for c in df.columns if any(k in c.lower() for k in keys)), None)

    c_t, c_x, c_f = col("time"), col("displacement", "extension", "position"), col("force", "load")
    if c_x is None or c_f is None:
        raise ValueError(f"need Displacement and Force columns; found {list(df.columns)}")
    kn = "kn" in units.get(c_f, "").lower()
    out = pd.DataFrame({"time_s": df[c_t] if c_t else np.arange(len(df)) * 0.02,
                        "disp_mm": df[c_x],
                        "force_N": df[c_f] * (1000.0 if kn else 1.0)}).dropna().reset_index(drop=True)
    return out, {"columns": list(df.columns), "used": {"time": c_t, "displacement": c_x, "force": c_f}}

# =============================================================================
# One test
# =============================================================================


def smooth(y):
    if SMOOTH_POINTS <= 1:
        return np.asarray(y, float)
    return pd.Series(y).rolling(SMOOTH_POINTS, center=True, min_periods=1).mean().to_numpy()


def analyse(df):
    F = df.force_N.to_numpy()
    x = df.disp_mm.to_numpy()
    if len(F) < 50:
        raise ValueError("too few data points")
    # compression sign: the probe starts pressed in, so the largest |force| is compressive
    sign = -1.0 if abs(F.min()) >= abs(F.max()) else 1.0      # sign of compression
    Fa = -sign * F                                             # adhesion (pull) positive
    if x[-1] < x[0]:
        x = -x                                                 # lifting = increasing displacement

    n = len(Fa)
    tail = Fa[int(n * (1 - BASELINE_FRACTION)):]
    base = float(np.median(tail))
    noise = float(np.std(tail, ddof=1))
    Fs = smooth(Fa) - base
    Fr = Fa - base

    ic = int(np.argmin(Fs[: max(5, n // 2)]))                  # deepest compression
    start_comp = float(-Fs[ic])
    after = np.where(Fs[ic:] >= 0)[0]
    if not len(after):
        raise ValueError("force never returns to zero after compression")
    i0 = ic + int(after[0])                                    # compression released
    ipk = i0 + int(np.argmax(Fs[i0:]))
    peak = float(max(Fs[ipk], 0.0))
    measurable = peak > MEASURABLE_FACTOR * noise

    # end of contact: force falls back to the noise band after the peak
    lim = max(MEASURABLE_FACTOR * noise, 1e-6)
    back = np.where(Fs[ipk:] <= lim)[0]
    iend = ipk + int(back[0]) if len(back) else n - 1
    seg = slice(i0, iend + 1)
    pos = np.clip(Fr[seg], 0, None)
    trap = getattr(np, "trapezoid", None) or np.trapz
    work = float(trap(pos, x[seg])) if measurable else 0.0
    detach = float(x[iend] - x[i0]) if measurable else 0.0
    rate = float(np.polyfit(df.time_s.to_numpy(), x, 1)[0] * 60) if n > 3 else np.nan

    res = {"peak_adhesion_N": peak if measurable else 0.0,
           "peak_adhesion_raw_N": peak,
           "work_of_adhesion_mJ": work,
           "detachment_distance_mm": detach,
           "measurable": "yes" if measurable else "no",
           "noise_N": noise, "start_compression_N": start_comp,
           "pull_rate_mm_min": rate}
    curve = {"x": x - x[i0], "F": Fr, "Fs": Fs, "i0": i0, "ipk": ipk, "iend": iend,
             "noise": noise, "measurable": measurable}
    return res, curve

# =============================================================================
# Plots
# =============================================================================


def qc_plot(c, r, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    x, F = c["x"], c["F"]
    ax = axes[0]
    ax.plot(x, F, color="0.5", lw=0.8)
    ax.axhline(0, color="0.7", lw=0.7)
    ax.set_xlabel("Displacement from release (mm)")
    ax.set_ylabel("Force (N), pull = +")
    ax.set_title("whole pull-off", fontsize=9)
    ax = axes[1]
    w = slice(max(c["i0"] - 20, 0), min(c["iend"] + 60, len(x)))
    ax.plot(x[w], F[w], ".", color="0.6", ms=2, label="raw")
    ax.plot(x[w], c["Fs"][w], color="#0072B2", lw=1.4, label="smoothed")
    ax.axhspan(-3 * c["noise"], 3 * c["noise"], color="0.85", alpha=0.6, lw=0, label="noise band (±3 SD)")
    if c["measurable"]:
        s = slice(c["i0"], c["iend"] + 1)
        ax.fill_between(x[s], 0, np.clip(F[s], 0, None), color="#0072B2", alpha=0.2, label="work of adhesion")
        ax.plot(x[c["ipk"]], c["Fs"][c["ipk"]], "x", color="k", ms=9, mew=2,
                label=f"peak {r['peak_adhesion_N'] * 1000:.1f} mN")
    else:
        ax.text(0.5, 0.9, "no measurable adhesion", transform=ax.transAxes, ha="center", fontsize=9)
    ax.axhline(0, color="0.5", lw=0.7)
    lo = min(-4 * c["noise"], np.nanmin(c["Fs"][w]) if c["measurable"] else -4 * c["noise"])
    hi = max(c["Fs"][c["ipk"]] * 1.4, 5 * c["noise"])
    ax.set_ylim(max(lo, -hi), hi)
    ax.set_xlabel("Displacement from release (mm)")
    ax.set_title("zoom: adhesion", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(f"{path}.png", dpi=FIG_DPI)
    fig.savefig(f"{path}.pdf")
    plt.close(fig)

# =============================================================================
# Folders
# =============================================================================
BATCH_RE = re.compile(r"\b([123])\s*(?:st|nd|rd)\b", re.I)


def parse_folder(name):
    n = name.strip()
    m = BATCH_RE.search(n)
    if not m:
        return None
    batch = int(m.group(1))
    if "control" in n.lower():
        return "Control", CONTROL_NAME, batch
    letter = re.match(r"\s*([A-Za-z])\b|\s*([A-Za-z])-", n)
    key = (letter.group(1) or letter.group(2)).upper() if letter else n
    return key, WAX_NAMES.get(key, f"Wax {key}"), batch


def find_root():
    for c in DATA_CANDIDATES:
        if c.is_dir():
            return c
    sys.exit("Tack data folder not found. Expected  raw/tackiness test/  with one folder per wax and batch.")


def find_groups():
    root = find_root()
    groups, skipped = [], []
    for d in sorted([d for d in root.iterdir() if d.is_dir()], key=natural_key):
        p = parse_folder(d.name)
        if p is None:
            skipped.append(d.name)
            continue
        key, label, batch = p
        groups.append({"folder": d, "key": key, "surface": label, "batch": batch})
    order = ["Control"] + sorted({g["key"] for g in groups if g["key"] != "Control"})
    groups.sort(key=lambda g: (order.index(g["key"]), g["batch"]))
    return root, groups, skipped, order


def batch_label(b):
    lo, hi = BATCH_WINDOW_MIN.get(b, (np.nan, np.nan))
    return f"Batch {b} ({lo:.0f}–{hi:.0f} min)" if np.isfinite(lo) else f"Batch {b}"

# =============================================================================
# Check / run
# =============================================================================


def check():
    root, groups, skipped, order = find_groups()
    print(f"Looking in: {root}\n")
    first = None
    for g in groups:
        fs = data_files(g["folder"])
        print(f"  {g['folder'].name:<24} -> {g['surface']:<20} batch {g['batch']}   {len(fs)} file(s)")
        first = first or (fs[0] if fs else None)
    if skipped:
        print(f"\nSkipped (no batch number in the name): {', '.join(skipped)}")
    surfaces = [k for k in order]
    missing = [f"{k} batch {b}" for k in surfaces for b in (1, 2, 3)
               if not any(g["key"] == k and g["batch"] == b for g in groups)]
    if missing:
        print(f"Missing combinations: {', '.join(missing)}")
    unnamed = [k for k in order if k != "Control" and WAX_NAMES.get(k, "").startswith("Wax ")]
    if unnamed:
        print(f"\nReminder: fill in the real wax names for {', '.join(unnamed)} in WAX_NAMES at the top of the script.")
    if first is None:
        sys.exit("\nNo .csv/.txt files found.")
    print(f"\nTest-reading {first.name} ...")
    df, info = read_bluehill(first)
    r, _ = analyse(df)
    print(f"  columns used: {info['used']}")
    print(f"  start compression {r['start_compression_N']:.2f} N, peak adhesion {r['peak_adhesion_raw_N'] * 1000:.1f} mN, "
          f"noise {r['noise_N'] * 1000:.1f} mN -> measurable: {r['measurable']}")
    print("\nIf that looks right, run:  python tack_test.py")


def load_exclusions(ids):
    cols = ["folder", "file", "exclude", "reason"]
    if EXCLUSIONS_FILE.exists():
        # sep=None: accepts commas, semicolons or tabs (Excel/Numbers in some regions save with ';')
        ex = pd.read_csv(EXCLUSIONS_FILE, dtype=str, keep_default_na=False, sep=None, engine="python")
        ex.columns = [c.strip().lower() for c in ex.columns]
        missing = [c for c in cols if c not in ex.columns]
        if missing:
            sys.exit(f"{EXCLUSIONS_FILE.name} is missing column(s) {missing}. Columns found: {list(ex.columns)}.\n"
                     "Save it as CSV with the columns: folder, file, exclude, reason.")
        for c in cols:
            ex[c] = ex[c].astype(str).str.strip()
    else:
        ex = pd.DataFrame(columns=cols)
    known = set(zip(ex.folder, ex.file))
    new = [{"folder": a, "file": b, "exclude": "N", "reason": ""} for a, b in ids if (a, b) not in known]
    if new:
        ex = pd.concat([ex, pd.DataFrame(new)], ignore_index=True)
        ex.to_csv(EXCLUSIONS_FILE, index=False)
    marked = {(r.folder, r.file): r.reason for r in ex.itertuples() if str(r.exclude).strip().upper().startswith("Y")}
    present = set(ids)
    applied = {k: v for k, v in marked.items() if k in present}
    unmatched = [k for k in marked if k not in present]
    print(f"Exclusions: {len(applied)} applied from {EXCLUSIONS_FILE.name}")
    if unmatched:
        print("WARNING - marked Y but no such file now (renamed or moved?) - these were NOT applied:")
        for f, n in unmatched:
            print(f"   {f} / {n}")
        print("   -> put the Y on the row with the file's current name (new rows were added at the bottom).")
    print()
    return applied


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def run():
    root, groups, skipped, order = find_groups()
    excluded = load_exclusions([(g["folder"].name, f.name) for g in groups for f in data_files(g["folder"])])
    out = RESULTS_FOLDER
    (out / "qc").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    if skipped:
        print(f"Skipped folders (no batch number): {', '.join(skipped)}\n")

    rows = []
    for g in groups:
        print(f"[{g['surface']} - batch {g['batch']}]  ({g['folder'].name})")
        for k, f in enumerate(data_files(g["folder"]), start=1):
            row = {"surface": g["surface"], "key": g["key"], "batch": g["batch"], "sample": k,
                   "sample_id": f"{g['surface']} B{g['batch']} #{k}", "folder": g["folder"].name,
                   "file": f.name, "status": "ok"}
            if (g["folder"].name, f.name) in excluded:
                row["status"] = f"excluded: {excluded[(g['folder'].name, f.name)]}"
                print(f"   #{k:<3} {f.name:<28} EXCLUDED")
                rows.append(row)
                continue
            try:
                df, _ = read_bluehill(f)
                r, c = analyse(df)
                row.update(r)
                safe = re.sub(r"[^\w\-]+", "_", f"{g['key']}_B{g['batch']}_{k}_{f.stem}")
                qc_plot(c, r, f"{row['sample_id']}  ({f.name})", out / "qc" / f"{safe}.png")
                print(f"   #{k:<3} {f.name:<28} ok   peak={r['peak_adhesion_raw_N'] * 1000:6.1f} mN  "
                      f"work={r['work_of_adhesion_mJ'] * 1000:7.2f} µJ  measurable={r['measurable']}")
            except Exception as e:
                row["status"] = f"error: {e}"
                print(f"   #{k:<3} {f.name:<28} ERROR: {e}")
            rows.append(row)
        print()

    res = pd.DataFrame(rows)
    res.to_csv(out / "per_sample_results.csv", index=False)
    ok = res[res.status == "ok"].copy()
    if ok.empty:
        sys.exit("No tests analysed successfully - see the errors above.")
    ok["peak_adhesion_mN"] = ok.peak_adhesion_N * 1000
    ok["work_of_adhesion_uJ"] = ok.work_of_adhesion_mJ * 1000
    ok["measurable_flag"] = (ok.measurable == "yes").astype(int)

    labels = {k: (CONTROL_NAME if k == "Control" else WAX_NAMES.get(k, f"Wax {k}")) for k in order}
    surf_order = [labels[k] for k in order if labels[k] in set(ok.surface)]
    batches = sorted(ok.batch.unique())
    metrics = {"peak_adhesion_mN": "Peak adhesion force (mN)",
               "work_of_adhesion_uJ": "Work of adhesion (µJ)",
               "detachment_distance_mm": "Detachment distance (mm)"}

    # summary table
    summ = ok.groupby(["surface", "batch"]).agg(
        n=("sample", "count"),
        measurable=("measurable_flag", lambda v: f"{int(v.sum())}/{len(v)}"),
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_SD": (m, "std") for m in metrics},
        start_compression_N_mean=("start_compression_N", "mean"))
    summ = summ.reindex([(s, b) for s in surf_order for b in batches if (s, b) in summ.index])
    summ.round(4).to_csv(out / "summary_by_surface_and_batch.csv")

    # statistics
    lines = [f"Batches = time the coating was applied after mixing: " +
             "; ".join(batch_label(b) for b in batches), ""]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m, label in metrics.items():
            lines.append(f"=== {label} ===")
            # two-way ANOVA surface x batch
            try:
                import statsmodels.formula.api as smf
                from statsmodels.stats.anova import anova_lm
                d = ok[["surface", "batch", m]].dropna().rename(columns={m: "y"})
                tab = anova_lm(smf.ols("y ~ C(surface) * C(batch)", data=d).fit(), typ=2)
                rdf = tab.loc["Residual", "df"]
                lines.append("  Two-way ANOVA (surface x batch):")
                for idx, r in tab.iterrows():
                    if idx != "Residual":
                        nm = idx.replace("C(", "").replace(")", "").replace(":", " x ")
                        lines.append(f"    {nm:<16} F({r['df']:.0f}, {rdf:.0f}) = {r['F']:.3f}, "
                                     f"p = {r['PR(>F)']:.4g} {stars(r['PR(>F)'])}")
            except ImportError:
                lines.append("  Two-way ANOVA skipped: install statsmodels")
            # per batch: Kruskal-Wallis + Dunnett vs control
            for b in batches:
                sub = ok[ok.batch == b]
                data = {s: sub.loc[sub.surface == s, m].dropna().to_numpy() for s in surf_order}
                data = {s: v for s, v in data.items() if len(v) >= 2}
                lines.append(f"  {batch_label(b)}:")
                for s, v in data.items():
                    lines.append(f"    {s:<22} n={len(v)}  mean={v.mean():.4g}  SD={v.std(ddof=1):.4g}")
                vals = [v for v in data.values() if np.ptp(v) > 0 or True]
                if len(vals) >= 2:
                    try:
                        lines.append(f"    Kruskal-Wallis across surfaces: p = {stats.kruskal(*vals).pvalue:.4g}")
                    except ValueError:
                        lines.append("    Kruskal-Wallis: not possible (all values identical)")
                if CONTROL_NAME in data:
                    treat = [s for s in data if s != CONTROL_NAME]
                    try:
                        dn = stats.dunnett(*[data[s] for s in treat], control=data[CONTROL_NAME])
                        lines.append("    Dunnett vs plastic control:")
                        for s, p in zip(treat, dn.pvalue):
                            lines.append(f"      {s:<22} p = {p:.4g} {stars(p)}")
                    except Exception as e:
                        lines.append(f"    Dunnett not possible ({e})")
            lines.append("")
    (out / "statistics.txt").write_text("\n".join(lines))

    # figure: grouped bars - surface on x, one bar per batch
    hue = "#0072B2"
    for m, label in metrics.items():
        fig, ax = plt.subplots(figsize=(max(7, 0.95 * len(surf_order) + 1.5), 4.0))
        w = 0.8 / len(batches)
        rng = np.random.default_rng(0)
        for j, b in enumerate(batches):
            for i, s in enumerate(surf_order):
                v = ok.loc[(ok.surface == s) & (ok.batch == b), m].dropna().to_numpy()
                if not len(v):
                    continue
                xpos = i - 0.4 + w * (j + 0.5)
                ax.bar(xpos, v.mean(), width=w * 0.9, color=hue, alpha=BATCH_SHADES[j % 3],
                       yerr=v.std(ddof=1) if len(v) > 1 else 0, capsize=2, ecolor="0.3",
                       label=batch_label(b) if i == 0 else None)
                ax.scatter(xpos + rng.uniform(-w * 0.2, w * 0.2, len(v)), v, s=7, color="k", zorder=3)
        ax.set_xticks(range(len(surf_order)))
        ax.set_xticklabels(surf_order, rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8, title="Coating applied after mixing", title_fontsize=8)
        save(fig, out / "figures" / f"bars_{m}")

    # figure: peak adhesion vs coating time, one line per surface, control highlighted
    mid = {b: np.mean(BATCH_WINDOW_MIN.get(b, (b, b))) for b in batches}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(surf_order):
        pts = [(mid[b], ok.loc[(ok.surface == s) & (ok.batch == b), "peak_adhesion_mN"]) for b in batches]
        pts = [(t, v) for t, v in pts if len(v)]
        if not pts:
            continue
        t = [p[0] for p in pts]
        y = [p[1].mean() for p in pts]
        e = [p[1].std(ddof=1) if len(p[1]) > 1 else 0 for p in pts]
        if s == CONTROL_NAME:
            ax.errorbar(t, y, yerr=e, color="k", ls="--", marker="s", lw=1.8, capsize=3, label=s, zorder=5)
        else:
            ax.errorbar(t, y, yerr=e, color=cmap(i % 10), marker="o", ms=4, lw=1.2, capsize=2, label=s, alpha=0.9)
    ax.set_xlabel("Time coating was applied after mixing (min, batch midpoint)")
    ax.set_ylabel("Peak adhesion force (mN)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, out / "figures" / "peak_adhesion_vs_coating_time")

    (out / "settings_used.txt").write_text(
        f"Adhesion = pull-off (tensile) force after the compressive contact was released, baseline-corrected\n"
        f"Baseline = median of the last {BASELINE_FRACTION:.0%} of each record; noise = SD of that segment\n"
        f"Measurable adhesion: smoothed peak > {MEASURABLE_FACTOR:g} x noise; otherwise recorded as 0\n"
        f"Smoothing: {SMOOTH_POINTS}-point centred moving average (peak detection)\n"
        f"Work of adhesion: area under positive pull-off force vs displacement until force returned to the noise band\n"
        f"Pull-off rate (median): {ok.pull_rate_mm_min.median():.1f} mm/min\n"
        f"Starting compression (median): {ok.start_compression_N.median():.2f} N\n"
        "Batch coating windows after mixing: " + "; ".join(batch_label(b) for b in batches) + "\n")

    print("Tests analysed: " + str(len(ok)) + f" of {len(res)}")
    print(f"Measurable adhesion in {int(ok.measurable_flag.sum())} of {len(ok)} tests")
    sc = ok.start_compression_N
    odd = ok[(sc - sc.median()).abs() > 0.5 * sc.median()]
    if len(odd):
        print("Check these - starting compression differs a lot from the others:")
        for r in odd.itertuples():
            print(f"   {r.sample_id} ({r.file}): {r.start_compression_N:.2f} N")
    print(f"\nDone. Results in: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tack test analysis")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    check() if a.check else run()