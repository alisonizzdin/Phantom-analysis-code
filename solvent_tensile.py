#!/usr/bin/env python3
#!/usr/bin/env python3
"""
SOLVENT STUDY - tensile analysis of soaked silicone vs unsoaked control

Each sample is analysed with EXACTLY the same method as the silicone study
(it uses the analysis code inside silicone_tensile.py, which must be in the same folder).
Then:
  - each soaked condition is compared with the control (Dunnett's test)
  - a two-way ANOVA tests the effect of solvent, soak time, and their interaction
  - results are also given as % change from the control

FOLDERS (keep your own names - solvent and time are read from the folder name):
  silicone_tensile.py
  solvent_tensile.py
  raw/
    solvent/              (a folder called "solvent test" also works)
      Control/
      Acetone 60min/
      Acetone 90min/
      Acetone 12h/
      Acetone 24h/
      Hexane 60min/
      ...
      White spirit 24h/

"""

import argparse
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

try:
    from silicone_tensile import (read_bluehill, analyse, qc_plot, save, common_grid,
                                  data_files, natural_key)
except ImportError as e:
    sys.exit("Could not load silicone_tensile.py - it must be in the same folder as this script.\n"
             f"({e})")

# =============================================================================
# SETTINGS
# =============================================================================
HERE = Path(__file__).resolve().parent
DATA_CANDIDATES = [HERE / "raw" / "solvent", HERE / "raw" / "solvent test", HERE / "raw" / "Solvent test"]
RESULTS_FOLDER = HERE / "results_solvent"
EXCLUSIONS_FILE = HERE / "solvent_exclusions.csv"

SOLVENT_COLOURS = {"Acetone": "#D55E00", "Hexane": "#0072B2", "White spirit": "#009E73"}
OTHER_COLOURS = ["#CC79A7", "#E69F00", "#56B4E9", "#000000"]
CONTROL_COLOUR = "0.35"
FIG_DPI = 300

METRICS = {"youngs_modulus_kPa": "Young's modulus (kPa)",
           "uts_kPa": "Ultimate tensile strength (kPa)",
           "strain_at_break_pct": "Strain at break (%)",
           "stress_at_100pct_kPa": "Stress at 100% strain (kPa)"}

plt.rcParams.update({"font.size": 10, "font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

# =============================================================================
# Folder names -> solvent + time
# =============================================================================
TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(minutes|minute|mins|min|m|hours|hour|hrs|hr|h)\b", re.I)


# Words in a folder name that identify each solvent (case does not matter).
SOLVENT_KEYWORDS = {
    "Acetone": ["acetone", "ace"],
    "Hexane": ["hexane", "hex"],
    "White spirit": ["white spirit", "whitespirit", "ws"],
}


def parse_condition(name):
    """Read solvent + soak time from any folder name, e.g.
    'test 00-30 10_ thinner -Acetone 60mins' -> ('Acetone', 1.0)
    'Test 00-30 10_ thinner WS after 60 mins' -> ('White spirit', 1.0)
    'test 00-30+10_ thinner Control' -> ('Control', 0.0)"""
    n = name.strip()
    low = n.lower()
    if "control" in low:
        return "Control", 0.0
    solvent = None
    for sname, words in SOLVENT_KEYWORDS.items():
        if any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", low) for w in words):
            solvent = sname
            break
    if solvent is None:
        raise ValueError(f"could not tell which solvent folder '{name}' is - add a keyword to SOLVENT_KEYWORDS")
    m = TIME_RE.search(n)
    if not m:
        raise ValueError(f"could not find a soak time (e.g. '60min' or '12h') in folder name '{name}'")
    val, unit = float(m.group(1)), m.group(2).lower()
    hours = val / 60 if unit.startswith("m") else val
    return solvent, hours


def time_label(h):
    if h == 0:
        return "control"
    return f"{h * 60:.0f} min" if h < 2 else f"{h:g} h"


def find_data_folder():
    for c in DATA_CANDIDATES:
        if c.is_dir():
            return c
    sys.exit("Solvent data folder not found. Make  raw/solvent/  (or raw/solvent test/) with one "
             "sub-folder per condition, e.g. 'Control', 'Acetone 60min', 'Acetone 12h'.")


def find_groups():
    root = find_data_folder()
    folders = [d for d in root.iterdir() if d.is_dir()]
    if not folders:
        sys.exit(f"No condition folders inside {root}")
    groups = []
    for d in folders:
        solvent, hours = parse_condition(d.name)
        groups.append({"folder": d, "solvent": solvent, "hours": hours,
                       "label": "Control" if solvent == "Control" else f"{solvent} {time_label(hours)}"})
    groups.sort(key=lambda g: (g["solvent"] != "Control", g["solvent"], g["hours"]))
    if not any(g["solvent"] == "Control" for g in groups):
        print("WARNING: no folder with 'control' in its name - comparisons with control will be skipped.")
    return root, groups

# =============================================================================
# Check
# =============================================================================


def check():
    root, groups = find_groups()
    print(f"Looking in: {root}\n")
    first = None
    for g in groups:
        fs = data_files(g["folder"])
        print(f"  {g['folder'].name:<26} -> solvent '{g['solvent']}', time {time_label(g['hours']):<8} "
              f"{len(fs)} file(s)")
        first = first or (fs[0] if fs else None)
    solvents = sorted({g["solvent"] for g in groups if g["solvent"] != "Control"})
    times = sorted({g["hours"] for g in groups if g["solvent"] != "Control"})
    print(f"\nSolvents: {', '.join(solvents)}")
    print(f"Soak times: {', '.join(time_label(t) for t in times)}")
    missing = [f"{s} {time_label(t)}" for s in solvents for t in times
               if not any(g["solvent"] == s and g["hours"] == t for g in groups)]
    if missing:
        print(f"Missing combinations: {', '.join(missing)}")
    if first is None:
        sys.exit("\nNo .csv or .txt files found.")
    print(f"\nTest-reading {first.name} ...")
    try:
        df, info = read_bluehill(first)
    except Exception as e:
        print(f"  could not read it: {e}")
        print("  Open this file in VS Code and paste its first ~15 lines to Claude.")
        return
    res, _ = analyse(df)
    print(f"  E = {res['youngs_modulus_kPa']:.1f} kPa, UTS = {res['uts_kPa']:.1f} kPa, "
          f"strain at break = {res['strain_at_break_pct']:.0f} %")
    print("\nIf that looks right, run:  python solvent_tensile.py")

# =============================================================================
# Run
# =============================================================================


def load_exclusions(all_ids):
    if EXCLUSIONS_FILE.exists():
        ex = pd.read_csv(EXCLUSIONS_FILE, dtype=str, keep_default_na=False)
    else:
        ex = pd.DataFrame(columns=["condition_folder", "file", "exclude", "reason"])
    known = set(zip(ex.condition_folder, ex.file))
    new = [{"condition_folder": g, "file": f, "exclude": "N", "reason": ""}
           for g, f in all_ids if (g, f) not in known]
    if new:
        ex = pd.concat([ex, pd.DataFrame(new)], ignore_index=True)
        ex.to_csv(EXCLUSIONS_FILE, index=False)
    return {(r.condition_folder, r.file): r.reason for r in ex.itertuples()
            if str(r.exclude).strip().upper().startswith("Y")}


def solvent_colour(solvents):
    col, k = {}, 0
    for s in solvents:
        if s in SOLVENT_COLOURS:
            col[s] = SOLVENT_COLOURS[s]
        else:
            col[s] = OTHER_COLOURS[k % len(OTHER_COLOURS)]
            k += 1
    return col


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def run():
    root, groups = find_groups()
    excluded = load_exclusions([(g["folder"].name, f.name) for g in groups for f in data_files(g["folder"])])
    out = RESULTS_FOLDER
    (out / "qc").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    rows, curves = [], {}
    print(f"Reading from {root}\n")
    for g in groups:
        lab = g["label"]
        print(f"[{lab}]  ({g['folder'].name})")
        for k, f in enumerate(data_files(g["folder"]), start=1):
            row = {"condition": lab, "solvent": g["solvent"], "soak_h": g["hours"],
                   "sample": k, "sample_id": f"{lab} #{k}", "folder": g["folder"].name,
                   "file": f.name, "status": "ok", "flags": ""}
            if (g["folder"].name, f.name) in excluded:
                row["status"] = f"excluded: {excluded[(g['folder'].name, f.name)]}"
                print(f"   #{k:<3} {f.name:<30} EXCLUDED ({excluded[(g['folder'].name, f.name)]})")
                rows.append(row)
                continue
            try:
                df, _ = read_bluehill(f)
                res, c = analyse(df)
                row.update(res)
                if np.isfinite(res["modulus_fit_R2"]) and res["modulus_fit_R2"] < 0.98:
                    row["flags"] += "low modulus-fit R2; "
                safe = re.sub(r"[^\w\-]+", "_", f"{lab}_{k}_{f.stem}")
                qc_plot(c, res, f"{row['sample_id']}  ({f.name})", out / "qc" / f"{safe}.png")
                curves.setdefault(lab, []).append(c)
                print(f"   #{k:<3} {f.name:<30} ok   E={res['youngs_modulus_kPa']:6.1f} kPa  "
                      f"UTS={res['uts_kPa']:6.0f} kPa  break={res['strain_at_break_pct']:5.0f} %")
            except Exception as e:
                row["status"] = f"error: {e}"
                print(f"   #{k:<3} {f.name:<30} ERROR: {e}")
            rows.append(row)
        print()

    res = pd.DataFrame(rows)
    ok = res[res.status == "ok"].copy()
    if ok.empty:
        sys.exit("No samples analysed successfully - see the errors above.")
    med = ok.groupby("condition").strain_at_break_pct.transform("median")
    res.loc[ok.index[ok.strain_at_break_pct < 0.6 * med], "flags"] += "early break - check grips; "
    res.to_csv(out / "per_sample_results.csv", index=False)

    order = [g["label"] for g in groups if g["label"] in curves]
    solvents = list(dict.fromkeys(g["solvent"] for g in groups if g["solvent"] != "Control"))
    times = sorted({g["hours"] for g in groups if g["solvent"] != "Control"})
    scol = solvent_colour(solvents)
    has_ctrl = "Control" in curves
    ctrl = ok[ok.condition == "Control"]

    # ---------- summary table with % change from control ----------
    summ = ok.groupby("condition")[list(METRICS)].agg(["mean", "std", "count"]).reindex(order)
    summ.columns = [f"{m}_{s}".replace("_std", "_SD").replace("_count", "_n") for m, s in summ.columns]
    if has_ctrl:
        for m in METRICS:
            cm = ctrl[m].mean()
            summ[f"{m}_pct_change_vs_control"] = (summ[f"{m}_mean"] - cm) / cm * 100
    first = ok.drop_duplicates("condition").set_index("condition")[["solvent", "soak_h"]]
    summ = first.join(summ, how="right").reindex(order)
    summ.round(3).to_csv(out / "summary_by_condition.csv")

    # ---------- statistics ----------
    dunnett_p = {}
    lines = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m, label in METRICS.items():
            lines.append(f"=== {label} ===")
            for lab in order:
                v = ok.loc[ok.condition == lab, m].dropna().to_numpy()
                if len(v) == 0:
                    continue
                sw = stats.shapiro(v).pvalue if len(v) >= 3 else np.nan
                pc = ""
                if has_ctrl and lab != "Control":
                    cm = ctrl[m].mean()
                    pc = f"  ({(v.mean() - cm) / cm * 100:+.1f}% vs control)"
                lines.append(f"  {lab:<22} n={len(v)}  mean={v.mean():.4g}  SD={v.std(ddof=1) if len(v) > 1 else 0:.4g}"
                             f"  Shapiro p={sw:.3f}{pc}")
            # Dunnett vs control
            if has_ctrl:
                treat = [lab for lab in order if lab != "Control"
                         and ok.loc[ok.condition == lab, m].notna().sum() >= 2]
                cv = ctrl[m].dropna().to_numpy()
                if treat and len(cv) >= 2:
                    dn = stats.dunnett(*[ok.loc[ok.condition == t, m].dropna().to_numpy() for t in treat],
                                       control=cv)
                    lines.append("  Dunnett's test vs control (* p<0.05, ** p<0.01, *** p<0.001):")
                    for t, p in zip(treat, dn.pvalue):
                        dunnett_p[(m, t)] = p
                        lines.append(f"    {t:<22} p = {p:.4g} {stars(p)}")
            # two-way ANOVA solvent x time (control excluded)
            sub = ok[ok.condition != "Control"][["solvent", "soak_h", m]].dropna()
            # the two-way ANOVA needs every solvent at every time: use only the soak times that
            # all solvents share (e.g. if one solvent has no 90 min group, 90 min is left out here)
            n_solv = sub.solvent.nunique()
            counts = sub.groupby(["soak_h", "solvent"]).size().unstack(fill_value=0)
            shared = [t for t in counts.index if (counts.loc[t] >= 2).sum() == n_solv]
            dropped = [time_label(t) for t in counts.index if t not in shared]
            sub = sub[sub.soak_h.isin(shared)]
            if n_solv >= 2 and len(shared) < 2:
                lines.append("  Two-way ANOVA not run for this measure: fewer than 2 soak times are shared "
                             "by all solvents.")
            elif n_solv >= 2:
                if dropped:
                    lines.append(f"  (two-way ANOVA uses soak times shared by all solvents; left out: "
                                 f"{', '.join(dropped)} - those groups are still in the control comparisons above)")
                try:
                    import statsmodels.formula.api as smf
                    from statsmodels.stats.anova import anova_lm
                    sub = sub.rename(columns={m: "y"})
                    sub["time"] = sub.soak_h.map(time_label)
                    model = smf.ols("y ~ C(solvent) * C(time)", data=sub).fit()
                    tab = anova_lm(model, typ=2)
                    lines.append("  Two-way ANOVA (solvent x soak time, control excluded):")
                    for idx, r in tab.iterrows():
                        if idx == "Residual":
                            lines.append(f"    residual df = {r['df']:.0f}")
                        else:
                            name = idx.replace("C(", "").replace(")", "").replace(":", " x ")
                            lines.append(f"    {name:<16} F({r['df']:.0f}, {tab.loc['Residual', 'df']:.0f}) = "
                                         f"{r['F']:.3f}, p = {r['PR(>F)']:.4g} {stars(r['PR(>F)'])}")
                except ImportError:
                    lines.append("  Two-way ANOVA skipped: install statsmodels  (python -m pip install statsmodels)")
            lines.append("")
    (out / "statistics.txt").write_text("\n".join(lines))

    # ---------- figure: property vs soak time, one line per solvent, control band ----------
    xs = {t: i for i, t in enumerate(times)}
    for m, label in METRICS.items():
        if ok[m].notna().sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        if has_ctrl:
            cm, cs = ctrl[m].mean(), ctrl[m].std(ddof=1)
            ax.axhspan(cm - cs, cm + cs, color=CONTROL_COLOUR, alpha=0.15, lw=0)
            ax.axhline(cm, color=CONTROL_COLOUR, ls="--", lw=1.2, label="Control (mean ± SD)")
        offs = np.linspace(-0.12, 0.12, max(len(solvents), 1))
        for j, s in enumerate(solvents):
            pts = [(t, ok.loc[(ok.solvent == s) & (ok.soak_h == t), m].dropna()) for t in times]
            pts = [(t, v) for t, v in pts if len(v)]
            if not pts:
                continue
            x = np.array([xs[t] for t, _ in pts]) + offs[j]
            y = np.array([v.mean() for _, v in pts])
            e = np.array([v.std(ddof=1) if len(v) > 1 else 0 for _, v in pts])
            ax.errorbar(x, y, yerr=e, color=scol[s], marker="o", ms=5, lw=1.6, capsize=3, label=s)
            for (t, v), xi, yi, ei in zip(pts, x, y, e):
                p = dunnett_p.get((m, f"{s} {time_label(t)}"))
                if p is not None and stars(p):
                    ax.text(xi, yi + ei, stars(p), ha="center", va="bottom", color=scol[s], fontsize=9)
        ax.set_xticks(range(len(times)))
        ax.set_xticklabels([time_label(t) for t in times])
        ax.set_xlabel("Soak time")
        ax.set_ylabel(label)
        ax.legend(frameon=False, fontsize=8)
        save(fig, out / "figures" / f"vs_time_{m}")

    # ---------- figure: % change from control ----------
    if has_ctrl:
        for m, label in METRICS.items():
            cm = ctrl[m].mean()
            if not np.isfinite(cm):
                continue
            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            w = 0.8 / max(len(solvents), 1)
            for j, s in enumerate(solvents):
                for t in times:
                    v = ok.loc[(ok.solvent == s) & (ok.soak_h == t), m].dropna()
                    if not len(v):
                        continue
                    pct = (v - cm) / cm * 100
                    x = xs[t] - 0.4 + w * (j + 0.5)
                    ax.bar(x, pct.mean(), width=w * 0.9, color=scol[s], alpha=0.8,
                           yerr=pct.std(ddof=1) if len(pct) > 1 else 0, capsize=3,
                           label=s if t == times[0] else None)
                    p = dunnett_p.get((m, f"{s} {time_label(t)}"))
                    if p is not None and stars(p):
                        top = pct.mean() + (pct.std(ddof=1) if len(pct) > 1 else 0) * np.sign(pct.mean() or 1)
                        ax.text(x, top, stars(p), ha="center",
                                va="bottom" if pct.mean() >= 0 else "top", fontsize=8)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(range(len(times)))
            ax.set_xticklabels([time_label(t) for t in times])
            ax.set_xlabel("Soak time")
            ax.set_ylabel(f"Change from control (%)\n{label.split(' (')[0]}")
            ax.legend(frameon=False, fontsize=8)
            save(fig, out / "figures" / f"pct_change_{m}")

    # ---------- figure: mean stress-strain curves, one panel per solvent ----------
    n = len(solvents)
    if n:
        fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.4), sharex=True, sharey=True, squeeze=False)
        shades = np.linspace(0.35, 1.0, len(times))
        for ax, s in zip(axes[0], solvents):
            if has_ctrl:
                x, ys = common_grid(curves["Control"])
                ax.plot(x * 100, ys.mean(0), color=CONTROL_COLOUR, ls="--", lw=1.5, label="Control")
            for t, a in zip(times, shades):
                lab = f"{s} {time_label(t)}"
                if lab not in curves:
                    continue
                x, ys = common_grid(curves[lab])
                ax.plot(x * 100, ys.mean(0), color=scol[s], alpha=a, lw=1.8, label=time_label(t))
            ax.set_title(s, fontsize=10)
            ax.set_xlabel("Strain (%)")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.legend(frameon=False, fontsize=7)
        axes[0][0].set_ylabel("Stress (kPa)")
        save(fig, out / "figures" / "fig_mean_curves_by_solvent")

    (out / "settings_used.txt").write_text(
        "Per-sample analysis identical to the silicone study (silicone_tensile.py settings).\n"
        "Comparisons: Dunnett's test of each soaked condition vs unsoaked control; two-way ANOVA "
        "(solvent x soak time, type II sums of squares) on soaked conditions only.\n"
        "% change = (condition mean - control mean) / control mean x 100.\n")

    print("Samples per condition: " + ", ".join(f"{lab} = {len(curves[lab])}" for lab in order))
    flagged = res[res["flags"].astype(str).str.len() > 0]
    if len(flagged):
        print("Check these QC plots:")
        for r in flagged.itertuples():
            print(f"   {r.sample_id} ({r.file}): {r.flags.strip('; ')}")
    print(f"\nDone. Results in: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Solvent study tensile analysis")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    check() if a.check else run()
    
