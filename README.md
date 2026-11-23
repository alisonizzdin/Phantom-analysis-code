# Mechanical characterisation pipeline for soft vessel phantoms

Analysis code for characterising the materials used in a lost-wax, perfusable silicone phantom of brain vasculature for focused-ultrasound and microbubble experiments (King's College London, St Thomas' Laboratories).

> The code is shared on its own. Raw data and results are not included.

## What the code does

| Script | Test | Outputs |
|---|---|---|
| `silicone_tensile.py` | Uniaxial tensile tests of silicone elastomers (Instron / Bluehill exports) | Young's modulus, UTS, strain at break; ANOVA + Tukey HSD |
| `solvent_tensile.py` | Tensile properties after soaking in different solvents for different times | % change vs control; Dunnett's test; two-way ANOVA (solvent × time) |
| `wax_bend.py` | 3-point bend tests of waxes and wax blends (ISO 178 / ASTM D790) | Flexural modulus, flexural strength, strain and work to break |
| `tack_test.py` | Probe tack (pull-off) tests of silicone cured against different surfaces | Peak adhesion force, work of adhesion, detachment distance; two-way ANOVA + Dunnett's test |

## Methods built into the pipeline

- **One reproducible process for every sample:** settings live at the top of each script and are written to `settings_used.txt` with each run.
- **QC plot for every sample** showing the analysed curve, the fit window and the detected break or echo, so every automated decision can be checked by eye.
- **Toe (slack) compensation:** when a sample starts slack, the first straight part of the curve is extrapolated back to the starting stress, so the modulus isn't underestimated.
- **Robust break detection:** finds the highest stress before final failure, and separates the maximum stress from the break point for materials that crack gradually.
- **Logged exclusions:** samples are excluded only through a CSV file that records the reason, so every exclusion is documented.
- **Statistics that respect the design:** assumption checks (Shapiro–Wilk, Levene's), non-parametric confirmation (Kruskal–Wallis), and repeated tests of the same specimen are not counted as separate samples.

## Requirements

Python 3 with `numpy`, `pandas`, `matplotlib`, `scipy`, `statsmodels`:

```bash
python -m pip install numpy pandas matplotlib scipy statsmodels
```

## Usage

Put the raw exports in a `raw/` folder (one sub-folder per group), then run:

```bash
python silicone_tensile.py --check   # lists the folders/files found and test-reads one file
python silicone_tensile.py           # full analysis -> results_silicone/
```

The other scripts work the same way. Each script's header describes the folder layout it expects.

## Author

Alis Izzdin, Research Engineer, King's College London
