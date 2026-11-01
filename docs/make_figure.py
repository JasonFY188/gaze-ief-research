"""
Draws docs/figures/result_summary.png from the saved evaluation summaries in results/
(no dataset images are used).

  left : mean angular error per subject (leave-one-subject-out, 15 folds), plain L2CS-Net vs + IEF head
  right: calibration: how often the predicted 50 / 90 / 95 % gaze region really contains the true gaze

Usage:  python docs/make_figure.py
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATIVE = os.path.join(ROOT, "results", "native_l2cs", "eval_all_folds.txt")
IEF = os.path.join(ROOT, "results", "ief_mpiigaze_backbone", "ief_mpiigaze_all_folds.txt")
OUT = os.path.join(ROOT, "docs", "figures", "result_summary.png")


def parse(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)", line)
        if m:
            f, err, nll, c95, c90, c50, ece = m.groups()
            rows[int(f)] = dict(err=float(err), c95=float(c95), c90=float(c90), c50=float(c50), ece=float(ece))
    return rows


def main():
    a, b = parse(NATIVE), parse(IEF)
    folds = sorted(set(a) & set(b))
    ea, eb = np.array([a[f]["err"] for f in folds]), np.array([b[f]["err"] for f in folds])
    print(f"folds {len(folds)} | mean error {ea.mean():.3f} -> {eb.mean():.3f} deg "
          f"({100 * (eb.mean() - ea.mean()) / ea.mean():+.1f}%) | IEF lower in {int((eb < ea).sum())}/{len(folds)} folds")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    x = np.arange(len(folds))
    ax[0].bar(x - 0.2, ea, 0.4, color="#b0b0b0", label=f"L2CS-Net (mean {ea.mean():.2f}°)")
    ax[0].bar(x + 0.2, eb, 0.4, color="#2a78d6", label=f"+ IEF head (mean {eb.mean():.2f}°)")
    ax[0].set(xticks=x, xticklabels=[f"p{f:02d}" for f in folds], ylabel="mean angular error (°)",
              title="Angular error per held-out subject (lower is better)")
    ax[0].legend(frameon=False)
    nominal = [50, 90, 95]
    for rows, col, lab in ((a, "#b0b0b0", "L2CS-Net"), (b, "#2a78d6", "+ IEF head")):
        act = [np.mean([rows[f][f"c{n}"] for f in folds]) for n in nominal]
        ax[1].plot(nominal, act, "o-", color=col, label=lab, lw=2)
    ax[1].plot([40, 100], [40, 100], "--", color="#666", lw=1, label="ideal")
    ax[1].set(xlim=(40, 100), ylim=(40, 102), xlabel="predicted region size (%)",
              ylabel="true gaze inside the region (%)", title="Calibration of the gaze region")
    ax[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150)


if __name__ == "__main__":
    main()
