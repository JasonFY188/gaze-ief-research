"""
Side-by-side per-subject comparison of all models.

Reads:
  output/eval_native/eval_all_folds.txt   — native MPIIGaze L2CS (28 bins)
  output/eval_ief/ief_all_folds.txt       — IEF MLP head (90 bins, frozen backbone)

Prints a per-fold table and saves a CSV + summary.

Usage:
  python compare_models.py
"""

import csv
import os
import re


NATIVE_TXT = r"C:\Users\keiso\Jason\Dealing with uncertainty from l2cs net\L2CS-Net\output\eval_native\eval_all_folds.txt"
IEF_TXT    = r"C:\Users\keiso\Jason\Dealing with uncertainty from l2cs net\L2CS-Net\output\eval_ief\ief_all_folds.txt"
OUT_DIR    = r"C:\Users\keiso\Jason\Dealing with uncertainty from l2cs net\L2CS-Net\output\comparison"


def parse_results_txt(path):
    """Parse a results .txt file. Returns {fold_int: {metric: value}}."""
    results = {}
    if not os.path.exists(path):
        return results
    with open(path, encoding="utf-8") as f:
        for line in f:
            # Match lines like: "0      3.190    3.080    99.9%  ..."
            m = re.match(r"^(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)", line)
            if m:
                fold = int(m.group(1))
                results[fold] = {
                    "ang_err": float(m.group(2)),
                    "nll":     float(m.group(3)),
                    "cov95":   float(m.group(4)) / 100,
                    "cov90":   float(m.group(5)) / 100,
                    "cov50":   float(m.group(6)) / 100,
                    "ece":     float(m.group(7)),
                }
    return results


def mean_std(vals):
    import statistics
    if not vals:
        return float("nan"), float("nan")
    return statistics.mean(vals), statistics.pstdev(vals)


def main():
    native = parse_results_txt(NATIVE_TXT)
    ief    = parse_results_txt(IEF_TXT)

    os.makedirs(OUT_DIR, exist_ok=True)

    all_folds = sorted(set(list(native.keys()) + list(ief.keys())))

    # Print header
    sep = "-" * 105
    print(sep)
    print(f"{'':>6} | {'── Native MPIIGaze L2CS (28-bin) ──':^35} | {'── IEF MLP head (90-bin, frozen backbone) ──':^43} |")
    print(f"{'Fold':>6} | {'Err°':>6} {'NLL':>6} {'C95%':>6} {'C50%':>6} {'ECE':>6} | "
          f"{'Err°':>6} {'NLL':>6} {'C95%':>6} {'C50%':>6} {'ECE':>6} | "
          f"{'ΔErr°':>7}")
    print(sep)

    csv_rows = []
    delta_errs = []

    for fold in all_folds:
        n = native.get(fold, {})
        i = ief.get(fold, {})

        n_err  = n.get("ang_err", float("nan"))
        n_nll  = n.get("nll",     float("nan"))
        n_c95  = n.get("cov95",   float("nan"))
        n_c50  = n.get("cov50",   float("nan"))
        n_ece  = n.get("ece",     float("nan"))

        i_err  = i.get("ang_err", float("nan"))
        i_nll  = i.get("nll",     float("nan"))
        i_c95  = i.get("cov95",   float("nan"))
        i_c50  = i.get("cov50",   float("nan"))
        i_ece  = i.get("ece",     float("nan"))

        delta = i_err - n_err if not (i_err != i_err or n_err != n_err) else float("nan")
        sign  = "+" if delta > 0 else ""
        if not (delta != delta):
            delta_errs.append(delta)

        print(f"p{fold:02d}    | "
              f"{n_err:>6.3f} {n_nll:>6.3f} {n_c95*100:>5.1f}% {n_c50*100:>5.1f}% {n_ece:>6.4f} | "
              f"{i_err:>6.3f} {i_nll:>6.3f} {i_c95*100:>5.1f}% {i_c50*100:>5.1f}% {i_ece:>6.4f} | "
              f"{sign}{delta:>6.3f}")

        csv_rows.append({
            "fold": f"p{fold:02d}",
            "native_ang_err": n_err, "native_nll": n_nll,
            "native_cov95": n_c95,   "native_cov50": n_c50, "native_ece": n_ece,
            "ief_ang_err": i_err,    "ief_nll": i_nll,
            "ief_cov95": i_c95,      "ief_cov50": i_c50,    "ief_ece": i_ece,
            "delta_ang_err": delta,
        })

    # Summary rows
    print(sep)
    n_errs  = [r["native_ang_err"] for r in csv_rows if r["native_ang_err"] == r["native_ang_err"]]
    i_errs  = [r["ief_ang_err"]    for r in csv_rows if r["ief_ang_err"]    == r["ief_ang_err"]]
    n_c95s  = [r["native_cov95"]   for r in csv_rows if r["native_cov95"]   == r["native_cov95"]]
    i_c95s  = [r["ief_cov95"]      for r in csv_rows if r["ief_cov95"]      == r["ief_cov95"]]
    n_c50s  = [r["native_cov50"]   for r in csv_rows if r["native_cov50"]   == r["native_cov50"]]
    i_c50s  = [r["ief_cov50"]      for r in csv_rows if r["ief_cov50"]      == r["ief_cov50"]]
    n_eces  = [r["native_ece"]     for r in csv_rows if r["native_ece"]     == r["native_ece"]]
    i_eces  = [r["ief_ece"]        for r in csv_rows if r["ief_ece"]        == r["ief_ece"]]

    import statistics
    def ms(v): return (statistics.mean(v), statistics.pstdev(v)) if v else (float("nan"), float("nan"))

    nm, ns = ms(n_errs); im, is_ = ms(i_errs)
    d_m, d_s = ms(delta_errs)

    print(f"{'Mean':>6} | "
          f"{nm:>6.3f} {'':>6} {statistics.mean(n_c95s)*100:>5.1f}% {statistics.mean(n_c50s)*100:>5.1f}% {statistics.mean(n_eces):>6.4f} | "
          f"{im:>6.3f} {'':>6} {statistics.mean(i_c95s)*100 if i_c95s else float('nan'):>5.1f}% {statistics.mean(i_c50s)*100 if i_c50s else float('nan'):>5.1f}% {statistics.mean(i_eces) if i_eces else float('nan'):>6.4f} | "
          f"{'+' if d_m > 0 else ''}{d_m:>6.3f}")
    print(f"{'Std':>6} | "
          f"{ns:>6.3f} {'':>6} {statistics.pstdev(n_c95s)*100:>5.1f}% {statistics.pstdev(n_c50s)*100:>5.1f}% {statistics.pstdev(n_eces):>6.4f} | "
          f"{is_:>6.3f} {'':>6} {statistics.pstdev(i_c95s)*100 if i_c95s else float('nan'):>5.1f}% {statistics.pstdev(i_c50s)*100 if i_c50s else float('nan'):>5.1f}% {statistics.pstdev(i_eces) if i_eces else float('nan'):>6.4f} | "
          f"{d_s:>7.3f}")
    print(sep)
    print(f"\nDelta = IEF - Native.  Negative = IEF is better.")

    if not i_errs:
        print("\nNote: IEF results not yet available. Run evaluate_ief_all_folds.py after training completes.")

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "per_subject_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved CSV -> {csv_path}")


if __name__ == "__main__":
    main()
