#!/usr/bin/env python3
import argparse, ast, os, re, csv
from pathlib import Path

import matplotlib.pyplot as plt

DICT_RE = re.compile(r"\{[^}]*\}")

def parse_log(log_path):
    """Extract dict blocks like {'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}."""
    metrics = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for m in DICT_RE.finditer(line):
                block = m.group(0)
                try:
                    d = ast.literal_eval(block)
                except Exception:
                    continue
                # keep only known keys; cast to float if present
                out = {}
                for k in ("loss", "grad_norm", "learning_rate", "epoch"):
                    if k in d:
                        try:
                            out[k] = float(d[k])
                        except Exception:
                            pass
                if out:
                    metrics.append(out)
    return metrics

def save_csv(metrics, out_csv):
    keys = ["step", "loss", "grad_norm", "learning_rate", "epoch"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for i, m in enumerate(metrics):
            row = {"step": i}
            row.update(m)
            w.writerow(row)

def plot_series(xs, ys, title, ylabel, out_png):
    plt.figure()
    plt.plot(xs, ys)  # no style/colors set (default)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def ema(values, alpha=0.1):
    if not values:
        return values
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to training log file")
    ap.add_argument("--out", required=True, help="Output directory for plots/CSV")
    ap.add_argument("--smooth", type=float, default=0.0, help="EMA smoothing alpha (0 disables)")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    metrics = parse_log(args.log)
    if not metrics:
        print("No metrics found in log. Make sure lines contain Python-like dicts.")
        return

    steps = list(range(len(metrics)))
    loss = [m.get("loss") for m in metrics if "loss" in m]
    lr   = [m.get("learning_rate") for m in metrics if "learning_rate" in m]
    gn   = [m.get("grad_norm") for m in metrics if "grad_norm" in m]

    # align series lengths to steps (in case some entries miss a field)
    def align(series):
        # fill missing with None to keep index consistent
        s = [None]*len(metrics)
        j = 0
        for i, m in enumerate(metrics):
            if ("loss" in m and series is loss) or \
               ("learning_rate" in m and series is lr) or \
               ("grad_norm" in m and series is gn):
                s[i] = series[j]; j += 1
        return s

    loss_s = align(loss)
    lr_s   = align(lr)
    gn_s   = align(gn)

    # Optionally smooth (EMA) ignoring None
    def smooth_or_identity(seq):
        if args.smooth <= 0:
            return seq
        # collapse None, smooth, then expand back with None
        vals = [v for v in seq if v is not None]
        sm   = ema(vals, alpha=args.smooth)
        out, k = [], 0
        for v in seq:
            if v is None:
                out.append(None)
            else:
                out.append(sm[k]); k += 1
        return out

    loss_s = smooth_or_identity(loss_s)
    lr_s   = smooth_or_identity(lr_s)
    gn_s   = smooth_or_identity(gn_s)

    # Save CSV
    # save_csv(metrics, os.path.join(args.out, "training_metrics.csv"))

    # Plot each series, skipping None
    def drop_none(xs, ys):
        x2, y2 = [], []
        for x, y in zip(xs, ys):
            if y is not None:
                x2.append(x); y2.append(y)
        return x2, y2

    x_l, y_l = drop_none(steps, loss_s)
    x_r, y_r = drop_none(steps, lr_s)
    x_g, y_g = drop_none(steps, gn_s)

    if y_l:
        plot_series(x_l, y_l, "Training Loss vs Step", "loss", os.path.join(args.out, "loss.png"))
    if y_r:
        plot_series(x_r, y_r, "Learning Rate vs Step", "learning_rate", os.path.join(args.out, "learning_rate.png"))
    if y_g:
        plot_series(x_g, y_g, "Grad Norm vs Step", "grad_norm", os.path.join(args.out, "grad_norm.png"))

    print(f"Done, bro. Wrote CSV + plots to: {args.out}")

if __name__ == "__main__":
    main()
