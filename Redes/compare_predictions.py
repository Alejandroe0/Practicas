#!/usr/bin/env python3
import csv
import glob
import math
import os
import numpy as np

OUT_DIR = "outs"
PATTERN = os.path.join(OUT_DIR, "predicciones_*.csv")


def compute_metrics(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        return None

    if not {"t", "y_real", "y_pred"}.issubset(rows[0].keys()):
        return None

    t = np.array([float(r["t"]) for r in rows], dtype=float)
    y_true = np.array([float(r["y_real"]) for r in rows], dtype=float)
    y_pred = np.array([float(r["y_pred"]) for r in rows], dtype=float)

    err = y_pred - y_true
    rmse = math.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    T = np.vstack([np.ones_like(t), t, t**2]).T
    coef_pred, *_ = np.linalg.lstsq(T, y_pred, rcond=None)
    coef_real, *_ = np.linalg.lstsq(T, y_true, rcond=None)
    g_pred = -2 * coef_pred[2]
    g_real = -2 * coef_real[2]

    return {
        "file": os.path.basename(path),
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "g_pred": g_pred,
        "g_real": g_real,
        "g_diff": g_pred - g_real,
    }


def main():
    files = sorted(glob.glob(PATTERN))
    metrics = []
    for path in files:
        result = compute_metrics(path)
        if result is not None:
            metrics.append(result)

    metrics.sort(key=lambda x: (x["rmse"], x["mae"]))

    print("RESULTADOS COMPARATIVOS")
    print("-" * 95)
    print(f"{'archivo':<35} {'rmse':>10} {'mae':>10} {'corr':>10} {'g_pred':>10} {'g_real':>10} {'diff':>10}")
    print("-" * 95)
    for item in metrics:
        print(
            f"{item['file']:<35} "
            f"{item['rmse']:>10.6f} {item['mae']:>10.6f} {item['corr']:>10.6f} "
            f"{item['g_pred']:>10.4f} {item['g_real']:>10.4f} {item['g_diff']:>10.4f}"
        )
    print("-" * 95)

    best = metrics[0]
    print(f"Mejor RMSE: {best['file']} ({best['rmse']:.6f})")


if __name__ == "__main__":
    main()
