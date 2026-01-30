#!/usr/bin/env python3
import glob
import os
import numpy as np
import pandas as pd

# ================================================================
# Configuración
# ================================================================
DATA_DIR = "/home/alejandro/Practicas/Toma_de_datos/datos"
OUTPUT_DIR = "/home/alejandro/Practicas/Toma_de_datos/datos_resample"
FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# Lectura de archivos Tracker
# ================================================================
def read_tracker_file(path):
    rows = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("t"):
                continue
            parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            try:
                vals = [float(p) for p in parts]
            except:
                continue
            if len(vals) < 7:
                vals += [np.nan] * (7 - len(vals))
            rows.append(vals[:7])
    if len(rows) == 0:
        return None
    return np.array(rows)  # t,x,y,vx,vy,ax,ay

# ================================================================
# Utilidades
# ================================================================
def uniform_resample(t, y):
    dt = np.median(np.diff(t))
    t_new = np.arange(t[0], t[-1] + 1e-12, dt)
    y_new = np.interp(t_new, t, y)
    return t_new, y_new

def rmse(a, b):
    return np.sqrt(np.nanmean((a - b) ** 2))

# ================================================================
# Newton – diferencias divididas
# ================================================================
def velocity_newton(t, y):
    v = np.full_like(y, np.nan)
    v[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
    v[0] = (y[1] - y[0]) / (t[1] - t[0])
    v[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    return v

def acceleration_newton(t, y):
    a = np.full_like(y, np.nan)
    a[1:-1] = (
        (y[2:] - y[1:-1]) / (t[2:] - t[1:-1])
        - (y[1:-1] - y[:-2]) / (t[1:-1] - t[:-2])
    ) / ((t[2:] - t[:-2]) / 2)
    return a

def g_newton_global(t, y):
    # Ajuste cuadrático global y = a t^2 + b t + c
    T = np.vstack([np.ones_like(t), t, t**2]).T
    coef, *_ = np.linalg.lstsq(T, y, rcond=None)
    return 2 * coef[2]

# ================================================================
# Loop principal
# ================================================================
summary = []

for p in FILES:
    arr = read_tracker_file(p)
    if arr is None:
        print("skip", p)
        continue

    t = arr[:, 0]
    y = arr[:, 2]
    vy_t = arr[:, 4]
    ay_t = arr[:, 6]

    mask = ~np.isnan(t) & ~np.isnan(y)
    t = t[mask]
    y = y[mask]
    vy_t = vy_t[mask] if not np.isnan(vy_t).all() else np.full_like(t, np.nan)
    ay_t = ay_t[mask] if not np.isnan(ay_t).all() else np.full_like(t, np.nan)

    if len(t) < 5:
        print("too short", p)
        continue

    # Re-muestreo si t no es uniforme
    dt_med = np.median(np.diff(t))
    if np.max(np.abs(np.diff(t) - dt_med)) > 1e-6:
        t, y = uniform_resample(t, y)
        vy_t = np.interp(t, arr[:, 0], arr[:, 4], left=np.nan, right=np.nan)
        ay_t = np.interp(t, arr[:, 0], arr[:, 6], left=np.nan, right=np.nan)

    # ============================================================
    # Newton
    # ============================================================
    vy_new = velocity_newton(t, y)
    ay_new = acceleration_newton(t, y)
    g_new = g_newton_global(t, y)

    # Errores contra Tracker
    rmse_vy_new = rmse(vy_t, vy_new) if not np.isnan(vy_t).all() else np.nan
    rmse_ay_new = rmse(ay_t, ay_new) if not np.isnan(ay_t).all() else np.nan

    print(
        f"{os.path.basename(p)} | "
        f"g_newton={g_new:.4f} | "
        f"RMSE ay={rmse_ay_new:.2e}"
    )

    # Guardar CSV
    df = pd.DataFrame({
        "t": t,
        "y": y,
        "vy_tracker": vy_t,
        "ay_tracker": ay_t,
        "vy_newton": vy_new,
        "ay_newton": ay_new
    })

    out_csv = os.path.splitext(os.path.basename(p))[0] + "_newton.csv"
    df.to_csv(os.path.join(OUTPUT_DIR, out_csv), index=False)

    summary.append({
        "file": os.path.basename(p),
        "n": len(t),
        "dt": dt_med,
        "g_newton": g_new,
        "rmse_ay_newton": rmse_ay_new
    })

# ================================================================
# Resumen global
# ================================================================
summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    os.path.join(OUTPUT_DIR, "newton_summary.csv"),
    index=False
)

print("Done.")
print("Resultados en:", OUTPUT_DIR)
