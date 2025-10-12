#!/usr/bin/env python3
import glob, os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

DATA_DIR = "/home/alejandro/Practicas/Toma_de_datos/datos" 
OUTPUT_DIR = "/home/alejandro/Practicas/Toma_de_datos/datos_resample"  
FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_tracker_file(path):
    rows=[]
    with open(path,"r",encoding="utf8") as f:
        for line in f:
            line=line.strip()
            if not line or line.lower().startswith("t"):
                continue
            parts=[p.strip() for p in line.split(",") if p.strip()!='']
            try:
                vals=[float(p) for p in parts]
            except:
                continue
            if len(vals) < 7:
                vals += [np.nan]*(7-len(vals))
            rows.append(vals[:7])
    if len(rows)==0:
        return None
    arr = np.array(rows)
    return arr  # columns: t,x,y,vx,vy,ax,ay

def uniform_resample(t, y, fs=None):
    # si t no es uniforme, resamplea a grid uniforme por dt medio
    if fs is None:
        dt = np.median(np.diff(t))
    else:
        dt = 1.0/fs
    t_new = np.arange(t[0], t[-1]+1e-12, dt)
    # interpolar y a t_new (linear)
    y_new = np.interp(t_new, t, y)
    return t_new, y_new

def compute_sg_derivatives(t, y, fps=None, window_time=0.06, polyorder=3):
    # window_time en segundos -> elegir ventana impar en samples
    dt = np.median(np.diff(t))
    if fps is None:
        fps = 1.0/dt
    win = max(5, int(round(window_time * fps)))  # tamaño ventana aproximado
    if win % 2 == 0:
        win += 1
    # si la serie es muy corta, reducir window
    if win >= len(y):
        win = len(y) - 1 if (len(y)-1)%2==1 else len(y)-2
    if win < 3:
        win = 3
    # suavizar y y calcular primeras y segundas derivadas con Savitzky-Golay
    y_s = savgol_filter(y, window_length=win, polyorder=polyorder, mode='interp')
    # derivadas: coefficients divide by dt^n
    dy = savgol_filter(y, window_length=win, polyorder=polyorder, deriv=1, delta=dt, mode='interp')
    d2y = savgol_filter(y, window_length=win, polyorder=polyorder, deriv=2, delta=dt, mode='interp')
    return y_s, dy, d2y, win, polyorder

def compute_spline_derivatives(t, y, s_factor=1e-6):
    # spline suavizante: s factor = smoothing; ajustar según ruido
    # s = s_factor * len(t) * var(y)
    s = s_factor * len(t) * np.var(y)
    try:
        sp = UnivariateSpline(t, y, s=s)
        dy = sp.derivative(n=1)(t)
        d2y = sp.derivative(n=2)(t)
        y_s = sp(t)
        return y_s, dy, d2y
    except Exception as e:
        return None, None, None

def rmse(a,b):
    return np.sqrt(np.nanmean((a-b)**2))

out_rows = []
for p in FILES:
    a = read_tracker_file(p)
    if a is None: 
        print("skip", p); continue
    t = a[:,0]; y = a[:,2]; vy_t = a[:,4]; ay_t = a[:,6]
    mask = ~np.isnan(t) & ~np.isnan(y)
    t = t[mask]; y = y[mask]
    vy_t = vy_t[mask] if np.isnan(vy_t).all()==False else np.full_like(t, np.nan)
    ay_t = ay_t[mask] if np.isnan(ay_t).all()==False else np.full_like(t, np.nan)

    if len(t) < 5:
        print("too short", p); continue

    # 1) resample uniform si es necesario
    dt_median = np.median(np.diff(t))
    inconsistent = np.max(np.abs(np.diff(t)-dt_median)) > 1e-6
    if inconsistent:
        t_u, y_u = uniform_resample(t, y)
        # map original vy/ay by interpolation (for comparison)
        vy_t_u = np.interp(t_u, t, vy_t, left=np.nan, right=np.nan)
        ay_t_u = np.interp(t_u, t, ay_t, left=np.nan, right=np.nan)
        t, y, vy_t, ay_t = t_u, y_u, vy_t_u, ay_t_u

    # 2) Savitzky–Golay
    y_s, dy_s, d2y_s, win, poly = compute_sg_derivatives(t, y, fps=1.0/dt_median, window_time=0.06, polyorder=3)

    # 3) Spline (alternativa)
    y_sp, dy_sp, d2y_sp = compute_spline_derivatives(t, y, s_factor=1e-6)

    # 4) comparaciones
    rmse_vy_sg = rmse(vy_t, dy_s) if not np.isnan(vy_t).all() else np.nan
    rmse_ay_sg = rmse(ay_t, d2y_s) if not np.isnan(ay_t).all() else np.nan

    print(f"{os.path.basename(p)}: len={len(t)} dt~{dt_median:.5f}s sg_win={win} rmse_vy_sg={rmse_vy_sg:.5e} rmse_ay_sg={rmse_ay_sg:.5e}")

    # 5) guardar CSV por archivo en la carpeta de OUTPUT
    df = pd.DataFrame({
        "t": t,
        "y": y,
        "vy_tracker": vy_t,
        "ay_tracker": ay_t,
        "y_sg": y_s,
        "vy_sg": dy_s,
        "ay_sg": d2y_s,
        "y_spline": y_sp,
        "vy_spline": dy_sp,
        "ay_spline": d2y_sp
    })
    out_csv = os.path.splitext(os.path.basename(p))[0] + "_deriv.csv"
    df.to_csv(os.path.join(OUTPUT_DIR, out_csv), index=False)
    out_rows.append({
        "file": os.path.basename(p),
        "n": len(t),
        "dt": float(dt_median),
        "win_sg": int(win),
        "rmse_vy_sg": float(rmse_vy_sg) if not np.isnan(rmse_vy_sg) else None,
        "rmse_ay_sg": float(rmse_ay_sg) if not np.isnan(rmse_ay_sg) else None
    })

# resumen global en la carpeta de OUTPUT
pd.DataFrame(out_rows).to_csv(os.path.join(OUTPUT_DIR, "derivatives_summary.csv"), index=False)
print(f"Done. Results written to {OUTPUT_DIR}")
print(f"Summary written to {os.path.join(OUTPUT_DIR, 'derivatives_summary.csv')}")