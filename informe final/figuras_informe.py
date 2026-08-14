#!/usr/bin/env python3
"""
Genera las figuras del analisis por clip para el informe final.
Salida: ~/Documentos/personal/Practicas/informe final/imgs/
"""
import glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.expanduser("~/Documentos/personal/Practicas")
OUT = os.path.join(BASE, "informe final", "imgs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def read_freefall_file(path):
    rows = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("t"):
                continue
            parts = [p for p in line.split(",") if p.strip() != ""]
            try:
                vals = [float(p) for p in parts]
            except ValueError:
                continue
            if len(vals) < 7:
                vals += [np.nan] * (7 - len(vals))
            rows.append(vals[:7])
    return np.array(rows)


def gfit(t, y):
    """Ajuste y = c0 + c1 t + c2 t^2 ; devuelve g = 2*c2."""
    T = np.vstack([np.ones_like(t), t, t ** 2]).T
    c, *_ = np.linalg.lstsq(T, y, rcond=None)
    return 2 * c.flatten()[2]


# ------------------------------------------------------------------
# 1. Reconstruccion de los limites de cada clip (mismo orden que usan
#    los scripts de entrenamiento: sorted(glob))
# ------------------------------------------------------------------
files = sorted(glob.glob(os.path.join(BASE, "Toma_de_datos/datos/*.txt")))
segments = []   # (nombre, n_puntos)
clips = []      # (t, y) de cada clip
for p in files:
    a = read_freefall_file(p)
    if a.size == 0:
        continue
    t, y = a[:, 0], a[:, 2]
    m = ~np.isnan(t) & ~np.isnan(y)
    t, y = t[m], y[m]
    if len(t) == 0:
        continue
    segments.append((os.path.basename(p), len(t)))
    clips.append((t, y))

n_total = sum(n for _, n in segments)
print(f"clips={len(segments)}  muestras={n_total}")

g_real = np.array([gfit(t, y) for t, y in clips if len(t) >= 5])

MODELOS = [
    ("predicciones_red1_ep700.csv",           r"Red simple $1\to2\to1$"),
    ("predicciones_red2_ep400.csv",           r"Red $3\to4\to1$"),
    ("predicciones_red_tanh_original_ep700.csv", "Red Tanh profunda"),
]

datos = {}
for fname, label in MODELOS:
    d = pd.read_csv(os.path.join(BASE, "Redes/outs", fname))
    t, yr, yp = d["t"].values, d["y_real"].values, d["y_pred"].values
    assert len(t) == n_total, (fname, len(t), n_total)
    gp, i = [], 0
    for _, n in segments:
        sl = slice(i, i + n)
        i += n
        if n >= 5:
            gp.append(gfit(t[sl], yp[sl]))
    datos[label] = dict(t=t, yr=yr, yp=yp, g=np.array(gp))

# ------------------------------------------------------------------
# FIGURA 1: por que el ajuste sobre el conjunto agregado no mide g
# ------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

t_all = np.concatenate([t for t, _ in clips])
y_all = np.concatenate([y for _, y in clips])
ax[0].scatter(t_all, y_all, s=1, alpha=0.15, color="#3b6ea5", rasterized=True)
tt = np.linspace(t_all.min(), t_all.max(), 300)
T = np.vstack([np.ones_like(t_all), t_all, t_all ** 2]).T
c, *_ = np.linalg.lstsq(T, y_all, rcond=None)
c = c.flatten()
ax[0].plot(tt, c[0] + c[1] * tt + c[2] * tt ** 2, color="#c1121f", lw=2,
           label=fr"ajuste global: $g=2c_2={2*c[2]:+.3f}$ m/s$^2$")
ax[0].set_xlabel("$t$ [s]")
ax[0].set_ylabel("$y$ [m]")
ax[0].set_title("(a) Conjunto agregado: 606 clips superpuestos")
ax[0].legend(fontsize=8, loc="upper right")

rng = np.random.default_rng(3)
idx = rng.choice(len(clips), 6, replace=False)
colores = plt.cm.viridis(np.linspace(0.05, 0.85, len(idx)))
for k, (j, col) in enumerate(zip(idx, colores)):
    t, y = clips[j]
    ax[1].plot(t, y, "o-", ms=3, lw=1, color=col,
               label=fr"{segments[j][0]}: $g={gfit(t,y):.2f}$")
ax[1].set_xlabel("$t$ [s]")
ax[1].set_ylabel("$y$ [m]")
ax[1].set_title("(b) Seis clips individuales")
ax[1].legend(fontsize=7, loc="best")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "agregado_vs_clip.png"), bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# FIGURA 2: distribucion de g por clip, datos reales vs predicciones
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
bins = np.linspace(-20, 10, 61)
for axi, (label, d) in zip(axes, datos.items()):
    axi.hist(g_real, bins=bins, color="#3b6ea5", alpha=0.55, label="datos experimentales")
    axi.hist(d["g"], bins=bins, color="#c1121f", alpha=0.55, label="predicciones de la red")
    axi.axvline(-9.81, color="k", ls="--", lw=1.2, label=r"$g=-9.81$ m/s$^2$")
    axi.set_title(label, fontsize=10)
    axi.set_xlabel(r"$g$ estimada por clip [m/s$^2$]")
axes[0].set_ylabel("número de clips")
axes[0].legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "hist_g_por_clip.png"), bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# FIGURA 3: prediccion vs valor real y trayectorias de ejemplo
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True)
for axi, (label, d) in zip(axes, datos.items()):
    axi.scatter(d["yr"], d["yp"], s=1, alpha=0.1, color="#3b6ea5", rasterized=True)
    lim = [-2.5, 2.2]
    axi.plot(lim, lim, "k--", lw=1)
    axi.set_xlim(lim); axi.set_ylim(lim)
    r = np.corrcoef(d["yr"], d["yp"])[0, 1]
    r2 = 1 - np.mean((d["yp"] - d["yr"]) ** 2) / np.var(d["yr"])
    axi.set_title(f"{label}\n$r={r:.3f}$,  $R^2={r2:.3f}$", fontsize=9)
    axi.set_xlabel("$y$ experimental [m]")
axes[0].set_ylabel("$y$ predicha [m]")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pred_vs_real.png"), bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# FIGURA 4: cuatro clips con las tres predicciones superpuestas
# ------------------------------------------------------------------
bordes = np.cumsum([0] + [n for _, n in segments])
sel = [5, 150, 320, 500]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
for axi, j in zip(axes, sel):
    a, b = bordes[j], bordes[j + 1]
    t, y = clips[j]
    axi.plot(t, y, "o", ms=3.5, color="k", label="experimental")
    for (label, d), col in zip(datos.items(), ["#c1121f", "#2a9d8f", "#e9c46a"]):
        axi.plot(d["t"][a:b], d["yp"][a:b], "-", lw=1.6, color=col, label=label)
    axi.set_title(segments[j][0], fontsize=9)
    axi.set_xlabel("$t$ [s]")
axes[0].set_ylabel("$y$ [m]")
axes[0].legend(fontsize=6.5, loc="best")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "trayectorias_ejemplo.png"), bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# Resumen numerico para las tablas del informe
# ------------------------------------------------------------------
print("\n--- g por clip (datos experimentales) ---")
print(f"media={np.nanmean(g_real):.3f}  mediana={np.nanmedian(g_real):.3f} "
      f"std={np.nanstd(g_real):.3f}  n={len(g_real)}")
print("\n--- por modelo ---")
for label, d in datos.items():
    rmse = np.sqrt(np.mean((d["yp"] - d["yr"]) ** 2))
    mae = np.mean(np.abs(d["yp"] - d["yr"]))
    r2 = 1 - np.mean((d["yp"] - d["yr"]) ** 2) / np.var(d["yr"])
    r = np.corrcoef(d["yr"], d["yp"])[0, 1]
    print(f"{label}: RMSE={rmse:.4f} MAE={mae:.4f} r={r:.4f} R2={r2:.4f} "
          f"| g_clip media={np.nanmean(d['g']):.3f} mediana={np.nanmedian(d['g']):.3f} "
          f"std={np.nanstd(d['g']):.3f}")
print(f"\nRMSE del predictor constante (media): {np.std(list(datos.values())[0]['yr']):.4f}")
