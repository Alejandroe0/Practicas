"""
Figuras del proyecto v2. Se ejecuta después de run_all.py.

    python3 figures.py

Lee ../outs/*.csv y escribe ../figs/*.png
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import baseline
from dataset import load_clips, quality_filter, split_clips

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(AQUI, "..", "..", "Toma_de_datos", "datos"))
OUTS = os.path.abspath(os.path.join(AQUI, "..", "outs"))
FIGS = os.path.abspath(os.path.join(AQUI, "..", "figs"))

G_REF = -9.81
COLOR = {"v1": "#c1121f", "tau": "#e9a020", "tau_v0": "#2a9d8f"}
ETIQUETA = {"v1": "v1:  t → y", "tau": "tau:  τ → Δy", "tau_v0": "tau_v0:  (τ, v₀) → Δy"}
ARQ_ETQ = {"simple": "simple (2)", "media": "media (4)", "tanh_deep": "tanh 32×32"}

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150})


def main():
    os.makedirs(FIGS, exist_ok=True)
    df = pd.read_csv(os.path.join(OUTS, "resultados.csv"))
    dg = pd.read_csv(os.path.join(OUTS, "g_por_clip.csv"))

    formulaciones = ["v1", "tau", "tau_v0"]
    arqs = ["simple", "media", "tanh_deep"]

    # -------------------------------------------------------------- FIG 1
    # g recuperada por formulación y arquitectura
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    w = 0.25
    xs = np.arange(len(arqs))
    for k, form in enumerate(formulaciones):
        m = [df[(df.formulacion == form) & (df.arquitectura == a)].g_mediana.mean()
             for a in arqs]
        s = [df[(df.formulacion == form) & (df.arquitectura == a)].g_mediana.std()
             for a in arqs]
        ax[0].bar(xs + (k - 1) * w, m, w, yerr=s, capsize=3,
                  color=COLOR[form], label=ETIQUETA[form], alpha=0.9)
    ax[0].axhline(G_REF, color="k", ls="--", lw=1.3, label=f"g real = {G_REF}")
    ax[0].set_xticks(xs); ax[0].set_xticklabels([ARQ_ETQ[a] for a in arqs])
    ax[0].set_ylabel(r"$g$ recuperada [m/s$^2$]")
    ax[0].set_title("(a) Recuperación del parámetro físico")
    ax[0].set_ylim(-11.5, 2.6)
    ax[0].legend(fontsize=7.5, loc="upper left", ncol=2, framealpha=0.95)

    for k, form in enumerate(formulaciones):
        m = [df[(df.formulacion == form) & (df.arquitectura == a)].rmse.mean() for a in arqs]
        s = [df[(df.formulacion == form) & (df.arquitectura == a)].rmse.std() for a in arqs]
        ax[1].bar(xs + (k - 1) * w, m, w, yerr=s, capsize=3,
                  color=COLOR[form], label=ETIQUETA[form], alpha=0.9)
    try:
        b = pd.read_csv(os.path.join(OUTS, "baselines.csv"))
        ax[1].axhline(b[b.modelo == "ajuste_por_clip"].rmse.mean(), color="k", ls=":",
                      lw=1.3, label="ajuste por clip (oráculo)")
        ax[1].axhline(b[b.modelo == "fisica_g_global"].rmse.mean(), color="#555", ls="--",
                      lw=1.3, label="MRUV, g global")
    except FileNotFoundError:
        pass
    ax[1].axhline(df.rmse_trivial.mean(), color="#999", ls="-.", lw=1.3,
                  label="predictor constante")
    ax[1].set_xticks(xs); ax[1].set_xticklabels([ARQ_ETQ[a] for a in arqs])
    ax[1].set_ylabel("RMSE en clips no vistos [m]")
    ax[1].set_title("(b) Error de predicción")
    ax[1].legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "01_formulaciones.png"), bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------- FIG 2
    # distribución de g por clip: v1 vs corregida
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    bins = np.linspace(-20, 8, 57)
    for axi, form in zip(axes, formulaciones):
        sub = dg[(dg.formulacion == form) & (dg.arquitectura == "tanh_deep")]
        axi.hist(sub.g_real, bins=bins, color="#3b6ea5", alpha=0.55,
                 label="datos experimentales")
        axi.hist(sub.g_pred, bins=bins, color=COLOR[form], alpha=0.6,
                 label="predicción de la red")
        axi.axvline(G_REF, color="k", ls="--", lw=1.2)
        axi.set_title(ETIQUETA[form], fontsize=9)
        axi.set_xlabel(r"$g$ por clip [m/s$^2$]")
    axes[0].set_ylabel("clips de prueba")
    axes[0].legend(fontsize=7)
    fig.suptitle("Red tanh 32×32, clips no vistos, 5 semillas", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "02_g_por_clip.png"), bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------- FIG 3
    # trayectorias: la misma red en las dos formulaciones
    clips = quality_filter(load_clips(DATA))
    _, _, te = split_clips(clips, seed=0)
    npz = np.load(os.path.join(OUTS, "predicciones.npz"))

    sel = [0, 1, 2, 3]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for fila, form in enumerate(["v1", "tau_v0"]):
        arr = npz[f"{form}|tanh_deep"]
        t, y_true, y_pred, cid = arr
        for col, j in enumerate(sel):
            axi = axes[fila, col]
            m = cid == j
            orden = np.argsort(t[m])
            axi.plot(t[m][orden], y_true[m][orden], "o", ms=3.5, color="k",
                     label="experimental")
            axi.plot(t[m][orden], y_pred[m][orden], "-", lw=1.8, color=COLOR[form],
                     label="red")
            if fila == 0:
                axi.set_title(te[j].nombre, fontsize=9)
            if col == 0:
                etq = "v1:  $y$ [m]" if form == "v1" else r"v2:  $\Delta y$ [m]"
                axi.set_ylabel(etq, color=COLOR[form], fontweight="bold")
            axi.set_xlabel("$t$ [s]" if form == "v1" else r"$\tau$ [s]")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Arriba: formulación v1 (una sola curva para todos los clips).  "
                 "Abajo: formulación corregida.", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "03_trayectorias.png"), bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------- FIG 4
    # error en g frente a RMSE, todas las corridas
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    marca = {"simple": "o", "media": "s", "tanh_deep": "^"}
    for form in formulaciones:
        for a in arqs:
            sub = df[(df.formulacion == form) & (df.arquitectura == a)]
            ax.scatter(sub.rmse, sub["err_rel_%"], s=45, marker=marca[a],
                       color=COLOR[form], alpha=0.8,
                       label=f"{ETIQUETA[form].split(':')[0]} / {ARQ_ETQ[a]}")
    ax.axhline(5, color="k", ls=":", lw=1)
    ax.text(ax.get_xlim()[1], 5.6, "5 % de error en $g$", ha="right", fontsize=8)
    ax.set_xlabel("RMSE en clips no vistos [m]")
    ax.set_ylabel(r"error relativo en $g$ [%]")
    ax.set_yscale("log")
    ax.set_title("Cada punto es una corrida (formulación × arquitectura × semilla)")
    ax.legend(fontsize=6.5, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "04_rmse_vs_error_g.png"), bbox_inches="tight")
    plt.close(fig)

    print(f"Figuras escritas en {FIGS}/")
    for f in sorted(os.listdir(FIGS)):
        print("  ", f)


if __name__ == "__main__":
    main()
