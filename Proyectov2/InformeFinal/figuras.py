"""
Figuras del informe final (versión de resultados).

Se ejecuta después de ``../src/run_all.py``; lee ``../outs/*.csv`` y escribe
``imgs/*.png``.

    python3 figuras.py

Es una variante de ``../src/figures.py`` con las etiquetas de las tres
formulaciones escritas tal como aparecen en el informe, más una figura
adicional de caracterización del conjunto de datos.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(AQUI, "..", "src"))
OUTS = os.path.abspath(os.path.join(AQUI, "..", "outs"))
IMGS = os.path.join(AQUI, "imgs")
DATA = os.path.abspath(os.path.join(AQUI, "..", "..", "Toma_de_datos", "datos"))

sys.path.insert(0, SRC)
from dataset import load_clips, quality_filter, split_clips  # noqa: E402

G_REF = -9.81

# Claves internas del código -> etiquetas del informe.
FORMS = ["v1", "tau", "tau_v0"]
COLOR = {"v1": "#c1121f", "tau": "#e9a020", "tau_v0": "#2a9d8f"}
ETIQUETA = {
    "v1": r"directa:  $t \rightarrow y$",
    "tau": r"referida:  $\tau \rightarrow \Delta y$",
    "tau_v0": r"completa:  $(\tau, v_0) \rightarrow \Delta y$",
}
CORTA = {"v1": "directa", "tau": "referida", "tau_v0": "completa"}
ARQS = ["simple", "media", "tanh_deep"]
ARQ_ETQ = {"simple": "simple (2)", "media": "media (4)", "tanh_deep": "tanh 32×32"}

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150})


def fig_datos():
    """Caracterización del conjunto experimental: clips y g por clip."""
    crudos = load_clips(DATA)
    clips = quality_filter(crudos)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    paso = max(1, len(clips) // 6)
    for c in clips[::paso][:6]:
        ax[0].plot(c.tau, c.dy, "o", ms=3, alpha=0.75)
        tt = np.linspace(0, c.tau[-1], 100)
        p = np.polyfit(c.tau, c.dy, 2)
        ax[0].plot(tt, np.polyval(p, tt), "-", lw=1, color="k", alpha=0.35)
    ax[0].set_xlabel(r"$\tau = t - t_0$ [s]")
    ax[0].set_ylabel(r"$\Delta y = y - y_0$ [m]")
    ax[0].set_title("(a) Seis clips referidos a su propio origen")

    g = np.array([c.g_fit for c in clips])
    ax[1].hist(g, bins=np.linspace(-20, 2, 45), color="#3b6ea5", alpha=0.8)
    ax[1].axvline(G_REF, color="k", ls="--", lw=1.3, label=f"$g$ real = {G_REF}")
    ax[1].axvline(np.median(g), color="#c1121f", ls="-", lw=1.3,
                  label=f"mediana = {np.median(g):.2f}")
    ax[1].set_xlabel(r"$g$ del ajuste por clip [m/s$^2$]")
    ax[1].set_ylabel("clips")
    ax[1].set_title(f"(b) Distribución sobre {len(clips)} clips")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(IMGS, "00_datos.png"), bbox_inches="tight")
    plt.close(fig)


def fig_formulaciones(df):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    w = 0.25
    xs = np.arange(len(ARQS))

    for k, form in enumerate(FORMS):
        m = [df[(df.formulacion == form) & (df.arquitectura == a)].g_mediana.mean()
             for a in ARQS]
        s = [df[(df.formulacion == form) & (df.arquitectura == a)].g_mediana.std()
             for a in ARQS]
        ax[0].bar(xs + (k - 1) * w, m, w, yerr=s, capsize=3,
                  color=COLOR[form], label=ETIQUETA[form], alpha=0.9)
    ax[0].axhline(G_REF, color="k", ls="--", lw=1.3, label=f"$g$ real = {G_REF}")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([ARQ_ETQ[a] for a in ARQS])
    ax[0].set_ylabel(r"$g$ recuperada [m/s$^2$]")
    ax[0].set_title("(a) Recuperación del parámetro físico")
    ax[0].set_ylim(-11.5, 2.6)
    ax[0].legend(fontsize=7.5, loc="upper left", ncol=2, framealpha=0.95)

    for k, form in enumerate(FORMS):
        m = [df[(df.formulacion == form) & (df.arquitectura == a)].rmse.mean()
             for a in ARQS]
        s = [df[(df.formulacion == form) & (df.arquitectura == a)].rmse.std()
             for a in ARQS]
        ax[1].bar(xs + (k - 1) * w, m, w, yerr=s, capsize=3,
                  color=COLOR[form], label=ETIQUETA[form], alpha=0.9)
    b = pd.read_csv(os.path.join(OUTS, "baselines.csv"))
    ax[1].axhline(b[b.modelo == "ajuste_por_clip"].rmse.mean(), color="k", ls=":",
                  lw=1.3, label="ajuste por clip (oráculo)")
    ax[1].axhline(b[b.modelo == "fisica_g_global"].rmse.mean(), color="#555",
                  ls="--", lw=1.3, label="MRUV, $g$ global")
    ax[1].axhline(df.rmse_trivial.mean(), color="#999", ls="-.", lw=1.3,
                  label="predictor constante")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([ARQ_ETQ[a] for a in ARQS])
    ax[1].set_ylabel("RMSE en clips no vistos [m]")
    ax[1].set_title("(b) Error de predicción")
    ax[1].legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(IMGS, "01_formulaciones.png"), bbox_inches="tight")
    plt.close(fig)


def fig_g_por_clip(dg):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    bins = np.linspace(-20, 8, 57)
    for axi, form in zip(axes, FORMS):
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
    fig.savefig(os.path.join(IMGS, "02_g_por_clip.png"), bbox_inches="tight")
    plt.close(fig)


def fig_trayectorias():
    clips = quality_filter(load_clips(DATA))
    _, _, te = split_clips(clips, seed=0)
    npz = np.load(os.path.join(OUTS, "predicciones.npz"))

    sel = [0, 1, 2, 3]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for fila, form in enumerate(["v1", "tau_v0"]):
        t, y_true, y_pred, cid = npz[f"{form}|tanh_deep"]
        for col, j in enumerate(sel):
            axi = axes[fila, col]
            m = cid == j
            orden = np.argsort(t[m])
            axi.plot(t[m][orden], y_true[m][orden], "o", ms=3.5, color="k",
                     label="experimental")
            axi.plot(t[m][orden], y_pred[m][orden], "-", lw=1.8,
                     color=COLOR[form], label="red")
            if fila == 0:
                axi.set_title(te[j].nombre, fontsize=9)
            if col == 0:
                etq = ("directa:  $y$ [m]" if form == "v1"
                       else r"completa:  $\Delta y$ [m]")
                axi.set_ylabel(etq, color=COLOR[form], fontweight="bold")
            axi.set_xlabel("$t$ [s]" if form == "v1" else r"$\tau$ [s]")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Arriba: formulación directa (una sola curva para todos los "
                 "clips).  Abajo: formulación completa.", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(IMGS, "03_trayectorias.png"), bbox_inches="tight")
    plt.close(fig)


def fig_scatter(df):
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    marca = {"simple": "o", "media": "s", "tanh_deep": "^"}
    for form in FORMS:
        for a in ARQS:
            sub = df[(df.formulacion == form) & (df.arquitectura == a)]
            ax.scatter(sub.rmse, sub["err_rel_%"], s=45, marker=marca[a],
                       color=COLOR[form], alpha=0.8,
                       label=f"{CORTA[form]} / {ARQ_ETQ[a]}")
    ax.axhline(5, color="k", ls=":", lw=1)
    ax.text(ax.get_xlim()[1], 5.6, "5 % de error en $g$", ha="right", fontsize=8)
    ax.set_xlabel("RMSE en clips no vistos [m]")
    ax.set_ylabel(r"error relativo en $g$ [%]")
    ax.set_yscale("log")
    ax.set_title("Cada punto es una corrida (formulación × arquitectura × semilla)")
    ax.legend(fontsize=6.5, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(IMGS, "04_rmse_vs_error_g.png"), bbox_inches="tight")
    plt.close(fig)


def fig_epocas():
    """Coste y beneficio de entrenar más épocas (formulación completa)."""
    de = pd.read_csv(os.path.join(OUTS, "epocas.csv"))
    col = {"simple": "#c1121f", "media": "#e9a020", "tanh_deep": "#2a9d8f"}

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    paneles = [("tiempo_s", "tiempo acumulado [s]", "(a) Coste"),
               ("rmse", "RMSE en clips no vistos [m]", "(b) Error de posición"),
               ("err_rel_%", r"error relativo en $g$ [%]",
                "(c) Recuperación de la física")]

    for axi, (col_y, etq, titulo) in zip(ax, paneles):
        for a in ARQS:
            sub = de[de.arquitectura == a].groupby("epoca")[col_y]
            m, s = sub.mean(), sub.std()
            axi.plot(m.index, m.values, "o-", ms=3.5, lw=1.5, color=col[a],
                     label=ARQ_ETQ[a])
            axi.fill_between(m.index, m - s, m + s, color=col[a], alpha=0.15)
        axi.set_xscale("log")
        axi.set_xlabel("épocas de entrenamiento")
        axi.set_ylabel(etq)
        axi.set_title(titulo, fontsize=10)
    ax[0].set_yscale("log")
    ax[2].axhline(5, color="k", ls=":", lw=1)
    ax[2].text(1.1, 5.6, "5 %", fontsize=8)
    ax[0].legend(fontsize=7.5)

    fig.suptitle("Formulación completa, media ± desv. est. sobre 5 semillas",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(IMGS, "05_epocas.png"), bbox_inches="tight")
    plt.close(fig)


def fig_pesos():
    """Inspección de la red tanh 32×32: ¿está la física en su estructura?"""
    z = np.load(os.path.join(OUTS, "pesos.npz"))
    g_exp = float(np.median(z["g_datos"]))

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))

    # (a) curvatura punto a punto
    tau, d2 = z["tau_d"], z["d2_d"]
    ax[0].scatter(tau, d2, s=4, alpha=0.15, color="#2a9d8f", edgecolors="none")
    bordes = np.linspace(0, 0.6, 13)
    ctr, med = [], []
    for a, b in zip(bordes[:-1], bordes[1:]):
        k = (tau >= a) & (tau < b)
        if k.sum() > 30:
            ctr.append(0.5 * (a + b))
            med.append(np.median(d2[k]))
    ax[0].plot(ctr, med, "o-", color="#c1121f", ms=4, lw=1.8,
               label="mediana por tramo")
    ax[0].axhline(g_exp, color="k", ls="--", lw=1.3,
                  label=f"$g$ de los datos = {g_exp:.2f}")
    ax[0].set_xlim(0, 0.65)
    ax[0].set_ylim(-28, 10)
    ax[0].set_xlabel(r"$\tau$ [s]")
    ax[0].set_ylabel(r"$\partial^2 \Delta y / \partial \tau^2$ [m/s$^2$]")
    ax[0].set_title("(a) Curvatura de la función aprendida", fontsize=10)
    ax[0].legend(fontsize=7.5, loc="lower right")

    # (b) g leída de la ecuación agrupada: la red sigue a los datos
    grupos = [("todo el dominio", z["coef_datos_todo"], z["coef_red_todo"]),
              (r"$\tau < 0.6$ s", z["coef_datos_06"], z["coef_red_06"])]
    xs = np.arange(len(grupos))
    w = 0.26
    for k, (etq, c) in enumerate([("física ideal", "#333333"),
                                  ("datos", "#3b6ea5"), ("red", "#2a9d8f")]):
        if k == 0:
            v = [G_REF] * len(grupos)
        else:
            j = 1 if k == 1 else 2
            v = [2 * gr[j][2] for gr in grupos]
        ax[1].bar(xs + (k - 1) * w, v, w, color=c, label=etq, alpha=0.9)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([g[0] for g in grupos])
    ax[1].set_ylabel(r"$g$ del ajuste agrupado [m/s$^2$]")
    ax[1].set_title(r"(b) $g$ leída de $\Delta y = a\tau + b\,v_0\tau + c\tau^2$",
                    fontsize=9.5)
    ax[1].legend(fontsize=7.5, loc="lower right")

    # (c) primera capa
    W1, contrib = z["W1"], z["contrib"]
    s = 20 + 400 * contrib / contrib.max()
    sc = ax[2].scatter(W1[:, 0], W1[:, 1], s=s, c=contrib, cmap="viridis",
                       alpha=0.8, edgecolors="k", linewidths=0.4)
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].axvline(0, color="k", lw=0.6)
    ax[2].set_xlabel(r"peso sobre $\tau$ (normalizado)")
    ax[2].set_ylabel(r"peso sobre $v_0$ (normalizado)")
    ax[2].set_title("(c) Las 32 neuronas de la primera capa", fontsize=10)
    fig.colorbar(sc, ax=ax[2], label="contribución a la salida")

    fig.tight_layout()
    fig.savefig(os.path.join(IMGS, "06_pesos.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(IMGS, exist_ok=True)
    df = pd.read_csv(os.path.join(OUTS, "resultados.csv"))
    dg = pd.read_csv(os.path.join(OUTS, "g_por_clip.csv"))

    fig_datos()
    fig_formulaciones(df)
    fig_g_por_clip(dg)
    fig_trayectorias()
    fig_scatter(df)
    fig_epocas()
    fig_pesos()

    print(f"Figuras escritas en {IMGS}/")
    for f in sorted(os.listdir(IMGS)):
        print("  ", f)


if __name__ == "__main__":
    main()
