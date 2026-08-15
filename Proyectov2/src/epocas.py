"""
Estudio del número de épocas: ¿cuánto cuesta y cuánto se gana entrenando más?

Se entrena cada arquitectura bajo la formulación ``tau_v0`` durante un número
máximo de épocas y se evalúa en puntos de control a lo largo del *mismo*
entrenamiento. Así el tiempo que se reporta para la época N es el tiempo real
acumulado hasta llegar a ella, y la curva de error es una trayectoria de
aprendizaje y no una colección de entrenamientos independientes.

Se mide en cada punto de control:

* el coste  --- tiempo acumulado de entrenamiento;
* el nivel 1 --- RMSE y R² sobre clips de prueba no vistos;
* el nivel 2 --- la ``g`` recuperada del ajuste por clip.

El nivel 2 es el que interesa: nada garantiza que la física aparezca al mismo
ritmo al que baja el error de posición.

Uso:
    python3 epocas.py                 # 5 semillas, hasta 800 épocas
    python3 epocas.py --seeds 2       # versión rápida

Salida:
    ../outs/epocas.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate as ev
from dataset import (Normalizer, build_arrays, load_clips, quality_filter,
                     split_clips)
from models import ARQUITECTURAS, build_model
from run_all import semilla_de
from train import predict, set_seed

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(AQUI, "..", "..", "Toma_de_datos", "datos"))
OUTS = os.path.abspath(os.path.join(AQUI, "..", "outs"))

G_REF = -9.81
FORM = "tau_v0"
CORTES = [1, 2, 5, 10, 20, 50, 100, 200, 400, 800]


def entrena_con_cortes(model, Xtr, Ytr, Xva, Yva, cortes, lr, batch_size,
                       evalua):
    """Entrena y llama a ``evalua`` en cada época de la lista ``cortes``.

    El cronómetro se detiene mientras se evalúa, de modo que el tiempo
    acumulado que se reporta es solo el de entrenamiento.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    xt, yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    xv, yv = torch.from_numpy(Xva), torch.from_numpy(Yva)
    n = len(xt)

    corte = set(cortes)
    t_acum = 0.0
    for ep in range(1, max(cortes) + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            loss = mse(model(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        t_acum += time.time() - t0

        if ep in corte:
            model.eval()
            with torch.no_grad():
                va = mse(model(xv), yv).item()
            evalua(ep, t_acum, tot / n, va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-resid", type=float, default=0.08)
    args = ap.parse_args()

    os.makedirs(OUTS, exist_ok=True)
    clips = quality_filter(load_clips(DATA), args.max_resid)
    print(f"{len(clips)} clips tras el filtro de calidad\n")

    filas = []
    t_ini = time.time()

    for seed in range(args.seeds):
        tr_c, va_c, te_c = split_clips(clips, seed=seed)
        Xtr, Ytr, _ = build_arrays(tr_c, FORM)
        Xva, Yva, _ = build_arrays(va_c, FORM)
        Xte, Yte, cid_te = build_arrays(te_c, FORM)
        nrm = Normalizer(Xtr, Ytr)
        t_eje = np.concatenate([c.tau for c in te_c])
        print(f"{'-' * 70}\nSemilla {seed}")

        for arq in ARQUITECTURAS:
            set_seed(semilla_de(seed, FORM, arq))
            model = build_model(arq, Xtr.shape[1])

            def evalua(ep, t_acum, tr_mse, va_mse, _arq=arq, _seed=seed):
                yhat = nrm.iy(predict(model, nrm.fx(Xte)))
                m = ev.metricas(Yte, yhat)
                g_pred, _ = ev.g_por_clip(te_c, cid_te, t_eje, yhat)
                rg = ev.resumen_g(g_pred, G_REF)
                filas.append(dict(seed=_seed, arquitectura=_arq, epoca=ep,
                                  tiempo_s=t_acum, train_mse=tr_mse,
                                  val_mse=va_mse, **m, **rg))

            entrena_con_cortes(model, nrm.fx(Xtr), nrm.fy(Ytr),
                               nrm.fx(Xva), nrm.fy(Yva), CORTES,
                               args.lr, args.batch_size, evalua)

            ult = filas[-1]
            print(f"  [{arq:9s}] {ult['epoca']:4d} ep  {ult['tiempo_s']:6.1f} s  "
                  f"RMSE={ult['rmse']:.4f}  g={ult['g_mediana']:+7.3f}")

    print(f"\nTiempo total: {time.time() - t_ini:.1f} s")

    df = pd.DataFrame(filas)
    destino = os.path.join(OUTS, "epocas.csv")
    df.to_csv(destino, index=False)
    print(f"Escrito {destino}  ({len(df)} filas)")

    # Resumen en pantalla: media sobre semillas.
    print("\nMedia sobre semillas (formulación tau_v0)")
    g = (df.groupby(["arquitectura", "epoca"])
           [["tiempo_s", "rmse", "r2", "g_mediana", "err_rel_%"]].mean())
    print(g.round(4).to_string())


if __name__ == "__main__":
    main()
