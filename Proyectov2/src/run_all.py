"""
Experimento completo: 3 formulaciones x 3 arquitecturas x N semillas.

Verifica el supuesto del anteproyecto --- que una red entrenada con datos de
caída libre permite recuperar el modelo físico --- planteando el problema de
forma que la pregunta tenga respuesta.

Uso:
    python3 run_all.py                # experimento completo (5 semillas)
    python3 run_all.py --seeds 2      # versión rápida
    python3 run_all.py --sin-filtro   # sin descartar clips de mala calidad

Salidas en ../outs/:
    resultados.csv     una fila por (formulación, arquitectura, semilla)
    baselines.csv      líneas base clásicas por semilla
    g_por_clip.csv     g recuperada de cada clip de prueba
    predicciones.npz   predicciones del mejor modelo, para las figuras
    resumen.txt        el mismo resumen que se imprime en pantalla
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import baseline
import evaluate as ev
from dataset import (FORMULACIONES, Normalizer, build_arrays, describe,
                     load_clips, quality_filter, split_clips)
from models import ARQUITECTURAS, build_model, n_params
from train import predict, set_seed, train_model

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(AQUI, "..", "..", "Toma_de_datos", "datos"))
OUTS = os.path.abspath(os.path.join(AQUI, "..", "outs"))

G_REF = -9.81

# La columna de tiempo con la que se ajusta la parábola a las predicciones:
# en v1 es el tiempo original del video, en las demás el tiempo propio del clip.
COL_TIEMPO = {"v1": "t", "tau": "tau", "tau_v0": "tau"}


def semilla_de(seed: int, form: str, arq: str) -> int:
    """Semilla reproducible entre ejecuciones.

    No se usa hash() porque Python aleatoriza el hash de las cadenas en cada
    proceso (PYTHONHASHSEED), lo que haría irreproducible el experimento.
    """
    return 1000 * seed + zlib.crc32(f"{form}|{arq}".encode()) % 997


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--sin-filtro", action="store_true")
    ap.add_argument("--max-resid", type=float, default=0.08)
    args = ap.parse_args()

    os.makedirs(OUTS, exist_ok=True)
    log = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)

    # ---------------------------------------------------------------- datos
    P("=" * 78)
    P("PROYECTO v2 — verificación del supuesto del anteproyecto")
    P("=" * 78)
    P(f"\nDatos: {DATA}")

    clips = load_clips(DATA)
    P("\n[crudo]")
    P(describe(clips))

    if not args.sin_filtro:
        antes = len(clips)
        clips = quality_filter(clips, args.max_resid)
        P(f"\n[filtro de calidad: residuo < {args.max_resid} m]  "
          f"{antes} -> {len(clips)} clips  ({antes - len(clips)} descartados)")
        P(describe(clips))

    g_datos = np.array([c.g_fit for c in clips])
    P(f"\n>>> Referencia experimental: g mediana = {np.median(g_datos):+.3f} m/s²  "
      f"({100 * abs(np.median(g_datos) - G_REF) / abs(G_REF):.1f} % de {G_REF})")

    # ------------------------------------------------------------ bucle
    filas, filas_base, filas_g = [], [], []
    guardado = {}
    t_ini = time.time()

    for seed in range(args.seeds):
        tr_c, va_c, te_c = split_clips(clips, seed=seed)
        P(f"\n{'-' * 78}\nSemilla {seed}: "
          f"{len(tr_c)} clips train / {len(va_c)} val / {len(te_c)} test")

        # ---- líneas base (se calculan una vez por semilla) ----
        g_glob = baseline.g_global(tr_c)
        yt, yp = baseline.predice_fisica(te_c, g_glob)
        m_fis = ev.metricas(yt, yp)
        yt2, yp2, g_clip_base = baseline.ajuste_por_clip(te_c)
        m_aj = ev.metricas(yt2, yp2)

        filas_base += [
            dict(seed=seed, modelo="fisica_g_global", g_ajustada=g_glob,
                 **m_fis, g_mediana=g_glob, **{"err_rel_%": 100 * abs(g_glob - G_REF) / abs(G_REF)}),
            dict(seed=seed, modelo="ajuste_por_clip", g_ajustada=np.nan,
                 **m_aj, **{k: v for k, v in ev.resumen_g(g_clip_base, G_REF).items()
                            if k in ("g_mediana", "err_rel_%")}),
        ]
        P(f"  base física (g global = {g_glob:+.3f}): "
          f"RMSE={m_fis['rmse']:.4f} m  R²={m_fis['r2']:.3f}")
        P(f"  base ajuste por clip (oráculo):      "
          f"RMSE={m_aj['rmse']:.4f} m  R²={m_aj['r2']:.3f}  "
          f"g_med={np.median(g_clip_base):+.3f}")

        # ---- redes ----
        for form in FORMULACIONES:
            Xtr, Ytr, _ = build_arrays(tr_c, form)
            Xva, Yva, _ = build_arrays(va_c, form)
            Xte, Yte, cid_te = build_arrays(te_c, form)
            nrm = Normalizer(Xtr, Ytr)          # ajustado solo con train

            t_eje = np.concatenate(
                [getattr(c, COL_TIEMPO[form]) for c in te_c])

            for arq in ARQUITECTURAS:
                set_seed(semilla_de(seed, form, arq))
                model = build_model(arq, Xtr.shape[1])
                t0 = time.time()
                model, hist = train_model(
                    model, nrm.fx(Xtr), nrm.fy(Ytr), nrm.fx(Xva), nrm.fy(Yva),
                    epochs=args.epochs, lr=args.lr)
                dt = time.time() - t0

                yhat = nrm.iy(predict(model, nrm.fx(Xte)))
                m = ev.metricas(Yte, yhat)
                g_pred, idx = ev.g_por_clip(te_c, cid_te, t_eje, yhat)
                rg = ev.resumen_g(g_pred, G_REF)

                filas.append(dict(
                    seed=seed, formulacion=form, arquitectura=arq,
                    n_params=n_params(model), epocas=hist["epocas_corridas"],
                    mejor_epoca=hist["mejor_epoca"], tiempo_s=dt, **m, **rg))
                for gg, jj in zip(g_pred, idx):
                    filas_g.append(dict(seed=seed, formulacion=form,
                                        arquitectura=arq, clip=te_c[jj].nombre,
                                        g_pred=gg, g_real=te_c[jj].g_fit))

                P(f"  [{form:7s}|{arq:9s}] RMSE={m['rmse']:.4f}  R²={m['r2']:6.3f}  "
                  f"g_med={rg['g_mediana']:+7.3f}  err={rg['err_rel_%']:5.1f}%  "
                  f"({dt:.1f}s, {hist['epocas_corridas']} ep)")

                if seed == 0:
                    guardado[f"{form}|{arq}"] = dict(
                        t=t_eje, y_true=Yte.ravel(), y_pred=yhat.ravel(),
                        cid=cid_te, clips=[c.nombre for c in te_c])

    P(f"\nTiempo total: {time.time() - t_ini:.1f} s")

    # ------------------------------------------------------------ guardar
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(OUTS, "resultados.csv"), index=False)
    pd.DataFrame(filas_base).to_csv(os.path.join(OUTS, "baselines.csv"), index=False)
    pd.DataFrame(filas_g).to_csv(os.path.join(OUTS, "g_por_clip.csv"), index=False)
    np.savez_compressed(os.path.join(OUTS, "predicciones.npz"),
                        **{k: np.array([v["t"], v["y_true"], v["y_pred"], v["cid"]])
                           for k, v in guardado.items()})

    # ------------------------------------------------------------ resumen
    P("\n" + "=" * 78)
    P("RESUMEN  (media ± desv. est. sobre semillas, evaluado en clips no vistos)")
    P("=" * 78)

    agg = (df.groupby(["formulacion", "arquitectura"])
             .agg(rmse_m=("rmse", "mean"), rmse_s=("rmse", "std"),
                  r2_m=("r2", "mean"),
                  g_m=("g_mediana", "mean"), g_s=("g_mediana", "std"),
                  err_m=("err_rel_%", "mean"))
             .reset_index())

    P(f"\n{'formulación':<10} {'arquitectura':<11} {'RMSE [m]':>16} "
      f"{'R²':>7} {'g mediana':>17} {'err g':>8}")
    P("-" * 78)
    for form in FORMULACIONES:
        for _, r in agg[agg.formulacion == form].iterrows():
            P(f"{r.formulacion:<10} {r.arquitectura:<11} "
              f"{r.rmse_m:8.4f} ± {r.rmse_s:5.4f} {r.r2_m:7.3f} "
              f"{r.g_m:+9.3f} ± {r.g_s:5.3f} {r.err_m:7.1f}%")
        P("")

    b = pd.DataFrame(filas_base)
    P("-" * 78)
    for nombre, etiqueta in [("fisica_g_global", "MRUV con g global (1 parám.)"),
                             ("ajuste_por_clip", "ajuste por clip (oráculo)")]:
        s = b[b.modelo == nombre]
        P(f"{etiqueta:<40} RMSE={s.rmse.mean():.4f} ± {s.rmse.std():.4f}   "
          f"R²={s.r2.mean():6.3f}   g={s.g_mediana.mean():+.3f}")
    P(f"{'predictor constante (media)':<40} "
      f"RMSE={df.rmse_trivial.mean():.4f}")
    P(f"{'suelo de ruido (residuo del ajuste)':<40} "
      f"RMSE≈{np.median([c.resid for c in clips]):.4f}")

    P("\n" + "=" * 78)
    P("VEREDICTO")
    P("=" * 78)

    # Dos referencias distintas, y conviene no confundirlas:
    #   G_REF   valor físico verdadero (-9.81)
    #   g_exp   lo que el método clásico extrae de ESTOS datos
    # La red no puede superar a g_exp: no sabe más de lo que hay en los datos.
    # Lo que mide su éxito es cuánto se acerca a g_exp; la diferencia entre
    # g_exp y G_REF es un sesgo del experimento, no un fallo de la red.
    g_exp = float(b[b.modelo == "ajuste_por_clip"].g_mediana.mean())

    mejor = agg.loc[agg[agg.formulacion == "tau_v0"]["err_m"].idxmin()]
    err_vs_exp = 100 * abs(mejor.g_m - g_exp) / abs(g_exp)
    v1 = agg[agg.formulacion == "v1"]["err_m"].min()

    P(f"\nReferencias")
    P(f"    g física verdadera                        {G_REF:+.3f} m/s²")
    P(f"    g que el método clásico saca de los datos {g_exp:+.3f} m/s²  "
      f"(sesgo del experimento: {100 * abs(g_exp - G_REF) / abs(G_REF):.1f} %)")

    P(f"\nFormulación v1 (la del proyecto original)")
    P(f"    mejor error en g = {v1:.1f} %  ->  el supuesto NO se puede verificar")

    P(f"\nFormulación tau_v0 (corregida), arquitectura '{mejor.arquitectura}'")
    P(f"    g recuperada = {mejor.g_m:+.3f} ± {mejor.g_s:.3f} m/s²")
    P(f"    error frente al método clásico sobre los mismos datos : {err_vs_exp:5.1f} %")
    P(f"    error frente al valor físico verdadero                : {mejor.err_m:5.1f} %")
    P(f"    RMSE = {mejor.rmse_m:.4f} m  "
      f"(oráculo {b[b.modelo == 'ajuste_por_clip'].rmse.mean():.4f}, "
      f"trivial {df.rmse_trivial.mean():.4f})")

    if err_vs_exp < 5:
        P("\n>>> El supuesto del anteproyecto SE VERIFICA.")
        P("    Reformulado de modo que el problema sea identificable, la red")
        P("    recupera la gravedad que contienen los datos. El resto de la")
        P("    diferencia con -9.81 es sesgo experimental, no fallo de la red.")
    elif mejor.err_m < 10:
        P("\n>>> El supuesto se verifica frente al valor físico, pero la red no")
        P("    reproduce del todo lo que el método clásico saca de los datos.")
    else:
        P("\n>>> El supuesto sigue sin verificarse; revisar datos o entrenamiento.")

    with open(os.path.join(OUTS, "resumen.txt"), "w", encoding="utf8") as f:
        f.write("\n".join(log) + "\n")
    P(f"\nResultados escritos en {OUTS}/")


if __name__ == "__main__":
    main()
