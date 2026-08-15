"""
Inspección de la red entrenada: ¿está la física *dentro* de la red?

El objetivo del anteproyecto no era solo predecir bien, sino «entender la
estructura interna de las redes neuronales aplicadas a la elaboración de
modelos matemáticos». Ajustar una parábola a las *salidas* de la red dice que
la red se comporta como la física; este script mira la función que la red
implementa, para ver si la física está en su estructura.

Tres análisis, de menos a más informativo:

1. **Red simple (2 unidades ReLU).** Con 9 parámetros la función se puede
   escribir a mano. Se imprime explícitamente y se ve por qué no puede ser una
   parábola.

2. **Derivadas de la función aprendida** (red tanh 32x32, por diferenciación
   automática y en unidades físicas). Si la red implementa
   ``dy = v0*tau + g*tau^2/2``, entonces necesariamente

       d2(dy)/dtau2 = g       (constante, no depende de tau ni de v0)
       d(dy)/dv0    = tau     (recta de pendiente 1 por el origen)
       d(dy)/dtau   = v0 + g*tau

   Son tres predicciones falsables sobre el interior de la red, mucho más
   exigentes que ajustar una parábola a sus salidas.

3. **Lectura de la ecuación aprendida.** Se ajusta la superficie de salida de
   la red al modelo ``dy = a*tau + b*v0*tau + c*tau^2`` y se comparan los
   coeficientes con los que dicta la física: ``(0, 1, g/2)``.

Uso:
    python3 pesos.py                # semilla 0
    python3 pesos.py --seed 3

Salidas:
    ../outs/pesos.txt      informe legible
    ../outs/pesos.npz      mallas de derivadas y pesos, para las figuras
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evaluate as ev
from dataset import (Normalizer, build_arrays, load_clips, quality_filter,
                     split_clips)
from models import build_model, n_params
from run_all import semilla_de
from train import predict, set_seed, train_model

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(AQUI, "..", "..", "Toma_de_datos", "datos"))
OUTS = os.path.abspath(os.path.join(AQUI, "..", "outs"))

G_REF = -9.81
FORM = "tau_v0"


# ---------------------------------------------------------------------------
# Derivadas en unidades físicas
# ---------------------------------------------------------------------------
def derivadas(model, nrm, tau, v0):
    """Devuelve (dy, d_dtau, d2_dtau2, d_dv0) en unidades físicas.

    La red trabaja normalizada, ``dy = ys*f((x - xm)/xs) + ym``. Aquí se deriva
    respecto de la entrada *física* ``x``, de modo que la diferenciación
    automática ya aplica el ``1/xs`` de la regla de la cadena; lo único que
    falta es el factor ``ys`` de la salida.
    """
    xm = torch.from_numpy(nrm.xm.astype(np.float64))
    xs = torch.from_numpy(nrm.xs.astype(np.float64))
    ys = float(nrm.ys.ravel()[0])

    X = torch.tensor(np.column_stack([tau, v0]), dtype=torch.float64,
                     requires_grad=True)
    model = model.double()
    f = model((X - xm) / xs).squeeze(-1)

    g1, = torch.autograd.grad(f.sum(), X, create_graph=True)
    d_dtau, d_dv0 = g1[:, 0], g1[:, 1]
    g2, = torch.autograd.grad(d_dtau.sum(), X, create_graph=False)

    return (nrm.iy(f.detach().numpy()[:, None]).ravel(),
            ys * d_dtau.detach().numpy(),
            ys * g2[:, 0].detach().numpy(),
            ys * d_dv0.detach().numpy())


def formula_relu(model, nrm) -> str:
    """Escribe explícitamente la función de la red de 2 unidades ReLU."""
    W1 = model.net[0].weight.detach().numpy()
    b1 = model.net[0].bias.detach().numpy()
    W2 = model.net[2].weight.detach().numpy().ravel()
    b2 = float(model.net[2].bias.detach().numpy()[0])

    xm, xs = nrm.xm.ravel(), nrm.xs.ravel()
    ys, ym = float(nrm.ys.ravel()[0]), float(nrm.ym.ravel()[0])

    lineas = ["  En unidades físicas, con tau en s y v0 en m/s:"]
    for i in range(W1.shape[0]):
        # w . (x - xm)/xs + b  =  (w/xs) . x + (b - w.xm/xs)
        a = W1[i] / xs
        c = b1[i] - float(np.dot(W1[i], xm / xs))
        lineas.append(f"    h{i} = max(0,  {a[0]:+.4f}*tau {a[1]:+.4f}*v0 {c:+.4f})")
    term = "  ".join(f"{ys * W2[i]:+.4f}*h{i}" for i in range(len(W2)))
    lineas.append(f"    dy = {term}  {ys * b2 + ym:+.4f}")
    return "\n".join(lineas)


def ajusta_mruv(tau, v0, dy):
    """Ajusta ``dy = a*tau + b*v0*tau + c*tau^2``; la física predice (0, 1, g/2)."""
    A = np.column_stack([tau, v0 * tau, tau ** 2])
    coef, *_ = np.linalg.lstsq(A, dy, rcond=None)
    r2 = 1 - np.var(dy - A @ coef) / np.var(dy)
    return coef, float(r2)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-resid", type=float, default=0.08)
    args = ap.parse_args()

    os.makedirs(OUTS, exist_ok=True)
    log = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.append(s)

    clips = quality_filter(load_clips(DATA), args.max_resid)
    tr_c, va_c, te_c = split_clips(clips, seed=args.seed)
    Xtr, Ytr, _ = build_arrays(tr_c, FORM)
    Xva, Yva, _ = build_arrays(va_c, FORM)
    Xte, Yte, cid_te = build_arrays(te_c, FORM)
    nrm = Normalizer(Xtr, Ytr)
    t_eje = np.concatenate([c.tau for c in te_c])

    P("=" * 78)
    P("INSPECCIÓN DE LA RED ENTRENADA")
    P("=" * 78)
    P(f"\nFormulación {FORM}, semilla {args.seed}, "
      f"{len(tr_c)} clips de entrenamiento / {len(te_c)} de prueba")

    modelos = {}
    for arq in ("simple", "tanh_deep"):
        set_seed(semilla_de(args.seed, FORM, arq))
        model = build_model(arq, Xtr.shape[1])
        model, hist = train_model(model, nrm.fx(Xtr), nrm.fy(Ytr),
                                  nrm.fx(Xva), nrm.fy(Yva), epochs=600, lr=1e-2)
        yhat = nrm.iy(predict(model, nrm.fx(Xte)))
        m = ev.metricas(Yte, yhat)
        g_pred, _ = ev.g_por_clip(te_c, cid_te, t_eje, yhat)
        modelos[arq] = model
        P(f"  [{arq:9s}] {n_params(model):4d} parám.  RMSE={m['rmse']:.4f}  "
          f"g={np.median(g_pred):+.3f}  ({hist['epocas_corridas']} ep)")

    # ---------------------------------------------------------------- 1
    P("\n" + "-" * 78)
    P("1. RED SIMPLE (2 unidades ReLU, 9 parámetros): la función completa")
    P("-" * 78)
    P(formula_relu(modelos["simple"], nrm))
    P("\n  Es una función lineal a trozos con dos quiebres. Su segunda derivada")
    P("  es cero en todas partes salvo en los quiebres, donde no existe: no hay")
    P("  ningún valor de los 9 parámetros que produzca una curvatura constante,")
    P("  que es lo que se necesita para representar g.")

    # ---------------------------------------------------------------- 2
    P("\n" + "-" * 78)
    P("2. RED TANH 32x32: derivadas de la función aprendida")
    P("-" * 78)

    # Las estadísticas se calculan sobre los puntos (tau, v0) que realmente
    # ocurren en los clips de prueba. Una malla rectangular incluiría
    # combinaciones que no existen en los datos, donde la red extrapola y sus
    # derivadas no significan nada.
    tau_d, v0_d = Xte[:, 0].astype(float), Xte[:, 1].astype(float)
    _, d1_d, d2_d, dv0_d = derivadas(modelos["tanh_deep"], nrm, tau_d, v0_d)

    P(f"\n  Evaluado en los {len(tau_d)} puntos experimentales de los "
      f"{len(te_c)} clips de prueba")
    P(f"  (tau en [{tau_d.min():.2f}, {tau_d.max():.2f}] s, "
      f"v0 en [{v0_d.min():+.2f}, {v0_d.max():+.2f}] m/s)")

    P("\n  a) Curvatura  d2(dy)/dtau2   [física: constante = g]")
    P(f"       media   = {d2_d.mean():+.3f} m/s²")
    P(f"       mediana = {np.median(d2_d):+.3f} m/s²")
    P(f"       desv.   =  {d2_d.std():.3f} m/s²  "
      f"({100 * d2_d.std() / abs(d2_d.mean()):.1f} % de la media)")
    P(f"       rango intercuartil = [{np.percentile(d2_d, 25):+.3f}, "
      f"{np.percentile(d2_d, 75):+.3f}]")
    P(f"     comparar con g del ajuste por clip a los datos: "
      f"{np.median([c.g_fit for c in te_c]):+.3f} m/s²")

    # ¿depende la curvatura de tau o de v0? No debería.
    pend_v0 = np.polyfit(v0_d, d2_d, 1)[0]
    pend_tau = np.polyfit(tau_d, d2_d, 1)[0]
    P(f"\n     dependencia de v0:  {pend_v0:+.3f} (m/s²)/(m/s)   [física: 0]")
    P(f"     dependencia de tau: {pend_tau:+.3f} (m/s²)/s       [física: 0]")

    P("\n  b) Sensibilidad a la velocidad inicial  d(dy)/dv0   [física: = tau]")
    c_dv0 = np.polyfit(tau_d, dv0_d, 1)
    r2_dv0 = 1 - np.var(dv0_d - np.polyval(c_dv0, tau_d)) / np.var(dv0_d)
    P(f"       ajuste  d(dy)/dv0 = {c_dv0[0]:+.4f}*tau {c_dv0[1]:+.4f}   "
      f"[física: 1.0*tau + 0.0]")
    P(f"       R² del ajuste = {r2_dv0:.4f}")

    P("\n  c) Velocidad  d(dy)/dtau   [física: = v0 + g*tau]")
    A = np.column_stack([np.ones_like(tau_d), v0_d, tau_d])
    c_d1, *_ = np.linalg.lstsq(A, d1_d, rcond=None)
    r2_d1 = 1 - np.var(d1_d - A @ c_d1) / np.var(d1_d)
    P(f"       ajuste  d(dy)/dtau = {c_d1[0]:+.4f} {c_d1[1]:+.4f}*v0 "
      f"{c_d1[2]:+.4f}*tau   [física: 0 + 1.0*v0 + g*tau]")
    P(f"       R² del ajuste = {r2_d1:.4f}")

    P("\n  Las tres derivadas tienen la magnitud y el signo correctos, pero")
    P("  ninguna es la constante ni la recta exacta que dicta la física: la")
    P("  curvatura fluctúa varios m/s² a lo largo del dominio. La red no")
    P("  implementa un término de curvatura constante; lo que reproduce bien es")
    P("  la curvatura PROMEDIADA sobre cada clip, que es la que recoge el ajuste")
    P("  parabólico por clip del nivel 2.")

    # Malla regular, solo para las figuras.
    tau_max = float(np.percentile(tau_d, 99))
    v0_lo, v0_hi = np.percentile([c.v0 for c in tr_c], [5, 95])
    malla_tau = np.linspace(0.02, tau_max, 60)
    malla_v0 = np.linspace(v0_lo, v0_hi, 25)
    TT, VV = np.meshgrid(malla_tau, malla_v0, indexing="ij")
    dy, d1, d2, dv0 = derivadas(modelos["tanh_deep"], nrm,
                                TT.ravel(), VV.ravel())
    D2 = d2.reshape(TT.shape)
    DV0 = dv0.reshape(TT.shape)

    # ---------------------------------------------------------------- 3
    P("\n" + "-" * 78)
    P("3. LECTURA DE LA ECUACIÓN APRENDIDA, CONTRA LOS DATOS")
    P("-" * 78)
    P("\n  Se ajusta  dy = a*tau + b*v0*tau + c*tau^2  a la superficie de salida")
    P("  de la red Y, como control, a los datos experimentales mismos. Sin ese")
    P("  control no se puede saber si una desviación respecto de (0, 1, g/2) es")
    P("  culpa de la red o una propiedad del conjunto de datos.")

    dy_real = Yte.ravel().astype(float)
    dy_net, _, _, _ = derivadas(modelos["tanh_deep"], nrm, tau_d, v0_d)

    coefs = {}
    for etq, k in (("todo el dominio", np.ones_like(tau_d, dtype=bool)),
                   ("tau < 0.6 s", tau_d < 0.6)):
        cD, rD = ajusta_mruv(tau_d[k], v0_d[k], dy_real[k])
        cN, rN = ajusta_mruv(tau_d[k], v0_d[k], dy_net[k])
        coefs[etq] = (cD, cN)
        P(f"\n  [{etq}]  ({k.sum()} puntos)")
        P(f"    {'':7} {'a':>9} {'b':>9} {'2c = g':>9} {'R²':>7}")
        P(f"    {'física':7} {0.0:+9.3f} {1.0:+9.3f} {G_REF:+9.3f} {'--':>7}")
        P(f"    {'datos':7} {cD[0]:+9.3f} {cD[1]:+9.3f} {2 * cD[2]:+9.3f} {rD:7.3f}")
        P(f"    {'red':7} {cN[0]:+9.3f} {cN[1]:+9.3f} {2 * cN[2]:+9.3f} {rN:7.3f}")

    P("\n  La red reproduce el coeficiente cuadrático de los datos casi")
    P("  exactamente, sesgos incluidos: cuando el ajuste agrupado de los datos")
    P("  da un valor alejado de -9.81, la red da ese mismo valor. Lo que la red")
    P("  aprendió es lo que el conjunto contiene, no la ley ideal.")

    # ---------------------------------------------------------------- 4
    P("\n" + "-" * 78)
    P("4. PRIMERA CAPA: qué mira cada neurona")
    P("-" * 78)
    md = modelos["tanh_deep"]
    W1 = md.net[0].weight.detach().numpy()
    b1 = md.net[0].bias.detach().numpy()
    # magnitud con que cada neurona de la 1a capa llega a la salida
    W2 = md.net[2].weight.detach().numpy()
    W3 = md.net[4].weight.detach().numpy().ravel()
    contrib = np.abs(W3 @ W2)

    ratio = np.abs(W1[:, 1]) / (np.abs(W1[:, 0]) + 1e-9)
    Xn = nrm.fx(np.column_stack([TT.ravel(), VV.ravel()]))
    pre = Xn @ W1.T + b1
    sat = np.mean(np.abs(np.tanh(pre)) > 0.9, axis=0)

    P(f"\n  32 neuronas. |w_v0/w_tau|: mediana={np.median(ratio):.3f}, "
      f"rango=[{ratio.min():.3f}, {ratio.max():.3f}]")
    P(f"  Fracción del dominio en saturación (|tanh|>0.9): "
      f"mediana={np.median(sat):.2f}, máx={sat.max():.2f}")
    P(f"  Neuronas con |tanh|>0.9 en menos del 10 % del dominio "
      f"(régimen casi lineal): {int(np.sum(sat < 0.1))}/32")
    P("\n  Las 6 neuronas de mayor contribución a la salida:")
    P(f"    {'idx':>4} {'w_tau':>9} {'w_v0':>9} {'sesgo':>9} "
      f"{'|contrib|':>10} {'sat':>6}")
    for i in np.argsort(-contrib)[:6]:
        P(f"    {i:4d} {W1[i, 0]:+9.3f} {W1[i, 1]:+9.3f} {b1[i]:+9.3f} "
          f"{contrib[i]:10.3f} {sat[i]:6.2f}")

    # ---------------------------------------------------------------- salidas
    np.savez_compressed(
        os.path.join(OUTS, "pesos.npz"),
        malla_tau=malla_tau, malla_v0=malla_v0,
        d2=D2, dv0=DV0, dy=dy.reshape(TT.shape), d1=d1.reshape(TT.shape),
        W1=W1, b1=b1, contrib=contrib, sat=sat, ratio=ratio,
        g_datos=np.array([c.g_fit for c in te_c]),
        tau_d=tau_d, v0_d=v0_d, d2_d=d2_d, dv0_d=dv0_d, d1_d=d1_d,
        # coeficientes de dy = a*tau + b*v0*tau + c*tau^2, para la figura
        coef_datos_todo=coefs["todo el dominio"][0],
        coef_red_todo=coefs["todo el dominio"][1],
        coef_datos_06=coefs["tau < 0.6 s"][0],
        coef_red_06=coefs["tau < 0.6 s"][1])
    with open(os.path.join(OUTS, "pesos.txt"), "w") as fh:
        fh.write("\n".join(log) + "\n")
    P(f"\nEscritos {OUTS}/pesos.txt y pesos.npz")


if __name__ == "__main__":
    main()
