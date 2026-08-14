"""
Líneas base clásicas.

En v1 no había ninguna, y sin ella un RMSE no significa nada. Aquí se usan tres
referencias, de la más tonta a la mejor posible:

1. **Predictor constante.** Devuelve siempre la media. Su RMSE es la desviación
   estándar de los datos. Cualquier modelo debe superarlo holgadamente.

2. **Modelo físico con g global.** Se estima una única ``g`` con los clips de
   entrenamiento y se predice ``dy = v0*tau + 0.5*g*tau^2`` en los clips de
   prueba. Es la ecuación de MRUV con **un solo parámetro ajustado**, y es el
   rival de verdad: si una red con decenas de parámetros no lo iguala, la red no
   está aportando nada.

3. **Ajuste parabólico por clip.** Ajusta cada clip de prueba por separado, con
   tres parámetros libres *por clip*. No es un predictor honesto (usa el propio
   clip que evalúa), pero marca el suelo de error alcanzable: lo que queda es
   ruido de medición.
"""

from __future__ import annotations

import numpy as np

from dataset import Clip, fit_quadratic


def g_global(clips: list[Clip], robusto: bool = True) -> float:
    """Estima una g única a partir de los clips de entrenamiento."""
    g = np.array([c.g_fit for c in clips])
    g = g[np.isfinite(g)]
    return float(np.median(g) if robusto else np.mean(g))


def predice_fisica(clips: list[Clip], g: float) -> tuple[np.ndarray, np.ndarray]:
    """Predice dy con la ecuación de MRUV y una g dada. Devuelve (y_true, y_pred)."""
    yt, yp = [], []
    for c in clips:
        yt.append(c.dy)
        yp.append(c.v0 * c.tau + 0.5 * g * c.tau ** 2)
    return np.concatenate(yt), np.concatenate(yp)


def ajuste_por_clip(clips: list[Clip]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ajusta una parábola independiente a cada clip.

    Devuelve (y_true, y_pred, g_por_clip).
    """
    yt, yp, gs = [], [], []
    for c in clips:
        coef = fit_quadratic(c.tau, c.dy)
        yt.append(c.dy)
        yp.append(coef[0] + coef[1] * c.tau + coef[2] * c.tau ** 2)
        gs.append(2 * coef[2])
    return np.concatenate(yt), np.concatenate(yp), np.array(gs)
