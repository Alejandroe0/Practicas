"""
Métricas de predicción y recuperación del parámetro físico.

Dos niveles, igual que en el informe v1, pero con el nivel 2 hecho como
corresponde: **un ajuste parabólico por clip**, nunca uno solo sobre todos los
clips agregados. Ese fue el error que invirtió la conclusión en v1.
"""

from __future__ import annotations

import numpy as np

from dataset import Clip, fit_quadratic


# ---------------------------------------------------------------------------
# Nivel 1: error de predicción
# ---------------------------------------------------------------------------
def metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE, MAE, R² y correlación. Todo en metros."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    err = y_pred - y_true
    var = np.var(y_true)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.mean(err ** 2) / var) if var > 0 else np.nan,
        "r": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "rmse_trivial": float(np.std(y_true)),   # predictor constante = media
    }


# ---------------------------------------------------------------------------
# Nivel 2: recuperación de g
# ---------------------------------------------------------------------------
def g_por_clip(clips: list[Clip], clip_id: np.ndarray, x_tiempo: np.ndarray,
               y_pred: np.ndarray, min_points: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Ajusta una parábola a las predicciones DE CADA CLIP y devuelve su g.

    Devuelve ``(g_pred, indices)`` donde ``indices`` son las posiciones dentro de
    ``clips`` de los clips efectivamente ajustados.
    """
    y_pred = np.asarray(y_pred).ravel()
    x_tiempo = np.asarray(x_tiempo).ravel()
    gs, idx = [], []
    for j in np.unique(clip_id):
        m = clip_id == j
        if m.sum() < min_points:
            continue
        c = fit_quadratic(x_tiempo[m], y_pred[m])
        gs.append(2 * c[2])
        idx.append(j)
    return np.array(gs), np.array(idx)


def resumen_g(g: np.ndarray, g_ref: float = -9.81) -> dict:
    """Estadística robusta de un conjunto de estimaciones de g."""
    g = np.asarray(g)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return {"g_mediana": np.nan, "g_media": np.nan, "g_std": np.nan,
                "g_iqr": np.nan, "err_rel_%": np.nan, "n": 0}
    med = float(np.median(g))
    return {
        "g_mediana": med,
        "g_media": float(np.mean(g)),
        "g_std": float(np.std(g)),
        "g_iqr": float(np.percentile(g, 75) - np.percentile(g, 25)),
        "err_rel_%": float(100 * abs(med - g_ref) / abs(g_ref)),
        "n": int(g.size),
    }


def tabla(filas: list[dict], columnas: list[str], titulo: str = "") -> str:
    """Formatea una lista de diccionarios como tabla de texto alineada."""
    if not filas:
        return "(sin filas)"
    anchos = {c: max(len(c), *(len(_fmt(f.get(c, ""))) for f in filas)) for c in columnas}
    sep = "-" * (sum(anchos.values()) + 3 * (len(columnas) - 1))
    out = []
    if titulo:
        out += [titulo, "=" * len(titulo)]
    out.append("   ".join(c.ljust(anchos[c]) for c in columnas))
    out.append(sep)
    for f in filas:
        out.append("   ".join(_fmt(f.get(c, "")).ljust(anchos[c]) for c in columnas))
    return "\n".join(out)


def _fmt(v) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "--"
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.3f}"
    return str(v)
