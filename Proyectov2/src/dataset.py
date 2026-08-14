"""
Carga de los clips de caída libre y construcción de los conjuntos de datos.

La diferencia central con la versión 1 del proyecto está aquí, no en las redes.

En v1 todos los clips se concatenaban y se pedía a la red aprender ``y = f(t)``.
Como cada clip tiene su propio origen temporal y su propia altura inicial, el
mismo valor de ``t`` aparecía asociado a valores de ``y`` que difieren en más de
un metro: el problema no era identificable y la mejor solución posible era la
media condicional ``E[y|t]``, que es casi una recta.

Aquí se ofrecen tres formulaciones, para poder medir el efecto del cambio en vez
de darlo por supuesto:

``v1``       entrada ``t``,           salida ``y``    (la de v1, como control)
``tau``      entrada ``tau``,         salida ``dy``   (cada clip a su origen)
``tau_v0``   entrada ``(tau, v0)``,   salida ``dy``   (identificable del todo)

con ``tau = t - t0`` y ``dy = y - y0`` medidos dentro de cada clip.

Bajo la formulación ``tau_v0``, la ecuación de MRUV

    dy = v0 * tau + 0.5 * g * tau^2

es una función bien definida de las entradas, con ``g`` como único parámetro
compartido por todos los clips. Esa es la hipótesis del anteproyecto planteada
de forma que se pueda verificar.

Nota sobre ``v0``: *Tracker* no puede calcular la velocidad en el primer
fotograma (necesita un fotograma previo), así que ``v0`` se estima con un ajuste
lineal a los primeros ``N_V0`` puntos del clip. Es una medición de la condición
inicial que usa solo el arranque del clip, no su curvatura, de modo que no
introduce circularidad en la estimación posterior de ``g``.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

# Puntos iniciales usados para estimar v0 por ajuste lineal.
N_V0 = 3

# Formulaciones disponibles: nombre -> (columnas de entrada, columna de salida)
FORMULACIONES = ("v1", "tau", "tau_v0")


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------
def read_clip_file(path: str) -> np.ndarray:
    """Lee un .txt de Tracker y devuelve un array Nx7 (t,x,y,vx,vy,ax,ay).

    Los archivos traen una cabecera de dos líneas y celdas vacías donde Tracker
    no pudo calcular una derivada; esas celdas se leen como NaN.
    """
    filas = []
    with open(path, encoding="utf8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.lower().startswith("t"):
                continue
            vals, ok = [], True
            for celda in linea.split(","):
                celda = celda.strip()
                if celda == "":
                    vals.append(np.nan)
                    continue
                try:
                    vals.append(float(celda))
                except ValueError:      # línea de cabecera tipo ",masa A,,,"
                    ok = False
                    break
            if not ok:
                continue
            if len(vals) < 7:
                vals += [np.nan] * (7 - len(vals))
            filas.append(vals[:7])
    return np.array(filas) if filas else np.empty((0, 7))


def fit_quadratic(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ajuste de mínimos cuadrados y = c0 + c1*t + c2*t^2. Devuelve [c0,c1,c2]."""
    T = np.vstack([np.ones_like(t), t, t ** 2]).T
    coef, *_ = np.linalg.lstsq(T, y, rcond=None)
    return coef.flatten()


# ---------------------------------------------------------------------------
# Clip
# ---------------------------------------------------------------------------
@dataclass
class Clip:
    """Una trayectoria individual, ya referida a su propio origen."""

    nombre: str
    t: np.ndarray        # tiempo original del video [s]
    y: np.ndarray        # posición original [m]
    tau: np.ndarray      # t - t[0]
    dy: np.ndarray       # y - y[0]
    v0: float            # velocidad inicial estimada [m/s]
    g_fit: float         # g del ajuste cuadrático a este clip [m/s^2]
    resid: float         # desv. est. del residuo de ese ajuste [m]

    @property
    def n(self) -> int:
        return len(self.tau)

    @property
    def duracion(self) -> float:
        return float(self.tau[-1])


def build_clip(nombre: str, t: np.ndarray, y: np.ndarray) -> Clip:
    tau = t - t[0]
    dy = y - y[0]
    # v0 por ajuste lineal a los primeros puntos (solo el arranque del clip)
    k = min(N_V0, len(tau))
    v0 = float(np.polyfit(tau[:k], dy[:k], 1)[0]) if k >= 2 else 0.0
    c = fit_quadratic(tau, dy)
    resid = float(np.std(dy - (c[0] + c[1] * tau + c[2] * tau ** 2)))
    return Clip(nombre, t, y, tau, dy, v0, float(2 * c[2]), resid)


def load_clips(data_dir: str, min_points: int = 8) -> list[Clip]:
    """Carga todos los clips utilizables de un directorio."""
    clips = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        arr = read_clip_file(path)
        if arr.size == 0:
            continue
        t, y = arr[:, 0], arr[:, 2]
        m = ~np.isnan(t) & ~np.isnan(y)
        if m.sum() < min_points:
            continue
        clips.append(build_clip(os.path.basename(path), t[m], y[m]))
    return clips


def quality_filter(clips: list[Clip], max_resid: float = 0.08) -> list[Clip]:
    """Descarta clips cuyo ajuste parabólico deja un residuo grande.

    Un residuo alto indica un fallo de seguimiento de Tracker (el marcador
    saltó de objeto, se perdió, etc.), no una trayectoria físicamente distinta.
    Con el umbral por defecto se conservan ~96 % de los clips.
    """
    return [c for c in clips if c.resid < max_resid]


# ---------------------------------------------------------------------------
# Partición POR CLIP (no por punto)
# ---------------------------------------------------------------------------
def split_clips(clips: list[Clip], seed: int = 0,
                frac_val: float = 0.15, frac_test: float = 0.15):
    """Reparte los clips completos en entrenamiento / validación / prueba.

    Cada clip cae entero de un solo lado. Repartir puntos sueltos al azar --- lo
    que se hizo en v1 --- deja puntos del mismo clip a ambos lados, con lo que
    el error de validación mide interpolación dentro de clips ya vistos y no
    generalización a un clip nuevo.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(clips))
    n_test = int(round(frac_test * len(clips)))
    n_val = int(round(frac_val * len(clips)))
    i_test, i_val, i_train = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    pick = lambda ii: [clips[i] for i in ii]
    return pick(i_train), pick(i_val), pick(i_test)


# ---------------------------------------------------------------------------
# Construcción de matrices
# ---------------------------------------------------------------------------
def build_arrays(clips: list[Clip], formulacion: str):
    """Devuelve (X, Y, clip_id) para la formulación pedida.

    ``clip_id[i]`` es el índice, dentro de ``clips``, del clip al que pertenece
    la fila ``i``. Se necesita para poder ajustar la parábola clip por clip
    sobre las predicciones.
    """
    if formulacion not in FORMULACIONES:
        raise ValueError(f"formulación desconocida: {formulacion}")

    X, Y, cid = [], [], []
    for j, c in enumerate(clips):
        if formulacion == "v1":
            xs = c.t[:, None]
            ys = c.y[:, None]
        elif formulacion == "tau":
            xs = c.tau[:, None]
            ys = c.dy[:, None]
        else:  # tau_v0
            xs = np.column_stack([c.tau, np.full(c.n, c.v0)])
            ys = c.dy[:, None]
        X.append(xs)
        Y.append(ys)
        cid.append(np.full(len(xs), j))

    return (np.concatenate(X).astype(np.float32),
            np.concatenate(Y).astype(np.float32),
            np.concatenate(cid))


class Normalizer:
    """Normalización z ajustada SOLO con los clips de entrenamiento.

    En v1 los estadísticos se calculaban sobre el conjunto completo, lo que
    filtra información del conjunto de prueba al de entrenamiento.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.xm, self.xs = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-9
        self.ym, self.ys = Y.mean(0, keepdims=True), Y.std(0, keepdims=True) + 1e-9

    def fx(self, X):  return (X - self.xm) / self.xs
    def fy(self, Y):  return (Y - self.ym) / self.ys
    def iy(self, Yn): return Yn * self.ys + self.ym


# ---------------------------------------------------------------------------
def describe(clips: list[Clip]) -> str:
    g = np.array([c.g_fit for c in clips])
    v0 = np.array([c.v0 for c in clips])
    n = np.array([c.n for c in clips])
    d = np.array([c.duracion for c in clips])
    r = np.array([c.resid for c in clips])
    return (
        f"clips={len(clips)}  muestras={n.sum()}\n"
        f"  duración   media={d.mean():.3f} s   rango=[{d.min():.2f}, {d.max():.2f}]\n"
        f"  puntos/clip media={n.mean():.1f}     rango=[{n.min()}, {n.max()}]\n"
        f"  v0         media={v0.mean():+.3f}    std={v0.std():.3f} m/s\n"
        f"  g por clip mediana={np.median(g):+.3f} media={g.mean():+.3f} "
        f"std={g.std():.3f} m/s²\n"
        f"  residuo del ajuste  mediana={np.median(r):.4f} m  (suelo de ruido)"
    )
