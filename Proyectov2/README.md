# Proyecto v2 — verificación del supuesto del anteproyecto

**Aquí las cosas se hacen de la forma correcta.** Esta carpeta rehace el
experimento del año de prácticas corrigiendo el defecto de planteamiento que se
documenta en [`../informe final/BITACORA-correcciones.md`](../informe%20final/BITACORA-correcciones.md).

El código de v1 (`../Redes/`) y el informe (`../informe final/`) se dejan
intactos: sirven de registro de lo que pasó y de control contra el cual medir.

---

## El supuesto que se quiere verificar

Del anteproyecto (mayo 2024):

> Estudiar la teoría […] para entender la estructura interna de las redes
> neuronales aplicadas a la elaboración de modelos matemáticos que describan el
> comportamiento de los datos de montajes experimentales de caída libre.

Operacionalizado como criterio medible:

> Si se entrena una red con trayectorias reales de caída libre y después se
> ajusta una parábola a sus predicciones, ¿aparece el valor correcto de `g`?

## Qué estaba mal en v1

Los 606 clips se concatenaban y se pedía a la red aprender `y = f(t)`. Como cada
clip tiene su propio origen temporal y su propia altura inicial, **el mismo `t`
aparecía asociado a valores de `y` que difieren en más de un metro**. Ninguna
función puede satisfacer eso: el problema no era identificable, y la mejor
solución posible del error cuadrático era la media condicional `E[y|t]`, que es
casi una recta y cuyo coeficiente cuadrático vale ≈ 0.

La red convergía a esa solución. Respondía bien una pregunta mal hecha.

## Qué se corrige aquí

| | v1 | v2 |
|---|---|---|
| **Formulación** | `t → y`, clips concatenados | `(τ, v₀) → Δy`, cada clip a su origen |
| **Partición** | 80/20 por **punto** (fuga entre clips) | por **clip completo**, 70/15/15 |
| **Normalización** | sobre todo el conjunto (fuga) | ajustada solo con clips de entrenamiento |
| **Semillas** | 1 | 5, con media ± desv. est. |
| **Parada** | número fijo de épocas | parada temprana sobre clips de validación no vistos |
| **Línea base** | ninguna | 3: predictor constante, MRUV con `g` global, ajuste por clip |
| **Estimación de `g`** | un ajuste sobre **todos los clips agregados** ← el error | un ajuste **por clip** |
| **Calidad de datos** | sin filtro | descarta clips con residuo > 8 cm (fallos de seguimiento) |

con `τ = t − t₀` y `Δy = y − y₀` medidos dentro de cada clip.

Bajo esa formulación la ecuación de MRUV

```
Δy = v₀·τ + ½·g·τ²
```

**sí** es una función bien definida de las entradas, con `g` como único parámetro
compartido por todos los clips. La pregunta pasa a tener respuesta.

### Por qué hace falta `v₀`

Al referir cada clip a su origen, `y₀` desaparece del problema. `v₀` no: la
velocidad inicial varía entre clips (media −0.22 m/s, desv. est. 0.49 m/s, rango
de −3.8 a +1.6), porque no todos los clips arrancan en el instante de la
soltada. Sin `v₀` como entrada el problema sigue siendo parcialmente ambiguo —
por eso se evalúan las dos variantes (`tau` y `tau_v0`) y se mide la diferencia
en vez de suponerla.

`v₀` se estima por ajuste lineal a los **3 primeros puntos** de cada clip.
*Tracker* no da velocidad en el primer fotograma (necesita uno previo), y usar
solo el arranque evita que la estimación absorba la curvatura, que es
precisamente lo que se quiere medir después.

---

## Cómo ejecutarlo

Requiere `numpy`, `pandas`, `matplotlib` y `torch` (CPU basta).

```bash
cd src
python3 run_all.py              # experimento completo, 5 semillas (~2 min)
python3 epocas.py               # estudio del número de épocas (~2 min)
python3 pesos.py                # inspección de la red entrenada (~10 s)
python3 figures.py              # figuras a partir de los resultados
```

Opciones útiles:

```bash
python3 run_all.py --seeds 2            # versión rápida
python3 run_all.py --sin-filtro         # sin descartar clips de mala calidad
python3 run_all.py --max-resid 0.15     # filtro más permisivo
python3 run_all.py --epochs 1500 --lr 5e-3
python3 epocas.py --seeds 2             # versión rápida
python3 pesos.py --seed 3               # inspeccionar la red de otra semilla
```

### Entorno

Los resultados publicados en `outs/` se reprodujeron bit a bit con
`/home/alejandro/Documentos/envs/practicas` (Python 3.12, torch 2.13 CPU,
numpy 2.5, pandas 3.0). Si se recrea el entorno conviene verificar que la
semilla 0 sigue dando `RMSE=0.1533  g=-9.865` para `tau_v0 | tanh_deep`; si no
coincide, el flujo de números aleatorios cambió y las cifras del informe habría
que regenerarlas.

## Estructura

```
Proyectov2/
├── README.md
├── src/
│   ├── dataset.py    carga de clips, las 3 formulaciones, partición POR CLIP
│   ├── models.py     las mismas 3 arquitecturas de v1 (para comparar de igual a igual)
│   ├── baseline.py   las 3 líneas base clásicas
│   ├── train.py      entrenamiento con parada temprana
│   ├── evaluate.py   métricas y recuperación de g CLIP POR CLIP
│   ├── run_all.py    experimento completo
│   ├── epocas.py     coste/beneficio del número de épocas
│   ├── pesos.py      inspección de la red entrenada
│   └── figures.py    figuras
├── outs/             resultados (csv, npz, resumen.txt, pesos.txt, log)
├── figs/             figuras
└── InformeFinal/     informe final, versión de solo-resultados (ver su README)
```

## Diseño experimental

**3 formulaciones × 3 arquitecturas × 5 semillas = 45 entrenamientos.**

Formulaciones:

| clave | entrada → salida | qué prueba |
|---|---|---|
| `v1` | `t → y` | control: reproduce el planteamiento original, debe fallar |
| `tau` | `τ → Δy` | ¿basta con referir cada clip a su origen? |
| `tau_v0` | `(τ, v₀) → Δy` | formulación identificable completa |

Arquitecturas (idénticas a las de v1, a propósito):

| clave | red | ~parámetros |
|---|---|---|
| `simple` | `entrada→2→1`, ReLU | 7–9 |
| `media` | `entrada→4→1`, ReLU | 13–17 |
| `tanh_deep` | `entrada→32→32→1`, tanh | ~1 150 |

Mantener las mismas redes es deliberado: si el resultado cambia, el cambio viene
de **cómo se plantea el problema**, no de haber usado una red distinta.

Líneas base:

| base | qué es | para qué sirve |
|---|---|---|
| predictor constante | devuelve siempre la media | mínimo que hay que superar |
| MRUV con `g` global | `Δy = v₀τ + ½gτ²`, una sola `g` ajustada en train | **el rival de verdad**: física con 1 parámetro |
| ajuste por clip | parábola independiente por clip de prueba | suelo de error alcanzable (lo que queda es ruido) |

## Métricas

- **Nivel 1 — predicción:** RMSE, MAE, R², correlación, sobre **clips no vistos**.
- **Nivel 2 — física:** se ajusta una parábola a las predicciones **de cada clip
  por separado** y se compara la mediana de las `g` resultantes con −9.81 m/s².

El nivel 2 es el que verifica el supuesto. Hacerlo sobre clips agregados —lo que
se hizo en v1— produce un número sin significado físico.

## Análisis adicionales

Además del experimento principal hay dos estudios que responden a la parte del
objetivo general que habla de *entender la estructura interna* de la red.

### `epocas.py` — coste y beneficio de entrenar más

Entrena bajo `tau_v0` evaluando en puntos de control **a lo largo del mismo
entrenamiento**, de modo que el tiempo reportado para la época N es el tiempo
real acumulado hasta llegar a ella. Escribe `outs/epocas.csv`.

Lo que sale:

- el coste es exactamente lineal en el número de épocas;
- el RMSE se estanca hacia la época 20;
- la mejor `g` de la red tanh llega hacia la época 200;
- **el RMSE y el error en `g` no son monótonos entre sí**: la red `media (4)`
  mejora su RMSE de 0.167 a 0.157 m entre las épocas 10 y 800 mientras su error
  en `g` empeora del 17.7 % al 29.1 %. Sigue bajando el error cuadrático, pero
  ajustando cosas que no son la curvatura.

Conclusión práctica: parar por el error de validación no garantiza parar donde
el parámetro físico es mejor.

### `pesos.py` — ¿está la física dentro de la red?

Escribe `outs/pesos.txt` y `outs/pesos.npz`. Cuatro análisis:

1. **La red simple, completa.** Con 9 parámetros la función se escribe a mano:
   dos quiebres ReLU, segunda derivada nula en todas partes. No puede
   representar `g`, y se ve por construcción y no por estadística.

2. **Derivadas de la función aprendida** (autograd, en unidades físicas). Si la
   red implementara `Δy = v₀τ + ½gτ²` entonces `∂²Δy/∂τ² = g` constante,
   `∂Δy/∂v₀ = τ` y `∂Δy/∂τ = v₀ + gτ`. Las tres tienen el signo y el orden de
   magnitud correctos, pero **la curvatura no es constante**: mediana −7.8,
   IQR [−13.9, −2.8]. Lo que la red reproduce bien es la curvatura *promediada
   sobre cada clip*, que es la que mide el nivel 2.

3. **La ecuación leída, contra los datos.** Se ajusta `Δy = aτ + b·v₀τ + cτ²` a
   la salida de la red **y, como control, a los datos experimentales**. Sin ese
   control no se sabe si una desviación es culpa de la red o del conjunto:

   | dominio | fuente | a | b | 2c = g |
   |---|---|---|---|---|
   | todo | física | 0 | 1 | −9.81 |
   | todo | datos | −0.81 | 0.81 | −4.67 |
   | todo | red | −0.98 | 0.49 | **−4.66** |
   | τ<0.6 s | datos | +0.07 | 1.04 | −9.00 |
   | τ<0.6 s | red | −0.15 | 0.65 | **−8.81** |

   La red sigue al coeficiente cuadrático de los datos con <2 % de diferencia
   en ambos dominios, incluso donde los datos se alejan mucho de −9.81.
   Aprendió lo que el conjunto contiene, sesgos incluidos.

4. **Primera capa.** No hay «neurona de la gravedad»: las 32 unidades se
   reparten por el plano (w_τ, w_v₀), 15 operan casi siempre en el régimen
   lineal de la tanh, y las de mayor contribución no comparten estructura.

La conclusión honesta: **la física está en el comportamiento agregado de la
red, no en una estructura interna legible.** Recuperar una forma cerrada
exigiría arquitecturas diseñadas para ello (cuello de botella de una unidad,
activaciones polinómicas, regresión simbólica).

## Resultados

Ver [`outs/resumen.txt`](outs/resumen.txt) para la corrida completa,
[`outs/pesos.txt`](outs/pesos.txt) para la inspección, y `figs/` para las
figuras. El resumen está también al final de este README tras cada ejecución de
`run_all.py`.

El informe que recoge todo esto es
[`InformeFinal/`](InformeFinal/) (versión de solo-resultados); la versión que
además narra el diagnóstico está en [`../informe final/`](../informe%20final/).
