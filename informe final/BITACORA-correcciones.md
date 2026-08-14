# Bitácora de correcciones — Informe final

> **Registro creado:** 2026-08-14, 16:25 CST · **Última entrada:** 2026-08-14, 17:11 CST
> **Autor del registro:** revisión asistida sobre los documentos de Alejandro Barillas
> **Alcance:** fusión de `anteproyecto/` + `Resultados/` → `informe final/`

Este archivo existe para que dentro de seis meses se pueda reconstruir **qué se
cambió y por qué**, sin tener que volver a derivar el análisis. La parte
importante es la sección [2](#2-errores-encontrados): los tres errores que
invertían la conclusión del trabajo. El desenlace ---el experimento rehecho y
el supuesto ya verificado--- está en la sección
[8](#8-cierre-el-supuesto-sí-se-verifica-2026-08-14-1711-cst).

---

## 0. Documentos de origen

| Archivo | Última modificación | Rol |
|---|---|---|
| `anteproyecto/anteproyectoPractica.tex` | 2024-05-01 21:20 | Anteproyecto aprobado |
| `Resultados/Resultados.tex` | 2026-07-13 12:31 | Análisis de resultados (versión con los errores) |
| `Redes/informe_practicas_redes.md` | — | Nota previa, misma tabla de métricas |
| `Redes/outs/epochs_analysis_all_models.csv` | — | Datos crudos del estudio de épocas |

Ninguno de estos archivos fue modificado. El informe unificado es un documento
nuevo en `informe final/`.

---

## 1. Resumen en una frase

`Resultados.tex` concluía que las redes **sí** aprendían el marco teórico de la
caída libre, con un error del 0.2 % en la estimación de `g`. Al verificar los
números, la conclusión correcta es la **opuesta**: las redes no recuperan `g`, y
el acuerdo aparente venía de compararlas contra un valor de referencia que no
era la gravedad.

---

## 2. Errores encontrados

### 2.1 El valor de referencia `g_real = -0.35 m/s²` no es la gravedad

**Gravedad del error:** crítico — invierte la conclusión del informe.

**Qué decía `Resultados.tex`:**

> Red 3→4→1: `g_pred = -0.3564` vs `g_real = -0.3500` (error: ≈ 0.2 %)
> […] Un error de 0.2 % en la estimación de una constante física fundamental no
> es coincidencia.

**Qué pasa en realidad:** ese `-0.35` sale de `Redes/test/g_real.py` y de
`red1.py`, que ajustan **una sola parábola a los 606 clips concatenados**. Cada
clip conserva el origen temporal del video del que se extrajo, así que los
20 210 puntos se reparten sobre un intervalo de 0 a 10.4 s aunque cada
trayectoria dure ~0.54 s. Además cada clip parte de una altura inicial distinta.

La parábola que mejor ajusta esa nube tiene su vértice en t ≈ 5.2 s, **donde no
hay datos**, y su coeficiente cuadrático no tiene relación con `g`. Es solo el
número que minimiza el error sobre una mezcla de trayectorias incompatibles.

Ver `imgs/agregado_vs_clip.png`, panel (a).

**Verificación:**

```
ajuste sobre los 606 clips agregados : g = +0.350 m/s²   ← el "g_real" del informe
ajuste clip por clip, 606 clips      : media   -9.00 m/s²
                                        mediana -9.38 m/s²
                                        desv.   2.36 m/s²
```

La mediana por clip está a un 4 % de −9.81 m/s². **Los datos experimentales son
buenos**; el problema estaba en cómo se agregaban.

---

### 2.2 Las redes no recuperan `g`

**Gravedad del error:** crítico.

Repitiendo el ajuste clip por clip sobre las **predicciones** de cada red:

| Fuente | media | mediana | desv. est. |
|---|---:|---:|---:|
| **Datos experimentales** | **−9.00** | **−9.38** | 2.36 |
| Red 3→4→1 | −3.25 | −2.99 | 2.41 |
| Red Tanh profunda | −3.60 | −2.89 | 2.80 |
| Red simple 1→2→1 | +0.17 | −0.00 | 0.59 |

La red simple devuelve **cero**: dentro de cada clip sus predicciones son
prácticamente una recta. Las otras dos llegan a un tercio del valor correcto.
Ver `imgs/hist_g_por_clip.png` — son dos poblaciones separadas, no un sesgo
pequeño.

**Además, el RMSE parecía mejor de lo que era.** `Resultados.tex` reportaba
RMSE ≈ 0.51 m sin referencia contra la cual juzgarlo:

| Modelo | RMSE [m] | R² |
|---|---:|---:|
| Red 3→4→1 | 0.5068 | 0.329 |
| Red Tanh | 0.5123 | 0.314 |
| Red simple 1→2→1 | 0.5140 | 0.310 |
| **Predictor constante `ŷ = ȳ`** | **0.6186** | 0.000 |

Las redes mejoran solo un **18 %** al de un modelo que ignora la entrada y
devuelve siempre la posición media. Para un fenómeno determinista, R² ≈ 0.31 es
señal de un problema mal planteado, no de un ajuste aceptable.

---

### 2.3 Causa raíz: el problema no es identificable

**Esto no es un error de las redes.** Es un defecto del planteamiento, y es la
parte que conviene recordar.

Al concatenar los 606 clips y pedir `y = f(t)`, se exige que un valor de entrada
produzca un valor de salida. Pero para cada instante `t` del rango útil hay
decenas de puntos de clips distintos, con alturas iniciales que difieren en más
de un metro. **Ninguna función puede satisfacer eso.**

La teoría de la regresión da la respuesta exacta: el minimizador del error
cuadrático es la media condicional

```
f*(t) = E[y | t]
```

Como los clips no están sincronizados ni comparten altura inicial, esa media es
una curva suave y casi lineal, con coeficiente cuadrático ≈ 0. **Es exactamente
lo que devuelven las redes.** Convergieron a la solución correcta del problema
que se les planteó; el problema planteado era el equivocado.

Confirmación visual en `imgs/trayectorias_ejemplo.png`: las tres redes producen
casi la *misma* curva para clips distintos, porque su única entrada es `t` y
todos los clips empiezan en `t = 0`.

Esto también explica dos cosas que en `Resultados.tex` quedaban sin explicación
coherente:

- **Por qué la red 3→4→1 es la mejor.** Sus entradas `v_y` y `a_y` sí distinguen
  un clip de otro. Pasa de `g ≈ 0` a `g ≈ −3`. No llega más lejos porque esas
  derivadas son numéricas (ruidosas, sobre todo `a_y`) y 4 unidades ReLU no dan
  para representar los productos necesarios.
- **Por qué la red Tanh no gana pese a tener 1153 parámetros** frente a 21 de la
  red 3→4→1. Cuando el límite es de *información* y no de *capacidad*, añadir
  parámetros no ayuda: solo aprende la media condicional con más fidelidad.

---

### 2.4 La oscilación de la red 1→2→1 no es sobreajuste

**Gravedad:** menor, pero la interpretación anterior era insostenible.

**Qué decía `Resultados.tex`:**

> A más épocas, la red intenta mejorar aún más su ajuste memorizando
> irregularidades en los datos. El sobreentrenamiento es un artefacto del
> algoritmo, no del conocimiento físico de la red.

**Por qué no se sostiene:**

1. El sobreajuste produce un patrón distinto: error de entrenamiento que baja
   mientras el de validación sube de forma *sostenida*. Lo observado es una
   alternancia no monótona (0.52 → 0.62 → 0.60 → 0.51 → 0.60).
2. Con **7 parámetros y 20 210 muestras**, el sobreajuste es prácticamente
   imposible.

**Explicación real:** cada fila de la tabla de épocas es un entrenamiento
independiente con su propia inicialización aleatoria. Con solo dos neuronas
ocultas ReLU, basta que una quede en la región de gradiente nulo (*ReLU muerta*)
para que la red se reduzca de hecho a una sola neurona. Los dos niveles de RMSE
observados (≈0.60 y ≈0.51) corresponden verosímilmente a esos dos regímenes.
**Es dispersión entre semillas, no dinámica de aprendizaje.**

**Confirmado** (ver §8): con 5 semillas, la red de 2 unidades ReLU muestra una
desviación estándar de **4.3 m/s²** en la `g` recuperada, frente a 0.3 de la red
tanh. La dispersión entre semillas es efectivamente del orden del efecto que se
había interpretado como dinámica de aprendizaje.

---

### 2.5 Errores menores de redacción

| Problema | Dónde | Corrección |
|---|---|---|
| `**negrita**` de Markdown dentro de LaTeX | `Resultados.tex`, varias líneas | Aparecía literalmente con asteriscos en el PDF → `\textbf{}` |
| «error ≈ 0.2 %» | Conclusiones | Los números daban 1.8–2 %, no 0.2 % (y frente a una referencia inválida) |
| «RMSE entre 0.5064 y 0.5140» vs. «0.5068» | Texto vs. tabla | Unificado |
| Fórmula del MAE con `|x_i − x|` | Marco teórico | → `|ŷ_i − y_i|` |
| Se decía TensorFlow | Marco teórico | La implementación usa **PyTorch** (`torch.nn`) |
| `.bib`: `url = \url{...}` y `url = url{...}` | `referencias.bib` | Rompía BibTeX → `url = {...}` |
| `.bib`: autor `Jes\'us Mart\'inez'` | `referencias.bib` | Apóstrofo suelto que salía impreso |

---

## 3. Cambios aplicados al marco teórico

Se eliminó lo que no se usa y se añadió lo que la discusión necesita.

**Eliminado:**

- **Divergencia** (`div F = ∇·F`). No aparece en ningún punto del trabajo; la
  función de coste empleada es MSE. La justificación que traía —«se emplea para
  cuantificar qué tan lejos está una distribución aproximada de la verdadera»—
  mezclaba la divergencia vectorial con la divergencia KL, que son cosas
  distintas y ninguna de las dos se usó.
- **MASE**. No se calculó en ningún experimento.

**Ampliado:**

- **Mínimos cuadrados y ajuste polinómico** — es *la* herramienta con la que se
  estima `g`. Se añadió la matriz de diseño, la identificación `c₀ = y₀`,
  `c₁ = v₀`, `g = 2c₂`, y la condición explícita de que todos los puntos de un
  ajuste deben pertenecer a una misma trayectoria (que es justo lo que se violó,
  §2.1).
- **MRUV** — se añadieron `y₀`, el convenio de signos de *Tracker* (eje `y` hacia
  arriba) y tres observaciones sobre la estructura de la ecuación, la tercera de
  las cuales es la semilla del diagnóstico de §2.3.
- **Backpropagation** — se conectó explícitamente con la regla de la cadena.

**Añadido:**

- **Coeficiente de determinación R²** — sin él, un RMSE «pequeño» no significa
  nada (§2.2).
- **Normalización z** — se usa en todos los scripts pero no estaba documentada.
- **Sobreajuste y validación**, incluyendo por qué particionar por punto
  individual es incorrecto cuando los datos vienen de experimentos repetidos.
- **ReLU muerta** — necesaria para explicar §2.4.
- **Identificabilidad del modelo y media condicional** — el resultado teórico
  que explica todo el trabajo (§2.3).

**Sustituido:**

- **TensorFlow → PyTorch**, con una nota que justifica el cambio (modo *eager*
  para inspeccionar la red, que era el objetivo del proyecto). La referencia a
  TensorFlow se conserva en la bibliografía comentada como herramienta consultada
  en la fase inicial.

---

## 4. Estructura del informe final

```
Resumen
1. Descripción de la Institución      ← del anteproyecto, sin cambios
2. Descripción del grupo de trabajo   ← del anteproyecto, sin cambios
3. Introducción + marco teórico       ← reescrito (§3 de esta bitácora)
4. Objetivos                          ← del anteproyecto, sin cambios
5. Justificación                      ← del anteproyecto, sin cambios
6. Metodología                        ← ampliada: montaje, datos, arquitecturas,
                                         protocolo de evaluación
7. Resultados y discusión             ← NUEVO
   7.1 Error de predicción
   7.2 El ajuste agregado no es una prueba válida
   7.3 Recuperación de g clip por clip
   7.4 Diagnóstico: identificabilidad
   7.5 Efecto del número de épocas
   7.6 Síntesis
8. Limitaciones del estudio           ← NUEVO
9. Conclusiones                       ← NUEVO
10. Trabajo futuro                    ← NUEVO
Bibliografía
```

Se eliminó el **plan de trabajo** (diagrama de Gantt): es propio de un
anteproyecto, no de un informe final.

Se conservan **ambas** tablas de estimación de `g` —la del ajuste agregado y la
del ajuste por clip— para dejar el contraste explícito en el documento, en vez
de borrar el resultado erróneo sin más.

---

## 5. Figuras nuevas

Generadas por `figuras_informe.py`, que lee `../Toma_de_datos/datos/*.txt` y
`../Redes/outs/predicciones_*.csv`:

| Archivo | Qué muestra |
|---|---|
| `imgs/agregado_vs_clip.png` | (a) la parábola del ajuste agregado alejándose de los datos; (b) seis clips individuales con su `g` propia |
| `imgs/hist_g_por_clip.png` | distribución de las 606 estimaciones de `g`: datos vs. predicciones |
| `imgs/pred_vs_real.png` | dispersión predicción/valor experimental; la nube se estrecha verticalmente = las redes promedian |
| `imgs/trayectorias_ejemplo.png` | cuatro clips con las tres predicciones superpuestas; las curvas son casi idénticas entre clips |

Las dos figuras heredadas (`epochs_analysis_*.png`) vienen de
`../Redes/plot_epochs_analysis.py`.

---

## 6. Cómo reproducir la verificación

```bash
cd ~/Documentos/personal/Practicas/"informe final"
python3 figuras_informe.py
```

El script imprime al final el resumen numérico que sustenta las tablas del
informe. Salida esperada:

```
clips=606  muestras=20210

--- g por clip (datos experimentales) ---
media=-9.002  mediana=-9.378 std=2.357  n=606

--- por modelo ---
Red simple 1→2→1 : RMSE=0.5140 MAE=0.4344 r=0.5565 R2=0.3095 | g_clip mediana=-0.000
Red 3→4→1        : RMSE=0.5068 MAE=0.4265 r=0.5735 R2=0.3288 | g_clip mediana=-2.989
Red Tanh profunda: RMSE=0.5123 MAE=0.4323 r=0.5610 R2=0.3142 | g_clip mediana=-2.888

RMSE del predictor constante (media): 0.6186
```

**Nota sobre el orden de los datos:** el script reconstruye los límites de cada
clip asumiendo que las predicciones en `Redes/outs/predicciones_*.csv` conservan
el orden de `sorted(glob(...))` con el que las generaron los scripts de
entrenamiento. Verificado: 20 210 filas en ambos lados. Si en el futuro se
cambia el orden de lectura en `red1.py` / `red2.py` / `redtanh.py`, hay que
volver a validar esa suposición.

---

## 7. Qué hacer para que la pregunta original pueda responderse

> **Estado: puntos 1–4 aplicados** en [`../Proyectov2/`](../Proyectov2/). Ver §8
> para el resultado. El punto 5 (calidad de los datos) sigue abierto.

En orden de prioridad. El punto 1 es imprescindible; sin él, ningún resultado
sobre «si la red aprende física» significa nada.

1. **Hacer el problema identificable.** Referir cada clip a su propio origen:
   entrada `t − t₀`, salida `y − y₀`. Así todas las trayectorias comparten
   condiciones iniciales y el único parámetro libre pasa a ser `g`.
   Alternativas: añadir `y₀` (y `v₀`) como entradas, o predecir los coeficientes
   `(c₀, c₁, c₂)` de cada clip en vez de puntos sueltos.
2. **Validación por clip completo** (*leave-one-clip-out*), no por punto
   individual.
3. **≥5 semillas por configuración**, reportando media y desviación estándar.
   Sin esto no se puede afirmar que una arquitectura sea mejor que otra: las
   diferencias reportadas (1.4 %) son menores que la dispersión entre semillas.
4. **Línea base clásica**: comparar contra el ajuste cuadrático por clip, que
   recupera `g` con 4 % de error. Una red solo es interesante si iguala eso.
5. **Mejorar los datos**: clips más largos o desde mayor altura (con 0.54 s el
   término cuadrático apenas destaca sobre el ruido); suavizar las derivadas o
   calcularlas por ajuste local en vez de diferencias finitas.

---

## 8. Cierre: el supuesto sí se verifica (2026-08-14, 17:11 CST)

Las secciones 1–7 quedaron escritas antes de rehacer el experimento. Este
apartado registra el desenlace.

Se construyó [`../Proyectov2/`](../Proyectov2/) aplicando las correcciones del
punto 7. Con las **mismas tres arquitecturas** y los **mismos 609 clips**:

| Formulación | Red tanh 32×32 | RMSE | R² | g recuperada | Error vs −9.81 |
|---|---|---:|---:|---:|---:|
| `v1` — `t → y` (la original) | control | 0.497 | 0.33 | −2.77 ± 0.56 | 71.8 % |
| `tau` — `τ → Δy` | | 0.167 | 0.90 | −9.07 ± 0.28 | 7.6 % |
| **`tau_v0`** — `(τ, v₀) → Δy` | | **0.142** | **0.93** | **−9.46 ± 0.33** | **3.8 %** |
| *referencia experimental* | ajuste por clip | 0.040 | 0.99 | −9.48 ± 0.11 | 3.4 % |

**La red coincide con el método clásico sobre los mismos datos dentro del
0.2 %.** El 3.4 % restante frente a −9.81 es un sesgo del experimento que
afecta por igual a la red y al ajuste clásico: es techo del conjunto de datos,
no fallo del modelo. Ver §7 de este archivo, punto «diagnosticar el sesgo».

Tres hallazgos que conviene no perder:

1. **El límite cambió de naturaleza.** Bajo `v1`, multiplicar por 165 los
   parámetros no mejoraba `g`. Bajo `tau_v0`, esa misma escalada la lleva de
   −4.68 a −9.46. Cuando el límite es de *información*, la capacidad no ayuda;
   cuando es de *representación*, sí. Es la confirmación más fuerte de que el
   diagnóstico de §2.3 era correcto.
2. **La red le gana al modelo analítico.** RMSE 0.142 contra 0.202 de
   `Δy = v₀τ + ½gτ²` con `g` global, con las mismas entradas. Probablemente
   compensa el sesgo de la estimación de `v₀`. Es el primer punto del proyecto
   donde la red aporta algo que el método clásico no da.
3. **La red 1→2→1 sigue fallando, pero ahora por otra razón.** En `v1` era
   falta de información; en `tau_v0` es falta de capacidad — dos unidades ReLU
   no forman una parábola. La distinción importa para no repetir la lectura
   equivocada de §2.4.

Instalado PyTorch 2.13 CPU en `~/Documentos/envs/main` (los scripts de v1 lo
importaban pero no estaba en el venv).

El informe final incorpora esto como **sección 8, «Verificación del supuesto:
reformulación del problema»**, con cuatro figuras en `imgs/v2/`. Se actualizaron
también el resumen, la introducción, las limitaciones (separadas entre
corregidas y persistentes), las conclusiones y el trabajo futuro.

---

## 9. La lección metodológica

> En la validación de un modelo de aprendizaje automático contra una teoría
> física, **el valor de referencia debe verificarse con el mismo rigor que el
> valor predicho.**

El error de §2.1 no fue de programación ni de física: fue aceptar un número
—`-0.35 m/s²`— como «el valor real de `g`» sin contrastarlo contra los
9.81 m/s² que se sabían de antemano. Un factor de 28 de discrepancia estaba a
la vista en la propia tabla.

Y el corolario de §2.3:

> Una red neuronal entrenada con MSE converge a la media condicional de los
> datos. Que esa media condicional coincida o no con la ley física depende
> enteramente de cómo se hayan estructurado las entradas.
