# Informe final — versión de resultados

Segunda redacción del informe final de año de prácticas. Toma el anteproyecto
(`../../anteproyecto/anteproyectoPractica.tex`, mayo 2024) y le añade la sección
de resultados con el experimento de `../` (Proyecto v2).

**Diferencia con [`../../informe final/`](../../informe%20final/):** esta versión
presenta únicamente el resultado final. No narra el planteamiento inicial ni su
diagnóstico; las tres formulaciones de las entradas aparecen como un estudio de
ablación deliberado —*directa*, *referida*, *completa*— para medir cuánto aporta
cada ingrediente de la representación. La otra versión documenta además el
recorrido que llevó hasta ahí. Las dos usan los mismos datos y los mismos
números; se mantienen en paralelo para decidir cuál conviene presentar.

| | `informe final/` | `Proyectov2/InformeFinal/` (este) |
|---|---|---|
| Narrativa | diagnóstico → corrección → verificación | resultado final directo |
| Secciones de resultados | formulación inicial + verificación | una sola |
| Formulaciones | `v1` / `tau` / `tau_v0` | directa / referida / completa |
| Estudio de épocas | sí, con datos de la formulación inicial | sí, de la formulación completa |
| Inspección de pesos | no | sí (§7.6) |
| Extensión | 31 págs. | 30 págs. |

## Compilación

```bash
latexmk -pdf informeFinal.tex
```

o, sin latexmk:

```bash
pdflatex informeFinal && bibtex informeFinal && pdflatex informeFinal && pdflatex informeFinal
```

## Contenido de la carpeta

| Archivo | Descripción |
|---|---|
| `informeFinal.tex` | Fuente del informe |
| `referencias.bib` | Bibliografía |
| `figuras.py` | Genera las figuras de `imgs/` a partir de `../outs/` |
| `imgs/` | Figuras del experimento |
| `usacAzul.jpg`, `ecfmAzul.png` | Logos de la portada |
| `sigmoide.png`, `tanh.png`, `relu.png` | Figuras del marco teórico |

## Regenerar las figuras

`figuras.py` es una variante de `../src/figures.py` con las etiquetas de las
formulaciones escritas como aparecen en el texto, más las figuras propias de
este informe. Lee `../outs/*.csv|npz` y `../../Toma_de_datos/datos/`:

```bash
python3 figuras.py
```

| Figura | Contenido | Requiere |
|---|---|---|
| `00_datos.png` | clips referidos a su origen y distribución de `g` experimental | — |
| `01_formulaciones.png` | `g` recuperada y RMSE por formulación y arquitectura | `run_all.py` |
| `02_g_por_clip.png` | distribución de `g` por clip, datos vs. predicciones | `run_all.py` |
| `03_trayectorias.png` | cuatro clips de prueba con las predicciones superpuestas | `run_all.py` |
| `04_rmse_vs_error_g.png` | las 45 corridas en un solo gráfico | `run_all.py` |
| `05_epocas.png` | tiempo, RMSE y error en `g` frente a épocas | `epocas.py` |
| `06_pesos.png` | curvatura aprendida, ecuación leída, pesos de la 1ª capa | `pesos.py` |

Es decir, el orden completo desde cero:

```bash
cd ../src && python3 run_all.py && python3 epocas.py && python3 pesos.py
cd ../InformeFinal && python3 figuras.py && latexmk -pdf informeFinal.tex
```

Si se reejecuta cualquiera de los tres experimentos hay que regenerar las
figuras y revisar los números de las tablas contra `../outs/resumen.txt`,
`../outs/epocas.csv` y `../outs/pesos.txt`.

## Resultados que reporta

**Verificación del supuesto (§7.2–7.3)**

| | |
|---|---|
| `g` de los datos (ajuste clásico por clip) | −9.43 m/s² |
| `g` recuperada por la red (τ, v₀ → Δy, tanh 32×32) | −9.46 ± 0.33 m/s² |
| Discrepancia frente al método clásico | 0.2 % |
| RMSE en clips no vistos | 0.142 m (R² = 0.93) |
| RMSE del MRUV con `g` global | 0.202 m |

**Coste-beneficio de las épocas (§7.5)**

- El tiempo es exactamente lineal en el número de épocas.
- El RMSE se estanca hacia la época 20; la mejor `g` llega hacia la 200.
- Las dos cosas **no son monótonas entre sí**: la red `media (4)` mejora su
  RMSE de 0.167 a 0.157 m entre las épocas 10 y 800 mientras su error en `g`
  empeora del 17.7 % al 29.1 %. Un criterio de parada basado solo en el error
  de validación no garantiza parar donde la física es mejor.
- Punto de operación razonable: 100–200 épocas (~2 s).

**Inspección de la red (§7.6)**

- La red simple (9 parámetros) se escribe explícitamente: dos quiebres ReLU,
  segunda derivada nula. No puede representar `g` por construcción.
- Derivadas de la red tanh: signo y orden de magnitud correctos, pero la
  curvatura **no es constante** (mediana −7.8, IQR [−13.9, −2.8]).
- Control clave: ajustando `Δy = aτ + b·v₀τ + cτ²` a la salida de la red **y a
  los datos**, el coeficiente cuadrático de la red sigue al de los datos con
  <2 % de diferencia en los dos dominios probados, sesgos incluidos. La red
  aprendió lo que el conjunto contiene, no la ley ideal.
- No hay «neurona de la gravedad»: 15 de 32 unidades operan en el régimen
  lineal de la tanh y la representación está repartida.
- Conclusión: la física está en el **comportamiento** de la red, no en una
  estructura interna legible.
