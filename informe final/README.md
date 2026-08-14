# Informe final de año de prácticas

Documento unificado: fusiona el anteproyecto (`../anteproyecto/anteproyectoPractica.tex`,
mayo 2024) con el análisis de resultados (`../Resultados/Resultados.tex`),
añadiendo una sección de resultados y discusión.

## Compilación

```bash
pdflatex informeFinal
bibtex   informeFinal
pdflatex informeFinal
pdflatex informeFinal
```

o simplemente `latexmk -pdf informeFinal.tex`.

## Contenido de la carpeta

| Archivo | Descripción |
|---|---|
| `informeFinal.tex` | Fuente del informe |
| `BITACORA-correcciones.md` | **Registro fechado de los errores encontrados y los cambios aplicados** |
| `referencias.bib` | Bibliografía (copia corregida de la del anteproyecto) |
| `figuras_informe.py` | Genera las cuatro figuras nuevas del análisis por clip |
| `imgs/` | Figuras del análisis de la formulación inicial |
| `imgs/v2/` | Figuras de la verificación (generadas por `../Proyectov2/`) |
| `usacAzul.jpg`, `ecfmAzul.png` | Logos de la portada |
| `sigmoide.png`, `tanh.png`, `relu.png` | Figuras del marco teórico |

## Regenerar las figuras

`figuras_informe.py` lee los datos de `../Toma_de_datos/datos/*.txt` y las
predicciones de `../Redes/outs/predicciones_*.csv`, y escribe en `imgs/`:

- `agregado_vs_clip.png` — por qué el ajuste sobre el conjunto agregado no mide `g`
- `hist_g_por_clip.png` — distribución de `g` por clip, datos vs. predicciones
- `pred_vs_real.png` — dispersión predicción/valor experimental
- `trayectorias_ejemplo.png` — cuatro clips con las predicciones superpuestas

```bash
python3 figuras_informe.py
```

Las otras dos figuras (`epochs_analysis_*.png`) provienen de
`../Redes/plot_epochs_analysis.py`.

## Cambios respecto a los documentos de origen

> Resumen. El detalle completo, con evidencia numérica y fechas, está en
> [`BITACORA-correcciones.md`](BITACORA-correcciones.md).

**Marco teórico.** Se eliminó la subsección de divergencia (no se usa en el
trabajo; la función de coste empleada es MSE). Se ampliaron mínimos cuadrados y
ajuste polinómico, por ser la herramienta con la que se estima `g`. Se añadieron
el coeficiente de determinación, la normalización z, sobreajuste y validación, e
identificabilidad del modelo. Se corrigió la fórmula del MAE, se eliminó el MASE
(no se usó) y se sustituyó TensorFlow por PyTorch, que es lo que se usó en la
implementación.

**Resultados.** El análisis de la versión anterior comparaba la `g` predicha
contra un ajuste cuadrático hecho sobre los 606 clips agregados, cuyo
coeficiente (`+0.35 m/s²`) no es la aceleración de la gravedad. Repitiendo el
ajuste clip por clip, los datos experimentales dan `g` con mediana
`-9.38 m/s²` mientras que las redes dan entre `0` y `-3.6 m/s²`: las redes no
recuperan el modelo físico. El informe documenta ese diagnóstico y su causa
(el mapeo `y = f(t)` no es identificable con los clips concatenados).

**Verificación (sección 8 del informe).** Rehecho el experimento en
[`../Proyectov2/`](../Proyectov2/) con las mismas redes y los mismos datos pero
formulado como `(τ, v₀) → Δy`, la red tanh recupera `g = -9.46 ± 0.33 m/s²`,
a un 0.2 % de lo que el método clásico saca de esos mismos datos. El supuesto
del anteproyecto queda verificado.
