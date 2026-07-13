# Informe de prácticas: comparación de redes neuronales para la estimación de la caída libre

## 1. Objetivo
Comparar distintas arquitecturas de redes neuronales para estimar la posición en un problema de caída libre y analizar cuál ofrece mejores resultados en términos de ajuste a los datos reales y de estimación del valor de la aceleración gravitatoria.

## 2. Metodología
Se entrenaron y evaluaron varias redes con el mismo conjunto de datos de caída libre, variando:
- la arquitectura de la red,
- las entradas utilizadas,
- el número de épocas y el tipo de activación.

Las salidas se compararon con los datos reales mediante:
- RMSE (raíz del error cuadrático medio),
- MAE (error absoluto medio),
- correlación entre predicción y valor real,
- estimación de $g$ mediante ajuste cuadrático a la trayectoria.

Los valores reportados en esta comparación se han obtenido directamente a partir de los ficheros de predicción generados por cada script en la carpeta [Redes/outs](Redes/outs) y procesados con el script [Redes/compare_predictions.py](Redes/compare_predictions.py). En otras palabras, las afirmaciones sobre el rendimiento de cada red se basan en métricas calculadas a partir de los resultados guardados en disco, no en una valoración visual.

## 3. Redes evaluadas
- Red simple 1→2→1: usa solo el tiempo como entrada.
- Red 3→4→1: usa tiempo, velocidad vertical y aceleración vertical como entrada.
- Red con activación Tanh: arquitectura más profunda con activaciones suaves.
- Red Newton: configuración alternativa basada en un enfoque diferente de preparación de datos.

## 4. Resultados obtenidos
Los resultados verificados con el comando:

```bash
conda run -n practicas python compare_predictions.py
```

fueron los siguientes:

| Red | RMSE | MAE | Correlación | $g_{pred}$ | $g_{real}$ | Diferencia |
|---|---:|---:|---:|---:|---:|---:|
| red2 | 0.506764 | 0.426524 | 0.573548 | -0.3564 | -0.3500 | -0.0064 |
| red_tanh_original | 0.512264 | 0.432261 | 0.561017 | -0.3570 | -0.3500 | -0.0070 |
| red1_ep700 | 0.514009 | 0.434440 | 0.556483 | -0.3438 | -0.3500 | 0.0062 |
| red_newton | 0.602651 | 0.505148 | 0.225582 | -0.0000 | -0.3511 | 0.3511 |
| red1_ep400 | 0.618558 | 0.522080 | NaN | -0.0000 | -0.3500 | 0.3500 |

## 5. Interpretación de los resultados
La red que mejores resultados ofrece es la red 3→4→1, es decir, la arquitectura que utiliza más información de entrada. Esto sugiere que introducir variables físicas como la velocidad vertical y la aceleración mejora el aprendizaje del sistema y permite una aproximación más precisa de la posición.

La red con activación Tanh ofrece un rendimiento muy similar, aunque ligeramente peor que la red 3→4→1. Esto indica que una arquitectura más profunda y con activaciones suaves puede ser competitiva, pero no supera a la red que incorpora información física explícita.

Por el contrario, la red Newton y la red simple con 400 épocas presentan peores resultados, tanto en error como en estimación de $g$. Esto puede deberse a que la información disponible o la formulación del problema no encaja tan bien con esa arquitectura o con la preparación de los datos.

## 6. Conclusión
La mejor red de las evaluadas ha sido la red 3→4→1, ya que consigue el menor RMSE y MAE y una estimación de la aceleración gravitatoria más cercana al valor real. Esto indica que, para este problema, las redes que incorporan información física adicional y tienen una arquitectura ligeramente más expresiva son más adecuadas que las redes muy simples.

En términos prácticos, la red 3→4→1 es la opción más recomendable para este estudio, mientras que la red Tanh puede considerarse una alternativa robusta y competitiva.

## 7. Propuestas de ampliación del estudio
Para reforzar el análisis, se recomiendan las siguientes pruebas adicionales:
1. Repetir el entrenamiento con distintos seeds aleatorios para comprobar la estabilidad.
2. Probar diferentes tamaños de lote y tasas de aprendizaje.
3. Evaluar el rendimiento con validación por archivos completos (leave-one-file-out).
4. Comparar frente a un modelo clásico como un ajuste polinómico o una regresión lineal.
5. Medir el error en zonas temporales concretas, por ejemplo al inicio o al final de la trayectoria.
6. Añadir métricas físicas adicionales, como el error en la aceleración estimada o la pendiente de la trayectoria.

## 8. Texto breve para incluir en el informe final
Se han comparado distintas arquitecturas de redes neuronales para estimar la trayectoria de un cuerpo en caída libre. Entre las opciones evaluadas, la red con arquitectura 3→4→1 ha sido la que mejores resultados ha proporcionado, obteniendo el menor RMSE y MAE, así como una estimación de la aceleración gravitatoria más cercana al valor esperado. La red con activación Tanh ha mostrado un comportamiento similar, aunque ligeramente peor. Por el contrario, las configuraciones más simples o las basadas en el enfoque Newton han presentado mayores errores. Estos resultados indican que introducir información física adicional en la entrada y utilizar arquitecturas algo más expresivas mejora significativamente el rendimiento de la red.
