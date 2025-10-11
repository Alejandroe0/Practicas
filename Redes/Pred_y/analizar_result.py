import numpy as np
import pandas as pd

# Cargar y ver predicciones
pred_df = pd.read_csv("out/predicciones_y.csv")
print(pred_df.head(10))

# Calcular error promedio
error = np.mean(np.abs(pred_df['y_real'] - pred_df['y_pred']))
print(f"Error absoluto promedio: {error:.4f}")