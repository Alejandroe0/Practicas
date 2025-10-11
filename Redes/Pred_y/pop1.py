import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pyplot as plt

# === 1. Cargar todos los archivos TXT ===
ruta = "/home/alejandro/Practicas/Toma_de_datos/datos/*.txt"
archivos = glob.glob(ruta)
todos = []

print(f"Encontré {len(archivos)} archivos .txt")

for f in archivos:
    try:
        # Leer archivo saltando la primera línea y usando la segunda como header
        df = pd.read_csv(f, skiprows=1, header=0, engine='python')
        
        print(f"Procesando {f}: {len(df.columns)} columnas, {len(df)} filas")
        
        # Limpiar nombres de columnas (eliminar comas extras)
        df.columns = [col.strip(',') for col in df.columns]
        
        # Verificar que tenemos las columnas necesarias
        if 't' in df.columns and 'y' in df.columns:
            # Seleccionar solo las columnas que necesitamos y eliminar filas vacías
            df_clean = df[['t', 'y']].dropna()
            
            # Convertir notación científica (E0, E-1, etc.) a float
            df_clean['t'] = pd.to_numeric(df_clean['t'], errors='coerce')
            df_clean['y'] = pd.to_numeric(df_clean['y'], errors='coerce')
            
            # Eliminar posibles NaN después de la conversión
            df_clean = df_clean.dropna()
            
            if len(df_clean) > 0:
                todos.append(df_clean)
                print(f"✓ {f}: {len(df_clean)} puntos válidos")
            else:
                print(f"✗ {f}: No hay datos válidos después de limpiar")
        else:
            print(f"✗ {f}: No tiene columnas 't' y 'y'")
            
    except Exception as e:
        print(f"✗ Error en {f}: {e}")

if not todos:
    raise Exception("No se cargaron datos válidos.")

datos = pd.concat(todos, ignore_index=True)
print(f"\n=== RESUMEN ===")
print(f"Total de datos combinados: {len(datos)}")
print(f"Rango de tiempo: {datos['t'].min():.4f} a {datos['t'].max():.4f}")
print(f"Rango de altura: {datos['y'].min():.4f} a {datos['y'].max():.4f}")
print(f"Primeros datos:")
print(datos.head())

# === 2. Preparar datos ===
X = torch.tensor(datos["t"].values.reshape(-1, 1), dtype=torch.float32)
y = torch.tensor(datos["y"].values.reshape(-1, 1), dtype=torch.float32)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDatos de entrenamiento: {len(X_train)}")
print(f"Datos de validación: {len(X_val)}")

# === 3. Definir red 1→2→1 ===
class NN_Y(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        return self.output(self.relu(self.hidden(x)))

model = NN_Y()
print(f"\nParámetros del modelo: {sum(p.numel() for p in model.parameters())}")

# === 4. Entrenamiento ===
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 1000

print("\n=== ENTRENAMIENTO ===")
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val)
        print(f"Época {epoch:4d} | Train: {loss.item():.6e} | Val: {val_loss.item():.6e}")

# === 5. Guardar resultados ===
model.eval()
with torch.no_grad():
    pred = model(X).numpy()

df_pred = pd.DataFrame({
    "t": X.numpy().flatten(),
    "y_real": y.numpy().flatten(),
    "y_pred": pred.flatten()
})

df_pred.to_csv("out/predicciones_y.csv", index=False)

# Pesos
pesos = {name: param.data.numpy() for name, param in model.named_parameters()}
pd.Series({k: v.flatten().tolist() for k, v in pesos.items()}).to_csv("out/pesos_y.csv")

print(f"\nEntrenamiento completado!")
print(f"Predicciones guardadas: predicciones_y.csv")
print(f"Pesos guardados: pesos_y.csv")

# === 6. Visualizar ===
plt.figure(figsize=(12, 6))
plt.scatter(df_pred["t"], df_pred["y_real"], s=1, alpha=0.5, label="Datos reales")
plt.plot(df_pred["t"], df_pred["y_pred"], 'r-', linewidth=2, label="Predicción NN")
plt.xlabel("Tiempo (s)")
plt.ylabel("Altura (y)")
plt.legend()
plt.title("Red Neuronal 1→2→1 - Predicción de Altura")
plt.grid(True, alpha=0.3)
plt.savefig("out/prediccion_altura.png", dpi=300, bbox_inches='tight')
plt.show()