#!/usr/bin/env python3
"""
Script para analizar el impacto de épocas en tiempo de ejecución y error.
Varía el número de épocas y registra RMSE y tiempo de entrenamiento para múltiples redes.
"""
import os
import glob
import time
import math
import csv
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

# ================================================================
# Configuración
# ================================================================
DATA_DIR = "/home/alejandro/Practicas/Toma_de_datos/datos"
FILE_GLOB = os.path.join(DATA_DIR, "*.txt")
OUT_DIR = "outs"
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS_LIST = [10, 20, 50, 100, 200, 400, 700]
BATCH_SIZE = 512
LR = 1e-3

# Definir las arquitecturas de redes
MODELS = {
    "red1_simple": ("SmallNet_1", "1→2→1"),
    "red2_medium": ("MediumNet", "3→4→1"),
    "redtanh_deep": ("TanhNet", "Tanh profunda"),
}

# ================================================================
# Dataset (igual que en red2.py)
# ================================================================
def read_freefall_file(path):
    rows = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("t"):
                continue
            parts = [p.strip() for p in line.split(",") if p.strip() != ""]
            try:
                vals = [float(p) for p in parts]
            except:
                continue
            if len(vals) < 7:
                vals += [np.nan] * (7 - len(vals))
            rows.append(vals[:7])
    return np.array(rows)

class FreefallDataset(Dataset):
    def __init__(self, file_glob, input_dim=3):
        """
        input_dim=1: solo tiempo
        input_dim=3: tiempo, velocidad, aceleración
        """
        self.X = []
        self.Y = []
        self.input_dim = input_dim
        
        for path in sorted(glob.glob(file_glob)):
            arr = read_freefall_file(path)
            if arr.size == 0:
                continue
            t = arr[:, 0]
            y = arr[:, 2]
            vy = arr[:, 4]
            ay = arr[:, 6]
            mask = ~np.isnan(t) & ~np.isnan(y)
            t = t[mask]
            y = y[mask]
            vy = vy[mask]
            ay = ay[mask]

            vy = np.where(np.isnan(vy), 0.0, vy)
            ay = np.where(np.isnan(ay), 0.0, ay)

            for ti, vyi, ayi, yi in zip(t, vy, ay, y):
                if input_dim == 1:
                    self.X.append([ti])
                else:  # input_dim == 3
                    self.X.append([ti, vyi, ayi])
                self.Y.append([yi])

        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.float32)

        self.x_mean = self.X.mean(axis=0, keepdims=True)
        self.x_std = self.X.std(axis=0, keepdims=True) + 1e-9
        self.y_mean = self.Y.mean(axis=0, keepdims=True)
        self.y_std = self.Y.std(axis=0, keepdims=True) + 1e-9

        self.Xn = (self.X - self.x_mean) / self.x_std
        self.Yn = (self.Y - self.y_mean) / self.y_std

    def __len__(self):
        return len(self.Xn)

    def __getitem__(self, idx):
        return torch.from_numpy(self.Xn[idx]), torch.from_numpy(self.Yn[idx])

# ================================================================
# Modelos
# ================================================================
class SmallNet_1(nn.Module):
    """Red simple 1→2→1 (solo tiempo como entrada)."""
    def __init__(self, input_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

class MediumNet(nn.Module):
    """Red 3→4→1 (tiempo, velocidad, aceleración)."""
    def __init__(self, input_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 4)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

class TanhNet(nn.Module):
    """Red profunda con activación Tanh."""
    def __init__(self, input_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# ================================================================
# Entrenamiento y evaluación
# ================================================================
def train_model(dataset, model_class, model_name, epochs=100, batch_size=512, lr=1e-3, input_dim=1):
    """Entrena el modelo y retorna el modelo, dispositivo y tiempo de entrenamiento."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model_class(input_dim=input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    start_time = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = mse(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        total_loss /= n_train

        if ep % 100 == 0 or ep == 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += mse(model(xb.to(device)), yb.to(device)).item() * xb.size(0)
            val_loss /= n_val

    elapsed_time = time.time() - start_time
    return model, device, elapsed_time

def compute_rmse(model, device, dataset):
    """Calcula el RMSE de las predicciones."""
    model.eval()
    Xn = torch.from_numpy(dataset.Xn).to(device)
    with torch.no_grad():
        y_pred_n = model(Xn).cpu().numpy()
    y_pred = y_pred_n * dataset.y_std + dataset.y_mean
    y_real = dataset.Y
    err = y_pred - y_real
    rmse = math.sqrt(np.mean(err**2))
    return rmse

def compute_mae(model, device, dataset):
    """Calcula el MAE de las predicciones."""
    model.eval()
    Xn = torch.from_numpy(dataset.Xn).to(device)
    with torch.no_grad():
        y_pred_n = model(Xn).cpu().numpy()
    y_pred = y_pred_n * dataset.y_std + dataset.y_mean
    y_real = dataset.Y
    err = y_pred - y_real
    mae = np.mean(np.abs(err))
    return mae

# ================================================================
# Main
# ================================================================
def main():
    print("Cargando datos...")
    
    # Definir configuración de modelos con parámetros específicos
    model_configs = [
        {
            "name": "red1_simple",
            "model_class": SmallNet_1,
            "input_dim": 1,
            "description": "Red simple 1→2→1"
        },
        {
            "name": "red2_medium",
            "model_class": MediumNet,
            "input_dim": 3,
            "description": "Red 3→4→1"
        },
        {
            "name": "redtanh_deep",
            "model_class": TanhNet,
            "input_dim": 1,
            "description": "Red Tanh profunda"
        },
    ]

    all_results = []

    for config in model_configs:
        print(f"\n{'='*70}")
        print(f"Modelo: {config['description']} ({config['name']})")
        print(f"{'='*70}")

        ds = FreefallDataset(FILE_GLOB, input_dim=config["input_dim"])
        print(f"Total muestras: {len(ds)}")

        for epochs in EPOCHS_LIST:
            print(f"\n  --- Épocas: {epochs} ---")
            model, device, training_time = train_model(
                ds,
                model_class=config["model_class"],
                model_name=config["name"],
                epochs=epochs,
                batch_size=BATCH_SIZE,
                lr=LR,
                input_dim=config["input_dim"]
            )
            rmse = compute_rmse(model, device, ds)
            mae = compute_mae(model, device, ds)

            print(f"    Tiempo: {training_time:.2f} s | RMSE: {rmse:.6f} | MAE: {mae:.6f}")

            all_results.append({
                "model": config["name"],
                "model_desc": config["description"],
                "epochs": epochs,
                "training_time": training_time,
                "rmse": rmse,
                "mae": mae,
            })

    # Guardar resultados en CSV único
    results_file = os.path.join(OUT_DIR, "epochs_analysis_all_models.csv")
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=["model", "model_desc", "epochs", "training_time", "rmse", "mae"]
        )
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*70}")
    print(f"Resultados guardados en: {results_file}")
    print(f"{'='*70}")

    # Mostrar resumen por modelo
    print("\nRESUMEN POR MODELO:")
    print("-" * 90)
    for config in model_configs:
        model_results = [r for r in all_results if r["model"] == config["name"]]
        print(f"\n{config['description']}:")
        print(f"{'Épocas':<10} {'Tiempo (s)':<15} {'RMSE':<15} {'MAE':<15}")
        for r in model_results:
            print(f"{r['epochs']:<10} {r['training_time']:<15.2f} {r['rmse']:<15.6f} {r['mae']:<15.6f}")

if __name__ == "__main__":
    main()
