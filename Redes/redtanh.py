#!/usr/bin/env python3
import os
import glob
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

NAME = "red_tanh_original"
EPOCHS = 700
BATCH_SIZE = 512
LR = 1e-3

# ================================================================
# Lectura de archivos Tracker (t,x,y,vx,vy,ax,ay)
# ================================================================
def read_tracker_file(path):
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

# ================================================================
# Dataset
# ================================================================
class FreefallDataset(Dataset):
    def __init__(self, file_glob):
        self.X = []
        self.Y = []

        files = sorted(glob.glob(file_glob))
        if len(files) == 0:
            raise RuntimeError("No se encontraron archivos de datos.")

        for path in files:
            arr = read_tracker_file(path)
            if arr.size == 0:
                continue

            t = arr[:, 0]
            y = arr[:, 2]

            mask = ~np.isnan(t) & ~np.isnan(y)
            t = t[mask]
            y = y[mask]

            for ti, yi in zip(t, y):
                self.X.append([ti])
                self.Y.append([yi])

        if len(self.X) == 0:
            raise RuntimeError("Dataset vacío: no hay muestras válidas.")

        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.float32)

        # Normalización (z-score)
        self.x_mean = self.X.mean(axis=0, keepdims=True)
        self.x_std  = self.X.std(axis=0, keepdims=True) + 1e-9
        self.y_mean = self.Y.mean(axis=0, keepdims=True)
        self.y_std  = self.Y.std(axis=0, keepdims=True) + 1e-9

        self.Xn = (self.X - self.x_mean) / self.x_std
        self.Yn = (self.Y - self.y_mean) / self.y_std

    def __len__(self):
        return len(self.Xn)

    def __getitem__(self, idx):
        return torch.from_numpy(self.Xn[idx]), torch.from_numpy(self.Yn[idx])

# ================================================================
# Red neuronal (no cuadrática explícita)
# ================================================================
class TanhNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# ================================================================
# Entrenamiento
# ================================================================
def train_model(dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TanhNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mse = nn.MSELoss()

    for ep in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = mse(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= n_train

        if ep % 10 == 0 or ep == 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += mse(model(xb.to(device)), yb.to(device)).item() * xb.size(0)
            val_loss /= n_val

            print(f"Epoch {ep:03d} | TrainLoss={train_loss:.6e} | ValLoss={val_loss:.6e}")

    return model, device

# ================================================================
# Ajuste cuadrático (solo para extraer g)
# ================================================================
def fit_quadratic(t, y):
    T = np.vstack([np.ones_like(t), t, t**2]).T
    coef, *_ = np.linalg.lstsq(T, y, rcond=None)
    return coef  # c0, c1, c2

# ================================================================
# Main
# ================================================================
def main():
    os.makedirs("outs", exist_ok=True)

    ds = FreefallDataset(FILE_GLOB)
    print("Total muestras:", len(ds))
    print(f"Rango t: {ds.X.min():.3f} – {ds.X.max():.3f} s")
    print(f"Rango y: {ds.Y.min():.3f} – {ds.Y.max():.3f} m")

    model, device = train_model(ds)

    # Guardar pesos
    torch.save(model.state_dict(), f"outs/{NAME}_ep{EPOCHS}.pt")

    # Predicciones
    model.eval()
    Xn = torch.from_numpy(ds.Xn).to(device)
    with torch.no_grad():
        y_pred_n = model(Xn).cpu().numpy()

    y_pred = y_pred_n * ds.y_std + ds.y_mean
    y_real = ds.Y
    t = ds.X[:, 0]

    df = pd.DataFrame({
        "t": t,
        "y_real": y_real.flatten(),
        "y_pred": y_pred.flatten()
    })
    df.to_csv(f"outs/predicciones_{NAME}_ep{EPOCHS}.csv", index=False)

    # Estimación de g
    coef_pred = fit_quadratic(t, y_pred.flatten())
    coef_real = fit_quadratic(t, y_real.flatten())

    print("\nAjuste cuadrático (solo diagnóstico):")
    print(f"Coef pred: {coef_pred}")
    print(f"g_red ≈ {-2 * coef_pred[2]:.4f} m/s²")

    print(f"Coef real: {coef_real}")
    print(f"g_datos ≈ {-2 * coef_real[2]:.4f} m/s²")

if __name__ == "__main__":
    main()
