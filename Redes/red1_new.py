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
DATA_DIR = "/home/alejandro/Practicas/Toma_de_datos/datos_resample"
FILE_GLOB = os.path.join(DATA_DIR, "*_newton.csv")

OUT_DIR = "outs"
os.makedirs(OUT_DIR, exist_ok=True)

name = "red_newton"
epocas = 700

# ================================================================
# Lectura de datos (CSV con columnas t, y)
# ================================================================
def read_newton_file(path):
    df = pd.read_csv(path)

    if not {"t", "y"}.issubset(df.columns):
        return None

    t = df["t"].values
    y = df["y"].values

    mask = ~np.isnan(t) & ~np.isnan(y)
    return t[mask], y[mask]

# ================================================================
# Dataset
# ================================================================
class FreefallDataset(Dataset):
    def __init__(self, file_glob):
        self.X = []
        self.Y = []

        for path in sorted(glob.glob(file_glob)):
            data = read_newton_file(path)
            if data is None:
                continue

            t, y = data
            for ti, yi in zip(t, y):
                self.X.append([ti])   # entrada: tiempo
                self.Y.append([yi])   # salida: posición

        self.X = np.array(self.X, dtype=np.float32)
        self.Y = np.array(self.Y, dtype=np.float32)

        # Normalización z-score
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
# Modelo 1 → 2 → 1
# ================================================================
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# ================================================================
# Entrenamiento
# ================================================================
def train_model(dataset, epochs=100, batch_size=512, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SmallNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for ep in range(1, epochs + 1):
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

        if ep == 1 or ep % 10 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += mse(model(xb.to(device)), yb.to(device)).item() * xb.size(0)
            val_loss /= n_val

            print(
                f"Epoch {ep:03d} | "
                f"TrainLoss={train_loss:.6e} | "
                f"ValLoss={val_loss:.6e}"
            )

    return model, device

# ================================================================
# Ajuste cuadrático
# ================================================================
def fit_quadratic(t, y):
    T = np.vstack([np.ones_like(t), t, t**2]).T
    coef, *_ = np.linalg.lstsq(T, y, rcond=None)
    return coef.flatten()

# ================================================================
# Exportar pesos
# ================================================================
def export_weights(model, path):
    rows = []
    for name, param in model.named_parameters():
        rows.append({
            "nombre": name,
            "valores": param.detach().cpu().numpy().ravel().tolist()
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print("Pesos guardados en:", path)

# ================================================================
# Main
# ================================================================
def main():
    ds = FreefallDataset(FILE_GLOB)

    print("Total muestras:", len(ds))
    print(f"Rango t: {ds.X.min():.3f} – {ds.X.max():.3f} s")
    print(f"Rango y: {ds.Y.min():.3f} – {ds.Y.max():.3f} m")

    model, device = train_model(
        ds,
        epochs=epocas,
        batch_size=512,
        lr=1e-3
    )

    # Guardar pesos
    export_weights(
        model,
        f"{OUT_DIR}/pesos_{name}_ep{epocas}.csv"
    )

    # Predicciones
    model.eval()
    Xn = torch.from_numpy(ds.Xn).to(device)
    with torch.no_grad():
        y_pred_n = model(Xn).cpu().numpy()

    y_pred = y_pred_n * ds.y_std + ds.y_mean
    y_real = ds.Y

    df = pd.DataFrame({
        "t": ds.X[:, 0],
        "y_real": y_real.flatten(),
        "y_pred": y_pred.flatten()
    })

    df.to_csv(
        f"{OUT_DIR}/predicciones_{name}_ep{epocas}.csv",
        index=False
    )

    # Ajuste cuadrático
    coef_pred = fit_quadratic(ds.X[:, 0], y_pred.flatten())
    coef_real = fit_quadratic(ds.X[:, 0], y_real.flatten())

    print("\nAjuste cuadrático:")
    print("Coef pred:", coef_pred)
    print(f"g_red ≈ {-2 * coef_pred[2]:.4f} m/s²")

    print("Coef real:", coef_real)
    print(f"g_datos ≈ {-2 * coef_real[2]:.4f} m/s²")

if __name__ == "__main__":
    main()
