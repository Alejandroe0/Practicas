import os
import glob
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

DATA_DIR = "/home/alejandro/Practicas/Toma_de_datos/datos"  
FILE_GLOB = os.path.join(DATA_DIR, "*.txt")

name = 'red1'
epocas = 400

# ================================================================
# 1. Lectura de datos
# ================================================================
def read_freefall_file(path):
    """
    Lee un archivo txt con columnas tipo t,x,y,vx,vy,ax,ay
    Devuelve un np.array Nx7.
    """
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
    def __init__(self, file_glob):
        self.X = []
        self.Y = []
        for path in sorted(glob.glob(file_glob)):
            arr = read_freefall_file(path)
            if arr.size == 0:
                continue
            t = arr[:,0]
            y = arr[:,2]
            vy = arr[:,4]
            ay = arr[:,6]
            mask = ~np.isnan(t) & ~np.isnan(y)
            t = t[mask]
            y = y[mask]
            vy = vy[mask]
            ay = ay[mask]

            # Rellenar NaN después del filtrado
            vy = np.where(np.isnan(vy), 0.0, vy)
            ay = np.where(np.isnan(ay), 0.0, ay)

            for ti, vyi, ayi, yi in zip(t, vy, ay, y):
                self.X.append([ti, vyi, ayi])
                self.Y.append([yi])
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
# 2. Modelo 3→2→1
# ================================================================
class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(2, 1)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# ================================================================
# 3. Entrenamiento simple
# ================================================================
def train_model(dataset, epochs=100, batch_size=512, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SmallNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for ep in range(1, epochs+1):
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

        if ep % 10 == 0 or ep == 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += mse(model(xb.to(device)), yb.to(device)).item() * xb.size(0)
            val_loss /= n_val
            print(f"Epoch {ep:03d} | TrainLoss={total_loss:.6e} | ValLoss={val_loss:.6e}")
    return model, device

# ================================================================
# 4. Evaluación y exportación
# ================================================================
def fit_quadratic(t, y):
    T = np.vstack([np.ones_like(t), t, t**2]).T
    coef, *_ = np.linalg.lstsq(T, y, rcond=None)
    return coef.flatten()  # [c0, c1, c2]

def export_weights(model, file_path="outs/pesos_red.csv"):
    rows = []
    for name, param in model.named_parameters():
        rows.append({"nombre": name, "valores": param.detach().cpu().numpy().ravel().tolist()})
    pd.DataFrame(rows).to_csv(file_path, index=False)
    print("Pesos guardados en:", file_path)

def main():
    ds = FreefallDataset(FILE_GLOB)
    print("Total muestras:", len(ds))

    model, device = train_model(ds, epochs=epocas, batch_size=512, lr=1e-3)

    # Exportar pesos
    export_weights(model, f"outs/pesos_{name}_ep{epocas}.csv")

    # Predicciones y comparación
    model.eval()
    Xn = torch.from_numpy(ds.Xn).to(device)
    with torch.no_grad():
        y_pred_n = model(Xn).cpu().numpy()
    y_pred = y_pred_n * ds.y_std + ds.y_mean
    y_real = ds.Y

    # Guardar predicciones
    df = pd.DataFrame({
        "t": ds.X[:,0],
        "vy": ds.X[:,1],
        "ay": ds.X[:,2],
        "y_real": y_real.flatten(),
        "y_pred": y_pred.flatten()
    })
    df.to_csv(f"outs/predicciones_{name}_ep{epocas}.csv", index=False)
    print("Predicciones guardadas en: predicciones.csv")

    # Ajuste cuadrático sobre las predicciones
    coef = fit_quadratic(ds.X[:,0], y_pred.flatten())
    print(f"Coeficientes cuadráticos (c0, c1, c2): {coef}")
    print(f"g_aprendido ≈ {2*coef[2]:.4f} m/s²")

if __name__ == "__main__":
    main()
