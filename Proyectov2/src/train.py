"""
Entrenamiento de una red, con parada temprana sobre clips de validación no
vistos.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

torch.set_num_threads(4)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_model(model: nn.Module,
                Xtr: np.ndarray, Ytr: np.ndarray,
                Xva: np.ndarray, Yva: np.ndarray,
                epochs: int = 600, lr: float = 1e-2,
                batch_size: int = 1024, patience: int = 80,
                verbose: bool = False):
    """Entrena con Adam + MSE y devuelve (modelo, historial).

    La parada temprana usa clips de validación completos que la red no ha visto,
    de modo que mide generalización a una trayectoria nueva y no interpolación
    dentro de trayectorias ya vistas.
    """
    dev = torch.device("cpu")
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    xt = torch.from_numpy(Xtr).to(dev)
    yt = torch.from_numpy(Ytr).to(dev)
    xv = torch.from_numpy(Xva).to(dev)
    yv = torch.from_numpy(Yva).to(dev)

    n = len(xt)
    hist = {"train": [], "val": []}
    mejor, mejor_ep, mejor_state = np.inf, 0, None

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            loss = mse(model(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        tr = tot / n

        model.eval()
        with torch.no_grad():
            va = mse(model(xv), yv).item()
        hist["train"].append(tr)
        hist["val"].append(va)

        if va < mejor - 1e-7:
            mejor, mejor_ep = va, ep
            mejor_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif ep - mejor_ep >= patience:
            break

        if verbose and (ep % 50 == 0 or ep == 1):
            print(f"    ep {ep:4d}  train={tr:.5e}  val={va:.5e}")

    if mejor_state is not None:
        model.load_state_dict(mejor_state)
    hist["mejor_epoca"] = mejor_ep
    hist["epocas_corridas"] = len(hist["train"])
    return model, hist


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(X)).cpu().numpy()
