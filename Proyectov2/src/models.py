"""
Arquitecturas de red.

Se conservan las tres de v1 para que la comparación sea directa: si el
resultado cambia, el cambio viene de la formulación del problema y no de haber
usado redes distintas.
"""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Perceptrón multicapa genérico."""

    def __init__(self, input_dim: int, hidden: list[int], act: str = "relu"):
        super().__init__()
        Act = {"relu": nn.ReLU, "tanh": nn.Tanh}[act]
        capas, d = [], input_dim
        for h in hidden:
            capas += [nn.Linear(d, h), Act()]
            d = h
        capas.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*capas)

    def forward(self, x):
        return self.net(x)


# nombre -> (capas ocultas, activación)
ARQUITECTURAS = {
    "simple":    ([2],      "relu"),   # equivalente a la 1->2->1 de v1
    "media":     ([4],      "relu"),   # equivalente a la 3->4->1 de v1
    "tanh_deep": ([32, 32], "tanh"),   # equivalente a la Tanh profunda de v1
}


def build_model(nombre: str, input_dim: int) -> MLP:
    hidden, act = ARQUITECTURAS[nombre]
    return MLP(input_dim, hidden, act)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
