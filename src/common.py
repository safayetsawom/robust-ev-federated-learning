"""Shared experiment settings and helpers."""
import numpy as np
from src.model import predict, metrics

ROUNDS = 100          # federated communication rounds
LOCAL_EPOCHS = 1     # local epochs per round
LR = 1e-3
SEEDS = [0, 1, 2]
TOTAL_EPOCHS = ROUNDS * LOCAL_EPOCHS   # centralized / local-only get the same epoch budget


def evaluate_all(model, data):
    """Per-station MAE/RMSE (kWh) on each station's own held-out test set."""
    return {n: metrics(d["y_test"], predict(model, d["X_test"])) for n, d in data.items()}


def summarize(per_station):
    """Average MAE/RMSE across stations."""
    return {k: float(np.mean([m[k] for m in per_station.values()])) for k in ("MAE", "RMSE")}