import numpy as np
from src.data import WINDOW, load_all
from src.model import DemandMLP, train_local, predict, metrics, set_seed
from src.common import LR, TOTAL_EPOCHS, summarize


def run_persistence(data):
    """Naive baseline: next hour = last hour."""
    return {n: metrics(d["y_test"], d["X_test"][:, WINDOW - 1:WINDOW]) for n, d in data.items()}


def run_seasonal_naive(data):
    """Naive baseline: next hour = same hour yesterday (t-24)."""
    return {n: metrics(d["y_test"], d["X_test"][:, 0:1]) for n, d in data.items()}


def run_local_only(data, seed=0):
    """Each station trains its own model alone. No sharing at all."""
    out = {}
    for n, d in data.items():
        set_seed(seed)
        m = train_local(DemandMLP(), d["X_train"], d["y_train"], epochs=TOTAL_EPOCHS, lr=LR)
        out[n] = metrics(d["y_test"], predict(m, d["X_test"]))
    return out


if __name__ == "__main__":
    data, _ = load_all(seed=0)
    for label, res in [("persistence", run_persistence(data)),
                       ("seasonal_naive", run_seasonal_naive(data)),
                       ("local_only", run_local_only(data))]:
        s = summarize(res)
        print(f"{label:15s} MAE={s['MAE']:.2f} RMSE={s['RMSE']:.2f}")