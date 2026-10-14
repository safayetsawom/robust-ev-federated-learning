import numpy as np
from src.data import load_all
from src.model import DemandMLP, train_local, set_seed
from src.common import LR, TOTAL_EPOCHS, evaluate_all, summarize


def run_centralized(data, seed=0):
    """Upper-reference: pool ALL stations' raw data on one machine."""
    set_seed(seed)
    X = np.concatenate([d["X_train"] for d in data.values()])
    y = np.concatenate([d["y_train"] for d in data.values()])
    model = train_local(DemandMLP(), X, y, epochs=TOTAL_EPOCHS, lr=LR)
    return evaluate_all(model, data)


if __name__ == "__main__":
    data, _ = load_all(seed=0)
    res = run_centralized(data)
    for n, m in res.items():
        print(f"{n:15s} MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f}")
    print("average        ", {k: round(v, 2) for k, v in summarize(res).items()})