import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data import SCALE, INPUT_DIM, load_all


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DemandMLP(nn.Module):
    def __init__(self, in_dim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_local(model, X, y, epochs=5, lr=1e-3, batch_size=64, device=None):
    device = device or get_device()
    model.to(device).train()
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model


@torch.no_grad()
def predict(model, X, device=None):
    device = device or get_device()
    model.to(device).eval()
    return model(torch.from_numpy(X).to(device)).cpu().numpy()


def metrics(y_true, y_pred):
    """MAE and RMSE in kWh (undoes the normalisation)."""
    yt, yp = y_true * SCALE, y_pred * SCALE
    return {"MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp)))}


if __name__ == "__main__":   # smoke test
    set_seed(0)
    print("device:", get_device())
    data, _ = load_all()
    first = next(iter(data))
    model = train_local(DemandMLP(), data[first]["X_train"], data[first]["y_train"], epochs=30)
    for name, d in data.items():
        m = metrics(d["y_test"], predict(model, d["X_test"]))
        print(f"trained on {first} -> tested on {name}: MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f}")