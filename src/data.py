"""Synthetic EV charging demand data. ALL DATA IN THIS PROJECT IS SIMULATED."""
from pathlib import Path
import numpy as np
import pandas as pd

SCALE = 100.0            # fixed constant (kWh) for normalising; no data statistics are shared
WINDOW = 24              # use the previous 24 hours to predict the next hour
INPUT_DIM = WINDOW + 4   # 24 lagged demands + sin/cos(hour) + sin/cos(day of week)
RESULTS = Path("results")

STATIONS = {
    "A_office":      dict(amp=60, weekend=0.35),
    "B_residential": dict(amp=50, weekend=1.20),
    "C_mall":        dict(amp=65, weekend=1.20),
}


def _bump(h, mu, sigma):
    """Circular Gaussian bump over the 24-hour clock."""
    d = np.abs(h - mu)
    d = np.minimum(d, 24 - d)
    return np.exp(-0.5 * (d / sigma) ** 2)


def _profile(name, h):
    if name == "A_office":        # morning + evening commuter peaks
        return 0.8 * _bump(h, 8.5, 1.5) + 0.5 * _bump(h, 17.5, 1.5) + 0.3 * _bump(h, 13, 3.0) + 0.05
    if name == "B_residential":   # evening home charging
        return 1.0 * _bump(h, 20, 2.2) + 0.25 * _bump(h, 7.5, 1.5) + 0.05
    if name == "C_mall":          # broad midday-to-evening plateau
        return 0.9 * _bump(h, 15, 4.5) + 0.15
    raise ValueError(name)


def generate_station(name, days=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    hour = idx.hour.values.astype(float)
    weekend = idx.dayofweek.values >= 5
    cfg = STATIONS[name]

    base = _profile(name, hour)
    base = np.where(weekend, base * cfg["weekend"], base)
    daily = np.repeat(rng.lognormal(0, 0.10, size=days), 24)   # day-to-day variation
    demand = cfg["amp"] * base * daily + rng.normal(0, 0.04 * cfg["amp"], len(idx))
    demand = np.clip(demand, 0, SCALE)

    return pd.DataFrame({"timestamp": idx, "station": name,
                         "demand_kwh": demand, "simulated": True})


def make_windows(df):
    d = df["demand_kwh"].values / SCALE
    ts = df["timestamp"]
    X, y = [], []
    for i in range(WINDOW, len(d)):
        h = 2 * np.pi * ts.iloc[i].hour / 24
        w = 2 * np.pi * ts.iloc[i].dayofweek / 7
        X.append(np.concatenate([d[i - WINDOW:i], [np.sin(h), np.cos(h), np.sin(w), np.cos(w)]]))
        y.append(d[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)


def load_all(days=120, test_frac=0.2, seed=0):
    """Returns (dict of per-station train/test arrays, full DataFrame). Chronological split."""
    data, frames = {}, []
    for i, name in enumerate(STATIONS):
        df = generate_station(name, days, seed + i)
        frames.append(df)
        X, y = make_windows(df)
        n_test = int(len(X) * test_frac)
        data[name] = dict(X_train=X[:-n_test], y_train=y[:-n_test],
                          X_test=X[-n_test:], y_test=y[-n_test:])
    return data, pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    RESULTS.mkdir(exist_ok=True)
    data, df = load_all()
    df.to_csv(RESULTS / "simulated_data.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, (name, g) in zip(axes, df.groupby("station")):
        ax.plot(g["timestamp"].iloc[:24 * 7], g["demand_kwh"].iloc[:24 * 7])
        ax.set_ylabel("kWh"); ax.set_title(name, loc="left", fontsize=9)
    axes[0].set_title("SIMULATED data: first week per station", fontsize=10)
    plt.tight_layout(); plt.savefig(RESULTS / "sample_week.png", dpi=150)

    for k, v in data.items():
        print(k, v["X_train"].shape, v["X_test"].shape)