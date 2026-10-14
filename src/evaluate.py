import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_all, STATIONS
from src.baseline import run_persistence, run_seasonal_naive, run_local_only
from src.centralized import run_centralized
from src.federated import fedavg
from src.dropout import run_dropout
from src.common import SEEDS, ROUNDS, LOCAL_EPOCHS, LR, evaluate_all

RESULTS = Path("results")
RATES = [0.25, 0.5, 0.75]
ORDER = ["persistence", "seasonal_naive", "local_only", "centralized", "fedavg"]


def add(rows, seed, method, res, dropped="none", p=0.0):
    for st, m in res.items():
        rows.append(dict(seed=seed, method=method, dropped=dropped, p=p,
                         station=st, MAE=m["MAE"], RMSE=m["RMSE"]))


def run_all():
    rows = []
    for seed in SEEDS:
        print(f"=== seed {seed} ===", flush=True)
        data, _ = load_all(seed=seed * 10)
        add(rows, seed, "persistence", run_persistence(data))
        add(rows, seed, "seasonal_naive", run_seasonal_naive(data))
        add(rows, seed, "local_only", run_local_only(data, seed))
        add(rows, seed, "centralized", run_centralized(data, seed))
        model, _ = fedavg(data, seed=seed)
        add(rows, seed, "fedavg", evaluate_all(model, data))
        for dropped in STATIONS:
            for p in RATES:
                print(f"  dropout: {dropped} offline {int(p * 100)}%", flush=True)
                add(rows, seed, "fedavg_dropout", run_dropout(data, dropped, p, seed), dropped, p)
    return pd.DataFrame(rows)


def station_avg(df):
    """Average over stations within each run."""
    return df.groupby(["seed", "method", "dropped", "p"], as_index=False)[["MAE", "RMSE"]].mean()


def plot_methods(df):
    sub = station_avg(df[df.method.isin(ORDER)])
    g = sub.groupby("method")[["MAE", "RMSE"]].agg(["mean", "std"]).loc[ORDER]
    g.round(3).to_csv(RESULTS / "summary_methods.csv")
    print("\nAverage over stations (mean/std over seeds):\n", g.round(2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        ax.bar(ORDER, g[metric]["mean"].values, yerr=g[metric]["std"].values, capsize=4)
        ax.set_ylabel(f"{metric} (kWh)")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Next-hour demand error, SIMULATED data (mean ± std over seeds)")
    plt.tight_layout(); plt.savefig(RESULTS / "fig_methods.png", dpi=150); plt.close()


def plot_per_station(df):
    methods = ["local_only", "centralized", "fedavg"]
    t = df[df.method.isin(methods)].groupby(["station", "method"])["MAE"].mean().unstack("method")[methods]
    gap = t[["local_only", "fedavg"]].sub(t["centralized"], axis=0).div(t["centralized"], axis=0) * 100
    print("\nMAE gap vs centralized (%):\n", gap.round(1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    t.plot(kind="bar", ax=axes[0]); axes[0].set_ylabel("MAE (kWh)"); axes[0].set_title("MAE per station")
    gap.plot(kind="bar", ax=axes[1]); axes[1].set_ylabel("MAE gap vs centralized (%)")
    axes[1].axhline(0, color="k", lw=0.8); axes[1].set_title("Difference vs centralized")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15); ax.set_xlabel("")
    fig.suptitle("Per-station comparison (SIMULATED data)")
    plt.tight_layout(); plt.savefig(RESULTS / "fig_per_station.png", dpi=150); plt.close()


def plot_dropout(df):
    base = df[df.method == "fedavg"]
    drop = df[df.method == "fedavg_dropout"]
    rows = []
    for dropped in STATIONS:
        for p in [0.0] + RATES:
            sub = base if p == 0 else drop[(drop.dropped == dropped) & (drop.p == p)]
            avg = sub.groupby("seed")["MAE"].mean()
            own = sub[sub.station == dropped].groupby("seed")["MAE"].mean()
            rows.append(dict(dropped=dropped, p=p, avg_MAE=avg.mean(), avg_MAE_std=avg.std(),
                             own_MAE=own.mean(), own_MAE_std=own.std()))
    t = pd.DataFrame(rows)
    t.round(3).to_csv(RESULTS / "summary_dropout.csv", index=False)
    print("\nDropout results:\n", t.round(2).to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for dropped in STATIONS:
        s = t[t.dropped == dropped]
        axes[0].errorbar(s.p * 100, s.avg_MAE, yerr=s.avg_MAE_std, marker="o", capsize=3, label=dropped)
        axes[1].errorbar(s.p * 100, s.own_MAE, yerr=s.own_MAE_std, marker="o", capsize=3, label=dropped)
    axes[0].set_title("Average MAE over all stations")
    axes[1].set_title("MAE on the disconnected station itself")
    for ax in axes:
        ax.set_xlabel("% of rounds the station is offline"); ax.set_ylabel("MAE (kWh)")
    axes[0].legend(title="Offline station")
    fig.suptitle("FedAvg under communication failures, SIMULATED data (mean ± std over seeds)")
    plt.tight_layout(); plt.savefig(RESULTS / "fig_dropout.png", dpi=150); plt.close()


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    csv = RESULTS / "all_results.csv"
    if "--plots-only" in sys.argv and csv.exists():
        df = pd.read_csv(csv)
    else:
        print(f"Config: ROUNDS={ROUNDS}, LOCAL_EPOCHS={LOCAL_EPOCHS}, LR={LR}, seeds={SEEDS}")
        df = run_all()
        df.to_csv(csv, index=False)
    plot_methods(df)
    plot_per_station(df)
    plot_dropout(df)
    print("\nSaved figures and CSVs in results/")