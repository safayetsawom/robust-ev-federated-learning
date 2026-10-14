import numpy as np
from src.data import load_all
from src.federated import fedavg
from src.common import ROUNDS, evaluate_all


def make_schedule(stations, dropped, p, rounds=ROUNDS, seed=0):
    """Each round, with probability p the `dropped` station fails to communicate."""
    rng = np.random.default_rng(seed)
    schedule = []
    for _ in range(rounds):
        offline = rng.random() < p
        schedule.append([s for s in stations if not (offline and s == dropped)])
    return schedule


def run_dropout(data, dropped, p, seed=0):
    schedule = make_schedule(list(data), dropped, p, seed=seed + 1000)
    model, _ = fedavg(data, schedule=schedule, seed=seed)
    return evaluate_all(model, data)


if __name__ == "__main__":
    data, _ = load_all(seed=0)
    for p in [0.0, 0.5]:
        res = run_dropout(data, "C_mall", p, seed=0)
        print(f"C_mall offline in {int(p * 100)}% of rounds:",
              {n: round(m["MAE"], 2) for n, m in res.items()})