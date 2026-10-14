import copy
from src.data import load_all
from src.model import DemandMLP, train_local, set_seed
from src.common import ROUNDS, LOCAL_EPOCHS, LR, evaluate_all, summarize


def fedavg(data, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS, lr=LR,
           schedule=None, seed=0, track=False):
    """FedAvg. Only model weights are exchanged; raw station data never leaves its station.

    schedule: optional list of length `rounds`; schedule[r] = names of stations that
              successfully communicate in round r. None = everyone, every round.
    If no station participates in a round, the global model is left unchanged.
    """
    set_seed(seed)
    global_model = DemandMLP()
    history = []

    for r in range(rounds):
        active = list(data) if schedule is None else schedule[r]
        if active:
            states, sizes = [], []
            for name in active:
                local = copy.deepcopy(global_model)           # station downloads global model
                train_local(local, data[name]["X_train"], data[name]["y_train"],
                            epochs=local_epochs, lr=lr)       # trains on its OWN data
                states.append({k: v.detach().cpu().clone()
                               for k, v in local.state_dict().items()})
                sizes.append(len(data[name]["X_train"]))      # only weights + count are sent
            total = sum(sizes)
            avg = {k: sum(s[k] * (n / total) for s, n in zip(states, sizes))
                   for k in states[0]}                        # server: weighted average
            global_model.load_state_dict(avg)
        if track:
            history.append(summarize(evaluate_all(global_model, data)))

    return global_model, history


if __name__ == "__main__":
    from src.centralized import run_centralized
    data, _ = load_all(seed=0)
    model, hist = fedavg(data, track=True)
    fed = evaluate_all(model, data)
    cen = run_centralized(data)
    print(f"{'station':15s} {'FedAvg MAE':>11s} {'Central MAE':>12s}")
    for n in data:
        print(f"{n:15s} {fed[n]['MAE']:11.2f} {cen[n]['MAE']:12.2f}")
    print("FedAvg avg:", {k: round(v, 2) for k, v in summarize(fed).items()})
    print("Central avg:", {k: round(v, 2) for k, v in summarize(cen).items()})
    print("MAE by round (every 5):", [round(h["MAE"], 2) for h in hist[::5]])