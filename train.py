import argparse
import json
import os
import re

import mlx.core as mx
import mlx.optimizers as optim

from data.dataset import Dataset, get_partitions
from modeling.cfm import str_to_cfm
from modeling.manager import Manager
from modeling.models import UNet
from modeling.odeint import NeuralODE
from plotting import plot_loss
from utils import get_learning_rate, load_model_and_optimizer

parser = argparse.ArgumentParser(description="Train models for arc-x project.")
parser.add_argument("--outdir", type=str, default="results", help="Output directory")
parser.add_argument("--desc", type=str, default="cfm-ns", help="Description")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--epochs", type=int, default=100, help="N epochs for training")


def setup(outdir, desc, c):
    prev_run_dirs = [
        x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))
    ]
    prev_run_ids = [
        int(re.match(r"^\d+", x).group()) for x in prev_run_dirs if re.match(r"^\d+", x)
    ]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join(outdir, f"{cur_run_id:05d}-{desc}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "training_options.json"), "wt") as f:
        json.dump(c, f, indent=2)

    return run_dir


def main(args):
    mx.random.seed(args.seed)

    X, T = get_partitions(train_frac=1)
    print(f"Dataset: {X.shape=}, {T.shape=}")

    model_kwargs = {
        "in_channels": X.shape[-1],
        "out_channels": X.shape[-1],
        "depth": 3,
        "initial_filters": 16,
        "activation_f": "silu",
        "time_varying": True,
    }

    batch_size, max_lr, min_lr = 32, 1e-3, 1e-6
    learning_rate = get_learning_rate(
        max_lr, min_lr, X.shape[0] // batch_size * args.epochs
    )
    optim_kwargs = {"learning_rate": learning_rate, "weight_decay": 1e-4}

    model, optimizer = load_model_and_optimizer(
        UNet, model_kwargs, optim.AdamW, optim_kwargs
    )
    model.summary()

    method = "cfm"
    sigma = 0.1
    solver = "dopri5"

    flow_matcher = str_to_cfm(method, sigma)
    integrator = NeuralODE(model, solver)
    print(flow_matcher, integrator, sep="\n")

    f = Manager(model, optimizer, method, flow_matcher, integrator)

    c = {
        "model_kwargs": model_kwargs,
        "method": method,
        "sigma": sigma,
        "solver": solver,
    }
    run_dir = setup(args.outdir, args.desc, c)

    data = Dataset(X, T, bs=batch_size, shuffle=True)

    f.train(data, args.epochs)
    f.save(run_dir, "f")

    print("Finished Training.")

    plot_loss(
        [f.train_error_trace],
        ["CFM"],
        set_ylim=False,
        filename=os.path.join(run_dir, "train_loss.png"),
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
