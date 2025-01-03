import argparse
import json
import os
import sys

import mlx.core as mx
import mlx.optimizers as optim
from tqdm import tqdm

from data.dataset import load_data
from modeling.cfm import str_to_cfm
from modeling.manager import Manager
from modeling.models import UNet
from modeling.odeint import NeuralODE
from utils import load_model_and_optimizer

parser = argparse.ArgumentParser(description="Generate.")
parser.add_argument("--results", type=str, required=True, help="Output directory")
parser.add_argument("--seed", type=int, default=42, help="Random seed")


def main(args):
    mx.random.seed(args.seed)

    with open(os.path.join(args.results, "training_options.json"), "rt") as f:
        c = json.load(f)

    optim_kwargs = {"learning_rate": 1e-4, "weight_decay": 1e-4}
    model, optimizer = load_model_and_optimizer(
        UNet, c["model_kwargs"], optim.AdamW, optim_kwargs
    )
    # model.summary()
    flow_matcher = str_to_cfm(c["method"], c["sigma"])
    integrator = NeuralODE(model, c["solver"])
    f = Manager(model, optimizer, c["method"], flow_matcher, integrator)
    f.load(args.results, "f")

    a, u, _ = load_data()
    # n = a.shape[0]
    # inds = mx.random.permutation(n)
    # n_train = int(0.8 * n)
    # a = a[inds[:n_train]][:3]
    # u = u[inds[:n_train]][:3]

    o = mx.zeros_like(u)  # (N, H, W, T)

    for i in tqdm(range(o.shape[0])):
        x = mx.expand_dims(a[i : i + 1], -1)
        for j in tqdm(range(o.shape[-1]), leave=False):
            o[i : i + 1, ..., j : j + 1] = x = f.sample(x, ts=20)[-1]

    mx.savez(os.path.join(args.results, "generated.npz"), a=a, u=u, o=o)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
