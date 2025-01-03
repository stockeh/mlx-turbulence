import os
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Optimizer, clip_grad_norm
from mlx.utils import tree_flatten, tree_unflatten
from tqdm import tqdm


class Manager:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        evaluator: str,
        flow_matcher=None,
        integrator=None,
    ):
        self.model = model
        self.optimizer = optimizer

        self.flow_matcher = flow_matcher
        self.integrator = integrator

        self.evaluator = evaluator
        self.evaluators = {
            "cfm": self.eval_cfm,
            "mse": self.eval_net,
        }

        # bookkeeping
        self.train_error_trace = []

    def eval_cfm(self, X, T):
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(X, T)
        vt = self.model(t, xt, repeat=False)
        return nn.losses.mse_loss(vt, ut, reduction="mean")

    def eval_net(self, X, T):
        Y = self.model(X)
        return nn.losses.mse_loss(Y, T, reduction="mean")

    def train(self, data, epochs: int, verbose: bool = True):
        assert self.optimizer is not None, "No optimizer provided."
        state = [self.model.state, self.optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(X, T):
            train_step_fn = nn.value_and_grad(
                self.model, self.evaluators[self.evaluator]
            )
            loss, grads = train_step_fn(X, T)
            grads, _ = clip_grad_norm(grads, max_norm=1.0)
            self.optimizer.update(self.model, grads)
            return loss

        epoch_bar = tqdm(
            range(epochs),
            desc="Training",
            unit="epoch",
            disable=not verbose,
        )
        self.model.train()
        for _ in epoch_bar:
            total_loss = 0
            for X, T in tqdm(data, total=data.n_batches, leave=False):
                loss = step(X, T)
                mx.eval(state)
                total_loss += loss.item() * X.shape[0]

            avg_loss = total_loss / data.n_samples
            self.train_error_trace.append(avg_loss)

            postfix = {"loss": f"{avg_loss:.3f}"}
            epoch_bar.set_postfix(postfix)

    def save(self, path, name):
        mx.save_safetensors(
            os.path.join(path, f"{name}-optimizer.safetensors"),
            dict(tree_flatten(self.optimizer.state)),
        )
        self.model.save_weights(os.path.join(path, f"{name}-model.safetensors"))

        with open(os.path.join(path, f"{name}-train_error_trace.txt"), "w") as f:
            for loss in self.train_error_trace:
                f.write(f"{loss}\n")

    def load(self, path, name):
        self.model.load_weights(os.path.join(path, f"{name}-model.safetensors"))
        self.optimizer.state = tree_unflatten(
            list(mx.load(os.path.join(path, f"{name}-optimizer.safetensors")).items())
        )

    def sample(self, X, ts=128):
        self.model.eval()
        return self.integrator.trajectory(X, t_span=mx.linspace(0, 1, ts))
