import mlx.core as mx
from mlx.optimizers import cosine_decay, join_schedules, linear_schedule


def load_model_and_optimizer(
    model_class, model_kwargs, optimizer_class, optimizer_kwargs
):
    model = model_class(**model_kwargs)
    optimizer = optimizer_class(**optimizer_kwargs)
    return model, optimizer


def rmse(x, t):
    return mx.sqrt(mx.mean((x - t) ** 2))


def get_cosine_schedule(max_lr, min_lr, n_warmup, decay_steps):
    learning_rate = join_schedules(
        [
            linear_schedule(min_lr, max_lr, n_warmup),
            cosine_decay(max_lr, decay_steps, min_lr),
        ],
        [n_warmup],
    )
    return learning_rate


def get_learning_rate(max_lr, min_lr, total_steps, percent_warmup=0.10):
    n_warmup = int(total_steps * percent_warmup)  # % of total steps
    decay_steps = total_steps - n_warmup
    return get_cosine_schedule(max_lr, min_lr, n_warmup, decay_steps)
