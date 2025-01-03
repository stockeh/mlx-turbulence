import mlx.core as mx
from randf import GaussianRF
from tqdm import tqdm

CONFIG = {
    "grid_size": 64,  # Grid resolution
    "num_samples": 32,  # Total number of samples
    "batch_size": 32,  # Number of samples per batch
    "time_horizon": 5,  # Total simulation time
    "delta_t": 1e-4,  # Time step for integration
    "record_steps": 200,  # Number of snapshots to record
    "grf_params": {  # Parameters for Gaussian Random Field
        "dim": 2,
        "alpha": 3 / 2,
        "tau": mx.sqrt(196),
        "sigma": mx.sqrt(14),
    },
    "viscosity": 1e-4,  # Viscosity parameter
}


@mx.compile
def fun(k_y, k_x, lap, dealias, w_h, f_h, delta_t, visc):
    """
    Update vorticity in Fourier space using the Crank-Nicholson scheme.
    """
    psi_h = w_h / lap  # Solve Poisson equation for stream function
    q = mx.real(mx.fft.ifft2(-2j * mx.pi * k_y * psi_h))  # Velocity field (x)
    v = mx.real(mx.fft.ifft2(2j * mx.pi * k_x * psi_h))  # Velocity field (y)
    w_x = mx.real(mx.fft.ifft2(-2j * mx.pi * k_x * w_h))  # ∂x(vorticity)
    w_y = mx.real(mx.fft.ifft2(-2j * mx.pi * k_y * w_h))  # ∂y(vorticity)
    F_h = mx.fft.fft2(q * w_x + v * w_y) * dealias  # Non-linear term + dealias
    w_h = (  # Crank-Nicholson update
        -delta_t * F_h + delta_t * f_h + (1.0 - 0.5 * delta_t * visc * lap) * w_h
    ) / (1.0 + 0.5 * delta_t * visc * lap)
    return w_h


def navier_stokes_2d(w0, f, config):
    """
    Solve the 2D Navier-Stokes equation in Fourier space.
    """
    N = w0.shape[-1]
    k_max = mx.floor(N / 2.0)  # max frequency
    steps = int(mx.ceil(config["time_horizon"] / config["delta_t"]))

    w_h = mx.fft.fft2(w0)
    f_h = mx.fft.fft2(f)
    if len(f_h.shape) < len(w_h.shape):
        f_h = mx.expand_dims(f_h, axis=0)

    record_time = mx.floor(steps / config["record_steps"])
    k_y = mx.tile(  # y-direction wavenumbers
        mx.concat((mx.arange(0, k_max), mx.arange(-k_max, 0)), 0),
        (N, 1),
    )
    k_x = k_y.swapaxes(0, 1)  # x-direction wavenumbers

    # negative Laplacian operator
    lap = 4 * (mx.pi**2) * (k_x**2 + k_y**2)
    lap[0, 0] = 1.0

    dealias = mx.logical_and(
        mx.abs(k_y) <= (2.0 / 3.0) * k_max, mx.abs(k_x) <= (2.0 / 3.0) * k_max
    ).astype(mx.float32)
    dealias = mx.expand_dims(dealias, axis=0)

    sol = mx.zeros((*w0.shape, config["record_steps"]))
    sol_t = mx.zeros((config["record_steps"]))

    c, t = 0, 0.0
    for j in tqdm(range(steps), leave=False):
        w_h = fun(
            k_y, k_x, lap, dealias, w_h, f_h, config["delta_t"], config["viscosity"]
        )
        t += config["delta_t"]
        if (j + 1) % record_time == 0:
            sol[..., c] = mx.real(mx.fft.ifft2(w_h))
            sol_t[c] = t
            c += 1
        mx.eval(sol, sol_t)

    return sol, sol_t


def run_simulation(config):
    """
    Run the Navier-Stokes simulation for all samples using batches.
    """
    s = config["grid_size"]
    grf = GaussianRF(size=s, **config["grf_params"])

    t = mx.linspace(0, 1, s + 1)[:-1]
    X, Y = mx.meshgrid(t, t, indexing="ij")
    f = 0.1 * (mx.sin(2 * mx.pi * (X + Y)) + mx.cos(2 * mx.pi * (X + Y)))

    num_samples = config["num_samples"]
    batch_size = config["batch_size"]

    a = mx.zeros((num_samples, s, s))
    u = mx.zeros((num_samples, s, s, config["record_steps"]))

    c = 0
    for _ in tqdm(range(num_samples // batch_size), desc="Batch"):
        w0 = grf.sample(batch_size)
        sol, sol_t = navier_stokes_2d(w0, f, config)

        a[c : (c + batch_size), ...] = w0
        u[c : (c + batch_size), ...] = sol
        c += batch_size

    return a, u, sol_t


def save(a, u, sol_t, filename):
    import numpy as np

    print(f"{a.shape=}, {u.shape=}, {sol_t.shape=}")
    np.savez(filename, a=a.cpu().numpy(), u=u.cpu().numpy(), t=sol_t.cpu().numpy())
    print(f"Saved to {filename}")


if __name__ == "__main__":
    import os
    import sys

    # Go up one level from "data/" => project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, BASE_DIR)
    from plotting import plot_nsgrid

    a, u, sol_t = run_simulation(CONFIG)
    # save(a, u, sol_t, f"ns_data_N{N}_s{s}_T{T}.npz")
    plot_nsgrid(a, u, sol_t, N=4, steps=4, shuffle=True, filename="../media/ns2d.png")
