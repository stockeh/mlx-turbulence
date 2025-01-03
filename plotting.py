import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np


def plot_loss(losses, labels, set_ylim=False, filename=None):
    colors = ["k", "b", "r", "g", "m", "c"]
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), constrained_layout=True)

    for i, (loss, label) in enumerate(zip(losses, labels)):
        ax.plot(loss, color=colors[i], label=label, lw=2)

    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(fontsize=12)

    if set_ylim:
        ymax = min([loss[0] for loss in losses])
        ymin, _ = ax.get_ylim()
        ax.set_ylim(ymin, ymax)

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def plot_nsgrid(a, u, sol_t, N=8, steps=5, shuffle=False, filename=None):
    """
    Plot the Navier-Stokes simulation results.

    Parameters
    ----------
    a : mx.ndarray
        The initial vorticity field.
    u : mx.ndarray
        The Navier-Stokes simulation results.
    sol_t : mx.ndarray
        The time steps corresponding to the simulation results.
    N : int, optional
        The number of samples to plot.
    steps : int, optional
        The number of time steps to plot.
    shuffle : bool, optional
        If True, shuffle the samples before plotting.
    filename : str, optional
        If provided, save the figure to this file.

    """
    fig, axes = plt.subplots(
        N,
        steps + 1,
        figsize=(2 * steps, 1.7 * N),
        gridspec_kw={"wspace": 0.05, "hspace": -0.15},
    )
    start = int(0.2 * (u.shape[-1] - 1))
    time_indices = mx.linspace(start, u.shape[-1] - 1, num=steps, dtype=mx.int8)

    inds = mx.random.permutation(u.shape[0])[:N] if shuffle else range(N)
    for i, sample_idx in enumerate(inds):
        ax = axes[i, 0]
        ax.imshow(a[sample_idx], cmap="twilight")
        ax.set_title(r"$\omega_0(x)$" if i == 0 else "", pad=5)
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            f"({chr(97 + i)})",
            color="white",
            weight="bold",
            transform=ax.transAxes,
            fontsize=10,
            ha="left",
            va="top",
        )

    for i, sample_idx in enumerate(inds):
        for plot_idx, time_idx in enumerate(time_indices):
            ax = axes[i, plot_idx + 1]
            ax.imshow(u[sample_idx, ..., time_idx], cmap="twilight")
            ax.set_title(f"$t={int(sol_t[time_idx])}$" if i == 0 else "", pad=5)
            ax.axis("off")

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def radial_plot_power_spectrum(image, normalize=False):
    """
    Compute the radially averaged 2D power spectrum of `image`.

    Parameters
    ----------
    image : 2D np.ndarray
        Input image or 2D field (can be rectangular).
    normalize : bool, optional
        If True, normalize the power spectrum by the area of each annulus.

    Returns
    -------
    k : 1D np.ndarray
        The radial frequencies (in pixel-based units).
    p_k : 1D np.ndarray
        The average power at each radial bin.
    """
    # 1. Compute the 2D FFT of the image
    F = np.fft.fftn(image)
    P = np.abs(F) ** 2

    if normalize:
        P /= image.size

    # 2. Build 2D arrays of frequencies
    ny, nx = image.shape
    kx = np.fft.fftfreq(nx) * nx  # frequencies along x
    ky = np.fft.fftfreq(ny) * ny  # frequencies along y
    kx2D, ky2D = np.meshgrid(kx, ky, indexing="ij")

    # Radial frequency
    kr = np.sqrt(kx2D**2 + ky2D**2).ravel()
    P_flat = P.ravel()

    # 3. Define bins and do the radial binning (up to the smaller Nyquist limit)
    Nmin = min(nx, ny)
    kbins = np.arange(0.5, Nmin // 2 + 1, 1.0)
    k = 0.5 * (kbins[1:] + kbins[:-1])

    p_k = np.zeros(len(kbins) - 1)
    for i in range(len(kbins) - 1):
        mask = (kr >= kbins[i]) & (kr < kbins[i + 1])
        if np.any(mask):
            p_k[i] = np.mean(P_flat[mask])
        else:
            p_k[i] = 0.0

    return k, p_k


def plot_power_spectrum(k, p_k, title=None):
    """
    Plot the radial power spectrum on log–log scales.

    Parameters
    ----------
    k : 1D np.ndarray
        The radial frequencies (in pixel-based units).
    p_k : 1D np.ndarray
        The average power at each radial bin.
    title : str, optional
        Optional title for the plot.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.loglog(k, p_k, color="k", lw=2)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$P(k)$")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    from data.dataset import load_data

    a, u, sol_t = load_data()

    plot_nsgrid(a, u, sol_t, N=4, steps=4, shuffle=True)

    k, p_k = radial_plot_power_spectrum(a[0], normalize=False)
    plot_power_spectrum(k, p_k)
