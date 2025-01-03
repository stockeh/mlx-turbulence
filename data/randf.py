from typing import Union

import matplotlib.pyplot as plt
import mlx.core as mx


class GaussianRF:
    """Gaussian Random Fields."""

    def __init__(
        self,
        dim: int,
        size: int,
        length: float = 1.0,
        alpha: float = 2.0,
        tau: float = 3.0,
        sigma: Union[float, None] = None,
        boundary: str = "periodic",
        constant_eig: bool = False,
    ) -> None:
        assert size & (size - 1) == 0, "size must be power of 2"
        self.dim = dim
        self.size = size
        self.length = length
        self.alpha = alpha
        self.tau = tau
        self.sigma = sigma or tau ** (0.5 * (2 * alpha - dim))
        self.constant_eig = constant_eig

        self.k_max = size // 2
        self.const = (4 * (mx.pi**2)) / (length**2)

        self.k = mx.concat(
            (
                mx.arange(0, self.k_max),
                mx.arange(-self.k_max, 0),
            ),
            axis=0,
        )

        self.sqrt_eig = self._compute_sqrt_eig()
        self.size = (size,) * dim

    def _compute_sqrt_eig(self) -> mx.array:
        mesh = mx.meshgrid(*(self.k,) * self.dim, indexing="ij")
        k_sq = sum(k_i**2 for k_i in mesh)
        sqrt_eig = (
            (self.size**self.dim)
            * mx.sqrt(2)
            * self.sigma
            * ((self.const * k_sq + self.tau**2) ** (-self.alpha / 2.0))
        )
        if self.constant_eig:
            sqrt_eig[(0,) * self.dim] = (
                (self.size**self.dim) * self.sigma * (self.tau ** (-self.alpha))
            )
        return sqrt_eig

    def sample(self, N: int) -> mx.array:
        a = (
            mx.random.normal((N, *self.size)) + 1j * mx.random.normal((N, *self.size))
        ) * self.sqrt_eig
        return mx.fft.irfftn(a, s=self.size, axes=(-2, -1))


if __name__ == "__main__":
    resolution = 256
    n_samples = 4

    grf = GaussianRF(
        dim=2, size=resolution, alpha=3 / 2, tau=mx.sqrt(196), sigma=mx.sqrt(14)
    )
    w0 = grf.sample(n_samples)

    #! plotting
    fig, axs = plt.subplots(1, n_samples, figsize=(2 * n_samples, 2))
    axs = axs.tolist() if n_samples > 1 else [axs]
    for i, ax in enumerate(axs):
        ax.imshow(w0[i], cmap="twilight")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    fig.savefig("../media/random_field_mlx.png", dpi=300, bbox_inches="tight")
