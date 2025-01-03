import os

import mlx.core as mx
from scipy.io import loadmat


def load_data(file="NavierStokes_V1e-3_N8_T50.mat"):
    """
    Load the Navier-Stokes dataset from a .mat file.

    Parameters
    ----------
    file : str
        The name of the .mat file in .data

    Returns
    -------
    a : mx.ndarray
        The initial vorticity field.
    u : mx.ndarray
        The Navier-Stokes simulation results.
    sol_t : mx.ndarray
        The time steps corresponding to the simulation results.
    """
    base_path = os.path.dirname(__file__)
    full_path = os.path.join(base_path, ".data", file)
    print("Loading: ", full_path)
    data = loadmat(full_path)
    a = mx.array(data["a"])
    u = mx.array(data["u"])
    sol_t = mx.array(data["t"].flatten())
    return a, u, sol_t


def create_sequences(O, Z=1, V=1):
    """
    Create sequences (X, T) from a 4D array O of shape (B, N, M, D).

    Parameters
    ----------
    O : np.ndarray
        A 4D array of shape (B, N, M, D).
    Z : int
        Number of past timesteps to include in each sample X.
    V : int
        Number of future timesteps to include in each target T.

    Returns
    -------
    X : np.ndarray
        4D array of shape (K, N, M, Z).
    T : np.ndarray
        4D array of shape (K, N, M, V).

    Notes
    -----
    - K = B * (D - (Z + V) + 1), which is the total number of
      (Z, V) windows along the time dimension for all batches.
    - For each batch b and each valid time index t (where Z <= t <= D - V),
      X includes timesteps [t-Z, ..., t-1] and T includes timesteps [t, ..., t+V-1].
    """

    B, N, M, D = O.shape

    assert Z > 0 and V > 0, "Z and V must be positive."
    assert Z + V <= D, f"Z + V cannot exceed the time dimension D={D}."

    n_windows = D - (Z + V) + 1

    K = B * n_windows

    X = mx.zeros((K, N, M, Z), dtype=O.dtype)
    T = mx.zeros((K, N, M, V), dtype=O.dtype)

    k = 0
    for b in range(B):
        for i in range(n_windows):
            t = i + Z
            X[k] = O[b, ..., t - Z : t]
            T[k] = O[b, ..., t : t + V]
            k += 1

    return X, T


def get_partitions(train_frac=0.8, shuffle=True):
    assert 0 < train_frac <= 1, "train_frac must be in (0, 1]."
    a, u, _ = load_data()

    O = mx.concat([mx.expand_dims(a, axis=-1), u], axis=-1)
    Z, V = 1, 1

    if train_frac == 1:
        return create_sequences(O, Z, V)

    n = a.shape[0]
    inds = mx.random.permutation(n) if shuffle else mx.arange(n)
    n_train = int(train_frac * n)

    Xtrain, Ttrain = create_sequences(O[inds[:n_train]], Z, V)
    Xtest, Ttest = create_sequences(O[inds[n_train:]], Z, V)

    return Xtrain, Ttrain, Xtest, Ttest


class Dataset:
    def __init__(self, x, t=None, bs=-1, shuffle=False):
        self.x = x
        self.t = t if t is not None else x
        self.bs = bs if bs > 0 else len(x)
        self.shuffle = shuffle
        self.c = 0
        self.n = len(x)

    @property
    def n_batches(self):
        return self.n // self.bs

    @property
    def n_samples(self):
        return self.n

    def __iter__(self):
        self.c = 0
        if self.shuffle:
            indices = mx.random.permutation(self.n)
            self.x = self.x[indices]
            self.t = self.t[indices]
        return self

    def __next__(self):
        if self.c >= self.n:
            raise StopIteration

        end_idx = min(self.c + self.bs, self.n)
        x = self.x[self.c : end_idx]
        t = self.t[self.c : end_idx]
        self.c = end_idx

        return x, t


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    a, u, sol_t = load_data()
    X, T = create_sequences(
        mx.concat([mx.expand_dims(a, axis=-1), u], axis=-1), Z=1, V=1
    )
    print(f"{X.shape=}, {T.shape=}")

    train = Dataset(X, T, bs=32, shuffle=True)
    for x, t in train:
        print(f"{x.shape=}, {t.shape=}")
        break

    #! plotting
    fig, axes = plt.subplots(2, 2, figsize=(5, 5))

    for i in [0, 1]:
        axes[0, i].imshow(X[i, ..., 0], cmap="twilight")
        axes[1, i].imshow(T[i, ..., 0], cmap="twilight")
    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()
    plt.show()
