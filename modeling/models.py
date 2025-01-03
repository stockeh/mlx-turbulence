import importlib
from typing import List, Type, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def get_activation(activation_f: str) -> Type:
    package_name = "mlx.nn.layers.activations"
    module = importlib.import_module(package_name)

    activations = [getattr(module, attr) for attr in dir(module)]
    activations = [
        cls
        for cls in activations
        if isinstance(cls, type) and issubclass(cls, nn.Module)
    ]
    names = [cls.__name__.lower() for cls in activations]

    try:
        index = names.index(activation_f.lower())
        return activations[index]
    except ValueError:
        raise NotImplementedError(
            f"get_activation: {activation_f=} is not yet implemented."
        )


class Base(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    @property
    def num_params(self):
        return sum(x.size for k, x in tree_flatten(self.parameters()))

    @property
    def shapes(self):
        return tree_map(lambda x: x.shape, self.parameters())

    def summary(self):
        print(self)
        print(f"Number of parameters: {self.num_params}")

    def __call__(self, x: mx.array) -> mx.array:
        raise NotImplementedError("Subclass must implement this method")


class TMLP(Base):
    def __init__(
        self,
        n_inputs: int,
        n_hiddens_list: Union[List, int],
        n_outputs: int,
        activation_f: str = "selu",
        time_varying=True,
    ):
        super().__init__()

        if isinstance(n_hiddens_list, int):
            n_hiddens_list = [n_hiddens_list]

        if n_hiddens_list == [] or n_hiddens_list == [0]:
            self.n_hidden_layers = 0
        else:
            self.n_hidden_layers = len(n_hiddens_list)

        activation = get_activation(activation_f)
        self.time_varying = time_varying
        self.layers = []
        ni = n_inputs + (1 if time_varying else 0)
        if self.n_hidden_layers > 0:
            for _, n_units in enumerate(n_hiddens_list):
                self.layers.append(nn.Linear(ni, n_units))
                self.layers.append(activation())
                ni = n_units
        self.layers.append(nn.Linear(ni, n_outputs))

    def __call__(self, t, x, repeat=True):
        x = x.reshape(x.shape[0], -1)

        if self.time_varying:
            if repeat:
                t = mx.repeat(t, x.shape[0])
            x = mx.concatenate([x, t[:, None]], axis=-1)

        for l in self.layers:
            x = l(x)
        return x


class UNet(Base):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int,
        initial_filters: int = 32,
        activation_f: str = "relu",
        time_varying=True,
    ):
        super().__init__()

        self.depth = depth
        self.time_varying = time_varying
        activation = get_activation(activation_f)

        # downsampling
        self.down_layers = []
        self.pooling_layers = []
        filters = initial_filters
        for _ in range(depth):
            self.down_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, filters, kernel_size=3, padding=1),
                    nn.BatchNorm(filters),
                    activation(),
                    nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                    nn.BatchNorm(filters),
                    activation(),
                )
            )
            self.pooling_layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = filters
            filters *= 2

        # latent
        self.latent_layer = nn.Sequential(
            nn.Conv2d(filters // 2, filters, kernel_size=3, padding=1),
            nn.BatchNorm(filters),
            activation(),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm(filters),
            activation(),
        )

        # time embedding
        if self.time_varying:
            self.time_embedding = nn.Sequential(
                nn.Linear(1, filters // 2),
                activation(),
                nn.Linear(filters // 2, filters),
            )

        # upsampling
        self.up_layers = []
        self.upsample_layers = []
        filters = filters // 2
        for _ in range(depth):
            self.upsample_layers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest", align_corners=True),
                    nn.Conv2d(filters * 2, filters, kernel_size=3, padding=1),
                )
            )
            self.up_layers.append(
                nn.Sequential(
                    nn.Conv2d(filters * 2, filters, kernel_size=3, padding=1),
                    nn.BatchNorm(filters),
                    activation(),
                    nn.Conv2d(filters, filters, kernel_size=3, padding=1),
                    nn.BatchNorm(filters),
                    activation(),
                )
            )
            filters //= 2

        # output
        self.output = nn.Sequential(
            nn.Conv2d(filters * 2, filters, kernel_size=3, padding=1),
            nn.BatchNorm(filters),
            activation(),
            nn.Conv2d(filters, out_channels, kernel_size=1),
        )

    def __call__(self, t: mx.array, x: mx.array, repeat=True) -> mx.array:
        encodings = []
        for i, layer in enumerate(self.down_layers):
            x = layer(x)
            encodings.append(x)
            x = self.pooling_layers[i](x)

        x = self.latent_layer(x)

        if self.time_varying:
            if repeat:
                t = mx.repeat(t, x.shape[0])
            t = mx.expand_dims(t, axis=-1)
            t = self.time_embedding(t)
            # x = x + t[:, None, None, :]
            t = t[:, None, None, :]
            x = x * mx.sigmoid(t) + x

        for i, layer in enumerate(self.up_layers):
            x = self.upsample_layers[i](x)
            x = mx.concat([x, encodings[-(i + 1)]], axis=-1)
            x = layer(x)

        x = self.output(x)
        return x


if __name__ == "__main__":
    model = UNet(3, 3, 4)
    model.summary()

    x = mx.random.uniform(0, 1, (2, 64, 64, 3))
    t = mx.random.uniform(0, 1, (1,))
    y = model(t, x)
    print(x.shape, y.shape)
