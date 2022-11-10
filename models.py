from typing import Union

import torch
import torch.nn as nn


class MLP(nn.Sequential):
    def __init__(
        self, k: int, l: int, m: int, D: int = 120, nonlin: nn.Module = nn.Tanh()
    ):
        super().__init__(
            nn.Linear(k * m, D),
            nonlin,
            nn.Linear(D, k * l),
            # TODO do I need a second nonlinearity here?
            #  added it for now for compatability with legacy code
            nonlin,
        )


# def MLPTanh(*args, **kwargs):
#     return MLP(*args, **kwargs.update(nonlin=nn.Tanh()))


# class MLPTanh(MLP):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs.update(nonlin=nn.Tanh()))


class CompositionalMLP(nn.Module):
    def __init__(
        self, k: int, l: int, m: int, D: int = 120, nonlin: nn.Module = nn.Tanh()
    ):
        super().__init__()
        self.mlps = nn.ModuleList(
            [MLP(1, l, m, round(D / k), nonlin) for _ in range(k)]
        )

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(
                0
            )  # because gradtracking tensors from functorch have no batch-dimension
        x = x.reshape(x.shape[0], len(self.mlps), -1)
        zs = [mlp(x[:, i, :]) for i, mlp in enumerate(self.mlps)]
        return torch.cat(zs, dim=1)


class SinkhornOperator(object):
    """
    With permission from https://github.com/rpatrik96/nl-causal-representations/blob/master/care_nl_ica/models/sinkhorn.py

    From http://arxiv.org/abs/1802.08665
    """

    def __init__(self, num_steps: int):

        if num_steps < 1:
            raise ValueError(f"{num_steps=} should be at least 1")

        self.num_steps = num_steps

    def __call__(self, matrix: torch.Tensor) -> torch.Tensor:
        def _normalize_row(matrix: torch.Tensor) -> torch.Tensor:
            return matrix - torch.logsumexp(matrix, 1, keepdim=True)

        def _normalize_column(matrix: torch.Tensor) -> torch.Tensor:
            return matrix - torch.logsumexp(matrix, 0, keepdim=True)

        S = matrix

        for _ in range(self.num_steps):
            S = _normalize_column(_normalize_row(S))

        return torch.exp(S)


class SinkhornNet(nn.Module):
    """With permission from https://github.com/rpatrik96/nl-causal-representations/blob/master/care_nl_ica/models/sinkhorn.py"""

    def __init__(self, num_dim: int, num_steps: int, temperature: float = 1):
        super().__init__()

        self.num_dim = num_dim
        self.temperature = temperature

        self.sinkhorn_operator = SinkhornOperator(num_steps)
        # self.weight = nn.Parameter(nn.Linear(num_dim, num_dim).weight+0.5*torch.ones(num_dim,num_dim), requires_grad=True)
        self.weight = nn.Parameter(torch.ones(num_dim, num_dim), requires_grad=True)

    @property
    def doubly_stochastic_matrix(self) -> torch.Tensor:
        return self.sinkhorn_operator(self.weight / self.temperature)

    def forward(self, x) -> torch.Tensor:
        if (dim_idx := x.shape.index(self.num_dim)) == 0 or len(x.shape) == 3:
            return self.doubly_stochastic_matrix @ x
        elif dim_idx == 1:
            return x @ self.doubly_stochastic_matrix

    def to(self, device):
        """
        Move the model to the specified device.
        :param device: The device to move the model to.
        """
        super().to(device)
        self.weight = self.weight.to(device)

        return self


def PermutableCompositionalMLP(
    k: int,
    l: int,
    m: int,
    D: int = 120,
    nonlin: nn.Module = nn.Tanh(),
    sinkhorn_steps: int = 20,
    sinkhorn_temp: Union[float, str] = 1e-4,
):
    if isinstance(sinkhorn_temp, str):
        sinkhorn_temp = float(sinkhorn_temp)
    return nn.Sequential(
        SinkhornNet(k * m, sinkhorn_steps, sinkhorn_temp),
        CompositionalMLP(k, l, m, D, nonlin),
    )


class Autoencoder(nn.Module):
    def __init__(
        self, k: int, l: int, m: int, D: int = 120, nonlin: nn.Module = nn.Tanh()
    ):
        super().__init__()
        self.f = MLP(k * m, k * l, D, nonlin)
        self.g = MLP(k * l, k * m, D, nonlin)

    def forward(self, x):
        z = self.f(x)
        return self.g(z), z


def factory(name: str, *args, **kwargs):
    return globals()[name](*args, **kwargs)
