import torch
import torch.nn as nn


class MLP(nn.Sequential):
    def __init__(self, k: int, l: int, m: int, D: int=120, nonlin: nn.Module=nn.Tanh()):
        super().__init__(
            nn.Linear(k * m, D),
            nonlin,
            nn.Linear(D, k * l),
            # TODO do I need a second nonlinearity here?
            #  added it for now for compatability with legacy code
            nonlin
        )


# def MLPTanh(*args, **kwargs):
#     return MLP(*args, **kwargs.update(nonlin=nn.Tanh()))


# class MLPTanh(MLP):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs.update(nonlin=nn.Tanh()))

class CompositionalMLP(nn.Module):
    def __init__(self, k: int, l: int, m: int, D: int=120, nonlin: nn.Module=nn.Tanh()):
        super().__init__()
        self.mlps = nn.ModuleList([MLP(k * m, l, round(D / k), nonlin) for _ in range(k)])
    
    def forward(self, x):
        x = x.reshape(x.shape[0], len(self.mlps), -1)
        zs = [mlp(x[:, i, :]) for i, mlp in enumerate(self.mlps)]
        return torch.cat(zs, dim=1)


class Autoencoder(nn.Module):
    def __init__(self, k: int, l: int, m: int, D: int=120, nonlin: nn.Module=nn.Tanh()):
        super().__init__()
        self.f = MLP(k * m, k * l, D, nonlin)
        self.g = MLP(k * l, k * m, D, nonlin)
    
    def forward(self, x):
        z = self.f(x)
        return self.g(z), z


def factory(name, *args, **kwargs):
    return globals()[name](*args, **kwargs)