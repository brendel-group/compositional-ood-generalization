from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import functorch
from torchmetrics import R2Score

if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
dev = torch.device(dev)

class Regularizer:
    def __init__(self, function: torch.nn.Module, weight: float=1, **kwargs):
        self.function = function
        self.weight = weight
        self.kwargs = kwargs

    def __call__(self, model, x):
        return self.weight * self.function(model, x, **self.kwargs)


def comp_contrast(func: torch.nn.Module, inputs: torch.Tensor, l: int, normalize: bool=False, p: float=2) -> torch.Tensor:
    """Calculate the compositional contrast for a function `func` with respect to `inputs`.

    The output is calculated as the mean over the batch dimension.
    `inputs` needs to be flattened except for the batch dimension and `requires_grad` needs to be set to `True`.
    """
    assert inputs.requires_grad == True, 'To calculate the derivative by `inputs` `requires_grad` needs to be set to `True`.'
    assert p > 0

    jac = functorch.vmap(functorch.jacrev(func))(inputs).transpose(1, 2)  # batch_size × obs_dim × n_slots*slot_dim

    if normalize:
        jac /= jac.square().sum(-1, keepdim=True)

    slot_rows = torch.stack(torch.split(jac, l, dim=-1))  # n_slots × batch_size × obs_dim × slot_dim
    slot_norms = torch.norm(slot_rows, p=p, dim=-1)  # n_slots × batch_size × obs_dim
    cc = (slot_norms.sum(0).square() - slot_norms.square().sum(0)).sum(-1).mean()
    return cc


def sparse_hess(func: torch.nn.Module, inputs: torch.Tensor, p: int=2) -> torch.Tensor:
    hess = functorch.vmap(functorch.hessian(func))(inputs)

    return torch.mean(hess.norm(dim=(1, 2, 3), p=p))


def train_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, optimizer, criterion: nn.Module=nn.MSELoss(), regularizers: List[Regularizer]=None):
    if regularizers is None: regularizers = []
    
    cum_loss = 0

    for batch, data in enumerate(loader, 0):
        x, z = data
        x = x.detach()

        optimizer.zero_grad()

        x.requires_grad = True
        x = x.flatten(1).to(dev)
        z = z.flatten(1).to(dev)

        z_hat = model(x)
        loss = criterion(z_hat, z)
        for regularizer in regularizers:
            loss += regularizer(model, x)
        cum_loss += loss
        loss.backward()
        optimizer.step()
    
    cum_loss /= (batch + 1)

    return cum_loss.to(torch.device('cpu')).item()


def test(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, metric: callable=None, **reg_kwargs):
    cum_score = 0

    for batch, data in enumerate(dataloader, 0):
        x, z = data
        x.requires_grad = True
        x = x.flatten(1).to(dev)
        z = z.flatten(1).to(dev)
        if metric is None:
            out = model(x)
            r2score = R2Score(out.size(1)).to(dev)
            score = r2score(out, z)
        else:
            score = metric(model, x, **reg_kwargs)
        cum_score += score
    
    cum_score /= (batch + 1)
    return cum_score.to(torch.device('cpu')).item()