from typing import Dict
import math
import torch
import torch.nn as nn

def init_min_cond(model: nn.Module, n_trials: int=7500) -> torch.Tensor:
    """Initialize model with random parameters with minimal condition number among `n_trials` trials.
    """
    if isinstance(model, nn.Linear):
        w = model.weight.data
        k = 1 / w.size(0)

        w = nn.functional.normalize(w, p=2)
        cond = torch.linalg.cond(w)

        for _ in range(n_trials):
            _w = 2 * math.sqrt(k) * torch.rand(w.size()) - math.sqrt(k)
            _w = nn.functional.normalize(_w, p=2)
            _cond = torch.linalg.cond(_w)

            if _cond < cond:
                w = _w
                cond = _cond
        
        model.weight.data = w


class Generator(nn.Module):
    """Generator mapping from `k`×`l`-dimensional latents to a `k`×`m` dimensional output.
    """
    def __init__(self, k: int, l: int, m: int, D: int=50, nonlin=nn.LeakyReLU(.2)):
        super().__init__()
        self.gis = nn.ModuleList([self.build_gi(l, m, D, nonlin) for _ in range(k)])
    
    def build_gi(self, l, m, D, nonlin):
        """Build sub generator g_i mapping from `l` to `m`."""
        g = nn.Sequential(
            nn.Linear(l, D),
            nonlin,
            nn.Linear(D, m)
        )
        g.apply(init_min_cond)
        return g
    
    def merge_gis(self):
        raise NotImplementedError
        # TODO to speed up computation, the parallel MLPs could be merged into a single one
        # like this:
        weight0 = torch.zeros(k*D, k*l)
        bias0 = torch.zeros(k*D)
        weight1 = torch.zeros(k*m, k*D)
        bias1 = torch.zeros(k*m)
        for i in range(k):
            weight0[D*i:D*(i+1), l*i:l*(i+1)] = generators[i][0].weight.data
            bias0[D*i:D*(i+1)] = generators[i][0].bias.data
            weight1[m*i:m*(i+1), D*i:D*(i+1)] = generators[i][2].weight.data
            bias1[m*i:m*(i+1)] = generators[i][2].bias.data
        gen[0].weight.data = weight0
        gen[0].bias.data = bias0
        gen[2].weight.data = weight1
        gen[2].bias.data = bias1

    def forward(self, z):
        # Mapping from [N, K, L] to [N, K, M]
        xs = [self.gis[i](z[:, i, :]) for i in range(z.shape[1])]
        return torch.stack(xs, dim=1)


def sample_latents(n:int, k: int, l: int, sample_mode: str='random', correlation: float=0, delta: float=0) -> tuple[torch.Tensor, torch.Tensor]:
    if sample_mode == 'random':
        # sample randomly in complete latent space
        z = torch.rand(n, k, l)
    
    elif sample_mode == 'diagonal':
        # sample on diagonal with random offset △z_i
        _n = 10*n
        z = torch.Tensor(0, k, l)
        while z.shape[0] < n:
            # sample randomly on diagonal
            _z = torch.repeat_interleave(torch.rand(n, l), k, dim=0).reshape(n, k, l)
            # apply random offset
            _z += torch.rand(n, k, l) * 2 * delta - delta
            # reject samples outside the domain
            mask = ~((_z - 0.5).abs() > 0.5).flatten(1).any(1)
            idx = mask.nonzero().squeeze(1)
            z = torch.cat([z, _z[idx]])
        z = z[:n]
    
    elif sample_mode == 'off_diagonal':
        # sample the opposite of the diagonal case
        _n = 10*n
        z = torch.Tensor(0, k, l)
        while z.shape[0] < n:
            # sample randomly in whole space
            _z = torch.rand(_n, k, l)  
            # compute distances between blocks
            triuidx = torch.triu_indices(k, k)  
            mutual_distances = (_z.unsqueeze(1).repeat(1, _z.shape[1], 1, 1) - _z.unsqueeze(1).repeat(1, _z.shape[1], 1, 1).transpose(1, 2))[:, triuidx[0], triuidx[1], :]
            # reject samples on diagonal
            mask = (mutual_distances.abs() > 2*delta).any(dim=1).any(dim=1)
            idx = mask.nonzero().squeeze(1)
            z = torch.cat([z, _z[idx]])
        z = z[:n]
    
    elif sample_mode == 'pure_off_diagonal':
        # sample points where no component lies within a `k`-cube with side length 2*`delta` from the diagonal
        # this is not the exact opposite of the diagonal case, where not all component must lie on the diagonal 
        _n = 10*n
        z = torch.Tensor(0, k, l)
        while z.shape[0] < n:
            # sample randomly in whole space
            _z = torch.rand(_n, k, l)  
            # compute distances between blocks
            triuidx = torch.triu_indices(k, k)  
            mutual_distances = (_z.unsqueeze(1).repeat(1, _z.shape[1], 1, 1) - _z.unsqueeze(1).repeat(1, _z.shape[1], 1, 1).transpose(1, 2))[:, triuidx[0], triuidx[1], :]
            # reject samples on diagonal
            mask = (mutual_distances.abs() > 2*delta).any(dim=1).all(dim=1)
            idx = mask.nonzero().squeeze(1)
            z = torch.cat([z, _z[idx]])
        z = z[:n]
    
    elif sample_mode == 'orthogonal':
        _z = torch.rand(n, l)
        mask = torch.stack([torch.arange(n), torch.randint(k, (n, 1)).squeeze(dim=1)], dim=1).long()
        z = torch.zeros(n, k, l)
        z[mask.chunk(chunks=2, dim=1)] = _z.unsqueeze(1)
    
    elif sample_mode == 'mix':
        n_diag = int(n * correlation,)
        _z_diag = torch.repeat_interleave(torch.rand(n_diag, l), k, dim=0).reshape(n_diag, k, l)
        _z_rand = torch.rand(n - n_diag, k, l)
        z = torch.cat([_z_diag, _z_rand])[torch.randperm(n)]
    
    else:
        raise Exception(f'no sampling mode {sample_mode}')
    
    return z


class Dataset(torch.utils.data.TensorDataset):
    def __init__(self, n: int, k: int, l: int, m: int, generator: nn.Module, sample_kwargs: Dict=None):
        z = sample_latents(n, k, l, **sample_kwargs)
        x = generator(z)
        super().__init__(x, z)


class GenDataset(torch.utils.data.IterableDataset):
    """Dataset that generates new samples every epoch.
    """
    def __init__(self, n: int, k: int, l: int, m: int, generator: nn.Module, sample_kwargs: Dict=None):
        super().__init__()
        self.n, self.k, self.l, self.sample_kwargs = n, k, l, sample_kwargs
        self.g = generator

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None: raise NotImplementedError

        self.reset()
        return iter(zip(self.x, self.z))
    
    def reset(self):
        self.z = sample_latents(self.n, self.k, self.l, **self.sample_kwargs)
        self.x = self.g(self.z)


# class OODDataset(Dataset):
#     def __init__(self, n: int, k: int, l: int, m: int, sample_kwargs: Dict=None, generator_kwargs: Dict=None):
#         assert sample_kwargs is not None and 'sample_mode' in sample_kwargs, 'sample mode needs to be given'
#         if sample_kwargs['sample_mode'] == 'diagonal':
#             sample_kwargs['sample_mode'] = 'off_diagonal'
#         else:
#             raise NotImplementedError
#         super().__init__(n, k, l, m, sample_kwargs, generator_kwargs)


class BatchDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int) -> None:
        super().__init__(dataset, batch_size=batch_size, shuffle=None if isinstance(dataset, GenDataset) else True)


class NBatchDataLoader(torch.utils.data.DataLoader):
    """BatchDataLoader that loads a fixed number of batches, oversampling if neccessary.
    """
    def __init__(self, dataset: torch.utils.data.Dataset, batch_size: int, n_batches: int) -> None:
        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset, num_samples=batch_size * n_batches),
            batch_size,
            False
        )
        super().__init__(dataset, batch_sampler=batch_sampler)


def build_dataloaders(
    k: int,
    l: int,
    m: int,
    n: int,
    batch_size: int=64,
    generative: bool=False,
    generator_kwargs: Dict=None,
    sample_kwargs: Dict=None,
    **kwargs):
    generator = Generator(k, l, m, **generator_kwargs)

    dataset = GenDataset if generative else Dataset

    id_ds = dataset(n, k, l, m, generator, sample_kwargs)
    id_ldr = torch.utils.data.DataLoader(id_ds, batch_size=batch_size)

    if sample_kwargs['sample_mode'] == 'diagonal':
        sample_kwargs['sample_mode'] = 'off_diagonal'
    else:
        raise NotImplementedError
    
    ood_ds = dataset(n, k, l, m, generator, sample_kwargs)
    ood_ldr = torch.utils.data.DataLoader(ood_ds, batch_size=batch_size)

    return id_ldr, ood_ldr