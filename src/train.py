import datetime
import random
import time
from math import prod
from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import MeanSquaredError, Metric, R2Score

import models
import wandb
from data import get_dataloader, get_dataloaders, sample_latents
from models import *
from vis import visualize_score_heatmaps, visualize_output_v_target
from utils import all_equal

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev = torch.device(dev)


def objectness_loss(input: torch.Tensor) -> torch.Tensor:
    batch_size, w, h, _ = input.shape

    # create pixel coordinates on grid
    x = torch.arange(w).view(1, -1).repeat(h, 1)
    y = torch.arange(h).view(-1, 1).repeat(1, w)
    grid = torch.stack([x, y], dim=0).unsqueeze(0)

    # compute pairwise distances between all pixels
    pairwise_diff = grid.view(1, 2, -1).unsqueeze(-1) - grid.view(1, 2, 1, -1)
    pairwise_dist = torch.sqrt(torch.sum(pairwise_diff**2, dim=1))

    # normalize input
    input = input.abs().sum(-1)
    input = (input - input.min()) / (input.max() - input.min())
    print(input.shape)

    # calculate weighted pairwise distance
    weighted_pairwise_dist = (
        pairwise_dist * input.view(batch_size, -1, 1) * input.view(batch_size, 1, -1)
    )

    return weighted_pairwise_dist.mean()


class ObjectnessLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return objectness_loss(input)


def _get_criterion(name: str, alpha: float = 1, **kwargs) -> nn.Module:
    if name == "L1":
        return nn.L1Loss(**kwargs)
    elif name in ["MSE", "L2"]:
        return nn.MSELoss(**kwargs)
    elif name == "crossentropy":
        return nn.CrossEntropyLoss(**kwargs)
    elif name == "MSE+sparsity":
        return lambda y, y_hat: nn.functional.mse_loss(y, y_hat) + alpha * torch.norm(
            y_hat, p=1
        )
    elif name == "MSE+objectness":
        return lambda y, y_hat: nn.functional.mse_loss(
            y, y_hat
        ) + alpha * objectness_loss(y_hat)
    else:
        raise ValueError(f"Unknown criterion {name}.")


def _get_metrics(
    metric_names: List[str], d_out: Union[int, List[int]]
) -> Dict[str, callable]:
    if isinstance(d_out, list):
        d_out = prod(d_out)

    metrics = {}
    for name in metric_names:
        if name == "R2Score":
            metrics[name] = R2Score(
                num_outputs=d_out, multioutput="uniform_average"
            ).to(dev)
        elif name == "MSE":
            metrics[name] = MeanSquaredError().to(dev)
        else:
            raise ValueError(f"Unknown metric {name}.")
    return metrics


def _train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
):
    model.train()
    loss_accum = 0
    compute_efficiency_accum = 0

    time_start = time.time()

    for batch, (x, z) in enumerate(loader):
        x = x.to(dev)
        z = z.to(dev)

        time_data = time_start - time.time()

        optimizer.zero_grad()

        x_hat = model(z)

        loss = criterion(x, x_hat)
        loss_accum += loss.item()

        loss.backward()
        optimizer.step()

        time_compute = time_start - time.time()
        compute_efficiency_accum += time_compute / (time_compute + time_data)

        time_start = time.time()

    loss_accum /= batch + 1
    compute_efficiency_accum /= batch + 1

    return loss_accum, compute_efficiency_accum


def evaluate(
    model: nn.Module, loaders: torch.utils.data.DataLoader, metrics=Dict[str, Metric]
):
    model.eval()

    scores = {}

    for loader_name, loader in loaders.items():
        for batch, (x, z) in enumerate(loader):
            x = x.to(dev).flatten(1)
            z = z.to(dev)

            with torch.no_grad():
                x_hat = model(z).flatten(1)

            for metric_name, metric in metrics.items():
                score_name = f"{metric_name}_{loader_name}"
                if score_name in scores:
                    scores[score_name] += metric(x, x_hat).item()
                else:
                    scores[score_name] = metric(x, x_hat).item()

    scores = {name: (val / (batch + 1)) for name, val in scores.items()}
    return scores


def visualize_reconstruction(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
) -> plt.Figure:
    model.eval()

    x, z = next(iter(loader))
    z = z.to(dev)

    x_hat, phi_hat = model(z, return_slot_outputs=True)
    phi_hat = [ph.cpu() for ph in phi_hat]

    fig = visualize_output_v_target(x.cpu(), x_hat.cpu(), phi_hat, logging=True)
    return fig


def visualize_mse_on_grid(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    dim_per_slot: List[int],
) -> plt.Figure:
    model.eval()

    all_mse = torch.Tensor().to(dev)
    all_z = torch.Tensor().to(dev)
    for x, z in loader:
        x = x.to(dev).flatten(1)
        z = z.to(dev)
        all_z = torch.cat([all_z, z], dim=0)

        with torch.no_grad():
            x_hat = model(z).flatten(1)
            mse = (x_hat - x).pow(2).mean(dim=-1)
            all_mse = torch.cat([all_mse, mse], dim=0)

    fig = visualize_score_heatmaps(
        all_z.cpu(), all_mse.cpu(), dim_per_slot, "MSE", logging=True
    )
    return fig


def run(**cfg):
    now_str = f"{datetime.datetime.now():%Y%m%d-%H%M%S}"
    save_dir = Path(cfg["save_dir"]) / (cfg["save_name"] + "_" + now_str)
    save_dir.mkdir(parents=True, exist_ok=False)
    cfg["save_dir"] = save_dir

    wandb.init(project="COOD", config=cfg)

    # TODO only set this if input/output has constant size, otherwise graph is optimized
    # each time
    # torch.backends.cudnn.benchmark = True
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    D, M = cfg["data"]["D"], cfg["data"]["M"]

    # data generator
    phi = []
    model = getattr(models, cfg["data"]["phi"])
    for d_in, d_out in zip(D, M):
        phi.append(model(d_in, d_out, **cfg["data"]["phi_kwargs"]))
    phi = ParallelSlots(phi)
    C = getattr(models, cfg["data"]["C"])(**cfg["data"]["C_kwargs"])
    f = CompositionalFunction(C, phi).to(dev)
    f.eval()

    # model
    model = getattr(models, cfg["model"]["phi"])
    if cfg["model"]["reuse_phi"]:
        assert all_equal(D) and all_equal(
            M
        ), f"Can't reuse modules for slots with different inputs/outputs {D, M}."
        phi_hat = [model(D[0], M[0], **cfg["model"]["phi_kwargs"])] * len(D)
    else:
        phi_hat = []
        for d_in, d_out in zip(D, M):
            phi_hat.append(model(d_in, d_out, **cfg["model"]["phi_kwargs"]))
    phi_hat = ParallelSlots(phi_hat)
    f_hat = CompositionalFunction(C, phi_hat).to(dev)

    # TODO check whether we need more workers or better background prefetching

    # data and metrics
    train_ldr = get_dataloader(f, dev, **cfg["train"]["data"])
    criterion = _get_criterion(cfg["train"]["loss"], **cfg["train"]["loss_kwargs"])

    if cfg["eval"]:
        eval_ldrs = get_dataloaders(f, dev, cfg["eval"]["data"])
        eval_metrics = _get_metrics(cfg["eval"]["metrics"], f_hat.d_out)

    if cfg["visualization"]:
        vis_ldrs = get_dataloaders(f, dev, cfg["visualization"]["data"])

    if cfg["wandb"]["watch"]:
        wandb.watch(f_hat, log="all", log_freq=cfg["wandb"]["watch_freq"])

    # TODO consider setting up multi-GPU training

    optimizer = getattr(torch.optim, cfg["train"]["optimizer"])(
        f_hat.parameters(),
        **cfg["train"]["optimizer_kwargs"],
    )
    scheduler = getattr(torch.optim.lr_scheduler, cfg["train"]["scheduler"])(
        optimizer, **cfg["train"]["scheduler_kwargs"]
    )

    # keep track of scores to save model
    best_scores = {score: float("inf") for score in cfg["eval"]["save_scores"]}

    for epoch in range(cfg["train"]["epochs"]):
        log = {}

        # train
        loss, compute_efficiency = _train_epoch(f_hat, train_ldr, optimizer, criterion)
        log.update({"loss": loss, "compute_efficiency": compute_efficiency})

        if epoch > 0:
            # evaluate
            if cfg["eval"] and epoch % cfg["eval"]["freq"] == 0:
                scores = evaluate(f_hat, eval_ldrs, eval_metrics)
                log.update(scores)

                # save best models
                for name, val in best_scores.items():
                    if scores[name] < val:
                        scores[name] = val
                        torch.save(f_hat.state_dict(), save_dir / "best_{name}.pt")

            # visualize
            if cfg["visualization"] and epoch % cfg["visualization"]["freq"] == 0:
                for vis_name, vis_cfg in cfg["visualization"]["data"].items():
                    vis_type = vis_cfg["type"]
                    if vis_type == "heatmap":
                        fig = visualize_mse_on_grid(f_hat, vis_ldrs[vis_name], D)
                    elif vis_type == "reconstruction":
                        fig = visualize_reconstruction(f_hat, vis_ldrs[vis_name])
                    else:
                        raise ValueError(f"Unsupported visualization type {vis_type}.")
                    log.update({vis_name: wandb.Image(fig)})
                    plt.close(fig)

        # call only once to get the correct number of steps in the interface
        wandb.log(log)

        scheduler.step()

    torch.save(f_hat.state_dict(), save_dir / "latest.pt")

    wandb.finish()
