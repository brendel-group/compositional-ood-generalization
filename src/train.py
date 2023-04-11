import datetime
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Metric, R2Score

import wandb
from data import BatchDataLoader, Dataset, InfiniteDataset, get_dataloaders
from models import (
    CompositionalFunction,
    InvertibleMLP,
    LinearComposition,
    ParallelSlots,
)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev = torch.device(dev)


def _get_metric_dict(metric_names: List[str], d_out: int) -> Dict[str, callable]:
    metrics = {}
    for name in metric_names:
        if name == "R2Score":
            metrics[name] = R2Score(
                num_outputs=d_out, multioutput="uniform_average"
            ).to(dev)
        else:
            raise ValueError(f"Unknown metric {name}.")
    return metrics


def _train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module = nn.MSELoss(),
):
    model.train()
    loss_accum = 0
    compute_efficiency_accum = 0

    time_start = time.time()

    # TODO consider using tqdm
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
            x = x.to(dev)
            z = z.to(dev)

            x_hat = model(z)

            for metric_name, metric in metrics.items():
                score_name = f"{metric_name}_{loader_name}"
                if score_name in scores:
                    scores[score_name] += metric(x, x_hat).item()
                else:
                    scores[score_name] = 0

    scores = {name: (val / (batch + 1)) for name, val in scores.items()}
    return scores


def run(**cfg):
    now_str = f"{datetime.datetime.now():%Y%m%d-%H%M%S}"
    save_dir = Path(cfg["save_dir"]) / (cfg["save_name"] + "_" + now_str)
    save_dir.mkdir(parents=True, exist_ok=False)

    wandb.init(project="COOD", config=cfg)

    # TODO only set this if input/output has constant size, otherwise graph is optimized
    #   each time
    # torch.backends.cudnn.benchmark = True
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    D, M = cfg["data"]["D"], cfg["data"]["M"]
    phi = ParallelSlots(
        [InvertibleMLP(d_in, d_out, d_hidden=10) for d_in, d_out in zip(D, M)]
    )
    C = LinearComposition()
    f = CompositionalFunction(C, phi)

    # TODO check whether we need more workers or better background prefetching
    
    train_ldr, eval_ldrs = get_dataloaders(f, cfg["train"], cfg["eval"])

    phi_hat = ParallelSlots(
        [InvertibleMLP(d_in, d_out, d_hidden=10) for d_in, d_out in zip(D, M)]
    )
    f_hat = CompositionalFunction(C, phi_hat)
    f_hat.to(dev)

    if cfg["wandb"]["watch"]:
        wandb.watch(f_hat, log="all", log_freq=cfg["wandb"]["watch_freq"])

    # TODO consider setting up multi-GPU training

    optimizer = getattr(torch.optim, cfg["train"]["optimizer"])(
        f_hat.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        **cfg["train"]["optimizer_kwargs"],
    )
    scheduler = getattr(torch.optim.lr_scheduler, cfg["train"]["scheduler"])(
        optimizer, **cfg["train"]["scheduler_kwargs"]
    )

    best_scores = {score: float("infg") for score in cfg["eval"]["save_scores"]}

    # TODO consider using tqdm
    for epoch in range(cfg["train"]["epochs"]):
        loss, compute_efficiency = _train_epoch(f_hat, train_ldr, optimizer)
        wandb.log({"loss": loss, "compute_efficiency": compute_efficiency})

        if epoch % cfg["eval"]["freq"] == cfg["eval"]["freq"] - 1:
            scores = evaluate(
                f_hat, eval_ldrs, _get_metric_dict(cfg["eval"]["metrics"], f_hat.d_out)
            )
            wandb.log(scores)

            for name, val in best_scores:
                if scores[name] < val:
                    scores[name] = val
                    torch.save(f_hat.state_dict(), save_dir / "best_{name}.pt")

        scheduler.step()

    torch.save(f_hat.state_dict(), save_dir / "latest.pt")

    wandb.finish()
