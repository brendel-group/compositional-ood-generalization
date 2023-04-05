from pathlib import Path
from typing import List

import matplotlib as plt
import pandas as pd
import seaborn as sb
import torch

from utils import get_digit_subscript


def visualize_latents(latents: torch.Tensor, dim_per_slot: List[int], out: Path = None):
    cols = [
        f"z{get_digit_subscript(i+1)}{get_digit_subscript(j+1)}"
        for i, d in enumerate(dim_per_slot)
        for j in range(d)
    ]
    df = pd.DataFrame(latents.numpy(), columns=cols)
    sb.pairplot(df, corner=True)
