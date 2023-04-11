import argparse
from pathlib import Path

from src.train import run
from src.utils import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default="cfg.yml",
        help="Path to config file. CLI arguments will overwrite config settings.",
    )

    args = parser.parse_args()

    # first load config
    cfg = load_config(args.config)

    # TODO supply defaults from defautl.yml

    # then overwrite with CLI arguments
    cfg.update(**vars(args))

    run(**cfg)
