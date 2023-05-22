import argparse
from pathlib import Path

from src.train import run
from src.utils import load_config, deep_update

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to config file. CLI arguments will overwrite config settings.",
    )

    args = parser.parse_args()

    # first load default config
    cfg = load_config(Path("cfgs/default.yml"))

    # then load config if specified
    # NOTE this doesn't work for lists in YAML, so use with care
    if args.config is not None:
        cfg_update = load_config(args.config)
        cfg = deep_update(cfg, cfg_update)

    run(**cfg)
