from typing import Dict
import yaml

def load_config(path: str) -> Dict:
    with open(path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)


def save_config(cfg: Dict, path: str):
    with open(path, 'w') as f:
        yaml.dump(cfg, f)