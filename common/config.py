from pathlib import Path
import yaml

INT_KEYS = {"seed", "batch_size", "epochs", "patience", "num_workers", "n_samples", "n_features", "n_classes", "hidden_dim"}
FLOAT_KEYS = {"lr", "weight_decay"}


def load_config(path="config.yaml", **overrides) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    cfg.update(overrides)
    for k in INT_KEYS & cfg.keys():
        cfg[k] = int(cfg[k])
    for k in FLOAT_KEYS & cfg.keys():
        cfg[k] = float(cfg[k])
    return cfg


def with_overrides(cfg: dict, **overrides) -> dict:
    new = dict(cfg)
    new.update(overrides)
    for k in INT_KEYS & new.keys():
        new[k] = int(new[k])
    for k in FLOAT_KEYS & new.keys():
        new[k] = float(new[k])
    return new