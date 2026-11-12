from pathlib import Path
import torch


def save_checkpoint(model, path) -> None:
    torch.save(model.state_dict(), Path(path))


def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def init_early_stopping(patience: int, path) -> dict:
    return {"best": float("inf"), "counter": 0, "patience": int(patience), "path": str(path), "best_epoch": 0}


def early_stop_step(state: dict, val_loss: float, model, epoch: int) -> dict:
    new = dict(state)
    if val_loss < new["best"]:
        new["best"] = float(val_loss)
        new["counter"] = 0
        new["best_epoch"] = epoch
        save_checkpoint(model, new["path"])
    else:
        new["counter"] += 1
    return new


def should_stop(state: dict) -> bool:
    return state["counter"] >= state["patience"]


def save_adapter(model, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)


def load_adapter(model, path):
    from peft import PeftModel
    return PeftModel.from_pretrained(model, path)