import torch
import torch.nn as nn
from tqdm.auto import tqdm
from common.checkpoint import early_stop_step, should_stop


def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs: int, es_state: dict, log_fn=print) -> tuple:
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for ep in tqdm(range(1, int(epochs) + 1), desc="epoch"):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        for key, val in zip(history, (tl, ta, vl, va)):
            history[key].append(val)
        log_fn(f"ep {ep} | train {tl:.4f}/{ta:.4f} | val {vl:.4f}/{va:.4f}")
        es_state = early_stop_step(es_state, vl, model, ep)
        if should_stop(es_state):
            log_fn(f"early stop at ep {ep}, best ep {es_state['best_epoch']}")
            break
    return model, history, es_state