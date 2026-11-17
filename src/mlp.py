import time, copy, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.preprocessing import build_preprocessor

ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, n_in, hidden, activation="relu", dropout=0.0):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), ACTS[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))          # tek logit, sigmoid yok
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_optimizer(model, cfg, momentum):
    p, lr, wd = model.parameters(), cfg["lr"], cfg["weight_decay"]
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(p, lr=lr, momentum=momentum, weight_decay=wd)
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(p, lr=lr, weight_decay=wd)
    return torch.optim.AdamW(p, lr=lr, weight_decay=wd)


def fit_mlp(cfg, X, y, tr, es_seed, init_seed):
    t0 = time.time()
    # 1) Eğitim kümesini fit (%85) ve early-stop monitörü (%15) olarak ayır
    X_fit, X_es, y_fit, y_es = train_test_split(
        X, y, test_size=tr["es_fraction"], stratify=y, random_state=es_seed)

    # 2) Ön işlemeyi yalnızca fit kısmında öğren
    pre = build_preprocessor(cfg["scaler"], cfg["clipping"])
    A = torch.tensor(pre.fit_transform(X_fit), dtype=torch.float32)
    B = torch.tensor(pre.transform(X_es), dtype=torch.float32)
    ya = torch.tensor(y_fit, dtype=torch.float32)
    yb = torch.tensor(y_es, dtype=torch.float32)

    # 3) Model, optimizer, scheduler
    set_seed(init_seed)
    model = MLP(A.shape[1], cfg["hidden_layers"], cfg["activation"], cfg["dropout"])
    opt = make_optimizer(model, cfg, tr["sgd_momentum"])
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=tr["sched_factor"],
                patience=tr["sched_patience"], min_lr=tr["min_lr"])
             if cfg["scheduler"] else None)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(A, ya), batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(init_seed))

    # 4) Eğitim döngüsü + early stopping
    best, best_state, best_epoch, wait, hist = np.inf, None, 0, 0, []
    for epoch in range(1, tr["max_epochs"] + 1):
        model.train(); total = 0.0
        for xb, yb_ in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb_)
            loss.backward(); opt.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(B), yb).item()
        hist.append(dict(epoch=epoch, train_loss=total / len(A), val_loss=val,
                         lr=opt.param_groups[0]["lr"]))
        if sched:
            sched.step(val)
        if val < best - 1e-6:
            best, best_epoch, wait = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= tr["es_patience"]:
                break
    model.load_state_dict(best_state)              # en iyi ağırlıklara dön
    return dict(model=model, pre=pre, history=pd.DataFrame(hist),
                best_epoch=best_epoch, epochs_run=epoch,
                seconds=time.time() - t0, n_params=n_params(model))


def predict_proba(res, X):
    res["model"].eval()
    with torch.no_grad():
        Z = torch.tensor(res["pre"].transform(X), dtype=torch.float32)
        return torch.sigmoid(res["model"](Z)).numpy()
