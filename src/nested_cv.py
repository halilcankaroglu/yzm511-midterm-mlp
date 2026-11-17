import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from src.mlp import fit_mlp, predict_proba


def inner_cv(cfg, X, y, tr, seeds, tag):
    skf = StratifiedKFold(tr["inner_splits"], shuffle=True, random_state=seeds["inner_cv"])
    aps, eps, secs, logs = [], [], [], []
    for j, (a, b) in enumerate(skf.split(X, y)):
        res = fit_mlp(cfg, X.iloc[a], y[a], tr, seeds["early_stop"], seeds["init"][0])
        aps.append(average_precision_score(y[b], predict_proba(res, X.iloc[b])))
        eps.append(res["best_epoch"]); secs.append(res["seconds"])
        logs.append(res["history"].assign(**tag, code=cfg["code"], inner=j,
                                           best_epoch=res["best_epoch"],
                                           epochs_run=res["epochs_run"]))
    row = dict(**tag, code=cfg["code"], mean_ap=np.mean(aps), sd_ap=np.std(aps, ddof=1),
               ap_folds=np.round(aps, 4).tolist(), best_epochs=eps, seconds=sum(secs),
               **{k: v for k, v in cfg.items() if k != "code"})
    return row, pd.concat(logs)


def run_search(cands, X, y, tr, seeds, tag):
    out = [inner_cv(c, X, y, tr, seeds, tag) for c in cands]
    rows = pd.DataFrame([r for r, _ in out])
    best = rows.loc[rows["mean_ap"].idxmax(), "code"]   # eşitlikte listede önce gelen
    return rows, pd.concat([l for _, l in out]), best
