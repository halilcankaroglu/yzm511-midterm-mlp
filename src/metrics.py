from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score,
                             precision_score, recall_score, f1_score,
                             balanced_accuracy_score, confusion_matrix)


def compute_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return dict(
        AP=average_precision_score(y, p), ROC_AUC=roc_auc_score(y, p),
        Accuracy=accuracy_score(y, yhat),
        Precision=precision_score(y, yhat, zero_division=0),
        Sensitivity=recall_score(y, yhat, zero_division=0),
        Specificity=tn / (tn + fp), F1=f1_score(y, yhat, zero_division=0),
        Balanced_Acc=balanced_accuracy_score(y, yhat))
