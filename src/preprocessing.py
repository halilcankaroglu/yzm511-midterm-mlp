import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   OneHotEncoder, FunctionTransformer)

NUM = ["age", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx",
       "cons.conf.idx", "euribor3m", "nr.employed"]
CAT = ["job", "marital", "education", "default", "housing", "loan", "contact",
       "month", "day_of_week", "poutcome"]
SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


def pdays_rule(X):
    """Sabit kural (veriden öğrenmez): 999 -> bayrak + NaN."""
    X = X.copy()
    X["pdays_kayitli"] = (X["pdays"] != 999).astype(int)
    X["pdays"] = X["pdays"].where(X["pdays"] != 999)
    return X


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Eşikleri fit edildiği kümeden öğrenir, diğerlerine yalnızca uygular."""
    def __init__(self, low=1, high=99):
        self.low, self.high = low, high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.percentile(X, self.low, axis=0)
        self.hi_ = np.percentile(X, self.high, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def build_preprocessor(scaler="standard", clipping=False):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if clipping:
        num_steps.append(("clip", PercentileClipper()))
    num_steps.append(("scale", SCALERS[scaler]()))
    ct = ColumnTransformer([
        ("num", Pipeline(num_steps), NUM),
        ("flag", "passthrough", ["pdays_kayitli"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    return Pipeline([("pdays", FunctionTransformer(pdays_rule)), ("ct", ct)])
