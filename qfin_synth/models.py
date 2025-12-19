import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from sklearn.ensemble import GradientBoostingClassifier

def train_classifier(df_trades: pd.DataFrame, feature_cols=None, test_size=0.2, seed=42):
    if feature_cols is None:
        feature_cols = ["mom20", "vol20", "ma_ratio", "hold_bars"]

    df = df_trades.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols + ["label"]).copy()
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    clf = GradientBoostingClassifier(random_state=seed)
    clf.fit(X_tr, y_tr)

    p = clf.predict_proba(X_te)[:, 1]
    yhat = (p >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_te, p)),
        "f1": float(f1_score(y_te, yhat)),
        "mcc": float(matthews_corrcoef(y_te, yhat)),
        "feature_cols": list(feature_cols),
    }
    return clf, metrics
