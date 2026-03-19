# Reject Inference Engine
# Parceling, fuzzy, weighting
print("Reject Inference Module")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class RejectInference:

    def __init__(self, features):
        self.features = features

    def parceling(self, df):
        approved = df[df["approved"] == 1].copy()
        rejected = df[df["approved"] == 0].copy()

        model = LogisticRegression(max_iter=1000)
        model.fit(approved[self.features], approved["target"])

        rejected["pd"] = model.predict_proba(rejected[self.features])[:, 1]

        approved["band"] = pd.qcut(
            model.predict_proba(approved[self.features])[:, 1], 10, labels=False, duplicates="drop"
        )
        rejected["band"] = pd.qcut(rejected["pd"], 10, labels=False, duplicates="drop")

        band_bad_rate = approved.groupby("band")["target"].mean()

        rejected["target"] = rejected["band"].map(band_bad_rate)
        rejected["target"] = (np.random.rand(len(rejected)) < rejected["target"].fillna(approved["target"].mean())).astype(int)

        return pd.concat([approved, rejected])