# Scorecard Builder
# Includes WOE, binning, logistic regression, score scaling
print("Scorecard Builder Module")

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from optbinning import OptimalBinning

class ScorecardBuilder:

    def __init__(self, target):
        self.target = target
        self.model = None
        self.binning_models = {}

    def calculate_iv(self, df, feature):
        col_dtype = "categorical" if df[feature].dtype.name in ['object', 'category'] else "numerical"
        optb = OptimalBinning(name=feature, dtype=col_dtype)
        optb.fit(df[feature], df[self.target])
        return optb.binning_table.build()["IV"].sum()

    def select_features(self, df, threshold=0.02):
        iv_dict = {}
        for col in df.columns:
            if col == self.target:
                continue
            try:
                iv_dict[col] = self.calculate_iv(df, col)
            except:
                continue

        iv_df = pd.DataFrame.from_dict(iv_dict, orient="index", columns=["IV"])
        return iv_df[iv_df["IV"] > threshold].index.tolist()

    def transform_woe(self, df, features):
        woe_df = pd.DataFrame()

        for col in features:
            col_dtype = "categorical" if df[col].dtype.name in ['object', 'category'] else "numerical"
            optb = OptimalBinning(name=col, dtype=col_dtype)
            optb.fit(df[col], df[self.target])
            self.binning_models[col] = optb
            woe_df[col] = optb.transform(df[col], metric="woe")

        woe_df[self.target] = df[self.target]
        return woe_df

    def train(self, df):
        features = self.select_features(df)
        woe_df = self.transform_woe(df, features)

        X = woe_df[features]
        y = woe_df[self.target]

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)

        return self.model, features

    def score(self, X, pdo=50, base_score=600, base_odds=50):
        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(base_odds)

        log_odds = self.model.intercept_[0] + np.dot(X, self.model.coef_[0])
        score = offset + factor * log_odds

        return score