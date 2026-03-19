# Auto Recalibration Engine
# Intercept shift, Platt scaling, isotonic regression
print("Recalibration Module")

import numpy as np

def intercept_shift(df):
    expected_pd = df["pd"].mean()
    actual_pd = df["target"].mean()

    adjustment = np.log(actual_pd / (1 - actual_pd)) - np.log(expected_pd / (1 - expected_pd))

    df["log_odds"] = np.log(df["pd"] / (1 - df["pd"]))
    df["log_odds_adj"] = df["log_odds"] + adjustment

    df["pd_calibrated"] = 1 / (1 + np.exp(-df["log_odds_adj"]))

    return df