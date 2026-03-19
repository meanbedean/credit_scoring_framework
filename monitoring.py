# Monitoring Engine
# KS, PSI, lender metrics
print("Monitoring Module")

import numpy as np
import pandas as pd

def calculate_ks(df, score_col, target_col):
    df = df.sort_values(score_col)

    df["cum_bad"] = df[target_col].cumsum() / df[target_col].sum()
    df["cum_good"] = ((1 - df[target_col]).cumsum()) / (1 - df[target_col]).sum()

    ks = np.max(np.abs(df["cum_bad"] - df["cum_good"]))
    return ks


def calculate_psi(expected, actual, bins=10):
    breakpoints = np.linspace(0, 1, bins + 1)

    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi = np.sum((actual_percents - expected_percents) *
                 np.log((actual_percents + 1e-6) / (expected_percents + 1e-6)))

    return psi


def lender_metrics(df):
    results = []

    for lender in df["lender_id"].unique():
        temp = df[df["lender_id"] == lender]

        ks = calculate_ks(temp, "score", "target")
        bad_rate = temp["target"].mean()

        results.append({
            "lender_id": lender,
            "ks": ks,
            "bad_rate": bad_rate,
            "volume": len(temp)
        })

    return pd.DataFrame(results)