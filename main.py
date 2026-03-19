import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from scorecard_builder import ScorecardBuilder
from monitoring import calculate_ks

df = pd.read_csv("data/sample_data.csv")

builder = ScorecardBuilder(target="default_flag")

model, features = builder.train(df)

woe_df = builder.transform_woe(df, features)

df["score"] = builder.score(woe_df[features])

ks = calculate_ks(df, "score", "default_flag")

print("KS:", ks)