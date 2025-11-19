import numpy as np
import pandas as pd

np.random.seed(42)

df = pd.DataFrame(
    np.random.rand(10, 5),
    columns=[f"col{i+1}" for i in range(5)]
)

df["mean_>0.3"] = df.apply(
    lambda row: row[row > 0.3].mean(),
    axis=1
)

print(df.to_string())
