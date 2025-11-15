import numpy as np
import pandas as pd

# for reproducibility
np.random.seed(42)

# number of samples
n = 200

# feature columns (2 features)
x1 = np.random.uniform(-5, 5, n)
x2 = np.random.uniform(-5, 5, n)

# ground-truth function
y = 3 * x1 - 2 * x2 + 0.5 + np.random.normal(0, 0.5, n)

# build dataframe
df = pd.DataFrame({
    "x1": x1,
    "x2": x2,
    "y": y
})

print(df.head())
df.to_csv("synthetic_dataset.csv", index=False)
