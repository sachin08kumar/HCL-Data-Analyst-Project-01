"""Generate a SYNTHETIC Haberman-style survival dataset for learning purposes.
No real patient data is used. Values are random and carry no medical meaning.
Run: python scripts/generate_data.py"""
import os
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 301   # + 5 duplicate rows added below = 306 records (same size as the original dataset)
age = np.clip(rng.normal(52.5, 10.8, n), 30, 83).round(0)
operation_year = rng.choice(np.arange(58, 70), n,           # year minus 1900 (58 = 1958)
                            p=[.12,.09,.09,.085,.075,.10,.10,.09,.09,.08,.04,.04])
has_nodes = rng.random(n) > 0.44                              # ~44% of records have 0 nodes
nodes = np.where(has_nodes, np.clip(rng.geometric(0.14, n), 1, 52), 0)   # right-skewed counts
# status: 1 = survived 5+ years, 2 = died within 5 years (more likely with higher node counts)
p_died = 1 / (1 + np.exp(-(-1.35 + 0.08*nodes + 0.01*(age - 52))))
status = np.where(rng.random(n) < p_died, 2, 1)
df = pd.DataFrame({'age': age, 'operation_year': operation_year,
                   'positive_axillary_nodes': nodes, 'survival_status': status})
# Introduce realistic data-quality problems for the cleaning exercise
for col, k in [('age', 5), ('operation_year', 4), ('positive_axillary_nodes', 4)]:
    df[col] = df[col].astype(float)
    df.loc[rng.choice(n, k, replace=False), col] = np.nan
df.loc[12, 'age'] = 150.0                                     # impossible value
df.loc[45, 'positive_axillary_nodes'] = -3.0                  # impossible (negative) count
df.loc[[60, 120], 'operation_year'] = [1962, 1965]            # inconsistent format (4-digit year)
df = pd.concat([df, df.sample(5, random_state=1)], ignore_index=True)    # 5 duplicate rows
os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/haberman_synthetic.csv', index=False)
print(df.shape)
