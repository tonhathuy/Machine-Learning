import numpy as np
import pandas as pd


df = pd.read_csv('data/test.csv', header=None)

print(df[3])

# df = df[3]
# df.to_csv('export_csv.csv')