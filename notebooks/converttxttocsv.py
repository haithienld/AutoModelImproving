import os
import pandas as pd

dirpath = 'dataset/'
output = 'poses.csv'

pd.concat(
    pd.read_csv(os.path.join(dirpath, fname), sep=',', index_col=0, header=None)
    for fname in sorted(os.listdir(dirpath))
).to_csv(output)
