import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import create_dataset as cd


def create_space(df, cols=['A', 'B']):
    res = pd.DataFrame()
    num_elements = len(df)
    for i, col in enumerate(cols):
        min_col = np.min(df[col])
        max_col = np.max(df[col])
        margin = .25
        min_val = min_col - (max_col - min_col) * margin
        max_val = max_col + (max_col - min_col) * margin
        vals = np.random.uniform(min_val, max_val, num_elements)
        res[col] = vals
    return res


if __name__ == '__main__':
    n_blobs = 5
    df = cd.create_nblobs(n_blobs, 5, 250)
    df_space = create_space(df)
    for i in range(n_blobs):
        plt.plot(df.loc[df['class'] == i, 'A'], df.loc[df['class'] == i, 'B'], 'o')
    plt.plot(df_space['A'], df_space['B'], 'k.', alpha=0.1)

    print(len(df))
    print(len(df_space))
    plt.show()
