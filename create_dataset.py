import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.random.seed(1)


def create_nblobs(n_blobs=2, std=1, n_per_cat=100):
    df = pd.DataFrame()
    for i in range(n_blobs):
        means = np.random.random_integers(-50, 50, 2)
        df = pd.concat([df, pd.DataFrame(
            {'A': np.random.normal(means[0], std, n_per_cat), 'B': np.random.normal(means[1], std, n_per_cat),
             'class': [i] * n_per_cat})], axis=0)
    return df


if __name__ == '__main__':
    n_blobs = 5
    df = create_nblobs(n_blobs, 5, 250)
    for i in range(n_blobs):
        plt.plot(df.loc[df['class'] == i, 'A'], df.loc[df['class'] == i, 'B'], 'o')
    plt.show()
