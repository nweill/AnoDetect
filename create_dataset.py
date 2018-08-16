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
             'cat': [i] * n_per_cat})], axis=0)

    def get_data():
        return df

    return get_data


def create_circular_data(radius=[0, 5, 10, 7, 20], n_per_class=250):
    df = pd.DataFrame()
    jitter = 0.2
    for i, rad in enumerate(radius):
        vec = np.arange(-np.pi, np.pi, step=np.pi / n_per_class)
        a = rad * np.cos(vec) + np.random.normal(0, jitter, len(vec))
        b = rad * np.sin(vec) + np.random.normal(0, jitter, len(vec))
        tmp = pd.DataFrame(
            {'A': a, 'B': b,
             'cat': [i] * len(vec)})
        df = pd.concat([df, tmp], axis=0, ignore_index=True)

        def get_data():
            return df
    return get_data


if __name__ == '__main__':
    df = create_circular_data()()

    # df = create_nblobs(n_blobs, 5, 250)()
    n_class = len(df['cat'].unique())
    for i in range(n_class):
        plt.plot(df.loc[df['cat'] == i, 'A'], df.loc[df['cat'] == i, 'B'], 'o')
    plt.show()
