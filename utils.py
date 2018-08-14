import numpy as np


def pd_split_train_test(df, ratio=0.5):
    mask = np.random.rand(len(df)) < ratio
    return df[mask].copy(), df[~mask].copy()
