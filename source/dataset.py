import pandas as pd
import numpy as np

class Chilean(object):
    """
    Create a dataset class for the Chilean dataset
    Sensitive attribute is either gender, school or elite
    """

    def __init__(self, data_folder, sensitive='elite', type='train'):

        x = pd.read_csv(f'{data_folder}/aware_{sensitive}_{type}.txt')
        y = pd.read_csv(f'{data_folder}/y_{type}.txt', header=None)

        self.s = np.array(x[sensitive])
        self.y = np.array(y).squeeze(1)
        x.drop(sensitive, inplace=True, axis=1)
        x.drop('36x', inplace=True, axis=1)


        x = np.array(x)
        x = self.center(x)
        x = self.normalize(x)

        self.x = x

    def center(self, X):
        xmean = X.mean(0)
        xmean = xmean[None, ...]

        return X - xmean

    def normalize(self, X):
        x2sum = (X ** 2).sum(0)
        x2sum = x2sum[None, ...]

        return X / np.sqrt(x2sum)