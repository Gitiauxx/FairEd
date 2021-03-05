import pandas as pd
import numpy as np

class Chilean(object):
    """
    Create a dataset class for the Chilean dataset
    Sensitive attribute is either gender, school or elite
    """

    def __init__(self, data_folder, sensitive=None, type='train', xmean=None, x2sum=None):

        if sensitive is not None:
            x = pd.read_csv(f'{data_folder}/aware_{sensitive}_{type}.txt')
            self.s = np.array(x[sensitive])
        else:
            x = pd.read_csv(f'{data_folder}/aware_{type}.txt')
            self.s = x[['gender', 'school', 'elite']]

        y = pd.read_csv(f'{data_folder}/y_{type}.txt', header=None)
        self.y = np.array(y).squeeze(1)

        self.features_dict = {'4x': "High School Size", '0x': 'Income Decile', '1x': 'Family Income',
                              '2x': 'Preference of Application', '6x': 'SAT Math',
                              '10x': 'GPA High School', '8x': 'SAT History',
                              '3x': 'Average SAT High School', '9x': "SAT Verbal", '14x': 'GPA business Sem 1 '}
        x.rename(columns=self.features_dict, inplace=True)

        if sensitive is not None:
            x.drop(sensitive, inplace=True, axis=1)

        x.drop('36x', inplace=True, axis=1)
        self.feature_names = np.array(x.columns)

        x = np.array(x)
        if type == 'train':
            x = self.center(x)
            x = self.normalize(x)
        else:
            x = x - xmean
            x = x / np.sqrt(x2sum)

        self.x = x


    def center(self, X):
        xmean = X.mean(0)
        xmean = xmean[None, ...]

        self.xmean = xmean

        return X - xmean

    def normalize(self, X):
        x2sum = (X ** 2).sum(0)
        x2sum = x2sum[None, ...]

        self.x2sum = x2sum

        return X / np.sqrt(x2sum)