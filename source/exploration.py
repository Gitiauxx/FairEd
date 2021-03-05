import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from source.dataset import Chilean

class FeatureImportance(object):

    def __init__(self,  datafile):

        self.datafile = datafile
        self.names =  None

    def compute_importance(self, sensitive):
        dset  = Chilean(self.datafile, sensitive=sensitive, type='train')
        X = dset.x
        s = dset.s
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, )
        rf.fit(X, s)

        sorted_idx = rf.feature_importances_.argsort()

        if self.names is None:
            self.names = dset.feature_names

        return sensitive, sorted_idx, rf.feature_importances_[sorted_idx]

    def plot_panel_importance(self, importance_list, outfolder, tag='rf'):

        fig, ax = plt.subplots(nrows=1, ncols=len(importance_list), figsize=(16, 5))

        for i,  scenario in enumerate(importance_list):
            features = self.names[scenario[1]][-10:]
            importance = scenario[2][-10:]
            ax[i].barh(features, importance)

            if scenario[0] == 'gender':
                name = 'Gender'
            elif scenario[0] == 'school':
                name = 'Public High School'
            else:
                name = 'Gender - Public High School'

            ax[i].set_title(f'{name}', size=14)
            ax[i].set_xlabel("Random Forest Feature Importance", fontsize=10)

        fig.tight_layout(pad=2.0)

        plt.savefig(f'{outfolder}/feature_importance_{tag}.png')

        plt.clf()



