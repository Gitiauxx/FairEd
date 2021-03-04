import numpy as np

from sklearn.metrics import roc_auc_score

class Model(object):
    """
    Generic model class for a model built off a learner. The learner needs a fit and predict method
    """
    def __init__(self, learner):

        self.learner = learner

    def fit(self, X, y):
        self.learner.fit(X, y)

    def predict(self, X):
        return self.learner.predict(X)

    def predict_score(self, X):
        return self.learner.predict_proba(X)

    def compute_accuracy(self, y, ypred):
        return np.mean((y == ypred).astype('int32'))

    def compute_true_positive(self, y, ypred):
        return ypred[(ypred == 1) & (y == 1)].sum()

    def compute_false_positive(self, y, ypred):
        return ypred[(ypred == 1) & (y == 0)].sum()

    def compute_false_negative(self, y, ypred):
        return np.sum(((ypred == 0) & (y == 1)).astype('int32'))

    def compute_true_negative(self, y, ypred):
        return np.sum(((ypred == 0) & (y == 0)).astype('int32'))

    def compute_recall(self, y, ypred):
        tp = self.compute_true_positive(y, ypred)
        fn = self.compute_false_negative(y, ypred)

        return tp / (tp + fn)

    def compute_precision(self, y, ypred):
        tp = self.compute_true_positive(y, ypred)
        fp = self.compute_false_positive(y, ypred)

        return tp / (tp + fp)

    def compute_f1(self, y, ypred):
        precision = self.compute_precision(y, ypred)
        recall = self.compute_recall(y, ypred)
        return 2 * precision * recall / (precision + recall)

    def compute_confusion(self, y, ypred):
        tp = self.compute_true_positive(y, ypred) / y.sum()
        tn = self.compute_true_negative(y, ypred) / (1 - y).sum()
        fn = self.compute_false_negative(y, ypred) / y.sum()
        fp = self.compute_false_positive(y, ypred) / (1 - y).sum()

        return np.array([[tp, fp], [fn, tn]])

    def compute_auc(self, y, X):
        return roc_auc_score(y, self.predict_score(X)[:, 1])

    def compute_auc_per_group(self, y, X, s):
        y0 = y[s == 0]
        y1 = y[s == 1]

        X0 = X[s == 0]
        X1 = X[s == 1]

        auc0 = self.compute_auc(y0, X0)
        auc1 = self.compute_auc(y1, X1)

        return np.array([auc0, auc1])

    def compute_confusion_per_group(self, y, ypred, s):
        y0 = y[s == 0]
        y1 = y[s == 1]

        ypred0 = ypred[s == 0]
        ypred1 = ypred[s == 1]

        confusion_matrix0 = self.compute_confusion(y0, ypred0)
        confusion_matrix1 = self.compute_confusion(y1, ypred1)

        return np.stack([confusion_matrix0, confusion_matrix1], 0)


    def compute_disparate_impact(self, y, s):
        return (y[s==1].sum() / s.sum()  / (y[s==0].sum() / (1 - s).sum()))

    def compute_equality_opportunity(self, y, ypred, s):
        y0 = y[s == 0]
        y1 = y[s == 1]

        ypred0 = ypred[s == 0]
        ypred1 = ypred[s == 1]

        tp0 = self.compute_true_positive(y0, ypred0)
        tp1 = self.compute_true_positive(y1, ypred1)

        return tp0 / y0.sum() - tp1 / y1.sum()