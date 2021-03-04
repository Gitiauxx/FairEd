from source.models.abstract_model import Model

class PostProcessing(Model):
    """
    Generic class for postprocessing techniques
    The class is a child of Model and takes a fitted model and adjust its
    prediction fairness to meet a constraint
    """

    def __init__(self, learner,
                 mitigation,
                 constraints='equalized_odds',
                 objective='balanced_accuracy_score'):
        super().__init__(learner)

        self.mitigation = mitigation(estimator=learner, constraints=constraints, objective=objective)

    def fit(self, X, y, s):
        self.mitigation.fit(X, y, sensitive_features=s)

    def predict(self, X, s):
        return self.mitigation.predict(X, sensitive_features=s)

class InProcessing(Model):
    """
    Generic class for in-processing techniques
    """

    def __init__(self, learner,
                 mitigation,
                 constraints):
        super().__init__(learner)

        self.mitigation = mitigation(learner, constraints)

    def fit(self, X, y, s):
        self.mitigation.fit(X, y, sensitive_features=s)
