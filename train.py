import numpy as np

from source import *
from source.dataset import Chilean
from source.models.abstract_model import Model
from source.mitigation import PostProcessing, InProcessing
from source.utils import get_logger

logger = get_logger(__name__)

train_file = '/home/mx/Documents/Xavier/Representation/Data/to-ml-chilea/train per group'
test_file = '/home/mx/Documents/Xavier/Representation/Data/to-ml-chilea/test per group'

sensitive = 'school'

train_dset = Chilean(train_file, sensitive=sensitive, type='train')
test_dset = Chilean(test_file, sensitive=sensitive, type='test')

learner = LogisticRegression(C=0.1, class_weight='balanced')
X = train_dset.x
y = train_dset.y
s = train_dset.s

model = Model(learner)

model.fit(X, y)
ypred = model.predict(test_dset.x)

logger.info(f'F1 score before mitigation {model.compute_f1(test_dset.y, ypred)}')
logger.info(f'Equality of Opportunity before mitigation '
            f'{model.compute_equality_opportunity(test_dset.y, ypred, test_dset.s)}')
logger.info(f'Confusion before mitigation '
            f'{model.compute_confusion_per_group(test_dset.y, ypred, test_dset.s)}')

post_mitigator = PostProcessing(model.learner,
                                ThresholdOptimizer,
                                constraints='equalized_odds',
                                objective='balanced_accuracy_score')
post_mitigator.fit(X, y, s)
ypred_fair = post_mitigator.predict(test_dset.x, test_dset.s)

logger.info(f'F1 score after postprocessing mitigation {model.compute_f1(test_dset.y, ypred_fair)}')
logger.info(f'Equality of Opportunity after postprocessing mitigation '
            f'{model.compute_equality_opportunity(test_dset.y, ypred_fair, test_dset.s)}')
logger.info(f'Confusion after postprocessing mitigation '
            f'{model.compute_confusion_per_group(test_dset.y, ypred_fair, test_dset.s)}')

constraint = EqualizedOdds()
in_mitigator = InProcessing(model.learner, ExponentiatedGradient, constraint)
in_mitigator.fit(X, y, s)

ypred_fair = in_mitigator.predict(test_dset.x)

logger.info(f'F1 score after in-processing mitigation {model.compute_f1(test_dset.y, ypred_fair)}')
logger.info(f'Equality of Opportunity after in-processing mitigation '
            f'{model.compute_equality_opportunity(test_dset.y, ypred_fair, test_dset.s)}')
logger.info(f'Confusion after in-processing mitigation '
            f'{model.compute_confusion_per_group(test_dset.y, ypred_fair, test_dset.s)}')

