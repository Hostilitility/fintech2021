import lightgbm as lgb
from utils import *


class BaseLGB(object):
    def __init__(self):
        self.params = {
            'learning_rate': 0.1,
            # 'objective': 'xentropy',
            'boosting_type': 'gbdt',
            'seed': 2019,
            'feature_fraction': 0.8,
            'verbosity': -1,
        }
        self.model = None

    def fit(self, X, y):
        y = y.astype(float)
        train_set = lgb.Dataset(X, y)
        self.model = lgb.train(self.params, train_set, num_boost_round=10)

    def predict(self, X: np.ndarray):
        return self.model.predict(X)
