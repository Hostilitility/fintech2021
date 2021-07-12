from utils import *
from sklearn.ensemble import IsolationForest


MAX_VALUE = 9999999999


class IF(object):
    def __init__(self):
        self.model = IsolationForest()
        self.n_models = 5

    def fit(self, train_data: pd.DataFrame):
        pass

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if 'label' in test_data.columns:
            test_data = test_data.drop(['label'], axis=1)
        ID = test_data["ID"]

        test_data = test_data.drop(["ID"], axis=1)
        test_data = test_data.fillna(0)
        test_data[test_data > MAX_VALUE] = MAX_VALUE
        test_data[test_data < -MAX_VALUE] = -MAX_VALUE

        self.model.fit(test_data)
        pred = 1 - self.model.decision_function(test_data)

        pred = pd.DataFrame({"ID": ID, "PD": pred})
        return pred
