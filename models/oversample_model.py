from utils import *
from . import BaseLGB


class OversampleModel(object):
    def __init__(self):
        self.models = []
        self.n_models = 5

    def fit(self, train_data: pd.DataFrame):
        # y 应为离散的 0/1 1代表异常
        y = train_data['label']
        X = train_data.drop(['label', 'ID'], axis=1)
        num_neg = sum(y == 0)
        neg_index = y.index[y == 0]
        pos_index = y.index[y == 1]
        # 5个模型 average
        for i in range(self.n_models):
            sample_pos_index = np.random.choice(pos_index, num_neg)
            sample_index = np.append(neg_index, sample_pos_index)
            np.random.shuffle(sample_index)
            sample_X = X.loc[sample_index].values
            sample_y = y.loc[sample_index].values
            over_model = BaseLGB()
            over_model.fit(sample_X, sample_y)
            self.models.append(over_model)

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if 'label' in test_data.columns:
            test_data = test_data.drop(['label'], axis=1)
        results = pd.DataFrame(index=test_data.index, columns=range(self.n_models))
        ID = test_data["ID"]
        test_data = test_data.drop(["ID"], axis=1).values
        for i_model in range(self.n_models):
            results[i_model] = self.models[i_model].predict(test_data)
        pred = post_precess(results.mean(axis=1))
        pred = pd.DataFrame({"ID": ID, "PD": pred})
        return pred
