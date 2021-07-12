from utils import *
from . import BaseLGB


class SubsampleModel(object):
    def __init__(self):
        self.models = []
        self.n_models = 5

    def fit(self, train_data: pd.DataFrame):
        # y 应为离散的 0/1 1代表异常
        y = train_data['label']
        X = train_data.drop(['label', 'ID'], axis=1)
        num_positives = sum(y == 1)
        neg_index = y.index[y == 0]
        pos_index = y.index[y == 1]
        # 5个模型 average
        for i_model in tqdm(range(self.n_models)):
            sample_neg_index = np.random.choice(neg_index, num_positives)
            sample_index = np.append(pos_index, sample_neg_index)
            np.random.shuffle(sample_index)
            sample_X = X.loc[sample_index].values
            sample_y = y.loc[sample_index].values
            submodel = BaseLGB()
            submodel.fit(sample_X, sample_y)
            self.models.append(submodel)

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
