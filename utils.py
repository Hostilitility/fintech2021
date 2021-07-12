import os
from typing import List

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from datetime import datetime
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score


t = datetime.now().strftime('%Y-%m-%d-%H')
logger.add(os.path.join('logs', f'{t}.log'))


def get_scores(y_true: np.ndarray, y_pred: np.ndarray, metrics: List[str]) -> List[float]:
    # metrics {'AUC_ROC','AUC_PR','AR'}
    scores = []
    for metric in metrics:
        if metric == "AUC_ROC":
            scores.append(roc_auc_score(y_true, y_pred))
        elif metric == "AUC_PR":
            scores.append(average_precision_score(y_true, y_pred))
        elif metric == "AR":
            roc = roc_auc_score(y_true, y_pred) if "AUC_PR" not in metrics else scores[0]
            scores.append(roc * 2 - 1)
        else:
            logger.warning(f"Unknown metric {metric}, return -1")
            scores.append(-1.)
    return scores


def join_and_mkdirs(*p: str) -> str:
    joined = os.path.join(*p)
    if '.' not in p[-1]:
        os.makedirs(joined, exist_ok=True)
    else:
        d = os.path.join(*p[:1])
        os.makedirs(d, exist_ok=True)
    return joined


def save_item(item, path: str):
    if path.endswith(".pkl"):
        with open(path, 'wb') as f:
            pickle.dump(item, f)
    elif path.endswith(".csv"):
        item.to_csv(path, index=False)
    else:
        raise ValueError("Unmatched path")


def load_item(path):
    if path.endswith(".pkl"):
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif path.endswith(".csv"):
        try:
            return pd.read_csv(path, encoding='gbk')
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding='utf-8')
    else:
        raise ValueError("Unmatched path")


def post_precess(pred):
    pred[pred > 1] = 1
    pred[pred < 0] = 0
    return pred


def mix_up(weight: np.ndarray, pos_data: np.ndarray, neg_data: np.ndarray) -> np.ndarray:
    """
    :param weight: <n,>
    :param pos_data: <n,>
    :param neg_data: <n,>
    :return:
    """
    total_data = np.row_stack([pos_data, neg_data])
    category_features = []
    unique_values = []
    for i in range(total_data.shape[1]):
        feature = total_data[:, i]
        unique_value = np.unique(feature)
        if len(unique_value) <= 10:
            category_features.append(i)
            unique_values.append(unique_value)

    mixup_data = weight * pos_data + (1 - weight) * neg_data
    for i, unique_value in zip(category_features, unique_values):
        feature = mixup_data[:, i]
        feature = [nearest_value(x, unique_value) for x in feature]
        mixup_data[:, i] = feature

    return mixup_data


def nearest_value(x, arr):
    max_diff, target = abs(x - arr[0]), arr[0]
    for z in arr:
        if abs(x - z) < max_diff:
            max_diff, target = abs(x - z), z
    return target


def get_feature_importance(result_dir: str):
    data_path = "data/data4.csv"
    data_head = pd.read_csv(data_path, nrows=1, encoding='gbk')
    col = list(data_head.columns)
    col.remove("label")
    col.remove("ID")

    model_path = os.path.join(result_dir, 'model.pkl')
    model = load_item(model_path)
    submodels = model.models
    fis = np.zeros([len(col), len(submodels)])
    for i, m in enumerate(submodels):
        fis[:, i] = m.model.feature_importance()
    fi = fis.mean(axis=1)

    cols = [(x, y) for x, y in zip(fi, col)]
    cols.sort()
    return cols


if __name__ == '__main__':
    print(get_feature_importance("task/v4"))
