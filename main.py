from utils import *
from models import *
from data_loader import DataLoader


# --------------- setting ----------------
header = "task/v0"
model_class = MixUpModel
validation = True
load_model = False
metrics = ["AUC_PR", "AUC_ROC"]

# ---------------- init --------------------
model_path = join_and_mkdirs(header, 'model.pkl')
result_path = join_and_mkdirs(header, 'result.csv')
validation_path = join_and_mkdirs(header, 'validation.csv')

# ---------------- load data ----------------
dl = DataLoader(val=validation)
train_data = dl.load_train()
logger.info(f"Train data loaded, shape: {train_data.shape}")

# --------------- train model ---------------
if load_model:
    model = load_item(model_path)
else:
    model = model_class()
    model.fit(train_data)
    save_item(model, model_path)
    logger.info(f"Model trained and saved to {model_path}")

# --------------- validation -------------------
dl.val = True
test_data = dl.load_test()
test_y = test_data['label']
pred = model.predict(test_data)
pred['truth'] = test_y
save_item(pred, validation_path)
logger.info(f"Validation result saved to {validation_path}")
scores = get_scores(test_y, pred["PD"], metrics)
for m, s in zip(metrics, scores):
    print(f"{m}: {s}")

# -------------------- get_result ----------------------
dl.val = False
test_data = dl.load_test()
pred = model.predict(test_data)
save_item(pred, result_path)
logger.info(f"Result saved to {result_path}")
