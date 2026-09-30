import joblib
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from config import *
from data_loading import data_load, make_target
from features import build_features


def train_model(session_path="data/ga_sessions.csv", hits_path="data/ga_hits.csv"):
    sessions, hits = data_load(session_path, hits_path)
    df = build_features(sessions, hits)
    y = make_target(hits)

    data = df.merge(y, on="session_id", how="left")
    data = data.dropna(subset=["target"])

    X = data[feature_cols]
    y = data.loc[X.index, "target"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostClassifier(
        iterations=100,
        depth=4,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        early_stopping_rounds=50
    )

    cat_features = category_cols
    model.fit(x_train, y_train, cat_features=cat_features, eval_set=(x_test, y_test), use_best_model=True)
    pred = model.predict_proba(x_test)[:, 1]
    roc_auc = roc_auc_score(y_test, pred)
    print(f"Model: CatBoostClassifier AUC: {roc_auc}")

    joblib.dump({
        "model": model,
        "feature_cols": feature_cols,
        "category_cols": category_cols,
        "numeric_cols": numeric_cols,
        "metadata": {
            "Name": "Make target action prediction",
            "Author": "Rostislav Shigapov",
            "Model type": "CatBoostClassifier",
            "Version": 3,
            "ROC_AUC": roc_auc
        }
    }, "model/make_target_action.pkl")


if __name__ == "__main__":
    train_model()