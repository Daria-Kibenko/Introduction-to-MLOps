import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train(train_csv: str) -> str:
    """
    Обучает модель логистической регрессии на тренировочных данных.
    Args:
        train_csv: Путь к iris_train.csv.
    Returns:
        str: Путь к сохранённой модели (model.pkl).
    """
    df = pd.read_csv(train_csv)
    X = df.drop('target', axis=1)
    y = df['target']

    # Обучаем модель (максимум итераций увеличен для сходимости)
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X, y)

    # Сохраняем
    model_path = "model.pkl"
    joblib.dump(model, model_path)
    return model_path
