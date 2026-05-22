import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def test(model_path: str, test_csv: str) -> str:
    """
    Тестирует модель на тестовой выборке, вычисляет accuracy и отчёт классификации,
    сохраняет метрики в JSON-файл.
    Args:
        model_path: Путь к model.pkl.
        test_csv: Путь к iris_test.csv.
    Returns:
        str: Путь к файлу с метриками (model_metrics.json).
    """

    # Загружаем модель и данные
    model = joblib.load(model_path)
    df = pd.read_csv(test_csv)
    X_test = df.drop('target', axis=1)
    y_test = df['target']

    # предсказания + метрики
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Сохраняем в json
    metrics = {
        "accuracy": accuracy,
        "report": report
    }

    metrics_path = "model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics_path
