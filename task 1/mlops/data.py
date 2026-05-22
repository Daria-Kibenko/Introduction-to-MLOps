import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data() -> str:
    """
    Загружает датасет Iris с помощью sklearn, сохраняет его в CSV-файл.
    Returns:
        str: Путь к сохранённому файлу.
    """
    # Создаём директорию dataset, если её нет
    os.makedirs("dataset", exist_ok=True)

    # Загружаем данные
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # Сохраняем
    file_path = "dataset/iris.csv"
    df.to_csv(file_path, index=False)
    return file_path


def prepare_data(csv_path: str) -> list[str]:
    """
    Читает исходный CSV, делит выборку на train (0.8) и test (0.2),
    сохраняет оба набора в отдельные файлы.
    Args:
        csv_path: Путь к исходному iris.csv.
    Returns:
        list[str]: [путь_к_train.csv, путь_к_test.csv]
    """
    df = pd.read_csv(csv_path)

    # Разделяем признаки и целевую переменную
    X = df.drop('target', axis=1)
    y = df['target']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Собираем обратно в DataFrame для удобства сохранения
    train_df = X_train.copy()
    train_df['target'] = y_train
    test_df = X_test.copy()
    test_df['target'] = y_test

    # Сохраняем
    train_path = "dataset/iris_train.csv"
    test_path = "dataset/iris_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return [train_path, test_path]
