"""
Airflow DAG для пайплайна обучения модели качества вина.
Порядок выполнения: загрузка данных → обучение модели → сохранение модели
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Аргументы по умолчанию
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Определение DAG
dag = DAG(
    "wine_quality_training",
    default_args=default_args,
    description="ML pipeline for wine quality prediction",
    schedule_interval="@daily",          # run once per day
    catchup=False,
    tags=["mlops", "wine-quality"],
)


# Функции тасок

def load_data(**context):
    """Загружает датасет о качестве вина из CSV-файла, отслеживаемого DVC."""
    import pandas as pd
    import yaml

    # Читаем путь к данным из конфига
    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)

    data_path = params["data"]["path"]
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} rows from {data_path}")
    print(f"Columns: {list(df.columns)}")
    print(df.describe())

    # Передаём количество строк следующим таскам через XCom
    context["ti"].xcom_push(key="row_count", value=len(df))
    return data_path


def train_model(**context):
    """Обучает модель (RandomForestRegressor) с параметрами из конфига."""
    import pickle

    import mlflow
    import pandas as pd
    import yaml
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    # Читаем конфиг с гиперпараметрами
    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)

    data_path = params["data"]["path"]
    model_cfg = params["model"]

    # Разделяем данные на признаки и целевую переменную
    df = pd.read_csv(data_path)
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Делим на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=model_cfg["test_size"],
        random_state=model_cfg["random_state"],
    )

    # Запускаем эксперимент в MLflow и логируем метрики
    mlflow.set_experiment("wine-quality")
    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=model_cfg["n_estimators"],
            max_depth=model_cfg["max_depth"],
            random_state=model_cfg["random_state"],
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae  = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2   = r2_score(y_test, preds)

        # Сохраняем параметры и метрики в MLflow
        mlflow.log_params(model_cfg)
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})

        print(f"MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    # Сериализуем модель на диск
    model_path = params["model"]["output_path"]
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_path}")
    # Передаём путь к модели следующей таске через XCom
    context["ti"].xcom_push(key="model_path", value=model_path)
    return model_path


def save_model(**context):
    """Регистрирует обученную модель в DVC и сохраняет метаданные."""
    import json
    import subprocess
    from datetime import datetime

    # Получаем путь к модели от предыдущей таски
    model_path = context["ti"].xcom_pull(key="model_path", task_ids="train_model")

    # Добавляем модель под версионирование DVC
    result = subprocess.run(
        ["dvc", "add", model_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"DVC warning: {result.stderr}")
    else:
        print(f"DVC: {result.stdout}")

    # Сохраняем метаданные об обученной модели
    metadata = {
        "model_path": model_path,
        "trained_at": datetime.utcnow().isoformat(),
        "version": "latest",
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved: {metadata}")


# Создание тасок
t_load = PythonOperator(
    task_id="load_data",
    python_callable=load_data,
    dag=dag,
)

t_train = PythonOperator(
    task_id="train_model",
    python_callable=train_model,
    dag=dag,
)

t_save = PythonOperator(
    task_id="save_model",
    python_callable=save_model,
    dag=dag,
)

# Порядок выполнения пайплайна
t_load >> t_train >> t_save
