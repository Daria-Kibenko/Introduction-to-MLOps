"""
Airflow DAG для пайплайна обучения модели качества вина.

Полный порядок выполнения:
  dvc pull (данные) → обучение модели → сохранение метаданных
  → dvc add модель+метаданные → dvc push
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
    description="ML-пайплайн для предсказания качества вина",
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops", "wine-quality"],
)


# Функции тасок

def load_data(**context):
    """
    Скачивает данные из удалённого хранилища через DVC,
    затем читает CSV и проверяет его корректность.
    """
    import subprocess

    import pandas as pd
    import yaml

    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)

    data_path = params["data"]["path"]

    # Тянем данные из удалённого хранилища DVC перед чтением
    print(f"Скачиваем данные через DVC: {data_path}.dvc")
    result = subprocess.run(
        ["dvc", "pull", data_path + ".dvc"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dvc pull для данных завершился с ошибкой:\n{result.stderr}")
    print(f"DVC pull данных: {result.stdout}")

    # Читаем данные только после успешного dvc pull
    df = pd.read_csv(data_path)

    print(f"Загружено {len(df)} строк из {data_path}")
    print(f"Колонки: {list(df.columns)}")
    print(df.describe())

    # Передаём путь к данным следующей таске через XCom
    context["ti"].xcom_push(key="data_path", value=data_path)
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

    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)

    model_cfg = params["model"]

    # Получаем путь к данным от предыдущей таски
    data_path = context["ti"].xcom_pull(key="data_path", task_ids="load_data")

    # Разделяем на признаки и целевую переменную
    df = pd.read_csv(data_path)
    X = df.drop("quality", axis=1)
    y = df["quality"]

    # Разбивка на обучающую и тестовую выборки (фиксируем random_state для воспроизводимости)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=model_cfg["test_size"],
        random_state=model_cfg["random_state"],
    )

    # Логируем эксперимент в MLflow
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

    # Сериализуем обученную модель на диск
    model_path = model_cfg["output_path"]
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Модель сохранена: {model_path}")
    context["ti"].xcom_push(key="model_path", value=model_path)
    return model_path


def save_metadata(**context):
    """Сохраняет метаданные об обученной модели в JSON-файл."""
    import json
    from datetime import datetime

    import yaml
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    model_path = context["ti"].xcom_pull(key="model_path", task_ids="train_model")

    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)

    # Формируем метаданные с описанием модели и параметров обучения
    metadata = {
        "model_path": model_path,
        "model_type": "RandomForestRegressor",
        "trained_at": datetime.utcnow().isoformat(),
        "version": "latest",
        "hyperparameters": {
            "n_estimators": params["model"]["n_estimators"],
            "max_depth": params["model"]["max_depth"],
            "random_state": params["model"]["random_state"],
        },
    }

    metadata_path = "models/metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Метаданные сохранены: {metadata}")
    context["ti"].xcom_push(key="metadata_path", value=metadata_path)
    return metadata_path


def dvc_push(**context):
    """
    Регистрирует модель и метаданные в DVC, затем отправляет
    всё в удалённое хранилище командой dvc push.
    """
    import subprocess

    model_path    = context["ti"].xcom_pull(key="model_path",    task_ids="train_model")
    metadata_path = context["ti"].xcom_pull(key="metadata_path", task_ids="save_metadata")

    for path in [model_path, metadata_path]:
        # dvc add: добавляем файл под версионирование DVC
        result = subprocess.run(
            ["dvc", "add", path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            # Ошибка dvc add - прерываем выполнение, задача не считается успешной
            raise RuntimeError(f"dvc add {path} завершился с ошибкой:\n{result.stderr}")
        print(f"dvc add {path}: {result.stdout.strip()}")

    # dvc push: отправляем все изменения в удалённое хранилище
    result = subprocess.run(
        ["dvc", "push"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dvc push завершился с ошибкой:\n{result.stderr}")
    print(f"dvc push выполнен: {result.stdout.strip()}")


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

t_save_meta = PythonOperator(
    task_id="save_metadata",
    python_callable=save_metadata,
    dag=dag,
)

t_dvc_push = PythonOperator(
    task_id="dvc_push",
    python_callable=dvc_push,
    dag=dag,
)

# Порядок выполнения пайплайна
# dvc pull данных → обучение → сохранение метаданных → dvc add + dvc push
t_load >> t_train >> t_save_meta >> t_dvc_push
