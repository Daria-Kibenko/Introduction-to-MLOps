"""
experiments/train_experiments.py

Запускает серию экспериментов с двумя типами моделей и логирует результаты в MLflow.
Использование:
    python experiments/train_experiments.py

Результаты сохраняются в MLflow и описываются в README.md.
"""

import mlflow
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Загрузка конфига и данных
with open("configs/params.yaml") as f:
    params = yaml.safe_load(f)

df = pd.read_csv(params["data"]["path"])

# Разделяем на признаки (X) и целевую переменную (y)
X = df.drop("quality", axis=1)
y = df["quality"]

# Фиксируем random_state для воспроизводимости результатов
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["data"]["test_size"], random_state=42
)

# Подключаемся к серверу MLflow и выбираем эксперимент
mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
mlflow.set_experiment(params["mlflow"]["experiment_name"])


def evaluate(model, X_test, y_test) -> dict:
    """Вычисляет метрики качества модели на тестовой выборке."""
    preds = model.predict(X_test)
    return {
        "mae":  mean_absolute_error(y_test, preds),   # средняя абсолютная ошибка
        "rmse": mean_squared_error(y_test, preds) ** 0.5,  # корень из MSE
        "r2":   r2_score(y_test, preds),              # коэффициент детерминации
    }


# Список экспериментов
experiments = [
    {
        # Базовая линейная модель - точка отсчёта для сравнения
        "name": "LinearRegression_baseline",
        "model": LinearRegression(),
        "params": {},
    },
    {
        # RandomForest с небольшой глубиной - меньше переобучения
        "name": "RandomForest_n100_d5",
        "model": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "params": {"n_estimators": 100, "max_depth": 5},
    },
    {
        # RandomForest с увеличенной глубиной - лучше захватывает нелинейности
        "name": "RandomForest_n100_d10",
        "model": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "params": {"n_estimators": 100, "max_depth": 10},
    },
    {
        # Больше деревьев при той же глубине - проверяем, стоит ли оно того
        "name": "RandomForest_n200_d10",
        "model": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "params": {"n_estimators": 200, "max_depth": 10},
    },
]

# Запуск экспериментов
results = []
for exp in experiments:
    with mlflow.start_run(run_name=exp["name"]):
        exp["model"].fit(X_train, y_train)
        metrics = evaluate(exp["model"], X_test, y_test)

        # Логируем параметры и метрики в MLflow
        mlflow.log_params({"model": exp["name"], **exp["params"]})
        mlflow.log_metrics(metrics)

        results.append({"name": exp["name"], **metrics})
        print(f"{exp['name']:40s}  MAE={metrics['mae']:.4f}  "
              f"RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}")

# Итоговая таблица результатов
print("\n=== Сводная таблица результатов ===")
results_df = pd.DataFrame(results).sort_values("r2", ascending=False)
print(results_df.to_string(index=False))
print(f"\nЛучшая модель: {results_df.iloc[0]['name']}")
