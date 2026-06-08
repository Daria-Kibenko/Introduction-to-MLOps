# Проект по качеству вина - MLOps

Полноценный MLOps-пайплайн для предсказания качества красного вина по физико-химическим характеристикам.

## Структура проекта

```
.
├── .gitlab-ci.yml                  # CI/CD пайплайн (линтер → DVC → тесты)
├── configs/
│   └── params.yaml                 # Централизованные гиперпараметры и пути
├── dags/
│   └── wine_quality_dag.py         # Airflow DAG: загрузка → обучение → сохранение
├── data/                           # Отслеживается DVC (в git только .dvc-файлы)
│   └── wine_quality.csv.dvc
├── experiments/
│   └── train_experiments.py        # Скрипт сравнения моделей
├── models/                         # Артефакты модели под версионированием DVC
│   ├── wine_quality_model.pkl.dvc
│   └── metadata.json
├── src/
│   └── api/
│       └── main.py                 # FastAPI-сервис
├── tests/
│   └── test_api.py                 # Независимые тесты API
└── requirements.txt
```

## Быстрый старт

### 1. Клонировать и установить зависимости

```bash
git clone <ссылка-на-репозиторий>
cd mlops-wine-project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Скачать данные и модель через DVC

```bash
# Настраиваем доступ к удалённому хранилищу
dvc remote modify myremote access_key_id     ВАШ_КЛЮЧ
dvc remote modify myremote secret_access_key ВАШ_СЕКРЕТ

dvc pull
```

### 3. Запустить API

```bash
uvicorn src.api.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

### 4. Запустить тесты

```bash
pytest tests/ -v --cov=src
```

### 5. Запустить Airflow локально

```bash
export AIRFLOW_HOME=$(pwd)/.airflow
airflow db init
airflow scheduler &
airflow webserver --port 8080
# Запуск DAG вручную:
airflow dags trigger wine_quality_training
```

---

## Данные

**Датасет:** [Red Wine Quality](https://github.com/aniruddhachoudhury/Red-Wine-Quality/blob/master/winequality-red.csv)

- 1 599 строк, 12 колонок (11 физико-химических признаков + оценка качества 0-10)
- Версионируется с помощью **DVC** (удалённое хранилище: MinIO S3)
- В git попадают только `.dvc`-файлы-указатели, сами данные - никогда

---

## Эксперименты и выбор модели

Две модели сравнивались на разбивке 80/20 с `random_state=42`.  
Все эксперименты залогированы в **MLflow** (эксперимент: `wine-quality`).

### Результаты

| Модель                              | MAE    | RMSE   | R²     |
|-------------------------------------|--------|--------|--------|
| LinearRegression (базовая)          | 0.5021 | 0.6481 | 0.3614 |
| RandomForest n=100 глубина=5        | 0.4312 | 0.5623 | 0.5287 |
| **RandomForest n=100 глубина=10**   | **0.3897** | **0.5102** | **0.6124** |
| RandomForest n=200 глубина=10       | 0.3901 | 0.5108 | 0.6118 |

### Победитель: `RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)`

**Почему RandomForest, а не LinearRegression?**
- R² вырос с 0.36 до 0.61 (+69%), потому что качество вина определяется нелинейными взаимодействиями признаков (например, алкоголь × кислотность).
- Увеличение числа деревьев со 100 до 200 даёт прирост R² всего +0.1% при удвоении времени обучения - не оправдано.
- `max_depth=10` предотвращает переобучение, сохраняя при этом способность улавливать главные закономерности.

---

## API-эндпоинты

| Метод | Путь           | Описание                                         |
|-------|----------------|--------------------------------------------------|
| POST  | `/predict`     | Предсказать качество вина по 11 признакам        |
| GET   | `/healthcheck` | Проверка работоспособности → `{"status": "ok"}` |
| GET   | `/model-info`  | Тип модели, версия, список признаков             |

### Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.70,
    "citric_acid": 0.00,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
# → {"quality_score": 5.62, "quality_label": "good"}
```

---

## CI/CD пайплайн (GitLab)

Запускается при каждом Merge Request в `main`:

```
lint ──► check_dvc ──► test_api
```

| Этап        | Что делает                                                         |
|-------------|--------------------------------------------------------------------|
| `lint`      | flake8 по `src/`, `dags/`, `tests/` (макс. длина строки 100)      |
| `check_dvc` | `dvc pull`, затем проверка наличия файлов данных и модели          |
| `test_api`  | `pytest tests/` с отчётом о покрытии (XML-артефакт)               |

**Секреты** хранятся как переменные GitLab CI/CD и никогда не коммитятся в репозиторий:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

---

## Git-процесс

```
main        ← продакшн; обновляется только через Merge Request
  └─ develop ← ветка интеграции; сюда вливаются feature-ветки
       └─ feature/xxx
```
