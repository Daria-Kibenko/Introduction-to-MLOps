from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weather import update_weather

default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),   # дата в прошлом
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'weather_fetcher',
    default_args=default_args,
    description='Сбор погоды в Москве каждую минуту',
    schedule_interval='* * * * *',   # каждую минуту
    catchup=False,
    tags=['weather'],
)

fetch_task = PythonOperator(
    task_id='fetch_and_save_weather',
    python_callable=update_weather,
    dag=dag,
)
