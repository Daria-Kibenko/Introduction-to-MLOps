import os
import csv
import requests
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")  # имя переменной должно быть таким
CITY = "Moscow"
CSV_FILE = "weather.csv"


def fetch_weather(api_key: str, city: str = "Moscow") -> dict:
    """Вызывает OpenWeatherMap API и возвращает сырые данные (JSON)."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()  # выбросит исключение при плохом статусе
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к API: {e}")
        return None


def extract_weather_data(raw_data: dict, city: str = "Moscow") -> dict:
    """Извлекает из сырого ответа API нужные поля."""
    if not raw_data:
        return None
    # Безопасно получаем значения, используя .get()
    weather_main = raw_data.get("weather", [{}])[0].get("description", "")
    temperature = raw_data.get("main", {}).get("temp")
    humidity = raw_data.get("main", {}).get("humidity")
    wind_speed = raw_data.get("wind", {}).get("speed")

    # Проверяем, что все ключевые данные есть
    if None in (temperature, humidity, wind_speed):
        print("Не все данные погоды получены, пропускаем запись.")
        return None

    return {
        "city": city,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def append_weather_to_csv(data: dict):
    """Добавляет одну строку в CSV-файл."""
    if not data:
        return
    headers = ["city", "temperature", "humidity", "wind_speed", "timestamp"]
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


def update_weather():
    """Главная функция, которая будет вызываться из DAG."""
    raw = fetch_weather(API_KEY, CITY)
    if raw:
        extracted = extract_weather_data(raw, CITY)
        if extracted:
            append_weather_to_csv(extracted)
            print(f"Данные записаны: {extracted}")
            return
    print("Не удалось получить или обработать данные погоды.")


if __name__ == "__main__":
    # Для локального тестирования
    load_dotenv(".env-example")
    test_key = os.getenv("OPENWEATHER_API_KEY")
    if not test_key:
        print("API ключ не найден. Проверьте .env файл.")
    else:
        update_weather()

