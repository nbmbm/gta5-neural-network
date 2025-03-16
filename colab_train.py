"""
Скрипт для запуска обучения нейросети для GTA 5 в Google Colab
"""

import os
from google.colab import drive
import subprocess
import sys

def setup_colab():
    """
    Настройка окружения Google Colab для обучения
    """
    print("Подключение к Google Drive...")
    drive.mount('/content/drive')
    
    print("Проверка наличия репозитория...")
    if not os.path.exists('/content/gta5-neural-network'):
        print("Клонирование репозитория...")
        subprocess.run(['git', 'clone', 'https://github.com/nbmbm/gta5-neural-network.git'], check=True)
    
    os.chdir('/content/gta5-neural-network')
    print(f"Текущая директория: {os.getcwd()}")
    
    print("Установка необходимых библиотек...")
    subprocess.run(['pip', 'install', 'torch', 'torchvision', 'numpy==1.23.3', 
                   'opencv-python==4.6.0.66', 'matplotlib', 'tqdm==4.64.1', 'h5py'], check=True)
    
    print("Подготовка данных...")
    if not os.path.exists('data'):
        os.makedirs('data', exist_ok=True)
    
    # Проверка наличия файла с данными
    if not os.path.exists('data/gameplay_data.h5'):
        print("Копирование данных из Google Drive...")
        drive_data_path = input("Введите путь к файлу gameplay_data.h5 в Google Drive (например, /content/drive/MyDrive/gameplay_data.h5): ")
        if os.path.exists(drive_data_path):
            subprocess.run(['cp', drive_data_path, 'data/gameplay_data.h5'], check=True)
            print("Данные успешно скопированы!")
        else:
            print(f"Файл {drive_data_path} не найден!")
            print("Пожалуйста, загрузите файл gameplay_data.h5 на Google Drive или укажите правильный путь.")
            sys.exit(1)
    
    # Создание директории для моделей
    os.makedirs('models', exist_ok=True)
    
    print("Окружение Google Colab настроено успешно!")

def train_model(model_type='lstm', epochs=50, batch_size=64):
    """
    Запуск обучения модели
    
    model_type: тип модели ('simple' или 'lstm')
    epochs: количество эпох обучения
    batch_size: размер мини-партии
    """
    print(f"Запуск обучения модели типа {model_type}...")
    
    command = [
        'python', 'train.py',
        '--mode', 'supervised',
        '--data', 'data/gameplay_data.h5',
        '--epochs', str(epochs),
        '--model_type', model_type,
        '--batch_size', str(batch_size)
    ]
    
    # Для LSTM модели добавляем длину последовательности
    if model_type == 'lstm':
        command.extend(['--sequence_length', '4'])
    
    # Запуск обучения
    subprocess.run(command, check=True)
    
    print("Обучение завершено!")
    
    # Копируем обученную модель в Google Drive
    drive_model_path = '/content/drive/MyDrive/gta5_model'
    os.makedirs(drive_model_path, exist_ok=True)
    
    print(f"Копирование модели в Google Drive ({drive_model_path})...")
    subprocess.run(['cp', 'models/final_model.pth', f'{drive_model_path}/final_model.pth'], check=True)
    subprocess.run(['cp', 'training_loss.png', f'{drive_model_path}/training_loss.png'], check=True)
    
    print("Модель успешно сохранена в Google Drive!")

if __name__ == "__main__":
    # Настройка окружения
    setup_colab()
    
    # Запуск обучения
    model_type = input("Выберите тип модели (simple/lstm) [по умолчанию: lstm]: ").strip() or 'lstm'
    epochs = int(input("Введите количество эпох [по умолчанию: 50]: ").strip() or '50')
    batch_size = int(input("Введите размер мини-партии [по умолчанию: 64]: ").strip() or '64')
    
    train_model(model_type, epochs, batch_size) 