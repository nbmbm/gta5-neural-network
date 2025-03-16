import os
import time
import numpy as np
import h5py
import cv2
from tqdm import tqdm
from pynput import keyboard
import threading

from utils import capture_screen, preprocess_image

# Глобальные переменные для отслеживания нажатых клавиш
keys_pressed = set()
recording = False
current_action = 0

# Словарь для преобразования клавиш в действия
key_to_action = {
    keyboard.Key.up: 1,         # Вперед
    keyboard.Key.left: 2,       # Влево
    keyboard.Key.right: 3,      # Вправо
    keyboard.Key.down: 4,       # Назад
    keyboard.KeyCode.from_char('e'): 7,  # Взаимодействие
}

# Комбинации клавиш
combined_actions = {
    frozenset([keyboard.Key.up, keyboard.Key.left]): 5,    # Вперед-влево
    frozenset([keyboard.Key.up, keyboard.Key.right]): 6,   # Вперед-вправо
}

def on_press(key):
    """Обработчик событий нажатия клавиш"""
    global keys_pressed, current_action, recording
    
    # Проверяем, не клавиша ли это для начала/остановки записи
    if key == keyboard.Key.f9:
        recording = not recording
        print(f"Запись {'включена' if recording else 'остановлена'}")
        return
    
    # Выход из программы по F10 вместо Esc
    if key == keyboard.Key.f10:
        print("Нажата клавиша F10. Завершение записи...")
        return False
    
    # Добавляем клавишу в набор нажатых клавиш
    if key in key_to_action:
        keys_pressed.add(key)
    
    # Проверяем комбинации клавиш
    for keys_combo, action in combined_actions.items():
        if keys_combo.issubset(keys_pressed):
            current_action = action
            return
    
    # Если есть хотя бы одна клавиша в наборе
    if keys_pressed:
        # Берем первую клавишу из набора
        first_key = next(iter(keys_pressed))
        current_action = key_to_action.get(first_key, 0)
    else:
        current_action = 0  # Ничего не нажато

def on_release(key):
    """Обработчик событий отпускания клавиш"""
    global keys_pressed, current_action
    
    # Удаляем клавишу из набора нажатых клавиш
    if key in key_to_action and key in keys_pressed:
        keys_pressed.remove(key)
    
    # Проверяем комбинации клавиш после отпускания
    for keys_combo, action in combined_actions.items():
        if keys_combo.issubset(keys_pressed):
            current_action = action
            return
    
    # Если еще есть нажатые клавиши
    if keys_pressed:
        first_key = next(iter(keys_pressed))
        current_action = key_to_action.get(first_key, 0)
    else:
        current_action = 0  # Ничего не нажато

def record_gameplay(duration_minutes=30, fps=20, output_file="gameplay_data.h5"):
    """
    Запись игрового процесса
    
    duration_minutes: продолжительность записи в минутах
    fps: частота записи кадров
    output_file: файл для сохранения данных
    """
    global recording, current_action
    
    # Создаем директорию для данных, если ее нет
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", output_file)
    
    # Инициализируем списки для хранения данных
    frames = []
    actions = []
    
    # Проверка захвата экрана
    print("Проверка захвата экрана...")
    test_frame = capture_screen()
    cv2.imwrite("test_capture.jpg", test_frame)
    print(f"Тестовый кадр сохранен в test_capture.jpg (размер: {test_frame.shape})")
    
    # Запускаем обработчики клавиатуры в отдельном потоке
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    # Автоматически начинаем запись
    recording = True
    print("Запись автоматически включена. Нажмите F9 для приостановки/возобновления записи")
    print("Нажмите F10 для завершения записи")
    
    # Основной цикл сбора данных
    frame_interval = 1.0 / fps
    start_time = time.time()
    
    with tqdm(total=duration_minutes*60*fps) as pbar:
        while len(frames) < duration_minutes * 60 * fps:
            if recording:
                # Захват экрана
                frame = capture_screen()
                
                # Сохраняем кадр и текущее действие
                frames.append(frame)
                actions.append(current_action)
                
                pbar.update(1)
                pbar.set_description(f"Записано кадров: {len(frames)}, Текущее действие: {current_action}")
            
            # Задержка для поддержания постоянной частоты кадров
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_interval - (elapsed % frame_interval))
            time.sleep(sleep_time)
            
            # Проверяем, активен ли еще поток обработки клавиатуры
            if not listener.is_alive():
                break
    
    # Останавливаем обработчик клавиатуры
    listener.stop()
    
    print(f"\nЗапись завершена. Записано {len(frames)} кадров.")
    
    # Если записаны кадры, сохраняем данные
    if len(frames) > 0:
        # Сохраняем данные в HDF5 файл
        save_to_hdf5(frames, actions, output_path)
        return frames, actions
    else:
        print("ОШИБКА: Не записано ни одного кадра! Проверьте настройки захвата экрана.")
        return [], []

def save_to_hdf5(frames, actions, output_file):
    """
    Сохранение данных в формате HDF5
    """
    print(f"Сохранение данных в {output_file}...")
    
    with h5py.File(output_file, 'w') as hf:
        # Сохраняем кадры
        hf.create_dataset("frames", data=np.array(frames), compression="gzip", compression_opts=9)
        
        # Сохраняем действия
        hf.create_dataset("actions", data=np.array(actions), compression="gzip")
    
    print(f"Данные успешно сохранены в {output_file}")

def load_from_hdf5(input_file):
    """
    Загрузка данных из HDF5 файла
    """
    print(f"Загрузка данных из {input_file}...")
    
    with h5py.File(input_file, 'r') as hf:
        frames = np.array(hf["frames"])
        actions = np.array(hf["actions"])
    
    print(f"Загружено {len(frames)} кадров и {len(actions)} действий")
    
    return frames, actions

if __name__ == "__main__":
    # Пример запуска записи игрового процесса
    print("Запуск записи игрового процесса...")
    print("Инструкция:")
    print("1. Запустите GTA 5")
    print("2. Переключитесь на игру")
    print("3. Нажмите F9 для приостановки/возобновления записи (запись начнется автоматически)")
    print("4. Играйте как обычно")
    print("5. Нажмите F10 для завершения записи")
    
    duration = int(input("Введите продолжительность записи в минутах (по умолчанию 30): ") or "30")
    output_file = input("Введите имя выходного файла (по умолчанию gameplay_data.h5): ") or "gameplay_data.h5"
    
    record_gameplay(duration_minutes=duration, output_file=output_file) 