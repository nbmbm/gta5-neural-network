import cv2
import numpy as np
import mss
import time
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
import torch

# Константы
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FRAME_RATE = 20
ACTION_REPEAT = 3  # Сколько кадров выполнять одно и то же действие

# Инициализация контроллеров клавиатуры и мыши
keyboard = KeyboardController()
mouse = MouseController()

def capture_screen():
    """
    Захват изображения с экрана
    """
    try:
        with mss.mss() as sct:
            # Настройка для захвата полного экрана GTA 5 в оконном режиме без рамки (1920x1080)
            monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
            img = np.array(sct.grab(monitor))
            img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    except Exception as e:
        print(f"Ошибка при захвате экрана: {e}")
        # Возвращаем пустое изображение в случае ошибки
        return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

def preprocess_image(image):
    """
    Предобработка изображения для нейросети
    """
    # Нормализация значений пикселей до диапазона [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Перестановка осей для PyTorch (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    
    return image

def perform_action(action_id):
    """
    Выполнение действия в игре
    action_id: целое число, соответствующее действию
    """
    # Сначала сбрасываем все клавиши
    reset_keys()
    
    # Словарь доступных действий
    actions = {
        0: lambda: None,                       # Ничего не делать
        1: lambda: keyboard.press(Key.up),     # Вперед
        2: lambda: keyboard.press(Key.left),   # Влево
        3: lambda: keyboard.press(Key.right),  # Вправо
        4: lambda: keyboard.press(Key.down),   # Назад
        5: lambda: (keyboard.press(Key.up), keyboard.press(Key.left)),    # Вперед-влево
        6: lambda: (keyboard.press(Key.up), keyboard.press(Key.right)),   # Вперед-вправо
        7: lambda: keyboard.press('e'),        # Взаимодействие
        8: lambda: mouse.press(Button.left),   # Стрельба
    }
    
    if action_id in actions:
        actions[action_id]()

def reset_keys():
    """
    Сбрасывает все клавиши
    """
    keyboard.release(Key.up)
    keyboard.release(Key.down)
    keyboard.release(Key.left)
    keyboard.release(Key.right)
    keyboard.release('e')
    mouse.release(Button.left)

def create_action_sequence(frames, sequence_length=4):
    """
    Создает последовательность кадров для подачи в LSTM
    """
    if len(frames) < sequence_length:
        # Если у нас недостаточно кадров, дублируем последний
        padding = [frames[-1]] * (sequence_length - len(frames))
        sequence = frames + padding
    else:
        sequence = frames[-sequence_length:]
    
    return np.array(sequence)

def calculate_reward(prev_state=None, curr_state=None):
    """
    Простая функция награды
    В реальном применении здесь будет более сложная логика
    """
    # Базовая награда за продолжение игры
    reward = 0.1
    
    # Здесь можно добавить более сложную логику вычисления награды
    # Например, детекция столкновений, прогресс миссии и т.д.
    
    return reward

def epsilon_greedy_policy(q_values, epsilon):
    """
    Эпсилон-жадная политика выбора действия
    """
    if np.random.random() < epsilon:
        # Случайное действие
        return np.random.randint(0, q_values.shape[-1])
    else:
        # Лучшее действие
        return np.argmax(q_values.cpu().numpy())

def save_model(model, optimizer, epoch, filename):
    """
    Сохранение модели и состояния оптимизатора
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)
    
def load_model(model, optimizer, filename, device):
    """
    Загрузка модели и состояния оптимизатора
    """
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch 