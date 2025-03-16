import numpy as np
import torch

# Константы
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FRAME_RATE = 20
ACTION_REPEAT = 3

def preprocess_image(image):
    """
    Предобработка изображения для нейросети
    """
    # Нормализация значений пикселей до диапазона [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Перестановка осей для PyTorch (C, H, W)
    image = np.transpose(image, (2, 0, 1))
    
    return image

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
    """
    return 0.1

def epsilon_greedy_policy(q_values, epsilon):
    """
    Эпсилон-жадная политика выбора действия
    """
    if np.random.random() < epsilon:
        return np.random.randint(0, q_values.shape[-1])
    else:
        return np.argmax(q_values.cpu().numpy()) 