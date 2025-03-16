import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from collections import deque
import random

# Проверяем, работаем ли в Colab
IN_COLAB = 'COLAB_GPU' in os.environ

from model import GTANet, GTANetWithMemory
if IN_COLAB:
    # Используем специальную версию утилит для Colab
    from colab_utils import preprocess_image, save_model, load_model
    from colab_data_collection import load_from_hdf5
else:
    # Используем обычную версию
    from utils import preprocess_image, save_model, load_model
    from data_collection import load_from_hdf5

# Константы
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
GAMMA = 0.99     # Коэффициент дисконтирования для будущих наград
MEMORY_SIZE = 10000  # Размер буфера опыта
EPSILON_START = 1.0  # Начальная вероятность случайного действия
EPSILON_END = 0.05   # Минимальная вероятность случайного действия
EPSILON_DECAY = 10000  # Скорость снижения epsilon
NUM_EPOCHS = 50
SEQUENCE_LENGTH = 4  # Длина последовательности для LSTM

class ExperienceBuffer:
    """
    Буфер опыта для хранения переходов (состояние, действие, награда, следующее состояние, терминальный флаг)
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Добавляет переход в буфер"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Выбирает случайную выборку из буфера"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

def create_dataset_from_gameplay(frames, actions, model_type='simple', sequence_length=4):
    """
    Создает обучающий набор данных из записанного геймплея
    
    model_type: тип модели ('simple' или 'lstm')
    sequence_length: длина последовательности для LSTM
    """
    states = []
    targets = []
    
    print("Подготовка данных для обучения...")
    
    if model_type == 'simple':
        # Для простой модели обрабатываем каждый кадр отдельно
        for i in tqdm(range(len(frames))):
            # Предобработка изображения
            state = preprocess_image(frames[i])
            states.append(state)
            
            # One-hot кодирование действия
            action_target = np.zeros(9)  # 9 возможных действий
            action_target[actions[i]] = 1
            targets.append(action_target)
        
        return np.array(states), np.array(targets)
    
    else:  # LSTM модель
        # Для LSTM модели создаем последовательности
        print(f"Создание последовательностей длиной {sequence_length} для LSTM...")
        # Минимальный индекс, с которого можно начать последовательность
        min_idx = sequence_length - 1
        
        for i in tqdm(range(min_idx, len(frames))):
            # Создаем последовательность кадров
            sequence = []
            for j in range(sequence_length):
                # Берем последние sequence_length кадров
                frame_idx = i - (sequence_length - 1) + j
                state = preprocess_image(frames[frame_idx])
                sequence.append(state)
            
            # Добавляем последовательность в набор данных
            states.append(np.array(sequence))
            
            # Действие, соответствующее последнему кадру в последовательности
            action_target = np.zeros(9)
            action_target[actions[i]] = 1
            targets.append(action_target)
        
        return np.array(states), np.array(targets)

def train_supervised(model, train_data, train_labels, epochs=10, batch_size=32, learning_rate=0.001, model_type='simple'):
    """
    Обучение модели на данных игрового процесса (имитационное обучение)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    model = model.to(device)
    
    # Создаем оптимизатор и функцию потерь
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Размер данных
    n_samples = len(train_data)
    n_batches = n_samples // batch_size
    
    # Для отслеживания потерь
    train_losses = []
    
    print(f"Начало обучения модели. Эпохи: {epochs}, Размер партии: {batch_size}, Обучающих примеров: {n_samples}")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # Перемешиваем данные
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Обучение по мини-партиям
        with tqdm(total=n_batches, desc=f"Эпоха {epoch+1}/{epochs}") as pbar:
            for i in range(n_batches):
                # Получаем индексы для текущей партии
                batch_indices = indices[i*batch_size:(i+1)*batch_size]
                
                # Формируем партию
                batch_data = torch.FloatTensor(train_data[batch_indices]).to(device)
                batch_labels = torch.LongTensor(np.argmax(train_labels[batch_indices], axis=1)).to(device)
                
                # Обнуляем градиенты
                optimizer.zero_grad()
                
                # Прямой проход
                if model_type == 'simple':
                    outputs = model(batch_data)
                else:  # LSTM модель
                    outputs, _ = model(batch_data)
                
                # Вычисляем потери
                loss = criterion(outputs, batch_labels)
                
                # Обратное распространение
                loss.backward()
                
                # Обновляем веса
                optimizer.step()
                
                # Обновляем потери
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update()
        
        # Средняя потеря за эпоху
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)
        
        print(f"Эпоха {epoch+1}/{epochs}, Средняя потеря: {avg_loss:.4f}")
        
        # Сохраняем модель каждые 5 эпох
        if (epoch + 1) % 5 == 0:
            save_model(model, optimizer, epoch, f"models/model_epoch_{epoch+1}.pth")
    
    # Сохраняем график потерь
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Потери при обучении')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.title('Динамика потерь при обучении')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    
    return model, train_losses

def train_reinforcement(model, env_function, num_episodes=1000, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=10000):
    """
    Обучение модели с подкреплением (DQN)
    
    model: модель нейронной сети
    env_function: функция, представляющая среду (GTA 5)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    model = model.to(device)
    
    # Создаем целевую сеть (для стабильности обучения)
    target_model = GTANet(9).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    
    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Буфер опыта
    memory = ExperienceBuffer(MEMORY_SIZE)
    
    # Счетчики и статистика
    steps = 0
    episode_rewards = []
    
    print(f"Начало обучения с подкреплением. Эпизоды: {num_episodes}")
    
    for episode in range(num_episodes):
        # Инициализация среды и получение начального состояния
        state = env_function()  # Здесь должна быть функция инициализации среды
        state = preprocess_image(state)
        
        done = False
        episode_reward = 0
        
        while not done:
            # Расчет эпсилон (вероятность случайного действия)
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                      np.exp(-steps / epsilon_decay)
            
            # Выбор действия (эпсилон-жадная стратегия)
            if np.random.random() < epsilon:
                # Случайное действие
                action = np.random.randint(0, 9)
            else:
                # Лучшее действие по модели
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item()
            
            # Выполнение действия и получение следующего состояния и награды
            next_state, reward, done = env_function(action)  # Здесь должна быть функция шага в среде
            next_state = preprocess_image(next_state)
            
            # Сохраняем опыт в буфере
            memory.add(state, action, reward, next_state, done)
            
            # Обновляем текущее состояние
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Обучение на мини-партии, если достаточно опыта
            if len(memory) > batch_size:
                # Выборка из буфера опыта
                batch = memory.sample(batch_size)
                
                # Распаковка выборки
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # Преобразование в тензоры
                states = torch.FloatTensor(np.array(states)).to(device)
                actions = torch.LongTensor(np.array(actions)).to(device)
                rewards = torch.FloatTensor(np.array(rewards)).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                dones = torch.FloatTensor(np.array(dones)).to(device)
                
                # Получение текущих Q-значений
                current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Получение максимальных Q-значений для следующих состояний
                with torch.no_grad():
                    max_next_q = target_model(next_states).max(1)[0]
                    target_q = rewards + gamma * max_next_q * (1 - dones)
                
                # Вычисление потерь
                loss = criterion(current_q, target_q)
                
                # Обновление весов
                optimizer.zero_grad()
                loss.backward()
                # Ограничение градиента для стабильности
                for param in model.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
            
            # Обновление целевой сети
            if steps % 1000 == 0:
                target_model.load_state_dict(model.state_dict())
        
        # Статистика за эпизод
        episode_rewards.append(episode_reward)
        
        print(f"Эпизод {episode+1}/{num_episodes}, Награда: {episode_reward:.2f}, Эпсилон: {epsilon:.4f}")
        
        # Сохранение модели
        if (episode + 1) % 10 == 0:
            save_model(model, optimizer, episode, f"models/dqn_model_episode_{episode+1}.pth")
    
    # Сохраняем график наград
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label='Награда за эпизод')
    plt.xlabel('Эпизод')
    plt.ylabel('Награда')
    plt.title('Динамика наград при обучении с подкреплением')
    plt.legend()
    plt.grid(True)
    plt.savefig('reinforcement_rewards.png')
    
    return model, episode_rewards

def main():
    parser = argparse.ArgumentParser(description='Обучение нейросети для игры в GTA 5')
    parser.add_argument('--mode', type=str, default='supervised', choices=['supervised', 'reinforcement'],
                        help='Режим обучения: supervised (имитационное) или reinforcement (с подкреплением)')
    parser.add_argument('--data', type=str, default='data/gameplay_data.h5',
                        help='Путь к файлу с данными геймплея')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Размер партии')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Скорость обучения')
    parser.add_argument('--model_type', type=str, default='simple', choices=['simple', 'lstm'],
                        help='Тип модели: simple (CNN) или lstm (CNN+LSTM)')
    parser.add_argument('--sequence_length', type=int, default=SEQUENCE_LENGTH,
                        help='Длина последовательности для LSTM модели')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Путь к предварительно обученной модели для продолжения обучения')
    
    args = parser.parse_args()
    
    # Создаем директорию для моделей, если ее нет
    os.makedirs("models", exist_ok=True)
    
    # Определяем устройство (CPU или GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Создаем модель нейросети
    if args.model_type == 'simple':
        model = GTANet(n_actions=9)
    else:
        model = GTANetWithMemory(n_actions=9, sequence_length=args.sequence_length)
    
    # Загружаем предобученную модель, если указана
    if args.load_model:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        model, optimizer, epoch = load_model(model, optimizer, args.load_model, device)
        print(f"Загружена модель из {args.load_model}, эпоха {epoch}")
    
    # Режим имитационного обучения
    if args.mode == 'supervised':
        # Загружаем данные
        print(f"Загрузка данных из {args.data}")
        frames, actions = load_from_hdf5(args.data)
        
        # Создаем набор данных
        states, targets = create_dataset_from_gameplay(frames, actions, args.model_type, args.sequence_length)
        
        # Обучаем модель
        model, losses = train_supervised(
            model, 
            states, 
            targets, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            model_type=args.model_type
        )
    
    # Режим обучения с подкреплением
    elif args.mode == 'reinforcement':
        print("Режим обучения с подкреплением временно недоступен")
        print("Для этого режима требуется интеграция с игрой GTA 5 через API или другие методы")
        print("Пожалуйста, используйте режим имитационного обучения (supervised)")
    
    # Сохраняем финальную модель
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    save_model(model, optimizer, args.epochs, "models/final_model.pth")
    print("Модель сохранена как models/final_model.pth")

if __name__ == "__main__":
    main() 