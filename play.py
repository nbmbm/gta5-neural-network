import os
import time
import torch
import numpy as np
import argparse
import cv2
from collections import deque

from model import GTANet, GTANetWithMemory
from utils import capture_screen, preprocess_image, perform_action, create_action_sequence, load_model, FRAME_RATE
from data_collection import on_press, on_release
from pynput import keyboard

# Константы
ACTION_DELAY = 1.0 / FRAME_RATE
SEQUENCE_LENGTH = 4  # Длина последовательности для LSTM

def play_with_model(model, model_type='simple', visualize=False):
    """
    Запускает модель для игры в GTA 5
    
    model: обученная модель
    model_type: тип модели ('simple' или 'lstm')
    visualize: отображать ли предсказанные действия
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Переключаем модель в режим оценки
    
    print("Запуск модели для игры...")
    print("Нажмите F9 для начала/остановки автоматической игры")
    print("Нажмите F10 для выхода")
    
    # Переменные для контроля
    running = False
    frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
    action_names = {
        0: "No action",           # Нет действия
        1: "Forward",             # Вперед
        2: "Left",                # Влево
        3: "Right",               # Вправо
        4: "Back",                # Назад
        5: "Forward-left",        # Вперед-влево
        6: "Forward-right",       # Вперед-вправо
        7: "Interact",            # Взаимодействие
        8: "Shoot"                # Стрельба
    }
    
    # Настраиваем обработчик клавиатуры для управления запуском/остановкой
    def on_press_control(key):
        nonlocal running
        if key == keyboard.Key.f9:
            running = not running
            print(f"Автоматическая игра {'включена' if running else 'остановлена'}")
        elif key == keyboard.Key.f10:
            running = False
            print("Нажата клавиша F10. Завершение программы...")
            return False
    
    # Запускаем обработчик клавиатуры
    listener = keyboard.Listener(on_press=on_press_control)
    listener.start()
    
    print("Ожидание запуска (нажмите F9)...")
    
    # Создаем окно для визуализации заранее, если нужно
    if visualize:
        cv2.namedWindow("GTA AI Vision", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("GTA AI Vision", 640, 480)
    
    try:
        # Основной игровой цикл
        while listener.is_alive():
            try:
                if running:
                    # Захват экрана
                    frame = capture_screen()
                    if frame is None or frame.size == 0:
                        print("Не удалось захватить экран, пропускаем кадр")
                        time.sleep(0.1)
                        continue
                    
                    # Предобработка изображения
                    processed_frame = preprocess_image(frame)
                    
                    # Добавляем кадр в буфер для LSTM моделей
                    frame_buffer.append(processed_frame)
                    
                    # Предсказание действия
                    action = 0  # По умолчанию - нет действия
                    try:
                        with torch.no_grad():
                            if model_type == 'simple':
                                # Для простой модели используем один кадр
                                state_tensor = torch.FloatTensor(processed_frame).unsqueeze(0).to(device)
                                action_probs = model(state_tensor)
                            else:
                                # Для LSTM модели используем последовательность кадров
                                if len(frame_buffer) < SEQUENCE_LENGTH:
                                    # Если недостаточно кадров, продолжаем собирать
                                    continue
                                
                                sequence = create_action_sequence(list(frame_buffer))
                                state_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
                                action_probs, _ = model(state_tensor)
                            
                            # Выбираем действие с наибольшей вероятностью
                            action = torch.argmax(action_probs).item()
                    except Exception as e:
                        print(f"Ошибка при предсказании действия: {e}")
                        action_probs = torch.zeros(1, 9)
                    
                    # Выполняем выбранное действие
                    perform_action(action)
                    
                    if visualize:
                        try:
                            # Отображаем информацию на экране
                            display_frame = frame.copy()
                            action_text = f"Action: {action_names[action]}"
                            cv2.putText(display_frame, action_text, (10, 30), 
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                            
                            # Показываем вероятности всех действий
                            probs = torch.softmax(action_probs, dim=1).cpu().numpy()[0]
                            for i, p in enumerate(probs):
                                prob_text = f"{action_names[i]}: {p:.2f}"
                                cv2.putText(display_frame, prob_text, (10, 60 + i*30), 
                                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                            
                            cv2.imshow("GTA AI Vision", display_frame)
                        except Exception as e:
                            print(f"Ошибка при отображении информации: {e}")
                    
                    # Задержка для стабильной частоты кадров
                    time.sleep(ACTION_DELAY)
                else:
                    # Если не запущено, просто ждем
                    time.sleep(0.1)
                
                # Обработка нажатия клавиш (в любом случае)
                if visualize:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # Выход по Esc
                        break
            
            except Exception as e:
                print(f"Ошибка в главном цикле: {e}")
                time.sleep(0.5)  # Пауза при ошибке
    
    except KeyboardInterrupt:
        print("Выход по запросу пользователя")
    
    finally:
        # Закрываем окно и завершаем программу
        if visualize:
            cv2.destroyAllWindows()
        
        # Убеждаемся, что все клавиши отпущены
        from utils import reset_keys
        reset_keys()
        
        listener.stop()
        print("Программа завершена")

def main():
    parser = argparse.ArgumentParser(description='Игра в GTA 5 с помощью нейросети')
    parser.add_argument('--model', type=str, default='models/final_model.pth',
                        help='Путь к файлу модели')
    parser.add_argument('--model_type', type=str, default='simple', choices=['simple', 'lstm'],
                        help='Тип модели: simple (CNN) или lstm (CNN+LSTM)')
    parser.add_argument('--visualize', action='store_true',
                        help='Показывать визуализацию действий модели')
    
    args = parser.parse_args()
    
    # Проверяем наличие модели
    if not os.path.exists(args.model):
        print(f"Ошибка: Модель {args.model} не найдена")
        print("Пожалуйста, сначала обучите модель с помощью train.py")
        return
    
    # Создаем модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_type == 'simple':
        model = GTANet(n_actions=9)
    else:
        model = GTANetWithMemory(n_actions=9, sequence_length=SEQUENCE_LENGTH)
    
    # Загружаем веса модели
    optimizer = torch.optim.Adam(model.parameters())
    model, _, _ = load_model(model, optimizer, args.model, device)
    
    print(f"Модель {args.model} загружена")
    
    # Запускаем игру с моделью
    play_with_model(model, model_type=args.model_type, visualize=args.visualize)

if __name__ == "__main__":
    main() 