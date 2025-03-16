import argparse
import os
import sys
import subprocess

def main():
    """
    Главная функция для запуска различных компонентов системы
    """
    parser = argparse.ArgumentParser(description='GTA 5 AI - обучение нейросети для игры в GTA 5')
    
    parser.add_argument('command', choices=['collect', 'train', 'play', 'setup'],
                        help='Команда для выполнения: collect - сбор данных, train - обучение модели, play - запуск игры, setup - установка зависимостей')
    
    # Аргументы для сбора данных
    parser.add_argument('--duration', type=int, default=30,
                        help='Продолжительность записи в минутах (для команды collect)')
    parser.add_argument('--output', type=str, default='gameplay_data.h5',
                        help='Имя выходного файла (для команды collect)')
    
    # Аргументы для обучения
    parser.add_argument('--mode', type=str, default='supervised',
                        choices=['supervised', 'reinforcement'],
                        help='Режим обучения (для команды train)')
    parser.add_argument('--data', type=str, default='data/gameplay_data.h5',
                        help='Путь к данным для обучения (для команды train)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Количество эпох обучения (для команды train)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Размер партии для обучения (для команды train)')
    parser.add_argument('--model_type', type=str, default='simple',
                        choices=['simple', 'lstm'],
                        help='Тип модели: simple (CNN) или lstm (CNN+LSTM)')
    
    # Аргументы для запуска игры
    parser.add_argument('--model', type=str, default='models/final_model.pth',
                        help='Путь к модели для запуска игры (для команды play)')
    parser.add_argument('--visualize', action='store_true',
                        help='Показывать визуализацию при запуске игры (для команды play)')
    
    args = parser.parse_args()
    
    # Проверка наличия папок
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Выполнение соответствующей команды
    if args.command == 'setup':
        print("Установка зависимостей...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("Зависимости установлены успешно!")
        
        # Проверка наличия CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"CUDA доступен. Найдено устройств: {torch.cuda.device_count()}")
                print(f"Текущее устройство: {torch.cuda.get_device_name(0)}")
            else:
                print("CUDA недоступен. Обучение будет выполняться на CPU, что намного медленнее.")
                print("Для ускорения рекомендуется установить CUDA.")
        except ImportError:
            print("Ошибка импорта PyTorch. Проверьте установку.")
    
    elif args.command == 'collect':
        print(f"Запуск сбора данных на {args.duration} минут. Выходной файл: {args.output}")
        from data_collection import record_gameplay
        record_gameplay(duration_minutes=args.duration, output_file=args.output)
    
    elif args.command == 'train':
        print(f"Запуск обучения модели в режиме {args.mode}. Данные: {args.data}")
        # Формируем команду для train.py с нужными аргументами
        cmd = [
            sys.executable, 'train.py',
            '--mode', args.mode,
            '--data', args.data,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--model_type', args.model_type
        ]
        subprocess.check_call(cmd)
    
    elif args.command == 'play':
        print(f"Запуск игры с моделью {args.model}")
        # Формируем команду для play.py с нужными аргументами
        cmd = [
            sys.executable, 'play.py',
            '--model', args.model,
            '--model_type', args.model_type
        ]
        if args.visualize:
            cmd.append('--visualize')
        subprocess.check_call(cmd)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрограмма остановлена пользователем")
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        sys.exit(1) 