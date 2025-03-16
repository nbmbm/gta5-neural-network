import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

# Заголовок
nb.cells.append(new_markdown_cell('# Обучение нейросети для игры в GTA 5\n\nЭтот ноутбук предназначен для обучения нейронной сети для игры в GTA 5 с использованием Google Colab.'))

# Шаг 1
nb.cells.append(new_markdown_cell('## Шаг 1: Подключение к Google Drive\n\nПодключим Google Drive для сохранения модели и доступа к данным геймплея.'))
nb.cells.append(new_code_cell('from google.colab import drive\ndrive.mount("/content/drive")'))

# Шаг 2
nb.cells.append(new_markdown_cell('## Шаг 2: Клонирование репозитория и настройка\n\nКлонируем репозиторий с GitHub и устанавливаем необходимые библиотеки.'))
nb.cells.append(new_code_cell('# Клонирование репозитория\n!git clone https://github.com/nbmbm/gta5-neural-network.git\n%cd gta5-neural-network'))
nb.cells.append(new_code_cell('# Установка необходимых библиотек\n!pip install torch torchvision numpy==1.23.3 opencv-python==4.6.0.66 matplotlib tqdm==4.64.1 h5py'))

# Шаг 3
nb.cells.append(new_markdown_cell('## Шаг 3: Подготовка данных\n\nПеред обучением нам необходимо скопировать данные геймплея из Google Drive.'))
nb.cells.append(new_code_cell('# Создание директории для данных\n!mkdir -p data\n\n# Путь к файлу с данными в Google Drive\n# Замените путь на актуальный\ndata_path = "/content/drive/MyDrive/gameplay_data.h5"\n\n# Копирование файла из Google Drive\n!cp $data_path data/gameplay_data.h5\n\n# Проверка наличия файла\n!ls -la data/'))

# Шаг 4
nb.cells.append(new_markdown_cell('## Шаг 4: Запуск обучения\n\nТеперь запустим процесс обучения нейронной сети.'))
nb.cells.append(new_code_cell('# Обучение модели LSTM\n!python train.py --mode supervised --data data/gameplay_data.h5 --epochs 50 --model_type lstm --batch_size 64 --sequence_length 4'))

# Шаг 5
nb.cells.append(new_markdown_cell('## Шаг 5: Визуализация результатов\n\nОтображение графика потерь для оценки процесса обучения.'))
nb.cells.append(new_code_cell('import matplotlib.pyplot as plt\nfrom IPython.display import Image\n\n# Отображение графика потерь\nImage(filename="training_loss.png")'))

# Шаг 6
nb.cells.append(new_markdown_cell('## Шаг 6: Сохранение модели в Google Drive\n\nПосле обучения сохраним модель в Google Drive для дальнейшего использования.'))
nb.cells.append(new_code_cell('# Создание директории в Google Drive для сохранения модели\n!mkdir -p "/content/drive/MyDrive/gta5_model"\n\n# Копирование модели в Google Drive\n!cp models/final_model.pth "/content/drive/MyDrive/gta5_model/final_model.pth"\n!cp training_loss.png "/content/drive/MyDrive/gta5_model/training_loss.png"\n\nprint("Модель успешно сохранена в Google Drive!")'))

# Советы
nb.cells.append(new_markdown_cell('## Полезные советы\n\n1. Если у вас возникают проблемы с памятью, уменьшите размер партии (`batch_size`).\n2. Для более быстрого обучения используйте GPU (Runtime -> Change runtime type -> Hardware accelerator -> GPU).\n3. Чтобы отслеживать использование GPU, используйте команду `!nvidia-smi`.\n4. Сохраняйте промежуточные результаты в Google Drive, чтобы не потерять их в случае разрыва соединения.'))

# Добавляем метаданные для Colab
nb.metadata = {
    "colab": {
        "name": "GTA5 Neural Network Training",
        "provenance": [],
        "gpuType": "T4"
    },
    "kernelspec": {
        "display_name": "Python 3",
        "name": "python3"
    },
    "language_info": {
        "name": "python"
    },
    "accelerator": "GPU"
}

# Сохранение ноутбука
with open('gta5_neural_network.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 