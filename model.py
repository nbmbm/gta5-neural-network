import torch
import torch.nn as nn
import torch.nn.functional as F

class GTANet(nn.Module):
    """
    Нейронная сеть для игры в GTA 5
    Принимает изображение с экрана и выдает действия
    """
    def __init__(self, n_actions=9):
        super(GTANet, self).__init__()
        
        # Сверточные слои для обработки изображений
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Вычисляем размер выхода сверточной части
        # Для изображения 320x240:
        # После conv1: (240-8)/4+1=59, (320-8)/4+1=79
        # После conv2: (59-4)/2+1=28, (79-4)/2+1=38
        # После conv3: (28-3)/1+1=26, (38-3)/1+1=36
        # Итоговый размер: 64 x 26 x 36 = 59904
        self.fc_input_size = 64 * 26 * 36
        
        # Полносвязные слои
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
    def forward(self, x):
        # x имеет размер [batch_size, 3, 240, 320]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Преобразуем в плоский вектор
        x = x.view(x.size(0), -1)
        
        # Применяем полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class GTANetWithMemory(nn.Module):
    """
    Расширенная версия модели с памятью (LSTM)
    для учета временной динамики игры
    """
    def __init__(self, n_actions=9, sequence_length=4):
        super(GTANetWithMemory, self).__init__()
        
        # Сверточные слои
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Для изображения 320x240
        # Вычисление размера выхода CNN:
        # После conv1: (240-8)/4+1=59, (320-8)/4+1=79
        # После conv2: (59-4)/2+1=28, (79-4)/2+1=38
        # После conv3: (28-3)/1+1=26, (38-3)/1+1=36
        # Итоговый размер: 64 x 26 x 36
        self.conv_output_size = 64 * 26 * 36
        
        # LSTM слой
        self.lstm = nn.LSTM(
            input_size=self.conv_output_size,
            hidden_size=512,
            num_layers=1,
            batch_first=True
        )
        
        # Выходной слой
        self.fc = nn.Linear(512, n_actions)
        
        self.sequence_length = sequence_length
        
    def forward(self, x, hidden=None):
        # x имеет размер [batch_size, sequence_length, 3, 240, 320]
        batch_size = x.size(0)
        sequence_length = x.size(1)
        
        # Изменяем форму для обработки последовательностей
        x = x.view(batch_size * sequence_length, 3, 240, 320)
        
        # Применяем сверточные слои
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Изменяем форму для LSTM
        x = x.view(batch_size, sequence_length, self.conv_output_size)
        
        # Применяем LSTM
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # Берем последний выход LSTM
        lstm_out = lstm_out[:, -1, :]
        
        # Применяем выходной слой
        output = self.fc(lstm_out)
        
        return output, hidden

    def init_hidden(self, batch_size, device):
        """Инициализация скрытого состояния LSTM"""
        return (
            torch.zeros(1, batch_size, 512).to(device),
            torch.zeros(1, batch_size, 512).to(device)
        ) 