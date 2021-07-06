import torch
import torch.nn as nn


class GestureClassifier(nn.Module):
    def __init__(self, input_size=20):
        super(GestureClassifier, self).__init__()
        self.input_size = input_size

        self.relu = nn.ReLU()

        hidden_size = 32
        self.hidden_size = hidden_size
        self.accel_lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, num_layers=5, batch_first=True)
        self.gyro_lstm = nn.LSTM(input_size=3, hidden_size=hidden_size, num_layers=5, batch_first=True)

        self.fc_1 = nn.Linear(input_size * hidden_size * 2, 128)  # fully connected 1
        self.fc_2 = nn.Linear(128, 5)  # fully connected last layer

    def forward(self, accel, gyro):
        accel_x, (_, _) = self.accel_lstm(accel)
        gyro_x, (_, _) = self.gyro_lstm(gyro)
        x = torch.cat((accel_x, gyro_x), dim=1)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)

        return x
