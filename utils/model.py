# utils/model.py
import torch.nn as nn

class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Output from last time step
