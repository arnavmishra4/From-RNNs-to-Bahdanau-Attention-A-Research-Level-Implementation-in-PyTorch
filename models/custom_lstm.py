import torch
import torch.nn as nn


class CustomLSTM(nn.Module):
    def __init__(self, input_layer=1, hidden_layer=16, out_layer=1, num_layers=1, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_layer, hidden_layer, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True)
        multi = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_layer * multi, out_layer)

    def forward(self, x, hx=None):
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
        elif x.dim() == 2:
            x = x.unsqueeze(-1)
        x = x.float()
        out, (h_n, c_n) = self.lstm(x, hx)
        return out, h_n, c_n
