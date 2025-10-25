import torch
import torch.nn as nn
from .custom_lstm import CustomLSTM
from .attention import CustomAttention


class LSTMAttention(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, output_dim=1, num_layers=1, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm_encoder = CustomLSTM(input_dim, hidden_dim, output_dim, num_layers, bidirectional)
        self.attention = CustomAttention(hidden_dim, bidirectional)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, x):
        lstm_out, h_n, c_n = self.lstm_encoder(x)
        if self.bidirectional:
            final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            final_hidden = h_n[-1]
        context, attn_weights = self.attention(lstm_out, final_hidden)
        out = self.fc(context)
        return out, attn_weights
