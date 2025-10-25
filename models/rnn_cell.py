import torch
import torch.nn as nn


class RNNCell(nn.Module):
   
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.Wxh = nn.Parameter(torch.randn(input_size, hidden_size) * (1.0 / hidden_size**0.5))
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * (1.0 / hidden_size**0.5))
        self.Why = nn.Parameter(torch.randn(hidden_size, output_size) * (1.0 / hidden_size**0.5))
        self.bh = nn.Parameter(torch.zeros(hidden_size))
        self.by = nn.Parameter(torch.zeros(output_size))

    def forward(self, x, h_prev):
        h_t = torch.tanh(x @ self.Wxh + h_prev @ self.Whh + self.bh)
        y_t = h_t @ self.Why + self.by
        return y_t, h_t
