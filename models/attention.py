import torch
import torch.nn as nn


class CustomAttention(nn.Module):
    def __init__(self, hidden_dim, bidirectional=True):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.total_dim = hidden_dim * self.num_directions
        self.W_a = nn.Linear(self.total_dim, hidden_dim)
        self.U_a = nn.Linear(self.total_dim, hidden_dim)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs, final_hidden_state):
        hidden_state_actual = final_hidden_state.unsqueeze(1)
        energy = torch.tanh(self.W_a(hidden_state_actual) + self.U_a(lstm_outputs))
        score = self.v_a(energy)
        attn_weight = torch.softmax(score, dim=1)
        context_vector = torch.bmm(attn_weight.transpose(1, 2), lstm_outputs).squeeze(1)
        return context_vector, attn_weight
