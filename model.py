import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, encoder_outputs, last_hidden):
        seq_len = encoder_outputs.size(1)
        last_expanded = last_hidden.unsqueeze(1).expand(-1, seq_len, -1)
        score = torch.tanh(self.W1(encoder_outputs) + self.W2(last_expanded))
        weights = F.softmax(self.v(score).squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, hidden_dim)
        return context, weights

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, attn_dim=64, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.attn = AttentionLayer(hidden_dim, attn_dim)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        outputs, (h_n, c_n) = self.lstm(x)  # outputs: (batch, seq_len, hidden_dim)
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        context, attn_weights = self.attn(outputs, last_hidden)
        combined = torch.cat([context, last_hidden], dim=1)
        out = self.regressor(combined)
        return out.squeeze(-1), attn_weights

class VanillaLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x):
        outputs, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.regressor(last_hidden)
        return out.squeeze(-1)


class SimpleLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(SimpleLSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # last timestep
        out = self.fc(out)

        return out.squeeze(-1)
