import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=256, lstm_layers=2, linear_dims=[256, 128, 64, 1], dropout=0.3):
        super(BiLSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        # Linear layer dimensions: [in1 -> out1 -> out2 -> out3 -> 1]
        fc_layers = []
        prev_dim = hidden_dim * 2  # because of bidirectional
        for dim in linear_dims[:-1]:
            fc_layers.append(nn.Linear(prev_dim, dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        # Final layer (binary classification)
        fc_layers.append(nn.Linear(prev_dim, linear_dims[-1]))

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # x shape: (batch_size, seq_len=52, input_dim=1280)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim * 2)

        # Option 1: Take only the last hidden state from both directions (shape: (batch_size, hidden_dim * 2))
        last_out = torch.cat([lstm_out[:, -1, :self.lstm.hidden_size], # last hidden state of forward LSTM
                              lstm_out[:, 0, self.lstm.hidden_size:]], dim=1) # last hidden state of backward LSTM

        out = self.fc(last_out)
        return torch.sigmoid(out).squeeze(1)  # shape: (batch_size,)
