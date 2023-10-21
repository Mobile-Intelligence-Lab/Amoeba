import torch
from torch import nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):
    def __init__(self, device, state_num, num_layers, hidden_size):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.state_num = state_num
        self.encoder = nn.GRU(
            input_size=state_num,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.decoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc_output = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(hidden_size / 2), state_num)
        )

    def init_hidden(self, batch_size):
        init_h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return init_h

    def forward(self, X):
        batch_size, seq_len, _ = X.size()

        # encoded
        h = self.init_hidden(batch_size)
        encoded, h = self.encoder(X, h)
        encoded = encoded.contiguous()
        last_encoded = encoded[:, -1, :].unsqueeze(dim=1)

        # [-1, 1]
        last_encoded = F.tanh(last_encoded)

        # decoded
        decoded = []
        last_decoded = last_encoded
        for len_i in range(seq_len):
            last_decoded, h = self.decoder(last_decoded, h)
            decoded.append(last_decoded)
        decoded = torch.cat(decoded, dim=1)
        output = self.fc_output(decoded)
        if self.state_num == 2:
            S = torch.tanh(output[:, :, 0])  # [-1, 1]
            T = torch.sigmoid(output[:, :, 1])  # [0, 1]
            normalized_output = torch.stack([S, T], dim=2)
        else:
            normalized_output = torch.tanh(output)
        return normalized_output


class StateEncoder(nn.Module):
    def __init__(self, device, state_num, num_layers, hidden_size):
        super(StateEncoder, self).__init__()
        self.num_layers = num_layers
        self.device = device
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(
            input_size=state_num,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def init_hidden(self, batch_size):
        init_h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return init_h

    def forward(self, X):
        batch_size, seq_len, state_num = X.size()

        # encoded
        h = self.init_hidden(batch_size)
        encoded, h = self.encoder(X, h)
        encoded = encoded.contiguous()
        last_encoded = encoded[:, -1, :]
        return last_encoded

    def step(self, X, h=None):
        batch_size, seq_len, state_num = X.size()
        assert seq_len == 1
        if h is None:
            h = self.init_hidden(batch_size)
        encoded, h = self.encoder(X, h)
        return encoded.squeeze(dim=1), h
