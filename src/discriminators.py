import torch
from torch import nn
from utils import generate_statistics
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def compute_statistics(X, batch_size, x_lengths):
    stats = np.zeros((batch_size, 166))
    for i in range(batch_size):
        x_len = x_lengths[i]
        stat_i = generate_statistics(X[i, :x_len].cpu().numpy())
        stats[i, :] = stat_i
    return torch.tensor(stats, dtype=torch.float32)


def convert_records_to_stat(X: list, y: list, MAX_UNIT: int, MAX_DELAY: int):
    stats_record = []
    y_list = []
    for idx, xi in enumerate(tqdm(X)):
        x_lengths = [xi.size(0)]
        yi = y[idx]
        y_list += [yi.view(1, )]
        xi[:, 0] = xi[:, 0] * MAX_UNIT
        xi[:, 1] = xi[:, 1] * MAX_DELAY
        xi = xi.unsqueeze(dim=0)
        f = compute_statistics(xi, 1, x_lengths)
        stats_record.append(f)
    stats_record = torch.cat(stats_record, dim=0)
    y_list = torch.cat(y_list, dim=0)
    return stats_record.numpy(), y_list.numpy()


class RecurrentDiscriminator(nn.Module):
    def __init__(self, device, input_size=2, hidden_size=256, num_layers=2):
        super(RecurrentDiscriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.output_class = 1

        self.model = None
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size * 2, self.output_class)
        )

    def init_hidden(self, batch_size):
        raise NotImplementedError

    def forward(self, X, batch_size=1, x_lengths=None):
        """

        :param X: shape (batch_size, seq_len, input_size)
        :return:
        generated seq: shape (batch_size, seq_len, output_class)
        """

        if batch_size > 1 and x_lengths is not None:
            hidden_states = self.init_hidden(batch_size)
            max_len = int(np.max(x_lengths))
            batch = torch.zeros((batch_size, max_len, self.input_size), device=self.device)
            for i in range(batch_size):
                x_len = x_lengths[i]
                batch[i, :x_len] = X[i, :x_len]
            batch_x = torch.nn.utils.rnn.pack_padded_sequence(batch, x_lengths, batch_first=True, enforce_sorted=False)
            out, hidden_states = self.model(batch_x, hidden_states)
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            last_indices = torch.LongTensor(x_lengths) - 1
            last_indices = last_indices.to(self.device)
            last_indices = last_indices.unsqueeze(-1)
            last_indices = last_indices.repeat(1, self.hidden_size)
            last_indices = last_indices.unsqueeze(1)
            last_out = torch.gather(out, dim=1, index=last_indices)
            last_out = last_out.squeeze(dim=1)
            last_out = last_out.contiguous()
            pred = torch.sigmoid(self.output_fc(last_out))
            return pred.view(-1, )
        else:
            hidden_states = self.init_hidden(batch_size)
            out, hidden_states = self.model(X, hidden_states)
            out = out.contiguous()  # (batch_size, seq_len, vocab_size), (1, seq_len, hidden)
            last_out = out[:, -1].view(1, -1)
            pred = torch.sigmoid(self.output_fc(last_out))
            return pred.view(-1, )


class LSTMDiscriminator(RecurrentDiscriminator):
    def __init__(self, device, input_size=2, hidden_size=256, num_layers=2):
        super(LSTMDiscriminator, self).__init__(device, input_size, hidden_size, num_layers)
        self.model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def init_hidden(self, batch_size):
        init_h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        init_c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        return init_h, init_c


class SkDiscriminator(nn.Module):
    def __init__(self, device, scalar, MAX_UNIT, MAX_DELAY):
        super(SkDiscriminator, self).__init__()
        self.model = None
        self.device = device
        self.scalar = np.load(scalar)
        self.scalar_mean = self.scalar[0]
        self.scalar_std = self.scalar[1]
        self.MAX_UNIT = MAX_UNIT
        self.MAX_DELAY = MAX_DELAY

    def fit(self, train_X, train_y):
        # * MAX UNIT in the method below
        stat_X, aug_y = convert_records_to_stat(train_X, train_y, self.MAX_UNIT, self.MAX_DELAY)
        # normalize
        normed_X = (stat_X - self.scalar_mean) / (self.scalar_std + 1e-8)
        self.model.fit(normed_X, aug_y)
        pred_y = self.model.predict(normed_X)
        return pred_y, aug_y

    def forward(self, X):
        X[..., 0] = X[..., 0] * self.MAX_UNIT
        X[..., 1] = X[..., 1] * self.MAX_DELAY
        x_len = [X.size(1)]
        stat_X = compute_statistics(X, 1, x_len).numpy()
        normed_stat_X = (stat_X - self.scalar_mean) / (self.scalar_std + 1e-8)
        pred_y = torch.tensor(self.model.predict(normed_stat_X), device=self.device)
        return pred_y


class RandomForestDiscriminator(SkDiscriminator):
    def __init__(self, device, scalar, MAX_UNIT, MAX_DELAY):
        super(RandomForestDiscriminator, self).__init__(device, scalar, MAX_UNIT, MAX_DELAY)
        self.model = RandomForestClassifier()


class DecisionTreeDiscriminator(SkDiscriminator):
    def __init__(self, device, scalar, MAX_UNIT, MAX_DELAY):
        super(DecisionTreeDiscriminator, self).__init__(device, scalar, MAX_UNIT, MAX_DELAY)
        self.model = DecisionTreeClassifier()


class CUMULDisciminator(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = SVC(kernel="rbf")
        self.device = device
        self.input_size = 2
        self.max_seq_len = 100

    def fit(self, train_X, train_y):
        x_lengths = [len(x) for x in train_X]
        x_lengths = np.minimum(x_lengths, self.max_seq_len)
        batch_size = len(x_lengths)
        padded_x = np.zeros((batch_size, self.max_seq_len, self.input_size))
        true_y = np.zeros((batch_size,))
        for i in range(len(train_X)):
            x_i = train_X[i][:x_lengths[i]]
            padded_x[i, :len(x_i)] = x_i[:len(x_i), :self.input_size].cpu().numpy()
            true_y[i] = train_y[i].cpu().numpy()
        padded_x = padded_x.reshape((batch_size, -1))  # (batch_size, 200)
        self.model.fit(padded_x, train_y)
        pred_y = self.model.predict(padded_x)
        return pred_y, true_y

    def forward(self, X):
        padded_x = np.zeros((self.max_seq_len, self.input_size))
        x_len = X.size(1)
        padded_x[:x_len] = X[0].cpu().numpy()
        padded_x = padded_x.reshape((1, -1))
        pred_y = torch.tensor(self.model.predict(padded_x), device=self.device)
        return pred_y


class DFDiscriminator(nn.Module):
    def __init__(self, device, input_size=2):
        super(DFDiscriminator, self).__init__()
        self.input_size = input_size
        self.device = device
        self.output_class = 1
        self.max_seq_len = 60
        dropout = 0.5
        channel_num = [16, 32, 64, 128]
        kernel_sizes = [8, 8, 8, 8]
        block_num = 4
        self.blocks = []
        for i in range(block_num):
            if i == 0:
                in_c = self.input_size
            else:
                in_c = channel_num[i - 1]
            out_c = channel_num[i]
            block = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_sizes[i], padding="same"),
                nn.BatchNorm1d(num_features=out_c),
                nn.ReLU(),
                nn.Conv1d(out_c, out_c, kernel_sizes[i], padding="same"),
                nn.BatchNorm1d(num_features=out_c),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.blocks.append(block)
        self.blocks = torch.nn.ModuleList(self.blocks)

        self.output_fc = nn.Sequential(
            nn.Linear(channel_num[-1] * self.max_seq_len, 512),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(dropout),
            nn.Linear(512, self.output_class)
        )

    def forward(self, X, batch_size=1, x_lengths=None):
        if batch_size == 1:
            x_len = X.size(1)
            padded_x = torch.zeros((1, self.max_seq_len, self.input_size), device=self.device)
            if x_len >= self.max_seq_len:
                padded_x[:, :self.max_seq_len] = X[:, :self.max_seq_len, :self.input_size]
            elif x_len < self.max_seq_len:
                padded_x[:, :x_len] = X[:, :x_len, :self.input_size]
        else:
            x_lengths = np.minimum(x_lengths, self.max_seq_len)
            augmented_batch_size = batch_size
            padded_x = torch.zeros((augmented_batch_size, self.max_seq_len, self.input_size), device=self.device)
            for i in range(X.size(0)):
                x_i = X[i, :x_lengths[i]]
                padded_x[i, :len(x_i)] = x_i
        padded_x = torch.permute(padded_x, [0, 2, 1])
        out = padded_x
        for block in self.blocks:
            out = block(out)
        out = out.flatten(1, 2)
        out = self.output_fc(out)
        return torch.sigmoid(out).view(-1, )


class SDAE(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.max_seq_len = 60
        self.input_size = 2
        self.encode_sizes = [100, 50, 30]
        self.decode_sizes = [50, 100]

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size * self.max_seq_len, 100),
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 30),
            nn.Tanh()
        )

        self.classifier = nn.Linear(30, 1)
        self.decoder = nn.Sequential(
            nn.Linear(30, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, self.input_size * self.max_seq_len)
        )

    def forward(self, X, batch_size=1, x_lengths=None):
        if batch_size == 1:
            x_len = X.size(1)
            padded_x = torch.zeros((1, self.max_seq_len, self.input_size), device=self.device)
            if x_len >= self.max_seq_len:
                padded_x[:, :self.max_seq_len] = X[:, :self.max_seq_len]
            elif x_len < self.max_seq_len:
                padded_x[:, :x_len] = X[:, :x_len]
            padded_x = torch.permute(padded_x, [0, 2, 1])
            padded_x = padded_x.reshape(1, -1)
        else:
            x_lengths = np.minimum(x_lengths, self.max_seq_len)
            padded_x = torch.zeros((batch_size, self.max_seq_len, self.input_size), device=self.device)
            for i in range(X.size(0)):
                x_i = X[i, :x_lengths[i]]
                padded_x[i, :len(x_i)] = x_i
            padded_x = torch.permute(padded_x, [0, 2, 1])
            padded_x = padded_x.reshape(batch_size, -1)
        latent_rep = self.encoder(padded_x)
        pred = torch.sigmoid(self.classifier(latent_rep)).view(-1)
        recon = self.decoder(latent_rep)
        return pred, recon, padded_x


if __name__ == "__main__":
    X = torch.randn(1, 40, 2)
    model = DFDiscriminator(torch.device("cpu"))
    model.eval()
    y = model(X, batch_size=1, x_lengths=[1])
    print(y)
