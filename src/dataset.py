import torch
import numpy as np
from torch.utils.data import Dataset


class RandomRecordDataset(Dataset):
    def __init__(self, state_num=2, total_num=500000, max_len=20, seed=10086):
        torch.manual_seed(seed)
        r_min = -1
        r_max = 1
        if state_num == 1:
            self.data = (r_max - r_min) * torch.rand(total_num, max_len, 1) + r_min
        else:
            self.timestamp = torch.rand(total_num, max_len, 1)
            self.packet_size = (r_max - r_min) * torch.rand(total_num, max_len, 1) + r_min
            self.data = torch.cat([self.packet_size, self.timestamp], dim=2)
        self.total_num = total_num

    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        return self.data[index]


class VariableRecordDataset(Dataset):
    def __init__(self, x_path, y_path, MAX_UNIT, MAX_DELAY, target=1):
        self.data = np.load(x_path, allow_pickle=True)
        labels = np.load(y_path, allow_pickle=True)
        self.x = []
        self.y = []
        for idx, xi in enumerate(self.data):
            label = labels[idx]
            xi = np.array(xi)
            pkt_sizes = xi[:, 0]
            timestamps = xi[:, 1]
            pkt_sizes = (pkt_sizes / MAX_UNIT).reshape((-1, 1))  # (-1, 1)
            ofr_idx = timestamps > MAX_DELAY
            timestamps[ofr_idx] = MAX_DELAY
            timestamps = (timestamps / MAX_DELAY).reshape((-1, 1))
            xi = np.concatenate([pkt_sizes, timestamps], axis=1)

            if label == target or target == -1:
                self.x.append(torch.tensor(xi, dtype=torch.float32))
                self.y.append(torch.tensor(label, dtype=torch.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class AgentTrainSet(Dataset):
    def __init__(self, x_path, MAX_UNIT, MAX_DELAY):
        self.data = np.load(x_path, allow_pickle=True)
        labels = np.zeros((self.data.shape[0],))
        self.x = []
        self.y = []
        for idx, xi in enumerate(self.data):
            label = labels[idx]
            xi = np.array(xi)
            pkt_sizes = xi[:, 0]
            timestamps = xi[:, 1]
            pkt_sizes = (pkt_sizes / MAX_UNIT).reshape((-1, 1))  # (-1, 1)
            ofr_idx = timestamps > MAX_DELAY
            timestamps[ofr_idx] = MAX_DELAY
            timestamps = (timestamps / MAX_DELAY).reshape((-1, 1))
            xi = np.concatenate([pkt_sizes, timestamps], axis=1)

            self.x.append(torch.tensor(xi, dtype=torch.float32))
            self.y.append(torch.tensor(label, dtype=torch.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
