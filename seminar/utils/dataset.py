import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, context_len, prediction_len):
        self.data = torch.FloatTensor(data) if not isinstance(data, torch.Tensor) else data
        self.context_len = context_len
        self.prediction_len = prediction_len
        self.n_samples = len(data) - context_len - prediction_len + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.context_len]
        y = self.data[idx + self.context_len:idx + self.context_len + self.prediction_len, 0]
        return x, y
