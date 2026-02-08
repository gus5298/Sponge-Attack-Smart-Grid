import torch
import numpy as np

from config import CONTEXT_LEN, NUM_FEATURES


def load_adversarial_inputs(file_map):
    results = {}
    for name, path in file_map.items():
        try:
            data = np.load(path).reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
            results[name] = data
        except FileNotFoundError:
            pass
    return results


class ACTEnergyWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        pred, ponder_cost = self.model(x)
        return ponder_cost + pred.abs().sum(dim=1, keepdim=True) * 0.01


class ACTLatencyWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        pred, ponder_cost = self.model(x)
        return pred.abs().sum(dim=1, keepdim=True) + pred.var(dim=1, keepdim=True) * 10


class DeepAREnergyWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return output.abs().sum(dim=1, keepdim=True) + (output ** 2).sum(dim=1, keepdim=True) * 0.1


def act_energy_proxy(x, model, device):
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        if len(x_tensor.shape) == 2:
            x_tensor = x_tensor.unsqueeze(0)
        pred, ponder = model(x_tensor)
        energy = ponder + pred.abs().sum(dim=1) * 0.01
        return energy.cpu().numpy()


def deepar_energy_proxy(x, model, device):
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        if len(x_tensor.shape) == 2:
            x_tensor = x_tensor.unsqueeze(0)
        output = model(x_tensor)
        energy = output.abs().sum(dim=1) + (output ** 2).sum(dim=1) * 0.1
        return energy.cpu().numpy()
