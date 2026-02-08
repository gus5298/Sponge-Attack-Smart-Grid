import torch
import torch.nn as nn
from config import MAX_PONDERS

class ACTLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.halting_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, state):
        batch_size = x.size(0)
        h, c = state
        h_acc = torch.zeros_like(h)
        c_acc = torch.zeros_like(c)
        remainders = torch.ones(batch_size, device=x.device)
        ponder_steps = torch.zeros(batch_size, device=x.device)
        active_mask = torch.ones(batch_size, device=x.device)

        for n in range(MAX_PONDERS):
            if active_mask.sum() == 0:
                break
            h_next, c_next = self.lstm_cell(x, (h, c))
            h_prob = self.sigmoid(self.halting_layer(h_next)).squeeze(1)
            still_running = (remainders > h_prob) * active_mask
            halting = (remainders <= h_prob) * active_mask
            step_prob = torch.where(still_running == 1, h_prob, remainders)
            h_acc = h_acc + (h_next * step_prob.unsqueeze(1))
            c_acc = c_acc + (c_next * step_prob.unsqueeze(1))
            remainders = remainders - step_prob
            active_mask = still_running
            ponder_steps = ponder_steps + 1
            h, c = h_next, c_next

        return (h_acc, c_acc), ponder_steps.mean()


class ACTModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.act_cell = ACTLSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        total_ponder = 0

        for t in range(seq_len):
            input_t = x[:, t, :]
            (h, c), ponders = self.act_cell(input_t, (h, c))
            total_ponder += ponders

        out = self.fc(h)
        return out, total_ponder / seq_len
