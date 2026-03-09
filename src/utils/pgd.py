import torch
import numpy as np


class PGDAttack:
    def __init__(self, model, loss_fn, epsilon=1.0, alpha=0.1, num_steps=50,
                 device="cpu", maximize=True):
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.device = device
        self.maximize = maximize

    def attack(self, x_init, callback=None):
        x_adv = x_init.clone().detach().requires_grad_(True)
        x_orig = x_init.clone().detach()
        best_loss = float('-inf') if self.maximize else float('inf')
        best_x = x_adv.clone().detach()

        for step in range(self.num_steps):
            self.model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.zero_()

            output = self.model(x_adv)
            loss = self.loss_fn(output, x_adv)

            loss.backward()

            with torch.no_grad():
                if self.maximize:
                    x_adv = x_adv + self.alpha * x_adv.grad.sign()
                else:
                    x_adv = x_adv - self.alpha * x_adv.grad.sign()

                perturbation = torch.clamp(x_adv - x_orig, -self.epsilon, self.epsilon)
                x_adv = x_orig + perturbation

            x_adv = x_adv.detach().requires_grad_(True)

            current_loss = loss.item()
            if (self.maximize and current_loss > best_loss) or \
               (not self.maximize and current_loss < best_loss):
                best_loss = current_loss
                best_x = x_adv.clone().detach()

            if callback:
                callback(step, current_loss, best_loss)

        return best_x, best_loss
