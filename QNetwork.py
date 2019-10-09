import torch
from torch import nn
import torch.nn.functional as F

import random

class QNetwork(nn.Module):
    def __init__(self, num_hidden=64):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(2, num_hidden)
        self.l2 = nn.Linear(num_hidden, 3)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))


def compute_q_val(model, state, action):
    out = model(state).gather(1, action.unsqueeze(1))
    return out.squeeze()


def compute_target(model, reward, next_state, done, discount_factor):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
    next_q_values = model(next_state)
    discounted_maxq_values = discount_factor * next_q_values.max(1)[0] * (1 - done).float()
    target = reward + discounted_maxq_values

    return target


def select_action(model, state, epsilon):
    with torch.no_grad():
        q_values = model.forward(torch.FloatTensor(state))

        if random.random() > epsilon:
            greedy_action = q_values.max(0)[1].item()
            return greedy_action

        return random.randrange(len(q_values))
