from torch import nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, env, num_hidden=64):
        nn.Module.__init__(self)
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.l1 = nn.Linear(self.state_space, num_hidden)
        self.l2 = nn.Linear(num_hidden, self.action_space)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))