import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

import random
import time
from collections import defaultdict

import gym

from replays.BasicReplayMemory import BasicReplayMemory
from QNetwork import *
from utils import *
import plot


def train(model, memory, optimizer, batch_size, discount_factor):
    if len(memory) < batch_size:
        return None

    transitions = memory.sample(batch_size)

    state, action, reward, next_state, done = zip(*transitions)

    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)

    q_val = compute_q_val(model, state, action)

    with torch.no_grad():
        target = compute_target(model, reward, next_state, done, discount_factor)

    loss = F.smooth_l1_loss(q_val, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def run_episodes(model, memory, env, num_episodes, batch_size, discount_factor, learn_rate):
    optimizer = optim.Adam(model.parameters(), learn_rate)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    for i in range(num_episodes):
        state = env.reset()
        episode_length = 0

        done = False
        while not done:
            epsilon = get_epsilon(global_steps)

            action = select_action(model, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            memory.push((state, action, reward, next_state, done))

            loss = train(model, memory, optimizer, batch_size, discount_factor)

            state = next_state
            episode_length += 1
            global_steps += 1

        episode_durations.append(episode_length)

    print(global_steps)
    return episode_durations


def main():
    num_episodes = 100
    batch_size = 64
    discount_factor = 0.8
    learn_rate = 1e-3
    num_hidden = 128

    seed = 42  # This is not randomly chosen
    random.seed(seed)
    torch.manual_seed(seed)

    env = gym.envs.make('CartPole-v0')
    env.seed(seed)

    memory = BasicReplayMemory(10000)
    model = QNetwork(num_hidden)

    episode_durations = run_episodes(model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)
    plot.episode_durations(episode_durations)


if __name__ == "__main__":
    main()
