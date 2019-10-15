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


def run_episodes(model, memory, env, num_episodes, batch_size, discount_factor, learn_rate, optim_steps):
    optimizer = optim.Adam(model.parameters(), learn_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []  #
    max_positions = []
    max_position = -1
    max_positions_per_ep = []
    successes = 0
    losses = []
    for i in range(num_episodes):
        state = env.reset()
        episode_length = 0
        print(f"episode {i}")
        ep_max_position = -1
        done = False
        while not done:
            epsilon = get_epsilon(global_steps, successes)

            action = select_action(model, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            memory.push((state, action, reward, next_state, done))

            state = next_state
            if state[0] > max_position:
                max_position = state[0]
            if state[0] > ep_max_position:
                ep_max_position = state[0]
            episode_length += 1
            global_steps += 1

        loss = 0
        for _ in range(optim_steps):
            loss += train(model, memory, optimizer, batch_size, discount_factor)

        if state[0] > 0.5:
            successes += 1
            scheduler.step()

        episode_durations.append(episode_length)
        losses.append(loss / optim_steps)
        max_positions.append(max_position)
        max_positions_per_ep.append(ep_max_position)

    print(global_steps)
    return episode_durations, max_positions, max_positions_per_ep


def main():
    num_episodes = 5000
    batch_size = 64
    discount_factor = 0.99
    learn_rate = 1e-3
    num_hidden = 256
    optimization_steps_per_episode = 50

    seed = 42  # This is not randomly chosen
    random.seed(seed)
    torch.manual_seed(seed)

    env = gym.envs.make('MountainCar-v0')
    env.seed(seed)

    memory = BasicReplayMemory(10000)
    model = QNetwork(env, num_hidden)

    episode_durations, max_pos, max_pos_per_ep = run_episodes(model, memory, env, num_episodes, batch_size, discount_factor, learn_rate, optimization_steps_per_episode)
    plot.episode_durations(episode_durations)
    plot.episode_durations(max_pos, max_pos_per_ep)


if __name__ == "__main__":
    main()
