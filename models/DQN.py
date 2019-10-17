import torch
from torch import optim
import torch.nn.functional as F
import random

from replay_memories.BasicReplayMemory import BasicReplayMemory
from QNetwork import *
import plot
from utils import *
import gym


class DQN:
    def __init__(self, env_name, replay_memory=BasicReplayMemory):
        self.num_episodes = 3000
        self.batch_size = 64
        self.discount_factor = 0.99
        self.learn_rate = 1e-3
        self.num_hidden = 256
        self.optimization_steps_per_episode = 50
        self.replay_memory_size = 10000

        self.seed = 42  # This is not randomly chosen
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.env = gym.envs.make(env_name)
        self.env.seed(self.seed)

        self.memory = replay_memory(self.replay_memory_size)
        self.model = QNetwork(self.env, self.num_hidden)
        self.optimizer = optim.Adam(self.model.parameters(), self.learn_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)

    @staticmethod
    def run(env_name):
        model, episode_durations, max_positions, max_positions_per_ep = DQN(env_name).__run_episodes()

        plot.episode_durations(episode_durations)
        plot.episode_durations(max_positions, max_positions_per_ep)
        plot.visualize_policy(model)
        
    def __train(self):
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)

        state, action, reward, next_state, done = zip(*transitions)

        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.int64)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.uint8)

        q_val = compute_q_val(self.model, state, action)

        with torch.no_grad():
            target = compute_target(self.model, reward, next_state, done, self.discount_factor)

        loss = F.smooth_l1_loss(q_val, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def __run_episodes(self):
        global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
        episode_durations = []
        max_positions = []
        max_position = -1
        max_positions_per_ep = []
        successes = 0
        losses = []
        for i in range(self.num_episodes):
            state = self.env.reset()
            episode_length = 0
            print(f"episode {i}")
            ep_max_position = -1
            done = False
            while not done:
                epsilon = get_epsilon(global_steps, successes)

                action = select_action(self.model, state, epsilon)
                next_state, reward, done, _ = self.env.step(action)

                self.memory.push((state, action, reward, next_state, done))

                state = next_state
                if state[0] > max_position:
                    max_position = state[0]
                if state[0] > ep_max_position:
                    ep_max_position = state[0]
                episode_length += 1
                global_steps += 1

            loss = 0
            for _ in range(self.optimization_steps_per_episode):
                loss += self.__train()

            if state[0] > 0.5:
                successes += 1
                self.scheduler.step()

            episode_durations.append(episode_length)
            losses.append(loss / self.optimization_steps_per_episode)
            max_positions.append(max_position)
            max_positions_per_ep.append(ep_max_position)

        print(global_steps)
        return self.model, episode_durations, max_positions, max_positions_per_ep


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
