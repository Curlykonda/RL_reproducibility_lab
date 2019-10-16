import random
import numpy as np

class PrioritizedExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.__to_update = None
        self.max_priority = 1

        self.epsilon = 1e-3

    def push(self, transition):
        if len(self.memory) == self.capacity:
            del self.memory[0]

        self.memory.append(transition + (self.max_priority,))

    def sample(self, batch_size):
        probability_distribution = self.__compute_probability_distribution()
        indexes = np.random.choice(range(self.__len__()), batch_size, replace=False, p=probability_distribution)

        self.__to_update = indexes
        return [self.memory[i] for i in indexes]

    def update_priorities(self, priorities):
        for index, priority in enumerate(priorities):
            transition = self.memory[self.__to_update[index]]
            new_transition = transition[:-1] + (priority.item() + self.epsilon,)
            self.memory[self.__to_update[index]] = new_transition

        max_priority = priorities.max().item()
        if max_priority > self.max_priority:
            self.max_priority = max_priority

    def __compute_probability_distribution(self, alpha=0.6):
        priorities = [transition[-1]**alpha for transition in self.memory]
        priority_sum = sum(priorities)
        distribution = [priority / priority_sum for priority in priorities]

        return distribution

    def __len__(self):
        return len(self.memory)
