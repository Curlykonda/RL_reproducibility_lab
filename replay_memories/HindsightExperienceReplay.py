import random
from copy import deepcopy

class HindsightExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.memory = []

    def __push(self, transition):
        if len(self.memory) == self.capacity:
            del self.memory[0]

        self.memory.append(transition)

    def push_to_buffer(self, transition):
        self.buffer.append(transition)

    def push_with_replay_goal(self, goal_state):
        for transition in self.buffer:
            self.__push(transition)

            if tuple(transition[3]) == tuple(goal_state):
                modified_transition = list(deepcopy(transition))
                modified_transition[2] = 0
                self.__push(tuple(modified_transition))
            else:
                self.__push(transition)

        self.buffer = []

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)