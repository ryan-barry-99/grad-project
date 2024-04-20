import random

class ExperienceBuffer:
    def __init__(self, batch_size, max_trajectory_length):
        self.batch_size = batch_size
        self.max_trajectory_length = max_trajectory_length
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def store(self, state=None, action=None, reward=None, log_prob=None, value=None):
        if state is not None: self.states.append(state)
        if action is not None: self.actions.append(action)
        if reward is not None: self.rewards.append(reward)
        if log_prob is not None: self.log_probs.append(log_prob)
        if value is not None: self.values.append(value)

    def at_capacity(self):
        return len(self.states) == self.max_trajectory_length

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def get_batch(self):
        indices = random.sample(range(self.max_trajectory_length), self.batch_size)
        states = [self.states[i] for i in indices]
        actions = [self.actions[i] for i in indices]
        rewards = [self.rewards[i] for i in indices]
        log_probs = [self.log_probs[i] for i in indices]
        values = [self.values[i] for i in indices]
        
        return states, actions, rewards, log_probs, values