import random

class ExperienceBuffer:
    def __init__(self, batch_size, max_trajectory_length):
        self.batch_size = batch_size
        self.max_trajectory_length = max_trajectory_length
        self.states = []
        self.actions = []
        self.rewards = []
        self.old_log_probs = [1] * self.max_trajectory_length
        self.new_log_probs = []
        self.values = []
        self.length = 0

    def store(self, state, action, reward, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.new_log_probs.append(log_prob)
        self.values.append(value)
        self.length += 1

    def at_capacity(self):
        return len(self.states) == self.max_trajectory_length

    def clear(self):
        self.length = 0
        self.states = []
        self.actions = []
        self.rewards = []
        self.old_log_probs = self.new_log_probs[:]
        self.new_log_probs = []
        self.values = []

    def get_batch(self):
        indices = random.sample(range(self.max_trajectory_length), self.batch_size)
        states = [self.states[i] for i in indices]
        actions = [self.actions[i] for i in indices]
        rewards = [self.rewards[i] for i in indices]
        old_log_probs = [self.old_log_probs[i] for i in indices]
        new_log_probs = [self.new_log_probs[i] for i in indices]
        values = [self.values[i] for i in indices]
        
        return states, actions, rewards, old_log_probs, new_log_probs, values