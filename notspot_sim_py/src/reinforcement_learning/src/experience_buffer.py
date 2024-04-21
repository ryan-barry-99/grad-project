import random

class ExperienceBuffer:
    def __init__(self, batch_size, max_trajectory_length):
        self.batch_size = batch_size
        self.max_trajectory_length = max_trajectory_length
        self.states = []
        self.actions = []
        self.rewards = []
        self.new_log_probs = []
        self.values = []
        self.length = 0
        self.old_log_probs = [1] * self.max_trajectory_length

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
        if len(self.new_log_probs) < self.max_trajectory_length:
            # If new_log_probs is shorter than max_trajectory_length,
            # recycle the data in old_log_probs until it reaches max_trajectory_length
            num_repeats = self.max_trajectory_length - len(self.new_log_probs)
            self.old_log_probs = (self.old_log_probs * num_repeats)[:self.max_trajectory_length]
        else:
            # If new_log_probs is longer than max_trajectory_length,
            # truncate old_log_probs to match max_trajectory_length
            self.old_log_probs = self.new_log_probs[:self.max_trajectory_length]
        self.new_log_probs = []
        self.values = []

    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        
        # Get the indices of the most recent batch
        start_index = max(0, self.length - batch_size)
        end_index = self.length
        indices = list(range(start_index, end_index))

        states = [self.states[i] for i in indices]
        actions = [self.actions[i] for i in indices]
        rewards = [self.rewards[i] for i in indices]
        old_log_probs = [self.old_log_probs[i] for i in indices]
        new_log_probs = [self.new_log_probs[i] for i in indices]
        values = [self.values[i] for i in indices]
        
        return states, actions, rewards, old_log_probs, new_log_probs, values
