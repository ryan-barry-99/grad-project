import torch
from torch.utils.data import Dataset

class ExperienceDataset(Dataset):
    def __init__(self, experiences):
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        return self.experiences[idx]


class ExperienceBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def store(self, state, action, reward, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def get_batch(self):
        # Convert lists to PyTorch tensors
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)
        
        return states, actions, rewards, log_probs, values