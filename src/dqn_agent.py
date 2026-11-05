"""
PyTorch Deep Q-Network (DQN) implementation for COVID-19 vaccine allocation.
Preserves the exact architecture (9 -> 64 -> 64 -> buckets) and hyperparameters
specified in Awasthi et al. (2022).
"""

import os
import random
from collections import deque
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    Q-Network architecture:
    Input (9 features) -> Linear(9, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, num_actions)
    """

    def __init__(self, state_dim: int = 9, action_dim: int = 1000):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experience Replay Buffer for storing and sampling transitions."""

    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append(
            (
                np.array(state, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
        )

    def __len__(self):
        return len(self.buffer)

    def state_dict(self):
        return list(self.buffer)

    def load_state_dict(self, state_list):
        self.buffer = deque(state_list, maxlen=self.buffer.maxlen)


class DQNAgent:
    """
    DQN Agent handling action selection, Q-network optimization,
    target network synchronization, and full state persistence.
    """

    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 1000,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.10,
        epsilon_decay_steps: int = 5000,
        buffer_capacity: int = 20000,
        batch_size: int = 64,
        target_update_interval: int = 200,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval

        # Exploration hyperparameters (10% exploration target as in Table 2)
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Auto device detection
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.steps_done = 0

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Selects an action using epsilon-greedy policy (or purely greedy if evaluate=True).
        """
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_t)
            action = int(q_values.argmax(dim=1).item())
        self.q_network.train()
        return action

    def step_decay_epsilon(self):
        """Linearly anneals epsilon down to epsilon_end."""
        if self.steps_done < self.epsilon_decay_steps:
            decay_frac = self.steps_done / float(self.epsilon_decay_steps)
            self.epsilon = self.epsilon_start - decay_frac * (self.epsilon_start - self.epsilon_end)
        else:
            self.epsilon = self.epsilon_end

    def update(self) -> Optional[float]:
        """Samples mini-batch and performs one step of gradient descent."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q estimates: Q(s, a)
        q_pred = self.q_network(states_t).gather(1, actions_t)

        # TD Targets with Target Network: y = r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q_max = self.target_network(next_states_t).max(1, keepdim=True)[0]
            target = rewards_t + (1.0 - dones_t) * self.gamma * next_q_max

        loss = self.loss_fn(q_pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_done += 1
        self.step_decay_epsilon()

        # Periodic target network sync
        if self.steps_done % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def save_checkpoint(self, filepath: str, extra_info: Optional[Dict[str, Any]] = None):
        """
        Saves a full training checkpoint atomically (write to temp file then rename)
        to prevent partial file corruption during Colab disconnections.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "replay_buffer": self.replay_buffer.state_dict(),
            "extra_info": extra_info or {},
        }
        temp_filepath = filepath + ".tmp"
        torch.save(checkpoint, temp_filepath)
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(temp_filepath, filepath)

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Loads network weights, optimizer state, exploration state, and replay buffer.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        except TypeError:
            # Fallback for earlier PyTorch versions where weights_only argument is not supported
            checkpoint = torch.load(filepath, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.steps_done = checkpoint.get("steps_done", 0)

        if "replay_buffer" in checkpoint and checkpoint["replay_buffer"]:
            self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])

        return checkpoint.get("extra_info", {})
