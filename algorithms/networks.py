from typing import Tuple

import torch
import torch.nn as nn
import torch.distributions as dist


class ActorNetwork(nn.Module):
    """策略网络（Actor），输出一维连续动作 in [-1, 1]."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # 输出在 [-1, 1]
        )

        # 简化：使用可学习的对数标准差（共享）
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def get_action_distribution(self, obs: torch.Tensor) -> dist.Normal:
        mean = self.forward(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return dist.Normal(mean, std)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作并返回 (action, log_prob).
        obs: [batch, obs_dim]
        """
        distribution = self.get_action_distribution(obs)
        action = distribution.rsample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        return action, log_prob


class CriticNetwork(nn.Module):
    """价值网络（Critic），基于全局观测。"""

    def __init__(self, global_obs_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        # 输出形状: [batch, 1]
        return self.net(global_obs)



