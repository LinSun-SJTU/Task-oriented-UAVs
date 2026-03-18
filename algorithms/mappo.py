from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import ActorNetwork, CriticNetwork


@dataclass
class MAPPOConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 10
    batch_size: int = 64
    hidden_dim: int = 64
    device: str = "cpu"


class RolloutBuffer:
    """简化版 PPO 经验缓冲区（on-policy）。"""

    def __init__(self) -> None:
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.global_obs: List[np.ndarray] = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        log_prob: float,
        global_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs.append(obs.copy())
        self.actions.append(np.array(action, copy=True))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.global_obs.append(global_obs.copy())

    def clear(self) -> None:
        self.__init__()


class MAPPO:
    """
    简化版 MAPPO 实现：每个智能体一个 Actor，共享一个 Critic（集中式训练，分布式执行）。
    """

    def __init__(self, env, config: MAPPOConfig) -> None:
        self.env = env
        self.cfg = config
        self.device = torch.device(config.device)

        # 多智能体 Actor，每个独立
        self.actors: List[ActorNetwork] = [
            ActorNetwork(env.obs_dim, env.action_dim, hidden_dim=config.hidden_dim).to(
                self.device
            )
            for _ in range(env.n_uavs)
        ]
        # 共享 Critic
        self.critic = CriticNetwork(env.global_obs_dim, hidden_dim=config.hidden_dim).to(
            self.device
        )

        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=config.actor_lr) for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # 为每个智能体维护一个 buffer
        self.buffers: List[RolloutBuffer] = [RolloutBuffer() for _ in range(env.n_uavs)]

    # ====== 交互采样 ====== #
    def collect_episodes(self, n_episodes: int, max_steps: int) -> None:
        """与环境交互，收集若干 episode 到 buffers 中。"""
        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False

            for _ in range(max_steps):
                actions = []
                log_probs = []

                for i in range(self.env.n_uavs):
                    obs_tensor = torch.as_tensor(
                        obs[i], dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        action, log_prob = self.actors[i].act(obs_tensor)
                    actions.append(action.cpu().numpy().reshape(-1))
                    log_probs.append(log_prob.cpu().item())

                actions_arr = np.stack(actions, axis=0)
                next_obs, rewards, done, _ = self.env.step(actions_arr)

                global_obs = self.env._get_global_obs()
                for i in range(self.env.n_uavs):
                    self.buffers[i].add(
                        obs=obs[i],
                        action=actions[i],
                        reward=rewards[i],
                        log_prob=log_probs[i],
                        global_obs=global_obs,
                        done=done,
                    )

                obs = next_obs
                if done:
                    break

    # ====== GAE & PPO 更新 ====== #
    def _compute_gae(
        self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据 rewards 和 values 计算 GAE 优势和 returns。
        输入均为 1D 向量 (T,)
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.cfg.gamma * next_value * next_non_terminal - values[t]
            last_gae = (
                delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last_gae
            )
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update_policy(self) -> None:
        """
        PPO 更新：先用各智能体 buffer 算 GAE；再集中用所有 (global_obs, return)
        训练共享 Critic；最后分别只更新各 Actor。
        """
        agent_updates: List[dict] = []

        for i in range(self.env.n_uavs):
            buffer = self.buffers[i]
            if len(buffer.rewards) == 0:
                continue

            obs = torch.as_tensor(
                np.stack(buffer.obs, axis=0), dtype=torch.float32, device=self.device
            )
            actions = torch.as_tensor(
                np.stack(buffer.actions, axis=0), dtype=torch.float32, device=self.device
            )
            old_log_probs = torch.as_tensor(
                np.array(buffer.log_probs), dtype=torch.float32, device=self.device
            )
            rewards = np.array(buffer.rewards, dtype=np.float32)
            dones = np.array(buffer.dones, dtype=np.float32)
            global_obs = torch.as_tensor(
                np.stack(buffer.global_obs, axis=0),
                dtype=torch.float32,
                device=self.device,
            )

            with torch.no_grad():
                values = self.critic(global_obs).squeeze(-1).cpu().numpy()

            advantages, returns = self._compute_gae(rewards, values, dones)
            advantages_t = torch.as_tensor(
                advantages, dtype=torch.float32, device=self.device
            )
            returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

            advantages_t = (advantages_t - advantages_t.mean()) / (
                advantages_t.std() + 1e-8
            )

            agent_updates.append(
                {
                    "agent_idx": i,
                    "obs": obs,
                    "actions": actions,
                    "old_log_probs": old_log_probs,
                    "advantages_t": advantages_t,
                    "global_obs": global_obs,
                    "returns_t": returns_t,
                }
            )

        if not agent_updates:
            return

        # 集中训练共享 Critic：拼接所有智能体的 (global_obs, return)
        all_global_obs = torch.cat([u["global_obs"] for u in agent_updates], dim=0)
        all_returns = torch.cat([u["returns_t"] for u in agent_updates], dim=0)
        crit_n = all_global_obs.size(0)
        bs = self.cfg.batch_size

        for _ in range(self.cfg.ppo_epochs):
            crit_perm = np.random.permutation(crit_n)
            for start in range(0, crit_n, bs):
                batch_idx = crit_perm[start : start + bs]
                batch_g = all_global_obs[batch_idx]
                batch_r = all_returns[batch_idx]
                values_pred = self.critic(batch_g).squeeze(-1)
                critic_loss = nn.functional.mse_loss(values_pred, batch_r)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()

        # 各 Actor 单独更新（不再在此轮次更新 Critic）
        for u in agent_updates:
            i = u["agent_idx"]
            obs = u["obs"]
            actions = u["actions"]
            old_log_probs = u["old_log_probs"]
            advantages_t = u["advantages_t"]
            actor = self.actors[i]
            actor_opt = self.actor_optimizers[i]
            dataset_size = obs.size(0)
            indices = np.arange(dataset_size)

            for _ in range(self.cfg.ppo_epochs):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, bs):
                    end = start + bs
                    batch_idx = indices[start:end]

                    batch_obs = obs[batch_idx]
                    batch_actions = actions[batch_idx]
                    batch_old_log_probs = old_log_probs[batch_idx]
                    batch_advantages = advantages_t[batch_idx]

                    dist_now = actor.get_action_distribution(batch_obs)
                    log_probs_now = dist_now.log_prob(batch_actions).sum(dim=-1)

                    ratio = torch.exp(log_probs_now - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = (
                        torch.clamp(
                            ratio,
                            1.0 - self.cfg.clip_epsilon,
                            1.0 + self.cfg.clip_epsilon,
                        )
                        * batch_advantages
                    )
                    actor_loss = -torch.min(surr1, surr2).mean()

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                    actor_opt.step()

            self.buffers[i].clear()



