import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from safety.safe_layer import SafeProjectionLayer
from tasks.scheduler import greedy_assign_pending_tasks
from tasks.task import Task, TaskGenerator


class MultiUAVEnv:
    """
    多无人机协作空域：可选「任务池 + 规则调度」模式。
    - 传统模式：reset 时每机给定起点/终点，全程 active。
    - 调度模式：任务动态到达；inactive 在停机位；分配后在任务起点 spawn，飞往终点。
    """

    def __init__(
        self,
        n_uavs: int = 5,
        world_size=(24.0, 24.0, 24.0),
        min_space_size: float = 2.0,
        max_space_size: float = 5.0,
        priority_range=(0.1, 1.0),
        urgency_range=(0.0, 1.0),
        safety_buffer: float = 0.2,
        task_weight: float = 1.0,
        completion_bonus: float = 0.0,
        safety_weight: float = 0.5,
        efficiency_weight: float = 0.1,
        random_seed: int = 0,
        navigator: Optional[Any] = None,
        task_scheduler: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.rng = np.random.RandomState(random_seed)
        self.navigator = navigator

        self.n_uavs = n_uavs
        self.world_size = np.array(world_size, dtype=float)
        self.space_dim = 3
        self.min_space_size = float(min_space_size)
        self.max_space_size = float(max_space_size)
        self.priority_range = priority_range
        self.urgency_range = urgency_range

        self.task_weight = task_weight
        self.completion_bonus = float(completion_bonus)
        self.safety_weight = safety_weight
        self.efficiency_weight = efficiency_weight

        ts = task_scheduler or {}
        self.task_scheduler_enabled = bool(ts.get("enabled", False))
        self._task_scheduler_cfg = ts

        # [pos(3), p, u, progress, active_flag, dist_to_others...]
        self.obs_dim = 7 + (self.n_uavs - 1)
        self.action_dim = 1

        self.safety_layer = SafeProjectionLayer(
            min_size=self.min_space_size,
            max_size=self.max_space_size,
            buffer=safety_buffer,
        )

        self.positions = np.zeros((self.n_uavs, 3), dtype=float)
        self.target_positions = np.zeros((self.n_uavs, 3), dtype=float)
        self.priorities = np.zeros(self.n_uavs, dtype=float)
        self.urgencies = np.zeros(self.n_uavs, dtype=float)
        self.space_sizes = np.ones(self.n_uavs, dtype=float)
        self.task_progress = np.zeros(self.n_uavs, dtype=float)
        self.start_positions = np.zeros((self.n_uavs, 3), dtype=float)
        self.max_projection_ratio = np.zeros(self.n_uavs, dtype=float)
        self.active = np.ones(self.n_uavs, dtype=bool)

        self.task_pool: List[Task] = []
        self._episode_step = -1
        self._tasks_completed_episode = 0
        self._depot_positions = self._make_depot_positions()
        self._initial_tasks_injected = False

        self._task_generator: Optional[TaskGenerator] = None
        if self.task_scheduler_enabled:
            m = float(ts.get("margin", 2.0))
            self._task_generator = TaskGenerator(
                rng=self.rng,
                world_size=self.world_size,
                margin=m,
                priority_range=tuple(ts.get("priority_range", list(priority_range))),
                urgency_range=tuple(ts.get("urgency_range", list(urgency_range))),
                arrival_every_n_steps=int(ts.get("arrival_every_n_steps", 30)),
                tasks_per_arrival=int(ts.get("tasks_per_arrival", 1)),
                min_seg_len=float(ts.get("min_seg_len", 3.0)),
                deadline_steps=ts.get("deadline_steps"),  # None ok
            )

    def _make_depot_positions(self) -> np.ndarray:
        w = self.world_size
        m = 2.0
        starts = np.array(
            [
                [m, m, m],
                [w[0] - m, m, m],
                [m, w[1] - m, m],
                [w[0] - m, w[1] - m, m],
                [w[0] * 0.5, w[1] * 0.5, w[2] * 0.5],
            ],
            dtype=float,
        )
        dep = np.zeros((self.n_uavs, 3), dtype=float)
        for i in range(self.n_uavs):
            base = starts[i % len(starts)]
            dep[i] = base + self.rng.uniform(-0.3, 0.3, size=3)
        return np.clip(dep, 0.0, self.world_size)

    @property
    def global_obs_dim(self) -> int:
        return self.n_uavs * self.obs_dim

    def reset(self) -> np.ndarray:
        self._episode_step = -1
        self._tasks_completed_episode = 0
        self._initial_tasks_injected = False
        if self.task_scheduler_enabled:
            self.task_pool = []
            if self._task_generator is not None:
                self._task_generator.reset()
            self.active[:] = False
            self.positions = self._depot_positions.copy()
            self.start_positions = self.positions.copy()
            self.target_positions = self.positions.copy()
            self.priorities[:] = 0.0
            self.urgencies[:] = 0.0
            self.space_sizes[:] = self.min_space_size
            self.max_projection_ratio[:] = 0.0
            self.task_progress[:] = 0.0
            return self._get_obs()

        # 传统模式：全 active，固定起点终点
        w = self.world_size
        margin = 2.0
        starts = np.array(
            [
                [margin, margin, margin],
                [w[0] - margin, margin, margin],
                [margin, w[1] - margin, margin],
                [w[0] - margin, w[1] - margin, margin],
                [w[0] * 0.5, w[1] * 0.5, w[2] * 0.5],
            ],
            dtype=float,
        )
        for i in range(self.n_uavs):
            base = starts[i % len(starts)]
            self.positions[i] = base + self.rng.uniform(-0.5, 0.5, size=3)
        self.positions = np.clip(self.positions, 0.0, self.world_size)
        targets = np.array(
            [
                [w[0] - margin, w[1] - margin, w[2] - margin],
                [margin, w[1] - margin, w[2] - margin],
                [w[0] - margin, margin, w[2] - margin],
                [margin, margin, w[2] - margin],
                [w[0] * 0.5, w[1] * 0.5, margin],
            ],
            dtype=float,
        )
        for i in range(self.n_uavs):
            self.target_positions[i] = targets[i % len(targets)] + self.rng.uniform(
                -0.3, 0.3, size=3
            )
        self.target_positions = np.clip(self.target_positions, 0.0, self.world_size)
        self.priorities = self.rng.uniform(
            self.priority_range[0], self.priority_range[1], size=self.n_uavs
        )
        self.urgencies = self.rng.uniform(
            self.urgency_range[0], self.urgency_range[1], size=self.n_uavs
        )
        self.space_sizes[:] = self.min_space_size
        self.start_positions = self.positions.copy()
        seg_len = np.linalg.norm(self.target_positions - self.start_positions, axis=1)
        self.max_projection_ratio = np.where(seg_len < 1e-6, 1.0, 0.0)
        self.task_progress = self.max_projection_ratio.copy()
        self.active[:] = True
        return self._get_obs()

    def _deactivate_uav(self, i: int) -> None:
        self.active[i] = False
        self.positions[i] = self._depot_positions[i].copy()
        self.start_positions[i] = self.positions[i].copy()
        self.target_positions[i] = self.positions[i].copy()
        self.priorities[i] = 0.0
        self.urgencies[i] = 0.0
        self.space_sizes[i] = self.min_space_size
        self.max_projection_ratio[i] = 0.0
        self.task_progress[i] = 0.0

    def _assign_task_to_uav(self, i: int, task: Task) -> None:
        task.status = "in_progress"
        task.assigned_uav_id = i
        self.active[i] = True
        self.positions[i] = np.clip(task.start.copy(), 0.0, self.world_size)
        self.start_positions[i] = self.positions[i].copy()
        self.target_positions[i] = np.clip(task.goal.copy(), 0.0, self.world_size)
        self.priorities[i] = float(task.priority)
        self.urgencies[i] = float(task.urgency)
        self.space_sizes[i] = self.min_space_size
        self.max_projection_ratio[i] = 0.0
        self.task_progress[i] = 0.0
        seg = np.linalg.norm(self.target_positions[i] - self.start_positions[i])
        if seg < 1e-6:
            self.max_projection_ratio[i] = 1.0
            self.task_progress[i] = 1.0

    def _scheduler_tick(self) -> None:
        assert self._task_generator is not None
        # episode 第 0 步一次性注入初始任务，用于开局激活更多 UAV
        if (not self._initial_tasks_injected) and self._episode_step == 0:
            init_n = int(self._task_scheduler_cfg.get("initial_tasks", 0) or 0)
            if init_n > 0:
                self.task_pool.extend(self._task_generator.spawn_n(init_n))
            self._initial_tasks_injected = True

        new_tasks = self._task_generator.step(self._episode_step)
        self.task_pool.extend(new_tasks)

        for t in self.task_pool:
            if t.status == "pending" and t.deadline_remaining is not None:
                t.deadline_remaining -= 1
                if t.deadline_remaining <= 0:
                    t.status = "expired"

        for i in range(self.n_uavs):
            if not self.active[i]:
                continue
            if self.task_progress[i] >= 0.95:
                for tk in self.task_pool:
                    if tk.assigned_uav_id == i and tk.status == "in_progress":
                        tk.status = "done"
                        self._tasks_completed_episode += 1
                        break
                self._deactivate_uav(i)

        assigns = greedy_assign_pending_tasks(
            self.task_pool, self.n_uavs, self.active
        )
        for uav_i, task in assigns:
            self._assign_task_to_uav(uav_i, task)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        actions = np.asarray(actions, dtype=float).reshape(self.n_uavs, -1)
        info: Dict[str, Any] = {}

        if self.task_scheduler_enabled:
            self._episode_step += 1
            self._scheduler_tick()

        prev_progress = self.task_progress.copy()

        act_mask = self.active.copy()
        if not self.task_scheduler_enabled:
            act_mask[:] = True

        movements = np.zeros((self.n_uavs, 3), dtype=float)
        if act_mask.any():
            if self.navigator is not None:
                movements_all = self.navigator.compute_movements(
                    self.positions, self.target_positions, self.world_size
                )
            else:
                movements_all = self.rng.normal(0.0, 0.1, size=(self.n_uavs, 3))
            movements[act_mask] = movements_all[act_mask]

        half_nav = self.space_sizes / 2.0
        max_axis = np.max(np.abs(movements), axis=1)
        scale = np.minimum(1.0, half_nav / np.maximum(max_axis, 1e-9))
        movements = movements * scale[:, np.newaxis]
        self.positions = np.clip(self.positions + movements, 0.0, self.world_size)

        self._enforce_min_separation_active_only(act_mask)

        self.space_sizes, safe_actions, proposed_sizes = self.safety_layer.safety_projection(
            actions, self._get_state()
        )
        proposed_sizes = np.asarray(proposed_sizes, dtype=float).reshape(-1).copy()
        for i in range(self.n_uavs):
            if not act_mask[i]:
                self.space_sizes[i] = self.min_space_size
                safe_actions[i] = np.array([-1.0], dtype=float)
                proposed_sizes[i] = self.min_space_size
        self.space_sizes = np.clip(
            self.space_sizes, self.min_space_size, self.max_space_size
        )

        for i in range(self.n_uavs):
            if not act_mask[i]:
                self.task_progress[i] = 0.0
                self.max_projection_ratio[i] = 0.0
                continue
            diff = self.target_positions[i] - self.start_positions[i]
            L_sq = float(np.sum(diff ** 2))
            L_sq = max(L_sq, 1e-12)
            t = float(
                np.sum((self.positions[i] - self.start_positions[i]) * diff)
            ) / L_sq
            t = float(np.clip(t, 0.0, 1.0))
            self.max_projection_ratio[i] = max(float(self.max_projection_ratio[i]), t)
            self.task_progress[i] = self.max_projection_ratio[i]

        delta_progress = np.maximum(0.0, self.task_progress - prev_progress)
        completed_now = (self.task_progress >= 0.95) & (prev_progress < 0.95) & act_mask
        rewards = self._calculate_rewards(
            safe_actions,
            proposed_sizes,
            delta_progress=delta_progress,
            completed_now=completed_now,
            active_mask=act_mask,
        )

        if self.task_scheduler_enabled:
            done = False
            info["tasks_completed"] = self._tasks_completed_episode
            info["n_pending"] = sum(1 for t in self.task_pool if t.status == "pending")
        else:
            completed = self.task_progress >= 0.95
            self.space_sizes[completed] = self.min_space_size
            done = bool(np.mean(self.task_progress) >= 0.95)

        if not self.task_scheduler_enabled:
            completed = self.task_progress >= 0.95
            self.space_sizes[completed] = self.min_space_size

        obs = self._get_obs()
        return obs, rewards, done, info

    def _get_state(self) -> Dict[str, np.ndarray]:
        return {
            "positions": self.positions.copy(),
            "space_sizes": self.space_sizes.copy(),
            "priorities": self.priorities.copy(),
            "urgencies": self.urgencies.copy(),
        }

    def _enforce_min_separation_active_only(self, act_mask: np.ndarray) -> None:
        n = self.n_uavs
        if n <= 1:
            return
        min_sep = self.min_space_size + self.safety_layer.buffer
        weights = self.priorities * self.urgencies
        max_iter = n * (n - 1)
        for _ in range(max_iter):
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (act_mask[i] and act_mask[j]):
                        continue
                    sep = np.abs(self.positions[i] - self.positions[j])
                    max_sep = float(np.max(sep))
                    if max_sep >= min_sep or max_sep < 1e-9:
                        continue
                    scale = min_sep / max_sep
                    delta = self.positions[i] - self.positions[j]
                    delta_new = delta * scale
                    wi, wj = weights[i], weights[j]
                    if wi >= wj:
                        winner, loser = i, j
                        self.positions[loser] = self.positions[winner] - delta_new
                    else:
                        winner, loser = j, i
                        self.positions[loser] = self.positions[winner] + delta_new
                    self.positions[loser] = np.clip(
                        self.positions[loser], 0.0, self.world_size
                    )
                    changed = True
            if not changed:
                break

    def _get_obs(self) -> np.ndarray:
        obs_list = []
        for i in range(self.n_uavs):
            if self.active[i]:
                pos = self.positions[i]
                p, u, pr = self.priorities[i], self.urgencies[i], self.task_progress[i]
                af = 1.0
            else:
                pos = np.zeros(3, dtype=float)
                p, u, pr = 0.0, 0.0, 0.0
                af = 0.0
            self_state = np.concatenate(
                [
                    pos,
                    np.array([p, u, pr, af], dtype=float),
                ]
            )
            distances = []
            for j in range(self.n_uavs):
                if i == j:
                    continue
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                distances.append(float(dist))
            obs_i = np.concatenate([self_state, np.array(distances, dtype=float)])
            obs_list.append(obs_i)
        return np.vstack(obs_list)

    def _get_global_obs(self) -> np.ndarray:
        return self._get_obs().flatten()

    def _calculate_rewards(
        self,
        safe_actions: np.ndarray,
        proposed_sizes: np.ndarray,
        delta_progress: np.ndarray,
        completed_now: np.ndarray,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        safe_actions = safe_actions.squeeze(-1)
        weights = self.priorities * self.urgencies
        task_reward = self.task_weight * (weights * np.asarray(delta_progress, dtype=float))
        completion_reward = self.completion_bonus * (
            weights * completed_now.astype(float)
        )
        proposed_sizes_clipped = np.clip(
            np.asarray(proposed_sizes, dtype=float).ravel(),
            self.min_space_size,
            self.max_space_size,
        )
        size_reduction = np.abs(proposed_sizes_clipped - self.space_sizes)
        safety_penalty = -self.safety_weight * size_reduction
        efficiency_penalty = -self.efficiency_weight * self.space_sizes
        rewards = task_reward + completion_reward + safety_penalty + efficiency_penalty
        rewards = rewards.astype(float)
        rewards[~active_mask] = 0.0
        return rewards


def test_environment() -> None:
    env = MultiUAVEnv()
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    actions = np.random.uniform(-1.0, 1.0, (env.n_uavs, 1))
    obs, rewards, done, info = env.step(actions)
    print(f"Rewards: {rewards}, done: {done}")
    assert not np.any(np.isnan(rewards))


if __name__ == "__main__":
    test_environment()
