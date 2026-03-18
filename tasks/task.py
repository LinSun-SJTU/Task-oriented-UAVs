from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Task:
    """动态任务：从起点 spawn 后飞往终点（激活即在起点）。"""

    id: int
    start: np.ndarray  # (3,)
    goal: np.ndarray  # (3,)
    priority: float
    urgency: float
    status: str = "pending"  # pending | in_progress | done | expired
    assigned_uav_id: int = -1
    deadline_remaining: Optional[int] = None  # 进入池后每步递减；None 表示不启用


@dataclass
class TaskGenerator:
    """按固定间隔在任务池中加入新任务。"""

    rng: np.random.RandomState
    world_size: np.ndarray
    margin: float
    priority_range: Tuple[float, float]
    urgency_range: Tuple[float, float]
    arrival_every_n_steps: int
    tasks_per_arrival: int = 1
    min_seg_len: float = 3.0
    deadline_steps: Optional[int] = None
    _next_id: int = field(default=0, repr=False)

    def reset(self) -> None:
        self._next_id = 0

    def _sample_point(self) -> np.ndarray:
        w = self.world_size
        m = self.margin
        return self.rng.uniform([m, m, m], [w[0] - m, w[1] - m, w[2] - m]).astype(
            np.float64
        )

    def _spawn_one(self) -> Task:
        for _ in range(50):
            s = self._sample_point()
            g = self._sample_point()
            if float(np.linalg.norm(g - s)) >= self.min_seg_len:
                break
        tid = self._next_id
        self._next_id += 1
        pr = self.priority_range
        ur = self.urgency_range
        p = float(self.rng.uniform(pr[0], pr[1]))
        u = float(self.rng.uniform(ur[0], ur[1]))
        dl = self.deadline_steps
        return Task(
            id=tid,
            start=s.copy(),
            goal=g.copy(),
            priority=p,
            urgency=u,
            status="pending",
            assigned_uav_id=-1,
            deadline_remaining=dl,
        )

    def spawn_n(self, n: int) -> List[Task]:
        """立即生成 n 个任务（用于 reset 时一次性注入）。"""
        out: List[Task] = []
        for _ in range(max(0, int(n))):
            out.append(self._spawn_one())
        return out

    def step(self, episode_step: int) -> List[Task]:
        """环境第 episode_step 步（从 0 起）调用；每隔 arrival_every_n_steps 生成任务。"""
        new_tasks: List[Task] = []
        n = int(self.arrival_every_n_steps)
        if n <= 0:
            return new_tasks
        if episode_step % n != 0:
            return new_tasks
        for _ in range(max(1, int(self.tasks_per_arrival))):
            new_tasks.append(self._spawn_one())
        return new_tasks
