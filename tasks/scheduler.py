from __future__ import annotations

from typing import List, Tuple

import numpy as np

from tasks.task import Task


def greedy_assign_pending_tasks(
    task_pool: List[Task],
    n_uavs: int,
    active: np.ndarray,
) -> List[Tuple[int, Task]]:
    """
    规则调度：按 priority*urgency 从高到低，为每个 pending 任务分配一架 inactive UAV。
    返回 [(uav_idx, task), ...]。
    """
    pending = [t for t in task_pool if t.status == "pending"]
    pending.sort(key=lambda t: t.priority * t.urgency, reverse=True)

    free_uavs: List[int] = [i for i in range(n_uavs) if not bool(active[i])]
    assignments: List[Tuple[int, Task]] = []

    for task in pending:
        if not free_uavs:
            break
        i = free_uavs.pop(0)
        assignments.append((i, task))
    return assignments
