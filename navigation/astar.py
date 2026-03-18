"""
基于 3D 网格的 A* 路径规划导航器。
每步为每架无人机计算朝向当前路径下一节点的位移，步长受 step_size 限制。
"""
import numpy as np
import heapq
from typing import List, Tuple

from .base import BaseNavigator


def _world_to_grid(pos: np.ndarray, resolution: float, world_bounds: np.ndarray) -> Tuple[int, int, int]:
    """世界坐标 -> 网格索引，裁剪到合法范围。"""
    g = (pos / resolution).astype(int)
    max_ij = (world_bounds / resolution).astype(int)
    g = np.clip(g, [0, 0, 0], np.maximum(max_ij - 1, 0))
    return int(g[0]), int(g[1]), int(g[2])


def _grid_to_world(i: int, j: int, k: int, resolution: float) -> np.ndarray:
    """网格索引 -> 世界坐标（格心）。"""
    return np.array([(i + 0.5) * resolution, (j + 0.5) * resolution, (k + 0.5) * resolution], dtype=float)


def _astar_path(
    start: np.ndarray,
    goal: np.ndarray,
    resolution: float,
    world_bounds: np.ndarray,
) -> List[np.ndarray]:
    """
    在 3D 网格上 A* 寻路，返回世界坐标路径 [start, ..., goal]。
    6 邻域，代价为步长，启发式为欧氏距离。
    """
    si, sj, sk = _world_to_grid(start, resolution, world_bounds)
    gi, gj, gk = _world_to_grid(goal, resolution, world_bounds)
    nx = max(1, int(world_bounds[0] / resolution))
    ny = max(1, int(world_bounds[1] / resolution))
    nz = max(1, int(world_bounds[2] / resolution))

    def heuristic(i: int, j: int, k: int) -> float:
        return float(np.linalg.norm(np.array([i - gi, j - gj, k - gk]) * resolution))

    open_heap: List[Tuple[float, int, Tuple[int, int, int]]] = []
    counter = 0
    g_score: dict = {}
    came_from: dict = {}
    start_node = (si, sj, sk)
    g_score[start_node] = 0.0
    heapq.heappush(open_heap, (heuristic(si, sj, sk), counter, start_node))
    counter += 1
    neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    step_cost = resolution

    while open_heap:
        _, _, (i, j, k) = heapq.heappop(open_heap)
        if (i, j, k) == (gi, gj, gk):
            path = []
            node = (gi, gj, gk)
            while node in came_from:
                path.append(_grid_to_world(node[0], node[1], node[2], resolution))
                node = came_from[node]
            path.append(_grid_to_world(si, sj, sk, resolution))
            path.reverse()
            path.append(goal.copy())
            return path
        for di, dj, dk in neighbors:
            ni, nj, nk = i + di, j + dj, k + dk
            if not (0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz):
                continue
            neighbor = (ni, nj, nk)
            tentative = g_score[(i, j, k)] + step_cost
            if neighbor not in g_score or tentative < g_score[neighbor]:
                g_score[neighbor] = tentative
                came_from[neighbor] = (i, j, k)
                heapq.heappush(open_heap, (tentative + heuristic(ni, nj, nk), counter, neighbor))
                counter += 1
    return [start.copy(), goal.copy()]


class AStarNavigator(BaseNavigator):
    """A* 网格导航：每步沿当前路径向下一节点移动 step_size。"""

    def __init__(self, grid_resolution: float = 1.0, step_size: float = 0.5):
        self.grid_resolution = float(grid_resolution)
        self.step_size = float(step_size)

    def compute_movements(
        self,
        positions: np.ndarray,
        targets: np.ndarray,
        world_bounds: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        n = positions.shape[0]
        movements = np.zeros((n, 3), dtype=float)
        for i in range(n):
            start, goal = positions[i], targets[i]
            if np.linalg.norm(goal - start) < 1e-6:
                continue
            path = _astar_path(start, goal, self.grid_resolution, world_bounds)
            if len(path) < 2:
                continue
            to_next = path[1] - start
            d = np.linalg.norm(to_next)
            if d < 1e-6:
                continue
            move_len = min(self.step_size, d)
            movements[i] = to_next * (move_len / d)
        return movements
