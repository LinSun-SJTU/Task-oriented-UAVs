import numpy as np
from typing import Tuple


def spheres_overlap(
    center_a: np.ndarray, radius_a: float, center_b: np.ndarray, radius_b: float, buffer: float
) -> bool:
    """
    简单双球体冲突检测。

    参数
    ----
    center_a, center_b : 形状为 (3,) 的坐标
    radius_a, radius_b : 半径
    buffer             : 安全缓冲区
    """
    center_a = np.asarray(center_a, dtype=float)
    center_b = np.asarray(center_b, dtype=float)
    dist = np.linalg.norm(center_a - center_b)
    safe_dist = radius_a + radius_b + buffer
    return bool(dist < safe_dist)


def min_safe_distance(radius_a: float, radius_b: float, buffer: float) -> float:
    """计算两球中心的最小安全距离。"""
    return float(radius_a + radius_b + buffer)


def pairwise_min_center_distance(
    positions: np.ndarray, radii: np.ndarray, buffer: float
) -> Tuple[float, Tuple[int, int]]:
    """
    返回所有无人机对中实际中心距离与安全距离的“最危险比值”及对应索引。

    返回
    ----
    ratio_min : 实际距离 / 安全距离 的最小值
    (i, j)    : 导致该比值的无人机索引对
    """
    n = positions.shape[0]
    ratio_min = float("inf")
    worst_pair = (-1, -1)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            safe_dist = radii[i] + radii[j] + buffer
            if safe_dist <= 0:
                continue
            ratio = dist / safe_dist
            if ratio < ratio_min:
                ratio_min = ratio
                worst_pair = (i, j)
    if ratio_min == float("inf"):
        ratio_min = 1.0
    return ratio_min, worst_pair


def pairwise_min_cube_ratio(
    positions: np.ndarray, half_sizes: np.ndarray, buffer: float
) -> Tuple[float, Tuple[int, int]]:
    """
    立方体约束：两 AABB 不重叠当且仅当在至少一轴上分离 >= half_i+half_j+buffer；
    ratio_ij = max over k of (|pos_i[k]-pos_j[k]|) / (half_i+half_j+buffer)，>=1 表示不重叠。
    ratio_min = min over pairs of ratio_ij。half_sizes[i] = space_sizes[i] / 2。
    """
    n = positions.shape[0]
    ratio_min = float("inf")
    worst_pair = (-1, -1)
    for i in range(n):
        for j in range(i + 1, n):
            required = half_sizes[i] + half_sizes[j] + buffer
            if required <= 0:
                continue
            sep = np.abs(positions[i] - positions[j])
            ratio_ij = float(np.max(sep) / required)
            if ratio_ij < ratio_min:
                ratio_min = ratio_ij
                worst_pair = (i, j)
    if ratio_min == float("inf"):
        ratio_min = 1.0
    return ratio_min, worst_pair


def cubes_overlap(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    half_i: float,
    half_j: float,
    buffer: float,
) -> bool:
    """
    两立方体（AABB）是否重叠。不重叠当且仅当存在至少一轴分离 >= half_i+half_j+buffer；
    即重叠当且仅当在所有轴上分离都 < required，故用 max(sep)。
    """
    sep = np.abs(np.asarray(pos_i, dtype=float) - np.asarray(pos_j, dtype=float))
    required = half_i + half_j + buffer
    return float(np.max(sep)) < required


def max_loser_cube_size(
    pos_i: np.ndarray,
    pos_j: np.ndarray,
    size_winner: float,
    buffer: float,
) -> float:
    """
    两立方体重叠时，在保持 winner 尺寸不变下，loser 不重叠的最大尺寸。
    """
    sep = np.abs(np.asarray(pos_i, dtype=float) - np.asarray(pos_j, dtype=float))
    sep_min = float(np.min(sep))
    return 2.0 * (sep_min - buffer) - size_winner

