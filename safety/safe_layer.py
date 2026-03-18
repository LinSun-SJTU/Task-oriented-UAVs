import numpy as np
from typing import Dict, Optional, Tuple

from .collision_checker import (
    pairwise_min_center_distance,
    pairwise_min_cube_ratio,
    cubes_overlap,
)


class SafeProjectionLayer:
    """
    安全投影层：将不安全动作投影到最近的安全动作。

    投影与冲突解决均采用全局等比例缩放（不使用权重），
    空间差异由策略根据 obs 中的 priority/urgency 自行学习。
    """

    def __init__(
        self,
        min_size: float = 1.0,
        max_size: Optional[float] = None,
        buffer: float = 0.1,
    ) -> None:
        self.min_size = float(min_size)
        self.max_size = float(max_size) if max_size is not None else None
        self.buffer = float(buffer)

    def _compute_per_uav_alphas(
        self,
        positions: np.ndarray,
        proposed_sizes: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        球体约束：对每对 (i,j) 有 alpha_i*half_i + alpha_j*half_j + buffer <= dist_ij；
        alpha_i : alpha_j = w_i : w_j（按优先级分配）。
        """
        n = positions.shape[0]
        half = proposed_sizes / 2.0
        w = np.asarray(weights, dtype=float).ravel()
        w = np.maximum(w, 1e-6)

        alpha = np.ones(n, dtype=float) * np.inf
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dist = float(np.linalg.norm(positions[i] - positions[j]))
                budget = dist - self.buffer
                denom = w[i] * half[i] + w[j] * half[j]
                if denom <= 0:
                    continue
                if budget <= 0:
                    t_max = 0.0
                else:
                    t_max = budget / denom
                alpha_i_ij = t_max * w[i]
                if alpha_i_ij < alpha[i]:
                    alpha[i] = alpha_i_ij

        alpha = np.where(np.isinf(alpha), 1.0, alpha)
        # 下界：保证缩放后尺寸 >= min_size
        alpha_min = self.min_size / np.maximum(proposed_sizes, 1e-6)
        alpha = np.maximum(alpha, alpha_min)
        return alpha

    def _apply_alphas_and_ensure_safe(
        self,
        positions: np.ndarray,
        proposed_sizes: np.ndarray,
        alpha: np.ndarray,
    ) -> np.ndarray:
        """应用 alpha 后，用球体约束校验；若仍重叠则再缩放至不重叠。"""
        sizes = proposed_sizes * alpha
        sizes = np.maximum(sizes, self.min_size)
        if self.max_size is not None:
            sizes = np.minimum(sizes, self.max_size)
        radii = sizes / 2.0
        ratio_min, _ = pairwise_min_center_distance(positions, radii, self.buffer)
        if ratio_min < 1.0:
            scale = max(ratio_min, self.min_size / np.max(sizes))
            sizes = sizes * scale
        sizes = np.maximum(sizes, self.min_size)
        if self.max_size is not None:
            sizes = np.minimum(sizes, self.max_size)
        return sizes

    def project_actions(self, raw_actions: np.ndarray, state: Dict[str, np.ndarray]) -> np.ndarray:
        """
        投影动作到安全空间。按球体约束做每机单独缩放因子（等权，不使用 priority/urgency）。
        """
        actions = np.asarray(raw_actions, dtype=float).copy().reshape(-1, 1)
        positions = np.asarray(state["positions"], dtype=float)
        current_sizes = np.asarray(state["space_sizes"], dtype=float)

        scale = 1.0 + actions.squeeze(-1) * 0.5
        proposed_sizes = current_sizes * scale
        proposed_sizes = np.maximum(proposed_sizes, self.min_size)
        if self.max_size is not None:
            proposed_sizes = np.minimum(proposed_sizes, self.max_size)

        n = positions.shape[0]
        alpha = self._compute_per_uav_alphas(positions, proposed_sizes, np.ones(n))
        # 只应用 per-UAV alpha，不再做二次球体验证（由 env 侧 resolve_conflicts 立方体兜底）
        proposed_sizes = proposed_sizes * alpha
        proposed_sizes = np.maximum(proposed_sizes, self.min_size)
        if self.max_size is not None:
            proposed_sizes = np.minimum(proposed_sizes, self.max_size)
        # proposed_sizes = self._apply_alphas_and_ensure_safe(positions, proposed_sizes, alpha)

        eps = 1e-6
        denom = np.maximum(current_sizes, eps)
        safe_actions = 2.0 * (proposed_sizes / denom - 1.0)
        safe_actions = np.clip(safe_actions, -1.0, 1.0)
        return safe_actions.reshape(-1, 1)

    def resolve_conflicts(
        self,
        positions: np.ndarray,
        sizes: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        消除冲突（立方体约束）：用立方体两两检测是否重叠，若有则全局等比例缩放至不重叠（不使用权重）。
        """
        positions = np.asarray(positions, dtype=float)
        sizes = np.asarray(sizes, dtype=float).copy()
        sizes = np.maximum(sizes, self.min_size)
        if self.max_size is not None:
            sizes = np.minimum(sizes, self.max_size)

        # 冲突检测与缩放：按立方体（两两立方体是否重叠）
        half_sizes = sizes / 2.0
        ratio_min, _ = pairwise_min_cube_ratio(positions, half_sizes, self.buffer)
        if ratio_min < 1.0:
            scale = max(ratio_min, self.min_size / np.max(sizes))
            sizes = sizes * scale
        sizes = np.maximum(sizes, self.min_size)
        if self.max_size is not None:
            sizes = np.minimum(sizes, self.max_size)
        return sizes

    def safety_projection(
        self,
        raw_actions: np.ndarray,
        state: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        由 actor 原始动作直接得到 proposed_sizes，再按 priority×urgency 两两冲突解决得到最终 space_sizes。
        返回 (final_space_sizes, safe_actions, proposed_sizes)。proposed_sizes 供奖励中安全惩罚使用。
        """
        actions = np.asarray(raw_actions, dtype=float).copy().reshape(-1, 1)
        positions = np.asarray(state["positions"], dtype=float)
        current_sizes = np.asarray(state["space_sizes"], dtype=float)
        priorities = np.asarray(state["priorities"], dtype=float).ravel()
        urgencies = np.asarray(state["urgencies"], dtype=float).ravel()

        scale = 1.0 + actions.squeeze(-1) * 0.5
        proposed_sizes = current_sizes * scale

        final_sizes = self.resolve_conflicts_priority(
            positions, proposed_sizes, priorities, urgencies
        )

        eps = 1e-6
        denom = np.maximum(current_sizes, eps)
        safe_actions = 2.0 * (final_sizes / denom - 1.0)
        safe_actions = np.clip(safe_actions, -1.0, 1.0)
        return final_sizes, safe_actions.reshape(-1, 1), proposed_sizes

    def resolve_conflicts_priority(
        self,
        positions: np.ndarray,
        sizes: np.ndarray,
        priorities: np.ndarray,
        urgencies: np.ndarray,
    ) -> np.ndarray:
        """
        消除冲突（立方体约束）：两两检测，重叠时按 priority*urgency 比例分配“不重叠预算”，
        双方一起缩小而非 winner 全拿、loser 全让。多轮直到无变化。
        """
        positions = np.asarray(positions, dtype=float)
        sizes = np.asarray(sizes, dtype=float).copy()
        sizes = np.maximum(sizes, self.min_size)
        if self.max_size is not None:
            sizes = np.minimum(sizes, self.max_size)
        n = positions.shape[0]
        if n <= 1:
            return sizes
        weights = np.asarray(priorities, dtype=float).ravel() * np.asarray(urgencies, dtype=float).ravel()
        weights = np.maximum(weights, 1e-6)
        half = sizes / 2.0
        max_iter = n * (n - 1)
        for _ in range(max_iter):
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not cubes_overlap(positions[i], positions[j], half[i], half[j], self.buffer):
                        continue
                    sep = np.abs(positions[i] - positions[j])
                    sep_min = float(np.min(sep))
                    budget = sep_min - self.buffer
                    if budget <= 0:
                        new_size_i = self.min_size
                        new_size_j = self.min_size
                    else:
                        wi, wj = weights[i], weights[j]
                        half_i_new = budget * wi / (wi + wj)
                        half_j_new = budget * wj / (wi + wj)
                        new_size_i = 2.0 * half_i_new
                        new_size_j = 2.0 * half_j_new
                    new_size_i = max(self.min_size, new_size_i)
                    new_size_j = max(self.min_size, new_size_j)
                    new_size_i = min(sizes[i], new_size_i)
                    new_size_j = min(sizes[j], new_size_j)
                    if new_size_i < sizes[i] or new_size_j < sizes[j]:
                        sizes[i] = new_size_i
                        sizes[j] = new_size_j
                        half[i] = sizes[i] / 2.0
                        half[j] = sizes[j] / 2.0
                        changed = True
            if not changed:
                break
        if self.max_size is not None:
            sizes = np.minimum(sizes, self.max_size)
        return sizes

