import numpy as np
from typing import Dict, Tuple, Any, Optional

from safety.safe_layer import SafeProjectionLayer


class MultiUAVEnv:
    """
    5 架无人机的协作空域管理环境（简化版 Demo）

    核心功能：
    1. 生成初始任务属性 (priority, urgency)
    2. 执行动作（调整独占空间大小）
    3. 计算奖励（任务奖励 + 安全 / 效率奖励）
    4. 更新状态（任务进度、无人机位置）

    说明：
    - 独占空间用球形或等边立方体的“边长/直径”等价标量 size 表示
    - 安全检查基于球形近似，使用安全缓冲区避免重叠
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
    ) -> None:
        self.rng = np.random.RandomState(random_seed)
        self.navigator = navigator

        # 基础参数
        self.n_uavs = n_uavs
        self.world_size = np.array(world_size, dtype=float)
        self.space_dim = 3  # 3D 空间
        self.min_space_size = float(min_space_size)
        self.max_space_size = float(max_space_size)

        # 任务属性范围
        self.priority_range = priority_range
        self.urgency_range = urgency_range

        # 奖励权重
        self.task_weight = task_weight
        self.completion_bonus = float(completion_bonus)
        self.safety_weight = safety_weight
        self.efficiency_weight = efficiency_weight

        # 状态空间：每个无人机观测
        # [pos_x, pos_y, pos_z, priority, urgency, progress,
        #  neighbor_dist_1, neighbor_dist_2, ...]
        self.obs_dim = 6 + (self.n_uavs - 1)

        # 动作空间：[-1, 1] -> 通过映射变为空间缩放因子
        self.action_dim = 1

        # 安全层（与环境共用 min/max 空间边界，不再单独配置）
        self.safety_layer = SafeProjectionLayer(
            min_size=self.min_space_size,
            max_size=self.max_space_size,
            buffer=safety_buffer,
        )

        # 运行时状态
        self.positions: np.ndarray = np.zeros((self.n_uavs, 3), dtype=float)
        self.target_positions: np.ndarray = np.zeros((self.n_uavs, 3), dtype=float)
        self.priorities: np.ndarray = np.zeros(self.n_uavs, dtype=float)
        self.urgencies: np.ndarray = np.zeros(self.n_uavs, dtype=float)
        self.space_sizes: np.ndarray = np.ones(self.n_uavs, dtype=float)
        self.task_progress: np.ndarray = np.zeros(self.n_uavs, dtype=float)
        self.start_positions: np.ndarray = np.zeros((self.n_uavs, 3), dtype=float)  # reset 时起点，用于投影
        self.max_projection_ratio: np.ndarray = np.zeros(self.n_uavs, dtype=float)  # 历史最大投影占比 [0,1]

    # 便于 MAPPO critic 的 global_obs 维度
    @property
    def global_obs_dim(self) -> int:
        # 简单做法：拼接所有局部观测
        return self.n_uavs * self.obs_dim

    # ====== Gym-like API ====== #
    def reset(self) -> np.ndarray:
        """重置环境，初始化无人机起点与目标（分散）、任务属性。"""
        w = self.world_size
        margin = 2.0
        # 1. 起点：在空间内分散布置（角点/边+中心附近），加小随机偏移
        #准备了一组起点和终点列表，然后无人机按照编号取模循环使用这些点
        starts = np.array([
            [margin, margin, margin],
            [w[0] - margin, margin, margin],
            [margin, w[1] - margin, margin],
            [w[0] - margin, w[1] - margin, margin],
            [w[0] * 0.5, w[1] * 0.5, w[2] * 0.5],
        ], dtype=float)
        for i in range(self.n_uavs):
            base = starts[i % len(starts)]
            self.positions[i] = base + self.rng.uniform(-0.5, 0.5, size=3)
        self.positions = np.clip(self.positions, 0.0, self.world_size)

        # 2. 目标点：与起点错开，分散在另一侧/对角
        targets = np.array([
            [w[0] - margin, w[1] - margin, w[2] - margin],
            [margin, w[1] - margin, w[2] - margin],
            [w[0] - margin, margin, w[2] - margin],
            [margin, margin, w[2] - margin],
            [w[0] * 0.5, w[1] * 0.5, margin],
        ], dtype=float)
        for i in range(self.n_uavs):
            self.target_positions[i] = targets[i % len(targets)] + self.rng.uniform(-0.3, 0.3, size=3)
        self.target_positions = np.clip(self.target_positions, 0.0, self.world_size)

        # 3. 任务属性
        # 在初始设置的优先级取值范围内均匀取一组赋值到任务上
        self.priorities = self.rng.uniform(
            self.priority_range[0], self.priority_range[1], size=self.n_uavs
        )
        self.urgencies = self.rng.uniform(
            self.urgency_range[0], self.urgency_range[1], size=self.n_uavs
        )

        # 4. 初始化独占空间大小（min 下 half_nav 仍 > 0，可直接从 min 起步）
        self.space_sizes = np.ones(self.n_uavs, dtype=float) * self.min_space_size

        # 5. 任务进度：起点-终点连线上的投影占比，取历史最大（只认“沿连线推进”，不认绕路）
        self.start_positions = self.positions.copy()
        seg_len = np.linalg.norm(self.target_positions - self.start_positions, axis=1)
        self.max_projection_ratio = np.where(seg_len < 1e-6, 1.0, 0.0)  # 起点即终点视为已到达
        self.task_progress = self.max_projection_ratio.copy()

        return self._get_obs()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        """
        执行一步动作。

        参数
        ----
        actions: [n_uavs] 或 [n_uavs, 1]，取值范围大致在 [-1, 1]
        """
        actions = np.asarray(actions, dtype=float).reshape(self.n_uavs, -1)

        prev_progress = self.task_progress.copy()

        # 1. 先移动：导航更新位置，每轮动完再保证安全
        if self.navigator is not None:
            movements = self.navigator.compute_movements(
                self.positions, self.target_positions, self.world_size
            )
        else:
            movements = self.rng.normal(0.0, 0.1, size=(self.n_uavs, 3))
        # 单步位移限制在安全空间内（中心不超出当前 space_sizes 对应立方体）
        half_nav = self.space_sizes / 2.0
        max_axis = np.max(np.abs(movements), axis=1)
        scale = np.minimum(1.0, half_nav / np.maximum(max_axis, 1e-9))
        movements = movements * scale[:, np.newaxis]
        self.positions = np.clip(self.positions + movements, 0.0, self.world_size)

        # 下一步校验：任意两机中心满足 max(sep) >= min_space_size + buffer，避免都压到 min 时仍重叠
        self._enforce_min_separation()

        # 2. 安全投影（原始动作 → proposed_sizes → 按 priority×urgency 冲突解决）
        self.space_sizes, safe_actions, proposed_sizes = self.safety_layer.safety_projection(
            actions, self._get_state()
        )
        self.space_sizes = np.clip(
            self.space_sizes, self.min_space_size, self.max_space_size
        )

        # 3. 任务进度 = 当前点到起点-终点连线投影占比，取历史最大（沿连线推进才涨进度）
        diff = self.target_positions - self.start_positions  # (n_uavs, 3)
        L_sq = np.sum(diff ** 2, axis=1)
        L_sq = np.maximum(L_sq, 1e-12)
        t = np.sum((self.positions - self.start_positions) * diff, axis=1) / L_sq
        t = np.clip(t, 0.0, 1.0)
        self.max_projection_ratio = np.maximum(self.max_projection_ratio, t)
        self.task_progress = self.max_projection_ratio.copy()

        # 4. 计算奖励
        delta_progress = np.maximum(0.0, self.task_progress - prev_progress)
        completed_now = (self.task_progress >= 0.95) & (prev_progress < 0.95)
        rewards = self._calculate_rewards(
            safe_actions,
            proposed_sizes,
            delta_progress=delta_progress,
            completed_now=completed_now,
        )

        # 5. 终止与已完成机空间：用同一阈值 0.95，避免分开两次判断产生偏差
        completed = self.task_progress >= 0.95
        self.space_sizes[completed] = self.min_space_size
        done = bool(np.mean(self.task_progress) >= 0.95)

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, rewards, done, info

    # ====== 状态与奖励 ====== #
    def _get_state(self) -> Dict[str, np.ndarray]:
        """供安全层使用的全局状态（含 priorities/urgencies 时启用按优先级每机缩放）。"""
        return {
            "positions": self.positions.copy(),
            "space_sizes": self.space_sizes.copy(),
            "priorities": self.priorities.copy(),
            "urgencies": self.urgencies.copy(),
        }

    def _enforce_min_separation(self) -> None:
        """
        下一步校验：保证任意两机满足 max(sep) >= min_space_size + buffer，
        避免两架都压到 min 时安全空间仍重叠。违反时将低 priority×urgency 的一方向外推。
        """
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
        """获取每个无人机的局部观测。"""
        obs_list = []
        for i in range(self.n_uavs):
            # 自身状态
            self_state = np.concatenate(
                [
                    self.positions[i],
                    np.array(
                        [
                            self.priorities[i],
                            self.urgencies[i],
                            self.task_progress[i],
                        ],
                        dtype=float,
                    ),
                ]
            )

            # 与其他无人机的距离
            distances = []
            for j in range(self.n_uavs):
                if i == j:
                    continue
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                distances.append(dist)

            obs_i = np.concatenate([self_state, np.array(distances, dtype=float)])
            obs_list.append(obs_i)

        return np.vstack(obs_list)

    def _get_global_obs(self) -> np.ndarray:
        """
        获取全局观测（供 MAPPO critic 使用）。

        简化实现：直接拼接所有局部观测向量。
        形状：(global_obs_dim,)
        """
        local_obs = self._get_obs()
        return local_obs.flatten()

    def _calculate_rewards(
        self,
        safe_actions: np.ndarray,
        proposed_sizes: np.ndarray,
        delta_progress: np.ndarray,
        completed_now: np.ndarray,
    ) -> np.ndarray:
        """计算综合奖励：任务奖励 + 安全惩罚 + 效率惩罚。"""
        safe_actions = safe_actions.squeeze(-1)

        # 1. 任务奖励：只奖励进度增量（避免到达后每步持续刷分） + 完成一次性奖励（首次到达阈值）
        weights = self.priorities * self.urgencies
        task_reward = self.task_weight * (weights * np.asarray(delta_progress, dtype=float))
        completion_reward = self.completion_bonus * (weights * completed_now.astype(float))

        # 2. 安全惩罚：actor 提出的尺寸（冲突解算前）vs 解算后无冲突的 space_sizes，被裁掉则惩罚
        proposed_sizes_clipped = np.clip(
            np.asarray(proposed_sizes, dtype=float).ravel(),
            self.min_space_size,
            self.max_space_size,
        )
        size_reduction = np.abs(proposed_sizes_clipped - self.space_sizes)
        safety_penalty = -self.safety_weight * size_reduction

        # 3. 效率惩罚：鼓励占用较紧凑的空间
        efficiency_penalty = -self.efficiency_weight * self.space_sizes

        rewards = task_reward + completion_reward + safety_penalty + efficiency_penalty
        return rewards.astype(float)


def test_environment() -> None:
    """简单测试环境是否正常工作。"""
    env = MultiUAVEnv()
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")

    # 执行随机动作
    actions = np.random.uniform(-1.0, 1.0, (env.n_uavs, 1))
    obs, rewards, done, info = env.step(actions)

    print(f"Rewards: {rewards}")
    print(f"Space sizes: {env.space_sizes}")
    assert not np.any(np.isnan(rewards)), "Rewards contain NaN!"


if __name__ == "__main__":
    test_environment()


