"""
导航算法基类：统一接口，便于接入 A*、最短路径等不同策略。

空间分配层（安全层）不干预导航输出，仅通过独占空间大小协调多机；
导航只负责“往哪走”，类似交通系统中车辆自身的路径规划，
与交通信号灯（空间分配）协同工作。
"""
import numpy as np
from abc import ABC, abstractmethod


class BaseNavigator(ABC):
    """导航器基类：根据当前位置与目标，计算本步期望位移 (n_uavs, 3)。"""

    @abstractmethod
    def compute_movements(
        self,
        positions: np.ndarray,
        targets: np.ndarray,
        world_bounds: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        计算每个无人机本步的期望位移（不包含空间分配逻辑）。

        参数
        ----
        positions : (n_uavs, 3) 当前坐标
        targets   : (n_uavs, 3) 目标坐标
        world_bounds : (3,) 世界边界 [x_max, y_max, z_max]，假设起点为 0

        返回
        ----
        movements : (n_uavs, 3) 本步位移向量，由环境做边界 clip 等
        """
        pass
