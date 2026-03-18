"""
可插拔导航模块：接入不同路径/导航算法，与空间分配策略解耦。
"""
from .base import BaseNavigator
from .astar import AStarNavigator


def get_navigator(name: str, **kwargs):
    """根据配置名创建导航器，便于后续扩展最短路径等算法。"""
    name = (name or "").strip().lower()
    if name in ("astar", "a*"):
        return AStarNavigator(
            grid_resolution=kwargs.get("grid_resolution", 1.0),
            step_size=kwargs.get("step_size", 0.5),
        )
    if name in ("none", ""):
        return None
    raise ValueError(f"Unknown navigator: {name!r}. Use 'astar' or 'none'.")
