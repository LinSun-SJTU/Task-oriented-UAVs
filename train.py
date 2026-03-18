import json
import os
import argparse
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from environments import MultiUAVEnv
from algorithms import MAPPO
from algorithms.mappo import MAPPOConfig
from navigation import get_navigator

# 轨迹 JSON 导出用常量（与前端展示格式一致）
STEP_INTERVAL_MS = 100
BASE_TS_MS = 1742393414649


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(cfg: Dict[str, Any]) -> MultiUAVEnv:
    env_cfg = cfg["environment"]
    safety_cfg = cfg.get("safety", {})
    rw_cfg = cfg.get("reward_weights", {})
    nav_cfg = cfg.get("navigation", {})

    navigator = None
    if nav_cfg.get("type"):
        navigator = get_navigator(
            nav_cfg.get("type", "none"),
            grid_resolution=nav_cfg.get("grid_resolution", 1.0),
            step_size=nav_cfg.get("step_size", 0.5),
        )

    ts = cfg.get("task_scheduler")
    env = MultiUAVEnv(
        n_uavs=env_cfg.get("n_uavs", 5),
        world_size=tuple(env_cfg.get("world_size", [10.0, 10.0, 10.0])),
        min_space_size=env_cfg.get("min_space_size", 2.0),
        max_space_size=env_cfg.get("max_space_size", 6.0),
        priority_range=tuple(env_cfg.get("priority_range", [0.1, 1.0])),
        urgency_range=tuple(env_cfg.get("urgency_range", [0.0, 1.0])),
        safety_buffer=safety_cfg.get("buffer", 0.2),
        task_weight=rw_cfg.get("task_weight", 1.0),
        completion_bonus=rw_cfg.get("completion_bonus", 0.0),
        safety_weight=rw_cfg.get("safety_weight", 0.5),
        efficiency_weight=rw_cfg.get("efficiency_weight", 0.1),
        navigator=navigator,
        task_scheduler=ts,
    )
    return env


def make_mappo(env: MultiUAVEnv, cfg: Dict[str, Any]) -> MAPPO:
    algo_cfg = cfg["algorithm"]

    mappo_cfg = MAPPOConfig(
        actor_lr=algo_cfg.get("actor_lr", 3e-4),
        critic_lr=algo_cfg.get("critic_lr", 1e-3),
        gamma=algo_cfg.get("gamma", 0.99),
        gae_lambda=algo_cfg.get("gae_lambda", 0.95),
        clip_epsilon=algo_cfg.get("clip_epsilon", 0.2),
        ppo_epochs=algo_cfg.get("ppo_epochs", 10),
        batch_size=algo_cfg.get("batch_size", 64),
        hidden_dim=algo_cfg.get("hidden_dim", 64),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return MAPPO(env, mappo_cfg)


def _position_to_include_area(
    position: np.ndarray, space_size: float, margin: float = 0.0
) -> List[float]:
    """由中心位置和边长得到 [minX, minY, minZ, maxX, maxY, maxZ]。margin>0 时边界内缩，便于可视化可动区。"""
    half = max(0.0, float(space_size) / 2.0 - float(margin))
    p = position.ravel()
    return [
        float(p[0] - half), float(p[1] - half), float(p[2] - half),
        float(p[0] + half), float(p[1] + half), float(p[2] + half),
    ]


def run_export_episode(
    env: MultiUAVEnv,
    algo: MAPPO,
    max_steps: int,
    step_interval_ms: int = STEP_INTERVAL_MS,
    base_ts_ms: int = BASE_TS_MS,
) -> List[Dict[str, Any]]:
    """用当前策略跑一个 episode，收集每步每机的状态，返回前端格式的 devices 列表。"""
    devices: List[Dict[str, Any]] = []
    obs = env.reset()
    sched = bool(getattr(env, "task_scheduler_enabled", False))
    # 每机上次写入 JSON 的位置；调度模式下 inactive 时清空，新任务首帧速度为 0
    last_exported: List[Any] = [None] * env.n_uavs

    for step in range(max_steps):
        ts_ms = base_ts_ms + step * step_interval_ms
        for i in range(env.n_uavs):
            act = bool(getattr(env, "active", np.ones(env.n_uavs, dtype=bool))[i])
            if sched and not act:
                last_exported[i] = None
                continue

            pos = env.positions[i]
            size = env.space_sizes[i]
            p = float(env.priorities[i])
            u = float(env.urgencies[i])
            uid = f"{i + 1:03d}"
            if last_exported[i] is None:
                vel = [0.0, 0.0, 0.0]
            else:
                le = last_exported[i]
                vel = [
                    float(pos[0] - le[0]),
                    float(pos[1] - le[1]),
                    float(pos[2] - le[2]),
                ]
            last_exported[i] = pos.copy()
            position_list = [float(pos[0]), float(pos[1]), float(pos[2])]
            include_area = _position_to_include_area(pos, size)
            target_list = [
                float(env.target_positions[i][0]),
                float(env.target_positions[i][1]),
                float(env.target_positions[i][2]),
            ]
            rec: Dict[str, Any] = {
                "uid": uid,
                "position": position_list,
                "velocity": vel,
                "ts": ts_ms,
                "include_area": include_area,
                "target_pos": target_list,
                "task_score": round(p * u, 4),
            }
            if sched:
                rec["active"] = True
            devices.append(rec)

        actions = []
        for i in range(env.n_uavs):
            obs_tensor = torch.as_tensor(obs[i], dtype=torch.float32, device=algo.device).unsqueeze(0)
            with torch.no_grad():
                action = algo.actors[i](obs_tensor).cpu().numpy().reshape(-1)
            actions.append(action)
        actions_arr = np.stack(actions, axis=0)
        obs, _, done, _ = env.step(actions_arr)
        if done:
            break
    return devices


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    env = make_env(cfg)
    algo = make_mappo(env, cfg)

    train_cfg = cfg["training"]
    n_episodes = int(train_cfg.get("n_episodes", 1000))
    max_steps = int(train_cfg.get("max_steps", 100))
    update_freq = int(train_cfg.get("update_freq", 5))
    save_freq = int(train_cfg.get("save_freq", 100))
    save_dir = train_cfg.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    reward_history = []
    reward_log_path = os.path.join(save_dir, "reward_log.txt")
    f_log = open(reward_log_path, "w", encoding="utf-8")

    for episode in range(1, n_episodes + 1):
        # 采样一条 episode
        obs = env.reset()
        done = False
        episode_rewards = np.zeros(env.n_uavs, dtype=np.float32)

        sched = getattr(env, "task_scheduler_enabled", False)
        for step_i in range(max_steps):
            actions = []
            log_probs = []

            # 对每个actor网络输出action
            for i in range(env.n_uavs):
                obs_tensor = torch.as_tensor(
                    obs[i], dtype=torch.float32, device=algo.device
                ).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob = algo.actors[i].act(obs_tensor)
                actions.append(action.cpu().numpy().reshape(-1))
                log_probs.append(log_prob.cpu().item())

            actions_arr = np.stack(actions, axis=0)
            next_obs, rewards, done, _ = env.step(actions_arr)

            done_buf = done or (
                sched and step_i >= max_steps - 1
            )

            global_obs = env._get_global_obs()
            for i in range(env.n_uavs):
                ai = float(env.active[i])
                algo.buffers[i].add(
                    obs=obs[i],
                    action=actions[i],
                    reward=rewards[i],
                    log_prob=log_probs[i],
                    global_obs=global_obs,
                    done=done_buf,
                    active=ai,
                )

            obs = next_obs
            episode_rewards += rewards

            if done and not sched:
                break

        mean_reward = float(episode_rewards.mean())
        reward_history.append(mean_reward)
        f_log.write(f"{episode},{mean_reward}\n")
        f_log.flush()

        # 更新策略
        if episode % update_freq == 0:
            algo.update_policy()

        if episode % 10 == 0:
            avg_recent = float(np.mean(reward_history[-10:]))
            print(
                f"Episode {episode}/{n_episodes}, "
                f"Avg Reward (last 10): {avg_recent:.3f}"
            )

        # 按 save_freq 导出轨迹 JSON（仅生成 json，不保存 pt）
        if episode % save_freq == 0:
            json_dir = os.path.join(save_dir, "json")
            os.makedirs(json_dir, exist_ok=True)
            json_path = os.path.join(json_dir, f"mappo_ep{episode}.json")
            devices = run_export_episode(env, algo, max_steps)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"devices": devices}, f, ensure_ascii=False, indent=2)
            print(f"Exported trajectory to {json_path} ({len(devices)} records)")

    f_log.close()
    print(f"Reward log written to {reward_log_path}")

    # 训练结束后保存 reward 曲线数据并生成收敛图
    reward_path = os.path.join(save_dir, "reward_history.npy")
    np.save(reward_path, np.asarray(reward_history, dtype=np.float32))
    print(f"Saved reward history to {reward_path}")

    try:
        fig_path = os.path.join(save_dir, "reward_curve.png")
        plot_reward_curve(reward_history, fig_path)
    except Exception as e:
        print(f"Could not generate plot (matplotlib missing?). Error: {e}")


def load_reward_log(log_path: str) -> List[float]:
    """从 reward 日志文件加载 reward 序列。每行格式：episode,reward"""
    rewards: List[float] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                rewards.append(float(parts[1]))
            else:
                rewards.append(float(parts[0]))
    return rewards


def plot_reward_curve(
    rewards: List[float],
    save_path: str,
    window: int = 10,
) -> None:
    """根据 reward 列表绘制收敛曲线并保存。"""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not installed, cannot plot.")
        return
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    episodes = np.arange(1, len(rewards_arr) + 1)
    if len(rewards_arr) >= window:
        kernel = np.ones(window, dtype=np.float32) / float(window)
        rewards_ma = np.convolve(rewards_arr, kernel, mode="valid")
        episodes_ma = np.arange(window, len(rewards_arr) + 1)
    else:
        rewards_ma = None
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, rewards_arr, label="Episode mean reward", linewidth=1.0, alpha=0.7)
    if rewards_ma is not None:
        plt.plot(episodes_ma, rewards_ma, label=f"Moving avg (window={window})", linewidth=2.0)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training convergence curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved reward curve to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/demo_config.yaml",
        help="Path to config YAML file.",
    )
    parser.add_argument(
        "--plot-from-log",
        type=str,
        default=None,
        metavar="PATH",
        help="从已有 reward 日志生成收敛曲线图，不训练。例如: --plot-from-log checkpoints/reward_log.txt",
    )
    args = parser.parse_args()

    if args.plot_from_log is not None:
        if not os.path.isfile(args.plot_from_log):
            print(f"Error: log file not found: {args.plot_from_log}")
            raise SystemExit(1)
        rewards = load_reward_log(args.plot_from_log)
        if not rewards:
            print("Error: no rewards in log file.")
            raise SystemExit(1)
        save_dir = os.path.dirname(args.plot_from_log) or "."
        fig_path = os.path.join(save_dir, "reward_curve.png")
        plot_reward_curve(rewards, fig_path)
    else:
        train(args.config)


