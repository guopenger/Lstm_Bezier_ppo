#!/usr/bin/env python
"""
test_phase1.py — Phase 1 验证脚本

验证目标:
  1. env.observation_space.shape == (3, 18)
  2. env.reset() 返回 shape (3, 18) 的 numpy array
  3. env.step() 返回 shape (3, 18) 的观测
  4. 8 区域感知值合理 (Δd > 0, 有车区域 Δd < 100)

使用方法:
  1. 先启动 CARLA 服务器: CarlaUE4.exe
  2. 运行: python test_phase1.py

预期输出:
  [Hierarchical Obs] state shape: (3, 18)
  Phase 1 验证通过！
"""

import sys
import os
import glob
import time
import numpy as np

# --- 添加 CARLA Python API 到路径 ---
carla_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'WindowsNoEditor')
carla_egg = glob.glob(os.path.join(carla_root, 'PythonAPI', 'carla', 'dist', 'carla-*-py3*-win_amd64.egg'))
if carla_egg:
    sys.path.insert(0, carla_egg[0])
    print(f"[Setup] CARLA egg found: {carla_egg[0]}")
else:
    # 尝试其他常见路径
    alt_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'WindowsNoEditor', 'PythonAPI', 'carla', 'dist'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'PythonAPI', 'carla', 'dist'),
    ]
    for p in alt_paths:
        eggs = glob.glob(os.path.join(p, 'carla-*-py3*.egg'))
        if eggs:
            sys.path.insert(0, eggs[0])
            print(f"[Setup] CARLA egg found: {eggs[0]}")
            break
    else:
        print("[Warning] CARLA .egg not found. Make sure carla is importable.")

import gym
from gym_carla.envs.carla_env import CarlaEnv

# ==============================================================
# Phase 1 测试参数 (hierarchical=True 是关键)
# ==============================================================
params = {
    # CARLA 连接
    'port': 2000,
    'town': 'Town03',

    # 显示
    'display_size': 256,
    'obs_range': 32,
    'lidar_bin': 0.125,
    'd_behind': 12,
    'display_route': True,

    # 仿真
    'dt': 0.1,
    'max_time_episode': 200,
    'max_waypt': 12,
    'max_past_step': 1,
    'max_ego_spawn_times': 200,

    # NPC (加一些车来测试 zone_detector)
    'number_of_vehicles': 20,
    'number_of_walkers': 0,

    # 动作空间 (Phase 1 暂时保持连续, Phase 4 改为 MultiDiscrete)
    'discrete': False,
    'continuous_accel_range': [-3.0, 3.0],
    'continuous_steer_range': [-0.3, 0.3],
    'discrete_acc': [-3.0, 0.0, 3.0],
    'discrete_steer': [-0.2, 0.0, 0.2],

    # 目标
    'desired_speed': 8,
    'out_lane_thres': 2.0,

    # 车辆
    'ego_vehicle_filter': 'vehicle.lincoln*',
    'task_mode': 'random',

    # ★ 关键: 开启分层 RL 模式 ★
    'hierarchical': True,
    'state_dim': 18,
    'seq_len': 3,
    'lane_width': 3.5,
}


def run_test():
    """Phase 1 验证流程。"""
    print("\n" + "=" * 60)
    print("Phase 1 验证: env 输出 18 维 × 3 步 状态序列")
    print("=" * 60)

    # --- 创建环境 ---
    print("\n[1/5] 创建环境...")
    env = CarlaEnv(params)

    # --- 检查观测空间 ---
    print(f"\n[2/5] 观测空间: {env.observation_space}")
    assert env.observation_space.shape == (3, 18), \
        f"观测空间形状错误! 期望 (3, 18), 得到 {env.observation_space.shape}"
    print(f"  ✓ observation_space.shape = {env.observation_space.shape}")

    # --- Reset ---
    print(f"\n[3/5] 执行 env.reset()...")
    obs = env.reset()
    print(f"  obs type: {type(obs)}")
    print(f"  obs shape: {obs.shape}")
    print(f"  obs dtype: {obs.dtype}")
    assert isinstance(obs, np.ndarray), f"obs 应为 numpy array, 得到 {type(obs)}"
    assert obs.shape == (3, 18), f"obs 形状错误! 期望 (3, 18), 得到 {obs.shape}"
    assert obs.dtype == np.float32, f"obs dtype 错误! 期望 float32, 得到 {obs.dtype}"
    print(f"  ✓ reset() 返回 shape={obs.shape}, dtype={obs.dtype}")

    # --- Step 多步 ---
    print(f"\n[4/5] 执行 10 步 env.step()...")
    from gym_carla.envs.zone_detector import ZoneDetector
    zone_det = ZoneDetector()

    for step in range(10):
        # 随机连续动作 (Phase 1 仍用原始 acc/steer)
        action = np.array([1.0, 0.0], dtype=np.float32)  # 轻微加速, 直行
        obs, reward, done, info = env.step(action)

        assert obs.shape == (3, 18), f"Step {step}: obs 形状错误 {obs.shape}"

        if step < 3 or done:
            latest = obs[-1]  # 最新一帧
            print(f"\n  Step {step}: reward={reward:.2f}, done={done}")
            print(f"    v_ego={latest[0]:.2f} m/s, lane_id={latest[1]:.0f}")
            # 检查是否有车辆被检测到
            dists = latest[10:18]
            detected = np.sum(dists < 99.0)
            print(f"    检测到 {detected}/8 个区域有车辆")
            print(f"    Δd: [{', '.join(f'{d:.1f}' for d in dists)}]")

        if done:
            print(f"\n  Episode 在 step {step} 终止, 执行 reset...")
            obs = env.reset()
            assert obs.shape == (3, 18)

    print(f"\n  ✓ 10 步全部通过, obs.shape 始终为 (3, 18)")

    # --- 数值合理性检查 ---
    print(f"\n[5/5] 数值合理性检查...")
    latest = obs[-1]
    assert latest[0] >= 0, f"v_ego 不应为负: {latest[0]}"
    assert all(latest[10:18] > 0), f"Δd 应全部为正: {latest[10:18]}"
    print(f"  ✓ v_ego={latest[0]:.2f} >= 0")
    print(f"  ✓ 所有 Δd > 0")

    print("\n" + "=" * 60)
    print("★ Phase 1 验证全部通过! ★")
    print("  observation_space: Box(3, 18)")
    print("  每步输出: np.ndarray shape=(3, 18), dtype=float32")
    print("  18维 = [v_ego, lane_id, Δv×8, Δd×8]")
    print("  3步 = 过去 3 个采样时刻的滑动窗口")
    print("=" * 60)

    # 清理
    # env._clear_all_actors(...)  # CarlaEnv 的析构会处理


if __name__ == '__main__':
    try:
        run_test()
    except KeyboardInterrupt:
        print("\n用户中断测试。")
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
