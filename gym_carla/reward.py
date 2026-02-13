#!/usr/bin/env python
"""
reward.py — 论文奖励函数实现

Ref: A Trajectory Planning and Tracking Method Based on Deep Hierarchical RL
- Eq.(3): r_t = m × R_i + (1-m) × R_e
- Eq.(4): R_i = σ_c + σ_v  (Intrinsic Reward)
- Eq.(5): σ_c = -5P_c
- Eq.(6): σ_v = 10exp(-err_v² / (5V_max))
- Eq.(7): R_e = σ_c + σ_lc (Extrinsic Reward)
- Eq.(8): σ_lc = 10exp(-err_v² / (5V_max)) × P_l
"""

import numpy as np
from gym_carla import config as cfg


def get_hierarchical_reward(ego, collision_hist, desired_speed, goal, last_lane_speed):
    """
    计算论文中的分层奖励函数
    
    Args:
        ego: CARLA ego vehicle object
        collision_hist: list of collision events
        desired_speed: V_max in paper (m/s)
        goal: Q1 action (0=left lane change, 1=keep lane, 2=right lane change)
        last_lane_speed: 上一次保持车道时的速度 (m/s)
    
    Returns:
        tuple: (reward, new_last_lane_speed)
    """
    # --- 0. 基础数据 ---
    v = ego.get_velocity()
    current_speed = np.sqrt(v.x**2 + v.y**2)  # m/s
    v_max = desired_speed  # 论文中的 V_max
    
    # --- 1. 碰撞因子 P_c (Eq.5) ---
    p_c = 1.0 if len(collision_hist) > 0 else 0.0
    
    # --- 2. 换道有效性因子 P_l (Eq.8) ---
    is_lane_change = (goal != 1)  # goal=1 表示保持车道
    p_l = 0.0
    new_last_lane_speed = last_lane_speed
    
    if is_lane_change:
        # 换道后速度是否提升（论文原文："the vehicle's speed has increased after changing lanes"）
        # 加一个小阈值 0.5 m/s 避免噪声
        if current_speed > last_lane_speed + 0.5:
            p_l = 1.0
    else:
        # 保持车道时，更新基准速度
        new_last_lane_speed = current_speed
    
    # --- 3. 计算各项 σ ---
    
    # σ_c: 碰撞惩罚 (Eq.5)
    sigma_c = -5.0 * p_c
    
    # σ_v: 速度跟踪奖励 (Eq.6)
    err_v = current_speed - v_max
    # 避免除零，虽然 v_max 通常 > 0
    sigma_v = 10.0 * np.exp(-(err_v**2) / (5.0 * max(v_max, 1.0)))
    
    # σ_lc: 换道有效性奖励 (Eq.8)
    # 注意：论文公式是 10exp(...) × P_l，和 σ_v 共享同一个 exp 项
    sigma_lc = sigma_v * p_l
    
    # --- 4. 组合 R_i 和 R_e ---
    
    # Intrinsic Reward (Eq.4)
    r_i = sigma_c + sigma_v
    
    # Extrinsic Reward (Eq.7)
    r_e = sigma_c + sigma_lc
    
    # --- 5. 最终加权 (Eq.3) ---
    # m 是权重参数，论文没有给出具体值
    # 建议: m=0.6 (偏重内在控制稳定性) 或 m=0.5 (均衡)
    m = getattr(cfg, 'REWARD_WEIGHT_M', 0.5)
    
    r_total = m * r_i + (1.0 - m) * r_e
    
    return r_total, new_last_lane_speed


# ======================================================================
# 辅助函数：用于调试和分析
# ======================================================================

def get_hierarchical_reward_with_details(ego, collision_hist, desired_speed, 
                                         goal, last_lane_speed):
    """
    返回详细的奖励分解，用于 TensorBoard 记录或调试
    
    Returns:
        dict: {
            'reward': float,
            'new_last_lane_speed': float,
            'sigma_c': float,
            'sigma_v': float,
            'sigma_lc': float,
            'r_i': float,
            'r_e': float,
            'p_c': float,
            'p_l': float,
        }
    """
    v = ego.get_velocity()
    current_speed = np.sqrt(v.x**2 + v.y**2)
    v_max = desired_speed
    
    p_c = 1.0 if len(collision_hist) > 0 else 0.0
    
    is_lane_change = (goal != 1)
    p_l = 0.0
    new_last_lane_speed = last_lane_speed
    
    if is_lane_change:
        if current_speed > last_lane_speed + 0.5:
            p_l = 1.0
    else:
        new_last_lane_speed = current_speed
    
    sigma_c = -5.0 * p_c
    
    err_v = current_speed - v_max
    sigma_v = 10.0 * np.exp(-(err_v**2) / (5.0 * max(v_max, 1.0)))
    
    sigma_lc = sigma_v * p_l
    
    r_i = sigma_c + sigma_v
    r_e = sigma_c + sigma_lc
    
    m = getattr(cfg, 'REWARD_WEIGHT_M', 0.5)
    r_total = m * r_i + (1.0 - m) * r_e
    
    return {
        'reward': r_total,
        'new_last_lane_speed': new_last_lane_speed,
        'sigma_c': sigma_c,
        'sigma_v': sigma_v,
        'sigma_lc': sigma_lc,
        'r_i': r_i,
        'r_e': r_e,
        'p_c': p_c,
        'p_l': p_l,
        'current_speed': current_speed,
        'err_v': err_v,
    }
