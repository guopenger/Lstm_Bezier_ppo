#!/usr/bin/env python
"""
config.py — 分层强化学习统一超参数配置

论文: A Trajectory Planning and Tracking Method Based on Deep Hierarchical RL

所有模块从此处读取默认值。训练脚本可通过 params dict 覆盖。
"""

# ==============================================================
# State Space (论文 §2.1.2)
# ==============================================================
STATE_DIM = 18              # 18维状态向量 [v_ego, lane_id, Δv×8, Δd×8]
SEQ_LEN = 3                 # 过去 3 个采样时刻 (论文 §2.2.1)

# ==============================================================
# Network Architecture (论文 §2.2.1, Figure 3)
# ==============================================================
HIDDEN_DIM = 64             # LSTM 隐藏层维度
NUM_GOALS = 3               # Q1 输出: 左换道 / 保持 / 右换道
# Q2 输出: 连续偏移值 p_off ∈ [-OFFSET_RANGE, OFFSET_RANGE] (论文 Eq.2)

# ==============================================================
# Zone Detection (论文 §2.1.2, Figure 2)
# ==============================================================
LANE_WIDTH = 3.5             # 标准车道宽度 (m), CARLA 默认
ZONE_FRONT_DIST = 50.0       # 前方最大感知距离 (m)
ZONE_REAR_DIST = 30.0        # 后方最大感知距离 (m)
ZONE_SIDE_THRESHOLD = 5.0    # 侧方区域纵向阈值 (m)
DEFAULT_REL_SPEED = 0.0      # 区域内无车时的默认相对速度 (m/s)
DEFAULT_REL_DIST = 100.0     # 区域内无车时的默认相对距离 (m)

# ==============================================================
# Planning (论文 §2.2.2)
# ==============================================================
PLAN_HORIZON = 30.0          # 贝塞尔曲线纵向规划距离 (m)
OFFSET_RANGE = 2.0           # Q2 连续偏移最大范围 (m), |p_off| ≤ OFFSET_RANGE
BEZIER_SAMPLES = 50          # 轨迹采样点数

# ==============================================================
# Control (PID / Pure Pursuit — 训练阶段替代 DMPC)
# ==============================================================
WHEELBASE = 2.85             # Lincoln MKZ 轴距 (m)
DESIRED_SPEED = 8.0          # 巡航目标速度 (m/s)
LOOKAHEAD_DIST = 5.0         # Pure Pursuit 基础前视距离 (m)
CONTROL_DT = 0.1             # 控制周期 (s), 与 CARLA 同步步长一致

# ==============================================================
# Training — PPO (Proximal Policy Optimization)
# ==============================================================
LEARNING_RATE = 3e-4
GAMMA = 0.99                 # 折扣因子
GAE_LAMBDA = 0.95            # GAE λ (Generalized Advantage Estimation)
PPO_CLIP_EPSILON = 0.2       # PPO clip ratio ε
PPO_EPOCHS = 4               # 每次 rollout 后的 PPO 更新轮数
NUM_MINI_BATCHES = 4         # 每轮分成的 mini-batch 数量
VALUE_COEF = 0.5             # Value loss 系数 c1
ENTROPY_COEF = 0.01          # Entropy bonus 系数 c2
MAX_GRAD_NORM = 0.5          # 梯度裁剪范数
ROLLOUT_STEPS = 1024         # 每次收集的交互步数
NUM_ITERATIONS = 300         # PPO 迭代次数 (总步数 ≈ ROLLOUT_STEPS × NUM_ITERATIONS)
MAX_STEPS_PER_EPISODE = 500  # 单 episode 最大步数
OFFSET_LOG_STD_INIT = 0.0    # Q2 高斯策略初始 log(σ)

# ==============================================================
# Reward Function 
# ==============================================================
REWARD_WEIGHT_M = 0.5       # Eq.(3) 中的 m，平衡内在和外在奖励
                            # m=0.5: 均衡速度跟踪和换道决策
                            # m=0.6: 偏重速度跟踪（内在稳定性）
                            # m=0.4: 偏重换道决策（外在任务完成）
# ==============================================================
# CARLA 连接
# ==============================================================
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CARLA_TOWN = 'Town03'
CARLA_DT = 0.1
NUMBER_OF_VEHICLES = 50
NUMBER_OF_WALKERS = 0
