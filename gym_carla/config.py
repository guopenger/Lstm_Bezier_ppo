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
NUM_OFFSETS = 3              # Q2 输出: 偏左 / 不偏 / 偏右

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
OFFSET_MAGNITUDE = 0.5       # Q2 偏移步长 (m)
BEZIER_SAMPLES = 50          # 轨迹采样点数

# ==============================================================
# Control (PID / Pure Pursuit — 训练阶段替代 DMPC)
# ==============================================================
WHEELBASE = 2.85             # Lincoln MKZ 轴距 (m)
DESIRED_SPEED = 8.0          # 巡航目标速度 (m/s)
LOOKAHEAD_DIST = 5.0         # Pure Pursuit 基础前视距离 (m)
CONTROL_DT = 0.1             # 控制周期 (s), 与 CARLA 同步步长一致

# ==============================================================
# Training
# ==============================================================
LEARNING_RATE = 3e-4
GAMMA = 0.99                 # 折扣因子
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 100000
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 500
TARGET_UPDATE_FREQ = 100     # Target network 更新频率

# ==============================================================
# Reward Weights (分层奖励)
# ==============================================================
# Q1 行为层奖励
W_COLLISION = -200.0
W_SPEED = 1.0               # 纵向速度奖励系数
W_LANE_CHANGE_SUCCESS = 5.0  # 成功换道到更快车道
W_UNNECESSARY_LC = -1.0      # 不必要换道惩罚
W_OUT_LANE = -1.0            # 出车道惩罚

# Q2 轨迹层奖励
W_TRACKING_ERROR = -1.0      # 横向跟踪误差惩罚
W_CURVATURE = -0.5           # 曲率惩罚 (平滑性)
W_COMFORT = -0.2             # 横向加速度惩罚 (舒适性)
W_STEP = -0.1                # 每步存在惩罚

# ==============================================================
# CARLA 连接
# ==============================================================
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CARLA_TOWN = 'Town03'
CARLA_DT = 0.1
NUMBER_OF_VEHICLES = 20
NUMBER_OF_WALKERS = 0
