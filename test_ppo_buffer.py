#!/usr/bin/env python
"""测试 PPOBuffer (GAE + 数据收集)"""
import numpy as np
import torch
import sys
sys.path.insert(0, r'd:\electron\CARLA_0.9.13\RL_Project\gym-carla')

# 直接 import PPOBuffer
# 由于 train_hierarchical.py 会尝试 import carla，我们手动模拟
# 先测试 PPOBuffer 的独立逻辑

print("=== Test 6: PPOBuffer (GAE) ===")

class PPOBuffer:
    """与 train_hierarchical.py 中相同的实现，用于独立测试"""
    def __init__(self, max_size, seq_len, state_dim, gamma=0.99, lam=0.95):
        self.max_size = max_size
        self.gamma = gamma
        self.lam = lam
        self.states = np.zeros((max_size, seq_len, state_dim), dtype=np.float32)
        self.goals = np.zeros(max_size, dtype=np.int64)
        self.offsets = np.zeros(max_size, dtype=np.float32)
        self.log_probs_goal = np.zeros(max_size, dtype=np.float32)
        self.log_probs_offset = np.zeros(max_size, dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.values = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)
        self.advantages = np.zeros(max_size, dtype=np.float32)
        self.returns = np.zeros(max_size, dtype=np.float32)
        self.ptr = 0
        self.path_start = 0
        self.size = 0

    def store(self, state, goal, offset, log_prob_goal, log_prob_offset,
              reward, value, done):
        assert self.ptr < self.max_size
        self.states[self.ptr] = state
        self.goals[self.ptr] = goal
        self.offsets[self.ptr] = offset
        self.log_probs_goal[self.ptr] = log_prob_goal
        self.log_probs_offset[self.ptr] = log_prob_offset
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)
        self.ptr += 1
        self.size = min(self.size + 1, self.max_size)

    def finish_path(self, last_value=0.0):
        path_slice = slice(self.path_start, self.ptr)
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]
        
        path_len = self.ptr - self.path_start
        advantages = np.zeros(path_len, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(path_len)):
            if t == path_len - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            last_gae = delta + self.gamma * self.lam * (1 - next_done) * last_gae
            advantages[t] = last_gae
        
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = advantages + values
        self.path_start = self.ptr

    def get(self):
        actual_size = self.ptr
        data = {
            'states': torch.FloatTensor(self.states[:actual_size]),
            'goals': torch.LongTensor(self.goals[:actual_size]),
            'offsets': torch.FloatTensor(self.offsets[:actual_size]),
            'old_log_probs_goal': torch.FloatTensor(self.log_probs_goal[:actual_size]),
            'old_log_probs_offset': torch.FloatTensor(self.log_probs_offset[:actual_size]),
            'advantages': torch.FloatTensor(self.advantages[:actual_size]),
            'returns': torch.FloatTensor(self.returns[:actual_size]),
        }
        adv = data['advantages']
        if adv.std() > 1e-8:
            data['advantages'] = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.ptr = 0
        self.path_start = 0
        self.size = 0
        return data

# 测试
buf = PPOBuffer(100, 3, 18, gamma=0.99, lam=0.95)
for i in range(10):
    buf.store(
        state=np.random.randn(3, 18).astype(np.float32),
        goal=i % 3,
        offset=float(i) * 0.1 - 0.5,
        log_prob_goal=-1.0,
        log_prob_offset=-0.8,
        reward=1.0 if i < 8 else -5.0,
        value=0.5,
        done=(i == 9)
    )
buf.finish_path(last_value=0.0)
data = buf.get()

print("  states shape:", data['states'].shape, "(expect [10, 3, 18])")
print("  advantages shape:", data['advantages'].shape, "(expect [10])")
print("  returns shape:", data['returns'].shape, "(expect [10])")
print("  goals:", data['goals'].tolist())
print("  offsets:", [round(v, 2) for v in data['offsets'].tolist()])
print("  advantages:", [round(v, 3) for v in data['advantages'].tolist()])
print("  returns:", [round(v, 3) for v in data['returns'].tolist()])

assert data['states'].shape == (10, 3, 18), "states shape wrong"
assert data['advantages'].shape == (10,), "advantages shape wrong"
assert data['returns'].shape == (10,), "returns shape wrong"
# Advantage 应该被标准化了
assert abs(data['advantages'].mean().item()) < 0.01, "advantages not normalized"
print("  Advantage mean ~0:", round(data['advantages'].mean().item(), 6))
print("  PASS")

print()
print("=== Test 7: BezierFitting with continuous offset ===")
from gym_carla.planning.frenet_transform import FrenetTransform
from gym_carla.planning.bezier_fitting import BezierFitting

waypoints = [[i * 5.0, 0.0, 0.0] for i in range(41)]
ft = FrenetTransform(ds=0.5)
ft.build_reference_line(waypoints)

planner = BezierFitting(frenet=ft, lane_width=3.5, plan_horizon=30.0, n_samples=50)

# 测试连续 offset
for offset_val in [-1.5, -0.5, 0.0, 0.5, 1.5]:
    traj = planner.generate_trajectory(10.0, 0.0, 0.0, goal=1, offset=offset_val)
    end_y = traj[-1, 1]
    print("  goal=1(keep), offset=%.1f -> end_y=%.3f" % (offset_val, end_y))
    # 保持车道时，终点 y 应该约等于 offset 值
    assert abs(end_y - offset_val) < 0.5, "Offset effect too far: %.3f vs %.3f" % (end_y, offset_val)

print("  PASS")

print()
print("=== ALL BUFFER + BEZIER TESTS PASSED ===")
