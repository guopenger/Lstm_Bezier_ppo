#!/usr/bin/env python
"""Phase 2 验证脚本: 分层网络前向推理测试"""
import sys
# Mock gym to avoid import error
sys.modules['gym'] = type(sys)('m')
sys.modules['gym.envs'] = type(sys)('m')
sys.modules['gym.envs.registration'] = type(sys)('m')
sys.modules['gym.envs.registration'].register = lambda **kw: None
sys.modules['gym.spaces'] = type(sys)('m')

import torch
from gym_carla.models.q1_decision import Q1Decision
from gym_carla.models.q2_decision import Q2Decision
from gym_carla.models.hierarchical_policy import HierarchicalPolicy

print("=" * 60)
print("Phase 2 Verification: random(1,3,18) -> goal(1,3) + offset(1,3)")
print("=" * 60)

policy = HierarchicalPolicy(state_dim=18, seq_len=3, hidden_dim=64, num_goals=3, num_offsets=3)
x = torch.randn(1, 3, 18)
result = policy.forward(x)

goal_q = result["goal_q"]
offset_q = result["offset_q"]
goal_idx = result["goal_idx"]
offset_idx = result["offset_idx"]

print(f"Input:       shape={x.shape}")
print(f"Goal Q:      shape={goal_q.shape}  values={[round(v, 4) for v in goal_q[0].tolist()]}")
print(f"Offset Q:    shape={offset_q.shape}  values={[round(v, 4) for v in offset_q[0].tolist()]}")

goal_names = ["left_LC", "keep", "right_LC"]
offset_names = ["left_off", "no_off", "right_off"]
print(f"Goal idx:    {goal_idx.item()} ({goal_names[goal_idx.item()]})")
print(f"Offset idx:  {offset_idx.item()} ({offset_names[offset_idx.item()]})")

# Action selection (single sample)
g, o = policy.select_action(x.squeeze(0), epsilon=0.0)
print(f"\nSingle-sample action: goal={g}, offset={o}")

# Batch action selection with exploration
goals, offsets = policy.select_action(torch.randn(8, 3, 18), epsilon=0.3)
print(f"Batch (8, eps=0.3): goals={goals.tolist()}, offsets={offsets.tolist()}")

# Parameters
p = policy.count_parameters()
print(f"\nParameters: Q1={p['q1']:,}  Q2={p['q2']:,}  Total={p['total_online']:,}")

# Softmax probs
gp, op = policy.get_probs(x)
print(f"Goal probs:   {[round(v, 4) for v in gp[0].tolist()]}  sum={gp.sum().item():.4f}")
print(f"Offset probs: {[round(v, 4) for v in op[0].tolist()]}  sum={op.sum().item():.4f}")

# Assertions
assert goal_q.shape == (1, 3), f"FAIL goal shape: {goal_q.shape}"
assert offset_q.shape == (1, 3), f"FAIL offset shape: {offset_q.shape}"
assert abs(gp.sum().item() - 1.0) < 1e-4, "FAIL goal probs dont sum to 1"
assert abs(op.sum().item() - 1.0) < 1e-4, "FAIL offset probs dont sum to 1"

print("\n*** ALL CHECKS PASSED ***")
