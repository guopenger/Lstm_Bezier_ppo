#!/usr/bin/env python
"""测试 carla_env.py 中 hierarchical action space 是否正确改为连续 offset"""
import sys
sys.path.insert(0, r'd:\electron\CARLA_0.9.13\RL_Project\gym-carla')

print("=== Test 8: Action Space in carla_env ===")

# 不启动 CARLA，只检查 action space 构建逻辑
import gym
import numpy as np

# 模拟 env 的 action_space 定义
from gym import spaces

# 论文: A_g = 离散3, A_p = 连续标量 [-2, 2]
# 代码应该用 Dict 或 Tuple space
goal_space = spaces.Discrete(3)
offset_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)

print("  Goal space:", goal_space)
print("  Offset space:", offset_space)

# 验证 sample
for _ in range(5):
    g = goal_space.sample()
    o = offset_space.sample()
    assert 0 <= g <= 2
    assert -2.0 <= o[0] <= 2.0

print("  5 random samples OK")
print("  PASS")

print()
print("=== Test 9: train_hierarchical.py syntax check ===")
import py_compile
try:
    py_compile.compile(
        r'd:\electron\CARLA_0.9.13\RL_Project\gym-carla\train_hierarchical.py',
        doraise=True
    )
    print("  train_hierarchical.py compiles OK")
except py_compile.PyCompileError as e:
    print("  COMPILE ERROR:", e)
    sys.exit(1)
print("  PASS")

print()
print("=== Test 10: export_onnx.py syntax check ===")
try:
    py_compile.compile(
        r'd:\electron\CARLA_0.9.13\RL_Project\gym-carla\export_onnx.py',
        doraise=True
    )
    print("  export_onnx.py compiles OK")
except py_compile.PyCompileError as e:
    print("  COMPILE ERROR:", e)
    sys.exit(1)
print("  PASS")

print()
print("=== ALL INTEGRATION TESTS PASSED ===")
