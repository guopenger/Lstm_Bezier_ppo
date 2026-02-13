#!/usr/bin/env python
"""
test_all_modules.py — 全模块离线测试

无需 CARLA 服务器运行。测试所有模块的接口一致性和基本功能。
模块:
  1. config.py             — 常量可访问
  2. zone_detector.py      — 区域分类逻辑
  3. state_buffer.py       — 滑动窗口缓冲
  4. frenet_transform.py   — Frenet⇌Cartesian 转换
  5. bezier_fitting.py     — 贝塞尔曲线生成
  6. trajectory_tracker.py — Pure Pursuit + PID 跟踪
  7. q1_decision.py        — Q1 LSTM+FC
  8. q2_decision.py        — Q2 LSTM+FC (Skip Connection)
  9. hierarchical_policy.py — 双层封装 + Target Network
 10. train_hierarchical.py  — ReplayBuffer + Trainer
 11. export_onnx.py         — ONNX 导出 (结构测试)
 12. End-to-end pipeline    — 全链路: obs → policy → bezier → tracker → control
"""

import sys
import os
import math
import numpy as np

# Mock gym to avoid import errors (gym 0.12.5 may not be in path)
mock_gym = type(sys)('gym')
mock_envs = type(sys)('gym.envs')
mock_reg = type(sys)('gym.envs.registration')
mock_reg.register = lambda **kw: None
mock_spaces = type(sys)('gym.spaces')
mock_utils = type(sys)('gym.utils')
mock_seeding = type(sys)('gym.utils.seeding')
mock_seeding.np_random = lambda seed=None: (np.random.RandomState(seed), seed)
mock_utils.seeding = mock_seeding

# Mock gym.Env base class
class MockEnv:
    pass
mock_gym.Env = MockEnv
mock_gym.spaces = mock_spaces

# Add Box, Dict, Discrete, MultiDiscrete to mock spaces
class MockBox:
    def __init__(self, *args, **kwargs): pass
class MockDict:
    def __init__(self, *args, **kwargs): pass
class MockDiscrete:
    def __init__(self, *args, **kwargs): pass
class MockMultiDiscrete:
    def __init__(self, *args, **kwargs): pass
mock_spaces.Box = MockBox
mock_spaces.Dict = MockDict
mock_spaces.Discrete = MockDiscrete
mock_spaces.MultiDiscrete = MockMultiDiscrete

sys.modules['gym'] = mock_gym
sys.modules['gym.envs'] = mock_envs
sys.modules['gym.envs.registration'] = mock_reg
sys.modules['gym.spaces'] = mock_spaces
sys.modules['gym.utils'] = mock_utils
sys.modules['gym.utils.seeding'] = mock_seeding

# Mock carla module (no server needed)
mock_carla = type(sys)('carla')
sys.modules['carla'] = mock_carla

# Mock pygame (already installed but avoid rendering)
# pygame is actually installed, don't mock it

# Mock skimage
mock_skimage = type(sys)('skimage')
mock_transform = type(sys)('skimage.transform')
mock_transform.resize = lambda *a, **kw: np.zeros((32, 32, 3))
mock_skimage.transform = mock_transform
sys.modules['skimage'] = mock_skimage
sys.modules['skimage.transform'] = mock_transform

import torch

# Project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0
total = 0


def test(name, condition, detail=""):
    global passed, failed, total
    total += 1
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    else:
        failed += 1
    extra = f" ({detail})" if detail else ""
    print(f"  [{status}] {name}{extra}")
    return condition


# ======================================================================
print("=" * 70)
print("TEST 1: config.py — Constants")
print("=" * 70)
from gym_carla import config as cfg

test("STATE_DIM = 18", cfg.STATE_DIM == 18)
test("SEQ_LEN = 3", cfg.SEQ_LEN == 3)
test("HIDDEN_DIM = 64", cfg.HIDDEN_DIM == 64)
test("NUM_GOALS = 3", cfg.NUM_GOALS == 3)
test("NUM_OFFSETS = 3", cfg.NUM_OFFSETS == 3)
test("LANE_WIDTH = 3.5", cfg.LANE_WIDTH == 3.5)
test("PLAN_HORIZON = 30.0", cfg.PLAN_HORIZON == 30.0)
test("WHEELBASE = 2.85", cfg.WHEELBASE == 2.85)
test("DESIRED_SPEED = 8.0", cfg.DESIRED_SPEED == 8.0)
test("W_COLLISION = -200", cfg.W_COLLISION == -200.0)

# ======================================================================
print("\n" + "=" * 70)
print("TEST 2: zone_detector.py — Zone Classification")
print("=" * 70)
from gym_carla.envs.zone_detector import ZoneDetector

zd = ZoneDetector(lane_width=3.5, side_threshold=5.0)
zone_tests = [
    (20.0,  0.0, 6, "前方中央→CF"),
    (-15.0, 0.0, 2, "后方中央→CR"),
    (20.0,  3.0, 5, "前方左侧→LF"),
    (20.0, -3.0, 7, "前方右侧→RF"),
    (-15.0, 3.0, 3, "后方左侧→LR"),
    (-15.0,-3.0, 1, "后方右侧→RR"),
    (2.0,   3.0, 4, "侧方左→LS"),
    (2.0,  -3.0, 8, "侧方右→RS"),
]
for lx, ly, expected, desc in zone_tests:
    result = zd._classify_zone(lx, ly)
    test(f"Zone({lx:+.0f},{ly:+.0f})={expected}", result == expected, desc)

# format_state test
state_vec = np.zeros(18, dtype=np.float32)
state_vec[0] = 8.5
state_vec[1] = -1
formatted = zd.format_state(state_vec)
test("format_state returns string", isinstance(formatted, str) and len(formatted) > 10)

# ======================================================================
print("\n" + "=" * 70)
print("TEST 3: state_buffer.py — Sliding Window")
print("=" * 70)
from gym_carla.planning.state_buffer import StateBuffer

buf = StateBuffer(state_dim=18, seq_len=3)
test("Initial shape (3,18)", buf.get_numpy().shape == (3, 18))
test("Initial all zeros", np.allclose(buf.get_numpy(), 0.0))

for i in range(5):
    s = np.random.randn(18).astype(np.float32)
    s[0] = float(i)
    buf.push(s)

arr = buf.get_numpy()
test("After 5 pushes shape (3,18)", arr.shape == (3, 18))
test("Latest state v_ego=4", arr[-1, 0] == 4.0, f"got {arr[-1, 0]}")
test("Oldest state v_ego=2", arr[0, 0] == 2.0, f"got {arr[0, 0]}")

tensor = buf.get_sequence(as_batch=True)
test("Batch tensor shape (1,3,18)", tensor.shape == (1, 3, 18))
test("Tensor dtype float32", tensor.dtype == torch.float32)

buf.reset()
test("Reset clears to zeros", np.allclose(buf.get_numpy(), 0.0))

# Error handling
try:
    buf.push(np.zeros(10))
    test("Push wrong dim raises", False, "No error raised")
except ValueError:
    test("Push wrong dim raises ValueError", True)

# ======================================================================
print("\n" + "=" * 70)
print("TEST 4: frenet_transform.py — Coordinate Transform")
print("=" * 70)
from gym_carla.planning.frenet_transform import FrenetTransform

# Straight reference line
wps_straight = [[i * 5.0, 0.0, 0.0] for i in range(20)]
ft = FrenetTransform(ds=0.5)
ft.build_reference_line(wps_straight)
test("Reference line built", ft._is_built)
test("Total length ~95m", abs(ft.total_length - 95.0) < 1.0, f"got {ft.total_length:.1f}")

s, d = ft.cartesian_to_frenet(25.0, 3.0)
test("cart→frenet s≈25", abs(s - 25.0) < 1.0, f"s={s:.2f}")
test("cart→frenet d≈3", abs(d - 3.0) < 0.5, f"d={d:.2f}")

x, y = ft.frenet_to_cartesian(25.0, 3.0)
test("frenet→cart x≈25", abs(x - 25.0) < 0.5, f"x={x:.2f}")
test("frenet→cart y≈3", abs(y - 3.0) < 0.5, f"y={y:.2f}")

# Batch conversion
s_arr = np.array([10.0, 20.0, 30.0])
d_arr = np.array([0.0, 1.0, -1.0])
pts = ft.frenet_to_cartesian_array(s_arr, d_arr)
test("Batch shape (3,2)", pts.shape == (3, 2))

# Curved reference line
R = 50.0
angles = np.linspace(0, np.pi / 4, 30)
wps_curve = [[R * np.sin(a), R * (1 - np.cos(a)), np.degrees(a)] for a in angles]
ft2 = FrenetTransform(ds=0.5)
ft2.build_reference_line(wps_curve)
test("Curved ref line built", ft2._is_built)

# Error handling
ft3 = FrenetTransform()
try:
    ft3.cartesian_to_frenet(0.0, 0.0)
    test("Unbuilt raises error", False)
except RuntimeError:
    test("Unbuilt raises RuntimeError", True)

# ======================================================================
print("\n" + "=" * 70)
print("TEST 5: bezier_fitting.py — Trajectory Generation")
print("=" * 70)
from gym_carla.planning.bezier_fitting import BezierFitting

planner = BezierFitting(frenet=ft, lane_width=3.5, plan_horizon=30.0,
                         offset_magnitude=0.5, n_samples=50)

# Lane keeping
traj_keep = planner.generate_trajectory(10.0, 0.0, 0.0, goal=1, offset=1)
test("Keep traj shape (50,2)", traj_keep.shape == (50, 2))
test("Keep start x≈10", abs(traj_keep[0, 0] - 10.0) < 2.0, f"x={traj_keep[0,0]:.1f}")
test("Keep end y≈0", abs(traj_keep[-1, 1]) < 1.0, f"y={traj_keep[-1,1]:.2f}")

# Left lane change
traj_left = planner.generate_trajectory(10.0, 0.0, 0.0, goal=0, offset=1)
test("Left LC end y≈3.5", abs(traj_left[-1, 1] - 3.5) < 1.0, f"y={traj_left[-1,1]:.2f}")

# Right lane change
traj_right = planner.generate_trajectory(10.0, 0.0, 0.0, goal=2, offset=1)
test("Right LC end y≈-3.5", abs(traj_right[-1, 1] + 3.5) < 1.0, f"y={traj_right[-1,1]:.2f}")

# Offset effect
traj_off_l = planner.generate_trajectory(10.0, 0.0, 0.0, goal=1, offset=0)
traj_off_r = planner.generate_trajectory(10.0, 0.0, 0.0, goal=1, offset=2)
test("Offset left > offset right end y",
     traj_off_l[-1, 1] > traj_off_r[-1, 1],
     f"left={traj_off_l[-1,1]:.3f}, right={traj_off_r[-1,1]:.3f}")

# Curvature
curv = planner.get_trajectory_curvature(traj_left)
test("Curvature array shape", len(curv) == 50)
test("Curvature positive", float(curv.max()) >= 0)

# Frenet version
s_arr, d_arr = planner.generate_trajectory_frenet(10.0, 0.0, 0.0, goal=0, offset=1)
test("Frenet traj s array len=50", len(s_arr) == 50)
test("Frenet traj d start≈0, end≈3.5",
     abs(d_arr[0]) < 0.5 and abs(d_arr[-1] - 3.5) < 1.0,
     f"d0={d_arr[0]:.2f}, df={d_arr[-1]:.2f}")

# ======================================================================
print("\n" + "=" * 70)
print("TEST 6: trajectory_tracker.py — Pure Pursuit + PID")
print("=" * 70)
from gym_carla.control.trajectory_tracker import TrajectoryTracker, EgoState

tracker = TrajectoryTracker(wheelbase=2.85, desired_speed=8.0, dt=0.1)

# Straight trajectory tracking
ego = EgoState(x=0.0, y=0.0, yaw=0.0, speed=5.0)
traj_st = np.column_stack([np.linspace(0, 50, 50), np.zeros(50)])
throttle, steer, brake = tracker.compute_control(traj_st, ego)
test("Straight: throttle>0", throttle > 0, f"throttle={throttle:.3f}")
test("Straight: steer≈0", abs(steer) < 0.1, f"steer={steer:.4f}")
test("Straight: brake=0", brake == 0.0)

# Left lane change tracking (ego slightly behind trajectory start)
ego2 = EgoState(x=8.0, y=0.0, yaw=0.0, speed=8.0)
throttle2, steer2, brake2 = tracker.compute_control(traj_left, ego2)
test("Left LC: steer>0", steer2 > 0, f"steer={steer2:.4f}")

# Tracking error
err = tracker.compute_tracking_error(traj_st, EgoState(5.0, 1.0, 0.1, 6.0))
test("Track error lat≈1.0", abs(err['lateral_error'] - 1.0) < 0.5,
     f"lat_err={err['lateral_error']:.3f}")
test("Track error head≈0.1", abs(err['heading_error'] - 0.1) < 0.05,
     f"head_err={err['heading_error']:.3f}")

# Reset
tracker.reset()
test("Tracker reset OK", tracker._error_integral == 0.0)

# ======================================================================
print("\n" + "=" * 70)
print("TEST 7: q1_decision.py — Q1 Network")
print("=" * 70)
from gym_carla.models.q1_decision import Q1Decision

q1 = Q1Decision(state_dim=18, seq_len=3, hidden_dim=64, num_goals=3)
x = torch.randn(4, 3, 18)
goal_q, raw_input = q1(x)
test("Q1 output shape (4,3)", goal_q.shape == (4, 3))
test("Q1 raw passthrough shape", raw_input.shape == (4, 3, 18))
test("Q1 raw is same tensor", raw_input is x)

probs = q1.get_goal_probs(x)
test("Q1 probs sum=1", all(abs(s - 1.0) < 1e-5 for s in probs.sum(dim=-1).tolist()))

action = q1.select_action(x, epsilon=0.0)
test("Q1 action shape (4,)", action.shape == (4,))
test("Q1 action in {0,1,2}", all(a in [0, 1, 2] for a in action.tolist()))

q1_params = sum(p.numel() for p in q1.parameters())
test("Q1 params = 23,683", q1_params == 23683, f"got {q1_params}")

# ======================================================================
print("\n" + "=" * 70)
print("TEST 8: q2_decision.py — Q2 Network (Skip Connection)")
print("=" * 70)
from gym_carla.models.q2_decision import Q2Decision

q2 = Q2Decision(state_dim=18, seq_len=3, hidden_dim=64, num_goals=3, num_offsets=3)
test("Q2 concat dim = 57", q2.concat_dim == 57)

goal_oh = torch.zeros(4, 3)
goal_oh[:, 1] = 1.0
offset_q = q2(x, goal_oh)
test("Q2 output shape (4,3)", offset_q.shape == (4, 3))

offset_probs = q2.get_offset_probs(x, goal_oh)
test("Q2 probs sum=1", all(abs(s - 1.0) < 1e-5 for s in offset_probs.sum(dim=-1).tolist()))

q2_params = sum(p.numel() for p in q2.parameters())
test("Q2 params = 33,667", q2_params == 33667, f"got {q2_params}")

# ======================================================================
print("\n" + "=" * 70)
print("TEST 9: hierarchical_policy.py — Full Pipeline")
print("=" * 70)
from gym_carla.models.hierarchical_policy import HierarchicalPolicy

policy = HierarchicalPolicy(state_dim=18, seq_len=3, hidden_dim=64,
                             num_goals=3, num_offsets=3)

# Forward
x1 = torch.randn(1, 3, 18)
result = policy.forward(x1)
test("Forward goal_q shape", result['goal_q'].shape == (1, 3))
test("Forward offset_q shape", result['offset_q'].shape == (1, 3))
test("Forward goal_idx in {0,1,2}", result['goal_idx'].item() in [0, 1, 2])
test("Forward offset_idx in {0,1,2}", result['offset_idx'].item() in [0, 1, 2])

# Action selection (single sample)
g, o = policy.select_action(x1.squeeze(0), epsilon=0.0)
test("Action select single: types", isinstance(g, int) and isinstance(o, int))

# Batch action selection
g_b, o_b = policy.select_action(torch.randn(8, 3, 18), epsilon=0.3)
test("Action select batch shape", g_b.shape == (8,) and o_b.shape == (8,))

# Compute Q-values for specific actions
batch4 = torch.randn(4, 3, 18)
g_act = torch.tensor([0, 1, 2, 1])
o_act = torch.tensor([1, 0, 2, 1])
gq, oq = policy.compute_q_values(batch4, g_act, o_act)
test("Compute Q goal shape (4,)", gq.shape == (4,))
test("Compute Q offset shape (4,)", oq.shape == (4,))

# Target network
res_online = policy.forward(x1, use_target=False)
res_target = policy.forward(x1, use_target=True)
test("Target network initially same",
     torch.allclose(res_online['goal_q'], res_target['goal_q']))

# Soft update
policy.q1.fc[0].weight.data += 1.0  # Perturb online
policy.update_target(tau=0.5)
res_target2 = policy.forward(x1, use_target=True)
test("Soft update changes target",
     not torch.allclose(res_target['goal_q'], res_target2['goal_q']))

# Probability output
gp, op = policy.get_probs(x1)
test("Probs goal sum=1", abs(gp.sum().item() - 1.0) < 1e-4)
test("Probs offset sum=1", abs(op.sum().item() - 1.0) < 1e-4)

# Parameter count
pc = policy.count_parameters()
test("Total online params = 57,350", pc['total_online'] == 57350, f"got {pc['total_online']}")

# ======================================================================
print("\n" + "=" * 70)
print("TEST 10: train_hierarchical.py — ReplayBuffer + Trainer")
print("=" * 70)
from train_hierarchical import ReplayBuffer, HierarchicalDQNTrainer, get_epsilon

# ReplayBuffer
rb = ReplayBuffer(capacity=100)
for i in range(50):
    rb.push(
        np.random.randn(3, 18).astype(np.float32),
        np.random.randint(3),
        np.random.randint(3),
        float(np.random.randn()),
        np.random.randn(3, 18).astype(np.float32),
        float(i == 49),
    )
test("Buffer len=50", len(rb) == 50)

states, goals, offsets, rewards, nexts, dones = rb.sample(16)
test("Sample states shape (16,3,18)", states.shape == (16, 3, 18))
test("Sample goals shape (16,)", goals.shape == (16,))
test("Sample dones shape (16,)", dones.shape == (16,))

# Epsilon schedule
test("Epsilon ep0 = 1.0", get_epsilon(0, 1000) == 1.0)
test("Epsilon ep250 = 0.525", abs(get_epsilon(250, 1000) - 0.525) < 0.01,
     f"got {get_epsilon(250, 1000):.3f}")
test("Epsilon ep500 = 0.05", get_epsilon(500, 1000) == 0.05)
test("Epsilon ep999 = 0.05", get_epsilon(999, 1000) == 0.05)

# Trainer (one update step)
policy2 = HierarchicalPolicy(18, 3, 64, 3, 3)
trainer = HierarchicalDQNTrainer(
    policy=policy2, lr=3e-4, gamma=0.99, batch_size=16,
    buffer_size=1000, target_update_freq=10, device='cpu')

# Fill buffer
for i in range(32):
    trainer.store_transition(
        np.random.randn(3, 18).astype(np.float32),
        np.random.randint(3), np.random.randint(3),
        np.random.randn(), np.random.randn(3, 18).astype(np.float32),
        float(np.random.random() < 0.1),
    )

losses = trainer.update()
test("Trainer update returns losses", 'loss_q1' in losses and 'loss_q2' in losses)
test("Loss Q1 is finite", np.isfinite(losses['loss_q1']),
     f"loss_q1={losses['loss_q1']:.4f}")
test("Loss Q2 is finite", np.isfinite(losses['loss_q2']),
     f"loss_q2={losses['loss_q2']:.4f}")

# ======================================================================
print("\n" + "=" * 70)
print("TEST 11: export_onnx.py — Structure Test")
print("=" * 70)
import tempfile
from export_onnx import export

# Export with random weights
with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
    onnx_path = f.name

try:
    export("nonexistent_checkpoint.pth", onnx_path, verify=False)
    test("ONNX export succeeds", os.path.exists(onnx_path))
    test("ONNX file size > 0", os.path.getsize(onnx_path) > 0)
finally:
    if os.path.exists(onnx_path):
        os.unlink(onnx_path)

# ======================================================================
print("\n" + "=" * 70)
print("TEST 12: End-to-End Pipeline (obs → policy → bezier → tracker)")
print("=" * 70)

# Simulate full pipeline without CARLA
policy_e2e = HierarchicalPolicy(18, 3, 64, 3, 3)
buffer_e2e = StateBuffer(18, 3)
ft_e2e = FrenetTransform(ds=0.5)
ft_e2e.build_reference_line([[i * 5.0, 0.0, 0.0] for i in range(41)])
bezier_e2e = BezierFitting(frenet=ft_e2e, lane_width=3.5,
                            plan_horizon=30.0, offset_magnitude=0.5, n_samples=50)
tracker_e2e = TrajectoryTracker(wheelbase=2.85, desired_speed=8.0, dt=0.1)

# Simulate 10 steps
ego_x, ego_y, ego_yaw, ego_speed = 10.0, 0.0, 0.0, 5.0
all_ok = True

for step in range(10):
    # 1. Generate fake 18-dim observation
    obs_18d = np.zeros(18, dtype=np.float32)
    obs_18d[0] = ego_speed
    obs_18d[1] = -1.0  # lane_id
    obs_18d[10:18] = 100.0  # all zones far away
    buffer_e2e.push(obs_18d)

    # 2. Get state sequence
    state_seq = buffer_e2e.get_sequence(as_batch=False)  # (3, 18)

    # 3. Policy selects action
    goal, offset = policy_e2e.select_action(state_seq, epsilon=0.3)

    # 4. Generate trajectory
    try:
        trajectory = bezier_e2e.generate_trajectory(
            ego_x, ego_y, ego_yaw, goal, offset)
    except Exception as e:
        all_ok = False
        print(f"  Step {step}: Bezier failed: {e}")
        break

    # 5. Track trajectory
    ego_state = EgoState(ego_x, ego_y, ego_yaw, ego_speed)
    throttle, steer, brake = tracker_e2e.compute_control(trajectory, ego_state)
    err = tracker_e2e.compute_tracking_error(trajectory, ego_state)

    # 6. Simple kinematic update
    ds = ego_speed * 0.1
    ego_yaw += ego_speed * math.tan(steer * math.radians(70)) / 2.85 * 0.1
    ego_x += ds * math.cos(ego_yaw)
    ego_y += ds * math.sin(ego_yaw)
    ego_speed = max(0.1, ego_speed + (throttle * 3.0 - brake * 5.0) * 0.1)

    if not (np.isfinite(throttle) and np.isfinite(steer) and np.isfinite(brake)):
        all_ok = False
        break

test("E2E: 10 steps complete", all_ok)
test("E2E: final position finite",
     np.isfinite(ego_x) and np.isfinite(ego_y),
     f"pos=({ego_x:.1f}, {ego_y:.1f})")
test("E2E: final speed positive", ego_speed > 0, f"speed={ego_speed:.1f}")

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 70)
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print("=" * 70)

if failed == 0:
    print("\n*** ALL TESTS PASSED ***\n")
else:
    print(f"\n*** {failed} TESTS FAILED ***\n")

sys.exit(0 if failed == 0 else 1)
