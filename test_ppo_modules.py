#!/usr/bin/env python
"""测试 PPO 改造后的所有模块"""
import torch

print("=== Test 1: CriticNetwork ===")
from gym_carla.models.critic import CriticNetwork
c = CriticNetwork(18, 3, 64)
x = torch.randn(4, 3, 18)
v = c(x)
print("  Value shape:", v.shape, "(expect [4])")
assert v.shape == (4,), "CriticNetwork output shape wrong"
print("  PASS")

print()
print("=== Test 2: Q1Decision (PPO Actor - Discrete) ===")
from gym_carla.models.q1_decision import Q1Decision
q1 = Q1Decision(18, 3, 64, 3)
logits, raw = q1(x)
print("  Logits shape:", logits.shape, "Raw shape:", raw.shape)
goal, lp, _ = q1.select_action(x)
print("  Select action goals:", goal.tolist())
lp2, ent, _ = q1.evaluate_actions(x, goal)
print("  Entropy:", [round(e, 3) for e in ent.tolist()])
print("  PASS")

print()
print("=== Test 3: Q2Decision (PPO Actor - Gaussian) ===")
from gym_carla.models.q2_decision import Q2Decision
q2 = Q2Decision(18, 3, 64, 3, log_std_init=0.0)
goal_oh = torch.zeros(4, 3)
goal_oh[:, 1] = 1.0
mean = q2(x, goal_oh)
print("  Mean shape:", mean.shape, "(expect [4])")
p_off, lp_off = q2.select_action(x, goal_oh)
print("  Sampled p_off:", [round(v, 3) for v in p_off.tolist()])
lp3, ent3 = q2.evaluate_actions(x, goal_oh, p_off)
print("  Entropy:", [round(v, 3) for v in ent3.tolist()])
print("  log_std:", round(q2.log_std.item(), 4))
print("  PASS")

print()
print("=== Test 4: HierarchicalPolicy (PPO) ===")
from gym_carla.models.hierarchical_policy import HierarchicalPolicy
pol = HierarchicalPolicy(18, 3, 64, 3, log_std_init=0.0)
params = pol.count_parameters()
print("  Params:", params)

# Single select
result = pol.select_action(x[0])
print("  Single: goal=%d, offset=%.3f, value=%.3f" % (
    result["goal"], result["offset"], result["value"]))

# Batch evaluate
goals_t = torch.tensor([0, 1, 2, 1])
offsets_t = torch.tensor([0.5, -0.3, 1.2, 0.0])
ev = pol.evaluate_actions(x, goals_t, offsets_t)
print("  Evaluate value shape:", ev["value"].shape)
print("  log_prob_goal:", [round(v, 3) for v in ev["log_prob_goal"].detach().tolist()])
print("  log_prob_offset:", [round(v, 3) for v in ev["log_prob_offset"].detach().tolist()])
print("  entropy_goal:", [round(v, 3) for v in ev["entropy_goal"].detach().tolist()])

# get_value
v_val = pol.get_value(x[0])
print("  V(s) = %.4f" % v_val)
print("  PASS")

print()
print("=== Test 5: Config ===")
from gym_carla import config as cfg
assert hasattr(cfg, 'PPO_CLIP_EPSILON'), 'Missing PPO_CLIP_EPSILON'
assert hasattr(cfg, 'GAE_LAMBDA'), 'Missing GAE_LAMBDA'
assert hasattr(cfg, 'OFFSET_RANGE'), 'Missing OFFSET_RANGE'
assert hasattr(cfg, 'NUM_ITERATIONS'), 'Missing NUM_ITERATIONS'
print("  PPO_CLIP_EPSILON =", cfg.PPO_CLIP_EPSILON)
print("  GAE_LAMBDA =", cfg.GAE_LAMBDA)
print("  OFFSET_RANGE =", cfg.OFFSET_RANGE)
print("  NUM_ITERATIONS =", cfg.NUM_ITERATIONS)
print("  PASS")

print()
print("=== ALL 5 MODULE TESTS PASSED ===")
