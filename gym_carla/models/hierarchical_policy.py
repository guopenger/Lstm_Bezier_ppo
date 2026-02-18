#!/usr/bin/env python
"""
hierarchical_policy.py — 分层决策策略网络 (PPO Actor-Critic)

论文依据 (§2.2.1, Figure 3):
  整体前向流:
    State(N, 3, 18) → Q1(LSTM₁+FC) → Softmax → Goal (离散 Categorical)
                 ↓ Skip Connection
    Concat(State_flat, Goal_onehot) → Q2(LSTM₂+FC) → μ → p_off (连续 Gaussian)

  训练模式 — PPO:
    - Q1 输出分类概率 (Categorical) — 离散行为决策 (Eq.1)
    - Q2 输出高斯均值+标准差 (Normal) — 连续轨迹偏移 (Eq.2)
    - Critic: LSTM → V(s) 状态价值估计

  部署模式:
    - 导出 ONNX: Q1 → argmax(goal), Q2 → mean(p_off)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from gym_carla.models.q1_decision import Q1Decision
from gym_carla.models.q2_decision import Q2Decision
from gym_carla.models.critic import CriticNetwork


class HierarchicalPolicy(nn.Module):
    """分层决策网络: Q1 (行为 Actor) + Q2 (偏移 Actor) + Critic。

    数据流:
        state_seq (batch, 3, 18)
            ↓
        Q1 → Categorical(logits) → sample goal ∈ {0,1,2}
            ↓ Skip Connection
        goal → one_hot → concat with raw_input_flat
            ↓
        Q2 → Normal(μ, σ) → sample p_off ∈ ℝ
            ↓
        Critic(state_seq) → V(s) 状态价值

    提供:
        - select_action():     采样动作 (训练时带探索)
        - evaluate_actions():  评估已有动作 (PPO 更新)
        - get_value():         仅获取 V(s) (GAE 末步)
        - export_onnx():       导出为 ONNX 文件
    """

    def __init__(self, state_dim: int = 18, seq_len: int = 3,
                 hidden_dim: int = 64, num_goals: int = 3,
                 log_std_init: float = 0.0):
        """
        Args:
            state_dim:    单步状态维度 (18)。
            seq_len:      时间步长 (3)。
            hidden_dim:   LSTM 隐藏层维度 (64)。
            num_goals:    Q1 输出类别数 (3)。
            log_std_init: Q2 高斯策略初始 log(σ)。
        """
        super().__init__()
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.num_goals = num_goals

        # Actor 网络
        self.q1 = Q1Decision(state_dim, seq_len, hidden_dim, num_goals)
        self.q2 = Q2Decision(state_dim, seq_len, hidden_dim, num_goals,
                             log_std_init=log_std_init)

        # Critic 网络 (独立 LSTM + FC, 估计 V(s))
        self.critic = CriticNetwork(state_dim, seq_len, hidden_dim)

    # ------------------------------------------------------------------
    # 动作选择 (训练 + 推理)
    # ------------------------------------------------------------------

    def select_action(self, state_seq: torch.Tensor,
                      deterministic: bool = False) -> dict:
        """采样分层动作。
        deterministic: bool
        # False: 随机采样（训练时，探索）
        # True:  确定性选择（测试/部署时，利用）

        Args:
            state_seq: (seq_len, state_dim) 单样本 或 (batch, seq_len, state_dim)。
            deterministic: True 时使用 argmax/mean (测试/部署)。

        Returns:
            dict: {
              'goal': int,              # 离散 goal 索引
              'offset': float,          # 连续 p_off 偏移量
              'log_prob_goal': float,   # goal 的对数概率
              'log_prob_offset': float, # p_off 的对数概率
              'value': float,           # V(s) 状态价值
            }
        """
        single = (state_seq.dim() == 2)
        if single:
            state_seq = state_seq.unsqueeze(0)

        with torch.no_grad():
            # Q1: 行为决策
            logits, raw_input = self.q1(state_seq)

            if deterministic:
                goal = logits.argmax(dim=-1)
                log_prob_goal = torch.zeros(goal.shape, device=goal.device)
            else:
                dist_goal = Categorical(logits=logits)
                goal = dist_goal.sample()
                log_prob_goal = dist_goal.log_prob(goal)

            # Goal → one_hot
            goal_onehot = F.one_hot(goal, self.num_goals).float()

            # Q2: 轨迹偏移 (连续高斯)
            offset_dist = self.q2.get_dist(raw_input, goal_onehot)

            if deterministic:
                p_off = offset_dist.mean
                log_prob_offset = torch.zeros(p_off.shape, device=p_off.device)
            else:
                p_off = offset_dist.sample()
                log_prob_offset = offset_dist.log_prob(p_off)

            # Critic: 状态价值
            value = self.critic(state_seq)

        if single:
            return {
                'goal': goal.item(),
                'offset': p_off.item(),
                'log_prob_goal': log_prob_goal.item(),
                'log_prob_offset': log_prob_offset.item(),
                'value': value.item(),
            }
        return {
            'goal': goal,
            'offset': p_off,
            'log_prob_goal': log_prob_goal,
            'log_prob_offset': log_prob_offset,
            'value': value,
        }

    # ------------------------------------------------------------------
    # PPO 更新用
    # ------------------------------------------------------------------

    def evaluate_actions(self, state_seq: torch.Tensor,
                         goals: torch.Tensor,
                         offsets: torch.Tensor) -> dict:
        """评估已有动作的 log_prob、entropy 和 value (PPO 更新用)。

        Args:
            state_seq: (batch, seq_len, state_dim)
            goals:     (batch,) LongTensor, 离散 goal 索引
            offsets:   (batch,) FloatTensor, 连续偏移值

        Returns:
            dict: {
              'log_prob_goal', 'log_prob_offset': (batch,),
              'entropy_goal', 'entropy_offset':   (batch,),
              'value':                            (batch,),
            }
        """
        # Q1: 评估 goal 动作
        log_prob_goal, entropy_goal, raw_input = self.q1.evaluate_actions(
            state_seq, goals)

        # Goal → one_hot (使用存储的 goals，不重新采样)
        goal_onehot = F.one_hot(goals, self.num_goals).float()

        # Q2: 评估 offset 动作
        log_prob_offset, entropy_offset = self.q2.evaluate_actions(
            raw_input, goal_onehot, offsets)

        # Critic
        value = self.critic(state_seq)

        return {
            'log_prob_goal': log_prob_goal,
            'log_prob_offset': log_prob_offset,
            'entropy_goal': entropy_goal,
            'entropy_offset': entropy_offset,
            'value': value,
        }

    def get_value(self, state_seq: torch.Tensor) -> float:
        """仅获取状态价值 V(s) (用于 GAE 计算最后一步)。

        Args:
            state_seq: (seq_len, state_dim) 或 (batch, seq_len, state_dim)。

        Returns:
            float (单样本) 或 Tensor (批量)。
        """
        single = (state_seq.dim() == 2)
        if single:
            state_seq = state_seq.unsqueeze(0)
        with torch.no_grad():
            value = self.critic(state_seq)
        if single:
            return value.item()
        return value

    # ------------------------------------------------------------------
    # 可视化 / 调试
    # ------------------------------------------------------------------

    def get_probs(self, state_seq: torch.Tensor,
                  temperature: float = 1.0) -> tuple:
        """获取 Goal 的 Softmax 概率和 Offset 的高斯参数 (用于可视化)。

        Returns:
            (goal_probs, offset_mean, offset_std)
        """
        with torch.no_grad():
            logits, raw_input = self.q1(state_seq)
            goal_probs = F.softmax(logits / temperature, dim=-1)

            goal_idx = logits.argmax(dim=-1)
            goal_onehot = F.one_hot(goal_idx, self.num_goals).float()

            offset_mean = self.q2(raw_input, goal_onehot)
            offset_std = self.q2.log_std.exp().expand_as(offset_mean)

        return goal_probs, offset_mean, offset_std

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def export_onnx(self, filepath: str = "hierarchical_policy.onnx") -> None:
        """导出为 ONNX 格式 (确定性推理: argmax goal + mean offset)。"""

        class OnnxWrapper(nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.q1 = policy.q1
                self.q2 = policy.q2
                self.num_goals = policy.num_goals

            def forward(self, state_seq):
                logits, raw_input = self.q1(state_seq)
                goal_idx = logits.argmax(dim=-1)
                goal_onehot = F.one_hot(goal_idx, self.num_goals).float()
                offset_mean = self.q2(raw_input, goal_onehot)
                return goal_idx, offset_mean

        wrapper = OnnxWrapper(self)
        wrapper.eval()
        dummy = torch.randn(1, self.seq_len, self.state_dim)

        torch.onnx.export(
            wrapper, dummy, filepath,
            input_names=["state_sequence"],
            output_names=["goal", "offset"],
            dynamic_axes={"state_sequence": {0: "batch"}},
            opset_version=11,
        )
        print(f"ONNX exported: {filepath}")

    # ------------------------------------------------------------------
    # 信息
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict:
        """统计各子网络参数量。"""
        q1_params = sum(p.numel() for p in self.q1.parameters())
        q2_params = sum(p.numel() for p in self.q2.parameters())
        critic_params = sum(p.numel() for p in self.critic.parameters())
        return {
            'q1_actor': q1_params,
            'q2_actor': q2_params,
            'critic': critic_params,
            'total': q1_params + q2_params + critic_params,
        }


# ======================================================================
# 快速自测
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("HierarchicalPolicy Self-Test (PPO Actor-Critic)")
    print("=" * 60)

    policy = HierarchicalPolicy(
        state_dim=18, seq_len=3, hidden_dim=64,
        num_goals=3, log_std_init=0.0,
    )
    params = policy.count_parameters()
    print(f"Q1 Actor params:  {params['q1_actor']:,}")
    print(f"Q2 Actor params:  {params['q2_actor']:,}")
    print(f"Critic params:    {params['critic']:,}")
    print(f"Total:            {params['total']:,}")

    # --- 测试 1: 采样动作 ---
    print("\n--- Test 1: Select action (stochastic) ---")
    x = torch.randn(1, 3, 18)
    result = policy.select_action(x.squeeze(0))
    print(f"  Single sample: goal={result['goal']}, offset={result['offset']:.4f}")
    print(f"    log_prob_goal={result['log_prob_goal']:.4f}, "
          f"log_prob_offset={result['log_prob_offset']:.4f}")
    print(f"    value={result['value']:.4f}")

    # --- 测试 2: 确定性动作 ---
    print("\n--- Test 2: Select action (deterministic) ---")
    result_det = policy.select_action(x.squeeze(0), deterministic=True)
    print(f"  Deterministic: goal={result_det['goal']}, offset={result_det['offset']:.4f}")

    # --- 测试 3: 批量采样 ---
    print("\n--- Test 3: Batch select action ---")
    xb = torch.randn(8, 3, 18)
    result_batch = policy.select_action(xb)
    print(f"  Batch goals:   {result_batch['goal'].tolist()}")
    print(f"  Batch offsets: {[f'{v:.3f}' for v in result_batch['offset'].tolist()]}")

    # --- 测试 4: evaluate_actions (PPO 更新核心) ---
    print("\n--- Test 4: Evaluate actions ---")
    batch = torch.randn(4, 3, 18)
    goals = torch.tensor([0, 1, 2, 1])
    offsets = torch.tensor([0.5, -0.3, 1.2, 0.0])
    eval_result = policy.evaluate_actions(batch, goals, offsets)
    print(f"  log_prob_goal:   {eval_result['log_prob_goal'].detach().tolist()}")
    print(f"  log_prob_offset: {eval_result['log_prob_offset'].detach().tolist()}")
    print(f"  entropy_goal:    {eval_result['entropy_goal'].detach().tolist()}")
    print(f"  entropy_offset:  {eval_result['entropy_offset'].detach().tolist()}")
    print(f"  value:           {eval_result['value'].detach().tolist()}")

    # --- 测试 5: get_value ---
    print("\n--- Test 5: Get value ---")
    v = policy.get_value(x.squeeze(0))
    print(f"  V(s) = {v:.4f}")

    # --- 测试 6: Softmax 概率 + 高斯参数 ---
    print("\n--- Test 6: Probability output ---")
    gp, om, os_ = policy.get_probs(x)
    print(f"  Goal probs:    {gp.tolist()}")
    print(f"  Offset mean:   {om.tolist()}")
    print(f"  Offset std:    {os_.tolist()}")

    print("\n✓ All HierarchicalPolicy (PPO) tests passed!")
