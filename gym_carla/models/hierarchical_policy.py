#!/usr/bin/env python
"""
hierarchical_policy.py — 分层决策策略网络 (封装 Q1 + Q2)

论文依据 (§2.2.1, Figure 3):
  整体前向流:
    State(N, 3, 18) → Q1(LSTM₁+FC) → Goal
                 ↓ Skip Connection
    Concat(State_flat, Goal_onehot) → Q2(LSTM₂+FC) → Offset

  训练模式:
    - DQN: Q1 和 Q2 各自有独立的 Target Network
    - 或 Actor-Critic: Softmax 概率用于 policy gradient

  部署模式:
    - 导出 ONNX: forward() 直接输出 (goal_idx, offset_idx)

本模块同时封装:
  - Online Network (用于选动作 + 计算当前 Q 值)
  - Target Network (用于计算 TD-target，定期从 Online 拷贝参数)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from gym_carla.models.q1_decision import Q1Decision
from gym_carla.models.q2_decision import Q2Decision


class HierarchicalPolicy(nn.Module):
    """分层决策网络: Q1 (行为) + Q2 (轨迹偏移)。

    数据流:
        state_seq (batch, 3, 18)
            ↓
        Q1 → goal_q_values (batch, 3) + raw_input (skip)
            ↓
        goal → one_hot → concat with raw_input_flat
            ↓
        Q2 → offset_q_values (batch, 3)

    提供:
        - forward():         获取双层 Q 值
        - select_action():   ε-greedy 选动作
        - get_probs():       Softmax 概率 (可视化/部署)
        - export_onnx():     导出为 ONNX 文件
    """

    def __init__(self, state_dim: int = 18, seq_len: int = 3,
                 hidden_dim: int = 64, num_goals: int = 3,
                 num_offsets: int = 3):
        super().__init__()
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.num_goals = num_goals
        self.num_offsets = num_offsets

        # Online Networks
        self.q1 = Q1Decision(state_dim, seq_len, hidden_dim, num_goals)
        self.q2 = Q2Decision(state_dim, seq_len, hidden_dim, num_goals, num_offsets)

        # Target Networks (延迟更新，用于 DQN 稳定训练)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        # 冻结 target 参数
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # 核心前向
    # ------------------------------------------------------------------

    def forward(self, state_seq: torch.Tensor,
                use_target: bool = False) -> dict:
        """完整的分层前向推理。

        Args:
            state_seq:  (batch, seq_len=3, state_dim=18)。
            use_target: True 时使用 Target Network (计算 TD-target)。

        Returns:
            dict:
                'goal_q':      (batch, 3) Q1 的 Q 值。
                'offset_q':    (batch, 3) Q2 的 Q 值。
                'goal_idx':    (batch,)   贪心选择的 Goal 索引。
                'offset_idx':  (batch,)   贪心选择的 Offset 索引。
                'goal_onehot': (batch, 3) Goal 的 one-hot 编码。
        """
        q1_net = self.q1_target if use_target else self.q1
        q2_net = self.q2_target if use_target else self.q2

        # Q1: 行为决策
        goal_q, raw_input = q1_net(state_seq)
        goal_idx = goal_q.argmax(dim=-1)  # (batch,)

        # Goal → one-hot
        goal_onehot = F.one_hot(goal_idx, self.num_goals).float()  # (batch, 3)

        # Q2: 轨迹偏移 (Skip Connection: 使用原始输入)
        offset_q = q2_net(raw_input, goal_onehot)
        offset_idx = offset_q.argmax(dim=-1)  # (batch,)

        return {
            'goal_q': goal_q,
            'offset_q': offset_q,
            'goal_idx': goal_idx,
            'offset_idx': offset_idx,
            'goal_onehot': goal_onehot,
        }

    # ------------------------------------------------------------------
    # 动作选择
    # ------------------------------------------------------------------

    def select_action(self, state_seq: torch.Tensor,
                      epsilon: float = 0.0) -> tuple:
        """ε-greedy 双层动作选择。

        Args:
            state_seq: (batch, seq_len, state_dim) 或 (seq_len, state_dim)。
            epsilon:   探索概率。

        Returns:
            (goal_idx, offset_idx): 各为 int (单样本) 或 Tensor (批量)。
        """
        # 处理单样本输入
        single = (state_seq.dim() == 2)
        if single:
            state_seq = state_seq.unsqueeze(0)

        with torch.no_grad():
            # Q1 选 Goal
            goal_q, raw_input = self.q1(state_seq)

            if epsilon > 0 and torch.rand(1).item() < epsilon:
                goal_idx = torch.randint(0, self.num_goals, (state_seq.size(0),),
                                         device=state_seq.device)
            else:
                goal_idx = goal_q.argmax(dim=-1)

            goal_onehot = F.one_hot(goal_idx, self.num_goals).float()

            # Q2 选 Offset
            offset_q = self.q2(raw_input, goal_onehot)

            if epsilon > 0 and torch.rand(1).item() < epsilon:
                offset_idx = torch.randint(0, self.num_offsets, (state_seq.size(0),),
                                           device=state_seq.device)
            else:
                offset_idx = offset_q.argmax(dim=-1)

        if single:
            return goal_idx.item(), offset_idx.item()
        return goal_idx, offset_idx

    # ------------------------------------------------------------------
    # 训练辅助
    # ------------------------------------------------------------------

    def compute_q_values(self, state_seq: torch.Tensor,
                         goal_actions: torch.Tensor,
                         offset_actions: torch.Tensor,
                         use_target: bool = False) -> tuple:
        """计算指定动作对应的 Q 值 (用于 DQN loss 计算)。

        Args:
            state_seq:      (batch, seq_len, state_dim)。
            goal_actions:   (batch,) 选择的 Goal 索引。
            offset_actions: (batch,) 选择的 Offset 索引。
            use_target:     是否使用 Target Network。

        Returns:
            (goal_q_selected, offset_q_selected):
                各为 (batch,) 对应动作的 Q 值。
        """
        result = self.forward(state_seq, use_target=use_target)

        # gather: 从 Q 值中取出对应动作的值
        goal_q_selected = result['goal_q'].gather(
            1, goal_actions.unsqueeze(1)
        ).squeeze(1)  # (batch,)

        offset_q_selected = result['offset_q'].gather(
            1, offset_actions.unsqueeze(1)
        ).squeeze(1)  # (batch,)

        return goal_q_selected, offset_q_selected

    def update_target(self, tau: float = 1.0) -> None:
        """更新 Target Network 参数。

        Args:
            tau: 软更新系数。
                 tau=1.0 → 硬拷贝 (标准 DQN)。
                 tau<1.0 → 软更新 θ_target = τ·θ_online + (1-τ)·θ_target。
        """
        for param, target_param in zip(self.q1.parameters(),
                                       self.q1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(),
                                       self.q2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def get_probs(self, state_seq: torch.Tensor,
                  temperature: float = 1.0) -> tuple:
        """获取双层 Softmax 概率 (用于可视化和部署)。

        Returns:
            (goal_probs, offset_probs): 各 (batch, 3)。
        """
        with torch.no_grad():
            goal_q, raw_input = self.q1(state_seq)
            goal_probs = F.softmax(goal_q / temperature, dim=-1)

            goal_idx = goal_q.argmax(dim=-1)
            goal_onehot = F.one_hot(goal_idx, self.num_goals).float()

            offset_q = self.q2(raw_input, goal_onehot)
            offset_probs = F.softmax(offset_q / temperature, dim=-1)

        return goal_probs, offset_probs

    def export_onnx(self, filepath: str = "hierarchical_policy.onnx") -> None:
        """导出为 ONNX 格式，供 C++/ROS2 部署。

        注意: ONNX 需要一个单一的 forward 方法。
        这里用一个 wrapper 将双层逻辑封装为单次调用。
        """
        class OnnxWrapper(nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.q1 = policy.q1
                self.q2 = policy.q2
                self.num_goals = policy.num_goals

            def forward(self, state_seq):
                goal_q, raw_input = self.q1(state_seq)
                goal_idx = goal_q.argmax(dim=-1)
                goal_onehot = F.one_hot(goal_idx, self.num_goals).float()
                offset_q = self.q2(raw_input, goal_onehot)
                offset_idx = offset_q.argmax(dim=-1)
                return goal_idx, offset_idx

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
        return {
            'q1': q1_params,
            'q2': q2_params,
            'total_online': q1_params + q2_params,
            'total_with_target': 2 * (q1_params + q2_params),
        }


# ======================================================================
# 快速自测
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("HierarchicalPolicy Self-Test")
    print("=" * 60)

    policy = HierarchicalPolicy(
        state_dim=18, seq_len=3, hidden_dim=64,
        num_goals=3, num_offsets=3,
    )
    params = policy.count_parameters()
    print(f"Q1 params:  {params['q1']:,}")
    print(f"Q2 params:  {params['q2']:,}")
    print(f"Total (online):      {params['total_online']:,}")
    print(f"Total (with target): {params['total_with_target']:,}")

    # --- 测试 1: 完整前向 ---
    print("\n--- Test 1: Full forward ---")
    x = torch.randn(1, 3, 18)
    result = policy.forward(x)
    print(f"  Input:       {x.shape}")
    print(f"  Goal Q:      {result['goal_q'].shape}      values={result['goal_q'].detach().tolist()}")
    print(f"  Offset Q:    {result['offset_q'].shape}    values={result['offset_q'].detach().tolist()}")
    print(f"  Goal idx:    {result['goal_idx'].tolist()}")
    print(f"  Offset idx:  {result['offset_idx'].tolist()}")

    # --- 测试 2: 动作选择 ---
    print("\n--- Test 2: Action selection ---")
    goal, offset = policy.select_action(x.squeeze(0), epsilon=0.0)
    print(f"  Single sample: goal={goal}, offset={offset}")

    goals, offsets = policy.select_action(torch.randn(8, 3, 18), epsilon=0.3)
    print(f"  Batch (8): goals={goals.tolist()}, offsets={offsets.tolist()}")

    # --- 测试 3: Q 值计算 (DQN loss 用) ---
    print("\n--- Test 3: Q-value for specific actions ---")
    batch = torch.randn(4, 3, 18)
    g_act = torch.tensor([0, 1, 2, 1])
    o_act = torch.tensor([1, 0, 2, 1])
    gq, oq = policy.compute_q_values(batch, g_act, o_act)
    print(f"  Goal Q selected:   {gq.detach().tolist()}")
    print(f"  Offset Q selected: {oq.detach().tolist()}")

    # --- 测试 4: Target network ---
    print("\n--- Test 4: Target network update ---")
    result_online = policy.forward(x, use_target=False)
    result_target = policy.forward(x, use_target=True)
    print(f"  Online goal Q:  {result_online['goal_q'].detach().tolist()}")
    print(f"  Target goal Q:  {result_target['goal_q'].detach().tolist()}")
    print(f"  Same? {torch.allclose(result_online['goal_q'], result_target['goal_q'])}")

    policy.update_target(tau=0.5)
    result_target2 = policy.forward(x, use_target=True)
    print(f"  After soft update (τ=0.5): {result_target2['goal_q'].detach().tolist()}")

    # --- 测试 5: Softmax 概率 ---
    print("\n--- Test 5: Probability output ---")
    gp, op = policy.get_probs(x)
    print(f"  Goal probs:   {gp.tolist()}  sum={gp.sum().item():.4f}")
    print(f"  Offset probs: {op.tolist()}  sum={op.sum().item():.4f}")

    print("\n✓ All tests passed!")
