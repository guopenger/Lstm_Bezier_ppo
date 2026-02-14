#!/usr/bin/env python
"""
q1_decision.py — Q¹ 上层行为决策模型 (Categorical Actor)

论文依据 (§2.2.1, Figure 3, §2.1.3 Eq.1):
  输入: 状态序列 (batch, seq_len=3, state_dim=18)
  处理: LSTM(input_size=18, hidden_size=64, num_layers=1) → FC → Softmax
  输出: Goal 概率分布 (batch, 3) — [左换道, 保持, 右换道]

  A_g = [g_l, g_r, g_s]  — 离散 3 类 (Eq.1)
  训练时通过 Categorical 分布采样 + PPO clip loss 更新。
  同时透传原始输入 (Skip Connection) 给 Q2 使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Q1Decision(nn.Module):
    """Q¹ 上层行为决策网络 (Categorical Actor)。

    Architecture:
        Input (batch, 3, 18) → LSTM → h_n (batch, 64) → FC(64→32) → ReLU → FC(32→3)
        → Softmax → Categorical 分布 → 采样 goal

    输出 logits (未经 Softmax)，由调用者决定如何使用:
        - PPO 训练:  构造 Categorical(logits=...) → sample / log_prob
        - 部署推理:  argmax (确定性)
    """

    def __init__(self, state_dim: int = 18, seq_len: int = 3,
                 hidden_dim: int = 64, num_goals: int = 3):
        """
        Args:
            state_dim:  单步状态维度 (论文: 18)。
            seq_len:    时间步长 (论文: 3)。
            hidden_dim: LSTM 隐藏层维度 (论文: 64)。
            num_goals:  输出类别数 (论文: 3 — 左换道/保持/右换道)。
        """
        super().__init__()
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_goals = num_goals

        # LSTM: 处理时序状态序列
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # FC Head: LSTM 隐状态 → logits (未经 Softmax)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_goals),
        )

    def forward(self, state_seq: torch.Tensor) -> tuple:
        """前向推理，输出 logits。

        Args:
            state_seq: (batch, seq_len=3, state_dim=18) 历史状态序列。

        Returns:
            logits:    (batch, 3) 未归一化的得分 (Categorical 分布的参数)。
            raw_input: (batch, seq_len, state_dim) 原始输入透传 — 供 Q2 Skip Connection。
        """
        lstm_out, (h_n, c_n) = self.lstm(state_seq)
        last_hidden = h_n.squeeze(0)  # (batch, 64)
        logits = self.fc(last_hidden)  # (batch, 3)
        return logits, state_seq

    # ------------------------------------------------------------------
    # PPO 接口
    # ------------------------------------------------------------------

    def get_dist(self, state_seq: torch.Tensor):
        """获取 Goal 的 Categorical 分布。

        Returns:
            dist:      Categorical 分布对象。
            raw_input: 原始输入透传 (Skip Connection)。
        """
        logits, raw_input = self.forward(state_seq)
        dist = Categorical(logits=logits)
        return dist, raw_input

    def evaluate_actions(self, state_seq: torch.Tensor,
                         actions: torch.Tensor):
        """评估给定动作的 log_prob 和 entropy (PPO 更新用)。

        Args:
            state_seq: (batch, seq_len, state_dim)。
            actions:   (batch,) LongTensor, 选择的 Goal 索引。

        Returns:
            log_probs: (batch,) 动作对数概率。
            entropy:   (batch,) 策略熵。
            raw_input: (batch, seq_len, state_dim) 透传给 Q2。
        """
        dist, raw_input = self.get_dist(state_seq)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, raw_input

    def select_action(self, state_seq: torch.Tensor):
        """从 Categorical 分布中采样动作 (PPO 训练用)。

        Args:
            state_seq: (batch, seq_len, state_dim)。

        Returns:
            goal:     (batch,) 采样的 Goal 索引。
            log_prob: (batch,) 对数概率。
            raw_input: 透传给 Q2。
        """
        dist, raw_input = self.get_dist(state_seq)
        goal = dist.sample()
        log_prob = dist.log_prob(goal)
        return goal, log_prob, raw_input

    def get_goal_probs(self, state_seq: torch.Tensor,
                       temperature: float = 1.0) -> torch.Tensor:
        """获取 Goal 的 Softmax 概率分布 (用于部署/可视化)。

        Args:
            state_seq:   (batch, seq_len, state_dim)。
            temperature: Softmax 温度。

        Returns:
            (batch, 3) 概率分布。
        """
        logits, _ = self.forward(state_seq)
        return F.softmax(logits / temperature, dim=-1)


# ======================================================================
# 快速自测
# ======================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Q1Decision Self-Test (PPO Categorical Actor)")
    print("=" * 50)

    model = Q1Decision(state_dim=18, seq_len=3, hidden_dim=64, num_goals=3)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(4, 3, 18)

    # forward
    logits, raw = model(x)
    print(f"\nInput shape:    {x.shape}")
    print(f"Logits shape:   {logits.shape}  (expect [4, 3])")
    print(f"Raw pass shape: {raw.shape}       (expect [4, 3, 18])")

    # Categorical 采样
    goal, lp, _ = model.select_action(x)
    print(f"Sampled goals:  {goal.tolist()}")
    print(f"Log probs:      {lp.tolist()}")

    # evaluate_actions
    lp2, ent, _ = model.evaluate_actions(x, goal)
    print(f"Eval log_probs: {lp2.tolist()}")
    print(f"Entropy:        {ent.tolist()}")

    # Softmax 概率
    probs = model.get_goal_probs(x)
    print(f"Probs sum:      {probs.sum(dim=-1).tolist()}")
    print("\n✓ Q1Decision (PPO) test passed!")
