#!/usr/bin/env python
"""
q1_decision.py — Q¹ 上层行为决策模型

论文依据 (§2.2.1, Figure 3):
  输入: 状态序列 (batch, seq_len=3, state_dim=18)
  处理: LSTM(input_size=18, hidden_size=64, num_layers=1) → FC → Softmax
  输出: Goal 概率分布 (batch, 3) — [左换道, 保持, 右换道]

  同时透传原始输入 (Skip Connection) 给 Q2 使用。

DQN 视角:
  Q1 本质上是一个 Q-network，输出 3 个动作的 Q 值。
  训练时用 argmax 选动作 + ε-greedy 探索。
  Softmax 可视为将 Q 值转为概率 (用于 policy gradient 或 Boltzmann 探索)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Q1Decision(nn.Module):
    """Q¹ 上层行为决策网络。

    Architecture:
        Input (batch, 3, 18) → LSTM → h_n (batch, 64) → FC(64→32) → ReLU → FC(32→3)

    输出有两种模式:
        - Q-values:   用于 DQN 训练 (不加 Softmax)
        - Softmax:    用于 policy gradient 或部署时的概率采样
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
        # input_size=18 (每步状态维度), hidden_size=64
        # batch_first=True → 输入 (batch, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # FC Head: LSTM 隐状态 → Q-values / Goal 概率
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_goals),
        )

    def forward(self, state_seq: torch.Tensor) -> tuple:
        """前向推理。

        Args:
            state_seq: (batch, seq_len=3, state_dim=18) 历史状态序列。

        Returns:
            goal_q_values: (batch, 3) Q 值 (未经 Softmax)。
            raw_input:     (batch, seq_len, state_dim) 原始输入透传 — 供 Q2 Skip Connection。
        """
        # LSTM 处理时序
        # lstm_out: (batch, seq_len, hidden_dim)
        # h_n:      (1, batch, hidden_dim) — 最后时刻的隐状态
        lstm_out, (h_n, c_n) = self.lstm(state_seq)

        # 取最后时刻的隐状态
        last_hidden = h_n.squeeze(0)  # (batch, 64)

        # FC 映射到 Q-values
        goal_q_values = self.fc(last_hidden)  # (batch, 3)

        # 透传原始输入 (Skip Connection for Q2)
        return goal_q_values, state_seq

    def get_goal_probs(self, state_seq: torch.Tensor,
                       temperature: float = 1.0) -> torch.Tensor:
        """获取 Goal 的 Softmax 概率分布 (用于部署/可视化)。

        Args:
            state_seq:   (batch, seq_len, state_dim)。
            temperature: Softmax 温度。越高分布越均匀，越低越尖锐。

        Returns:
            (batch, 3) 概率分布。
        """
        q_values, _ = self.forward(state_seq)
        return F.softmax(q_values / temperature, dim=-1)

    def select_action(self, state_seq: torch.Tensor,
                      epsilon: float = 0.0) -> torch.Tensor:
        """ε-greedy 动作选择 (用于 DQN 训练)。

        Args:
            state_seq: (batch, seq_len, state_dim)。
            epsilon:   探索概率。

        Returns:
            (batch,) 选择的 Goal 索引 ∈ {0, 1, 2}。
        """
        q_values, _ = self.forward(state_seq)

        if epsilon > 0 and torch.rand(1).item() < epsilon:
            # 随机探索
            return torch.randint(0, self.num_goals, (state_seq.size(0),),
                                 device=state_seq.device)
        else:
            # 贪心选择
            return q_values.argmax(dim=-1)


# ======================================================================
# 快速自测
# ======================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Q1Decision Self-Test")
    print("=" * 50)

    model = Q1Decision(state_dim=18, seq_len=3, hidden_dim=64, num_goals=3)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 随机输入 (batch=4, seq=3, dim=18)
    x = torch.randn(4, 3, 18)
    q_values, raw = model(x)
    probs = model.get_goal_probs(x)
    action = model.select_action(x, epsilon=0.1)

    print(f"\nInput shape:    {x.shape}")
    print(f"Q-values shape: {q_values.shape}  (expect [4, 3])")
    print(f"Raw pass shape: {raw.shape}       (expect [4, 3, 18])")
    print(f"Probs shape:    {probs.shape}     (expect [4, 3])")
    print(f"Probs sum:      {probs.sum(dim=-1).tolist()}")  # 每行应为 1.0
    print(f"Action shape:   {action.shape}    (expect [4])")
    print(f"Actions:        {action.tolist()}")
