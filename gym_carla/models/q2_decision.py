#!/usr/bin/env python
"""
q2_decision.py — Q² 下层轨迹偏移决策模型

论文依据 (§2.2.1, Figure 3):
  Q2 的输入是 **原始环境输入 (Skip Connection)** 与 **Q1 的 Goal** 的拼接。
  这是论文的关键设计: "concatenate the input and the output of Q1 as the input
  of the lower decision model"。

  架构:
    Input = Concat(Original_State_flat, Goal_onehot)
          = Concat((batch, 54), (batch, 3))
          = (batch, 57)
    → unsqueeze → (batch, 1, 57)  [单步 LSTM 输入]
    → LSTM(input_size=57, hidden_size=64)
    → FC(64→32) → ReLU → FC(32→3)
    → Offset Q-values (batch, 3)

  Offset 映射:
    0 = 偏左 (+offset_magnitude)
    1 = 不偏 (0)
    2 = 偏右 (-offset_magnitude)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Q2Decision(nn.Module):
    """Q² 下层轨迹偏移决策网络。

    ★ 关键修正 ★
      原错误设计: Q2 Input = Concat(Q1_hidden_h, Goal)    → 67维
      论文正确:   Q2 Input = Concat(Original_Input_flat, Goal) → 57维
      即 Q2 通过 Skip Connection 直接接收原始环境观测。

    Architecture:
        Concat(state_flat(54), goal_onehot(3)) = 57
        → LSTM(57, 64) single step
        → FC(64→32) → ReLU → FC(32→3)
    """

    def __init__(self, state_dim: int = 18, seq_len: int = 3,
                 hidden_dim: int = 64, num_goals: int = 3,
                 num_offsets: int = 3):
        """
        Args:
            state_dim:   单步状态维度 (18)。
            seq_len:     时间步数 (3)。
            hidden_dim:  LSTM 隐藏层维度 (64)。
            num_goals:   Q1 输出类别数 (3)，用于 one-hot 编码。
            num_offsets: Q2 输出类别数 (3 — 偏左/不偏/偏右)。
        """
        super().__init__()
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_goals = num_goals
        self.num_offsets = num_offsets

        # Skip Connection 后的拼接维度
        # state_flat = state_dim × seq_len = 18 × 3 = 54
        # goal_onehot = num_goals = 3
        # total = 54 + 3 = 57
        self.concat_dim = state_dim * seq_len + num_goals

        # LSTM: 单步输入，融合 "原始状态 + 行为意图"
        self.lstm = nn.LSTM(
            input_size=self.concat_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # FC Head: LSTM 输出 → Offset Q-values
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_offsets),
        )

    def forward(self, original_state_seq: torch.Tensor,
                goal_onehot: torch.Tensor) -> torch.Tensor:
        """前向推理。

        Args:
            original_state_seq: (batch, seq_len=3, state_dim=18)
                                来自 Q1 透传的原始环境输入 (Skip Connection)。
            goal_onehot:        (batch, num_goals=3)
                                Q1 输出的 Goal one-hot 编码。

        Returns:
            offset_q_values: (batch, 3) Offset Q 值。
        """
        batch_size = original_state_seq.size(0)

        # 展平原始输入: (batch, 3, 18) → (batch, 54)
        state_flat = original_state_seq.reshape(batch_size, -1)

        # 拼接: (batch, 54) + (batch, 3) → (batch, 57)
        concat = torch.cat([state_flat, goal_onehot], dim=-1)

        # 升维为单步 LSTM 输入: (batch, 57) → (batch, 1, 57)
        lstm_input = concat.unsqueeze(1)

        # LSTM 处理
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)

        # 取输出: (batch, 1, 64) → (batch, 64)
        last_hidden = lstm_out.squeeze(1)

        # FC 映射到 Offset Q-values
        offset_q_values = self.fc(last_hidden)  # (batch, 3)

        return offset_q_values

    def get_offset_probs(self, original_state_seq: torch.Tensor,
                         goal_onehot: torch.Tensor,
                         temperature: float = 1.0) -> torch.Tensor:
        """获取 Offset 的 Softmax 概率分布。"""
        q_values = self.forward(original_state_seq, goal_onehot)
        return F.softmax(q_values / temperature, dim=-1)

    def select_action(self, original_state_seq: torch.Tensor,
                      goal_onehot: torch.Tensor,
                      epsilon: float = 0.0) -> torch.Tensor:
        """ε-greedy 动作选择。"""
        q_values = self.forward(original_state_seq, goal_onehot)

        if epsilon > 0 and torch.rand(1).item() < epsilon:
            return torch.randint(0, self.num_offsets,
                                 (original_state_seq.size(0),),
                                 device=original_state_seq.device)
        else:
            return q_values.argmax(dim=-1)


# ======================================================================
# 快速自测
# ======================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Q2Decision Self-Test")
    print("=" * 50)

    model = Q2Decision(state_dim=18, seq_len=3, hidden_dim=64,
                        num_goals=3, num_offsets=3)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Concat dim: {model.concat_dim} (expect 57 = 18×3 + 3)")

    # 模拟 Q1 的输出
    batch = 4
    raw_input = torch.randn(batch, 3, 18)       # Skip Connection 的原始输入
    goal_onehot = torch.zeros(batch, 3)
    goal_onehot[:, 1] = 1.0                      # 全部选"保持"

    q_values = model(raw_input, goal_onehot)
    probs = model.get_offset_probs(raw_input, goal_onehot)
    action = model.select_action(raw_input, goal_onehot, epsilon=0.1)

    print(f"\nRaw input shape:    {raw_input.shape}")
    print(f"Goal onehot shape:  {goal_onehot.shape}")
    print(f"Q-values shape:     {q_values.shape}  (expect [4, 3])")
    print(f"Probs shape:        {probs.shape}     (expect [4, 3])")
    print(f"Probs sum:          {probs.sum(dim=-1).tolist()}")
    print(f"Action shape:       {action.shape}    (expect [4])")
    print(f"Actions:            {action.tolist()}")
