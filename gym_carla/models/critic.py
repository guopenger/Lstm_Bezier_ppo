#!/usr/bin/env python
"""
critic.py — 状态价值网络 (Critic / Value Network)

PPO 算法中的 Critic 网络，估计状态价值函数 V(s)。
用于计算 GAE Advantage: A_t = δ_t + γλ δ_{t+1} + ...

架构与 Q1 Actor 对称:
  Input (batch, 3, 18) → LSTM(18, 64) → h_n → FC(64→32) → ReLU → FC(32→1) → V(s)
"""

import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    """状态价值网络 V(s)。

    与 Q1 的 Actor 网络共享相同的输入结构 (LSTM 处理时序状态),
    但输出单个标量表示当前状态的期望回报。

    Architecture:
        Input (batch, 3, 18) → LSTM → h_n (batch, 64) → FC(64→32) → ReLU → FC(32→1)
    """

    def __init__(self, state_dim: int = 18, seq_len: int = 3,
                 hidden_dim: int = 64):
        """
        Args:
            state_dim:  单步状态维度 (18)。
            seq_len:    时间步长 (3)。
            hidden_dim: LSTM 隐藏层维度 (64)。
        """
        super().__init__()
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state_seq: torch.Tensor) -> torch.Tensor:
        """前向推理。

        Args:
            state_seq: (batch, seq_len=3, state_dim=18) 历史状态序列。

        Returns:
            value: (batch,) 状态价值估计。
        """
        _, (h_n, _) = self.lstm(state_seq)
        last_hidden = h_n.squeeze(0)  # (batch, 64)
        value = self.fc(last_hidden).squeeze(-1)  # (batch,)
        return value


# ======================================================================
# 快速自测
# ======================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("CriticNetwork Self-Test")
    print("=" * 50)

    critic = CriticNetwork(state_dim=18, seq_len=3, hidden_dim=64)
    n_params = sum(p.numel() for p in critic.parameters())
    print(f"Parameters: {n_params:,}")

    x = torch.randn(4, 3, 18)
    v = critic(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Value shape:  {v.shape}  (expect [4])")
    print(f"Values:       {v.detach().tolist()}")

    # 测试单样本
    x1 = torch.randn(1, 3, 18)
    v1 = critic(x1)
    print(f"\nSingle input: {x1.shape} → {v1.shape} = {v1.item():.4f}")

    print("\n✓ CriticNetwork test passed!")
