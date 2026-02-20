#!/usr/bin/env python
"""
q2_decision.py — Q² 下层轨迹偏移决策模型 (Gaussian Actor — 连续动作)

论文依据 (§2.2.1, Figure 3, §2.1.3 Eq.2):
  Q2 的输入是 **原始环境输入 (Skip Connection)** 与 **Q1 的 Goal** 的拼接。
  这是论文的关键设计: "concatenate the input and the output of Q1 as the input
  of the lower decision model"。

  动作空间 (论文 Eq.2):
    A_p = [p_off]  — 单个连续值！
    p_off 表示轨迹的横向偏移量 (meters)，不是离散档位。

  架构:
    Input = Concat(Original_State_seq, Goal_onehot_expanded)
          = Concat((batch, 3, 18), (batch, 3, 3))
          = (batch, 3, 21)
    → LSTM(input_size=21, hidden_size=64) 处理 3 个时间步
    → 取最后时间步 → (batch, 64)
    → FC(64→32) → ReLU → FC(32→1) → mean μ
    → 高斯策略: p_off ~ N(μ, σ²)，σ 为可学习参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Q2Decision(nn.Module):
    """Q² 下层轨迹偏移决策网络 — 连续高斯策略。

    修正
      1. Skip Connection: Q2 直接接收原始环境观测 (非 Q1 隐状态)
      2. 连续输出: 输出高斯分布 N(μ, σ²) 而非离散 Q 值
      3. 时序建模: 保持 (batch, 3, 21) 的时序结构，充分利用 LSTM

    Architecture:
        Concat(state_seq(3,18), goal_expanded(3,3)) = (3, 21)
        → LSTM(21, 64) 处理 3 个时间步
        → 取最后时间步 → (batch, 64)
        → FC(64→32) → ReLU → FC(32→1) → μ
        + learnable log_std → σ = exp(log_std)
        → p_off ~ N(μ, σ²)
    """

    def __init__(self, state_dim: int = 18, seq_len: int = 3,
                 hidden_dim: int = 64, num_goals: int = 3,
                 log_std_init: float = 0.0):
        """
        Args:
            state_dim:    单步状态维度 (18)。
            seq_len:      时间步数 (3)。
            hidden_dim:   LSTM 隐藏层维度 (64)。
            num_goals:    Q1 输出类别数 (3)，用于 one-hot 编码。
            log_std_init: 高斯策略初始 log(σ)，0.0 表示初始 σ=1.0。
        """
        super().__init__()
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_goals = num_goals

        # Skip Connection 后的拼接维度（每个时间步）
        # 每个时间步: state_dim + goal_onehot = 18 + 3 = 21
        # 序列长度: seq_len = 3
        self.concat_dim = state_dim + num_goals

        # LSTM: 多步时序输入，融合 "原始状态序列 + 行为意图"
        self.lstm = nn.LSTM(
            input_size=self.concat_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # FC Head: LSTM 输出 → 连续偏移量均值 μ
        self.fc_mean = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # 可学习的对数标准差 log(σ)
        # 初始 log_std=0 → σ=1.0，训练过程中自动调整探索程度
        self.log_std = nn.Parameter(torch.full((1,), log_std_init))

    def forward(self, original_state_seq: torch.Tensor,
                goal_onehot: torch.Tensor) -> torch.Tensor:
        """前向推理，输出高斯分布的均值 μ。

        Args:
            original_state_seq: (batch, seq_len=3, state_dim=18)
                                来自 Q1 透传的原始环境输入 (Skip Connection)。
            goal_onehot:        (batch, num_goals=3)
                                Q1 输出的 Goal one-hot 编码。

        Returns:
            mean: (batch,) 偏移量均值 μ。
        """
        batch_size = original_state_seq.size(0)

        # 扩展 goal 到每个时间步: (batch, 3) → (batch, seq_len=3, 3)
        goal_expanded = goal_onehot.unsqueeze(1).expand(-1, self.seq_len, -1)

        # 拼接: (batch, 3, 18) + (batch, 3, 3) → (batch, 3, 21)
        concat = torch.cat([original_state_seq, goal_expanded], dim=-1)

        # LSTM 处理
        lstm_out, (h_n, c_n) = self.lstm(concat)
        # 这个地方给维度问题说清楚
        # 输入state_seq.shape = (4, 3, 18)， batch seq feature
        # lstm_out.shape = (4, 3, 64)，batch seq hidden
        # h_n.shape = (1, 4, 64)，layers batch hidden
        # c_n.shape = (1, 4, 64)，layers batch hidden

        # 取输出: (batch, 1, 64) → (batch, 64)
        # 三种方式等价last_hidden = h_n.squeeze(0),last_hidden = lstm_out[:, -1, :],last_hidden = lstm_out.squeeze(1)
        last_hidden = h_n.squeeze(0)

        # FC 映射到均值 μ: (batch, 64) → (batch, 1) → (batch,)
        mean = self.fc_mean(last_hidden).squeeze(-1)

        return mean

    # ------------------------------------------------------------------
    # PPO 接口
    # ------------------------------------------------------------------

    def get_dist(self, original_state_seq: torch.Tensor,
                 goal_onehot: torch.Tensor):
        """获取 Offset 的高斯分布 N(μ, σ²)。

        Returns:
            Normal 分布对象。
        """
        mean = self.forward(original_state_seq, goal_onehot)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def evaluate_actions(self, original_state_seq: torch.Tensor,
                         goal_onehot: torch.Tensor,
                         actions: torch.Tensor):
        """评估给定连续动作的 log_prob 和 entropy (PPO 更新用)。

        Args:
            original_state_seq: (batch, seq_len, state_dim)。
            goal_onehot:        (batch, num_goals)。
            actions:            (batch,) FloatTensor, 连续偏移值。

        Returns:
            log_probs: (batch,) 动作对数概率。
            entropy:   (batch,) 策略熵。
        """
        dist = self.get_dist(original_state_seq, goal_onehot)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy

    def select_action(self, original_state_seq: torch.Tensor,
                      goal_onehot: torch.Tensor):
        """从高斯策略中采样偏移量 (PPO 训练用)。

        Returns:
            p_off:    (batch,) 采样的连续偏移值。
            log_prob: (batch,) 对数概率。
        """
        dist = self.get_dist(original_state_seq, goal_onehot)
        p_off = dist.sample()
        log_prob = dist.log_prob(p_off)
        return p_off, log_prob


# ======================================================================
# 快速自测
# ======================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Q2Decision Self-Test (PPO Gaussian Actor)")
    print("=" * 50)

    model = Q2Decision(state_dim=18, seq_len=3, hidden_dim=64,
                       num_goals=3, log_std_init=0.0)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Concat dim: {model.concat_dim} (expect 21 = 18 + 3)")
    print(f"Initial σ:  {model.log_std.exp().item():.4f} (expect 1.0)")

    # 模拟 Q1 的输出
    batch = 4
    raw_input = torch.randn(batch, 3, 18)       # Skip Connection 的原始输入
    goal_onehot = torch.zeros(batch, 3)
    goal_onehot[:, 1] = 1.0                      # 全部选"保持"

    # forward → mean
    mean = model(raw_input, goal_onehot)
    print(f"\nRaw input shape:    {raw_input.shape}")
    print(f"Goal onehot shape:  {goal_onehot.shape}")
    print(f"Mean shape:         {mean.shape}  (expect [4])")
    print(f"Mean values:        {mean.detach().tolist()}")

    # 采样
    p_off, lp = model.select_action(raw_input, goal_onehot)
    print(f"Sampled p_off:      {p_off.tolist()}")
    print(f"Log probs:          {lp.tolist()}")

    # evaluate
    lp2, ent = model.evaluate_actions(raw_input, goal_onehot, p_off)
    print(f"Eval log_probs:     {lp2.tolist()}")
    print(f"Entropy:            {ent.tolist()}")

    # 高斯分布验证
    dist = model.get_dist(raw_input, goal_onehot)
    print(f"Dist mean:          {dist.mean.tolist()}")
    print(f"Dist std:           {dist.stddev.tolist()}")

    print("\n✓ Q2Decision (PPO Gaussian) test passed!")
