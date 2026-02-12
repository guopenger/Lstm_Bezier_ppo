#!/usr/bin/env python
"""
state_buffer.py — 历史状态滑动窗口缓冲区

论文依据 (§2.1.2 + §2.2.1):
  - 状态空间 18 维: [v_ego, lane_id, Δv_1..Δv_8, Δd_1..Δd_8]
  - LSTM 输入为过去 3 个采样时刻的状态序列
  - 输出张量形状: (seq_len=3, state_dim=18)

使用方式:
  buf = StateBuffer(state_dim=18, seq_len=3)
  buf.reset()
  buf.push(state_18d)          # 每个 env.step() 后调用
  seq = buf.get_sequence()     # → torch.Tensor (3, 18)
"""

import numpy as np
import torch
from collections import deque


class StateBuffer:
    """维护固定长度的历史状态滑动窗口，供 LSTM 网络消费。

    Attributes:
        state_dim (int): 每步状态向量维度，论文要求 18。
        seq_len (int): 时间步长，论文 §2.2.1 要求 3。
    """

    def __init__(self, state_dim: int = 18, seq_len: int = 3, device: str = "cpu"):
        """
        Args:
            state_dim: 单步状态向量维度 (论文: 18)。
            seq_len:   保留的历史时间步数 (论文: 3)。
            device:    输出张量所在设备 ("cpu" / "cuda")。
        """
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.device = device
        self._buffer: deque = deque(maxlen=seq_len)
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """清空缓冲区并用零向量填满，保证任何时刻 get_sequence() 都能返回合法张量。"""
        self._buffer.clear()
        for _ in range(self.seq_len):
            self._buffer.append(np.zeros(self.state_dim, dtype=np.float32))

    def push(self, state: np.ndarray) -> None:
        """将新的单步状态推入缓冲区 (FIFO: 最老的自动弹出)。

        Args:
            state: 形状 (state_dim,) 的 numpy 数组。
                   论文 18 维: [v_ego, lane_id, Δv_1..Δv_8, Δd_1..Δd_8]

        Raises:
            ValueError: state 形状不匹配。
        """
        state = np.asarray(state, dtype=np.float32).flatten()
        if state.shape[0] != self.state_dim:
            raise ValueError(
                f"Expected state dim {self.state_dim}, got {state.shape[0]}. "
                f"State vector must be: [v_ego, lane_id, Δv×8, Δd×8]."
            )
        self._buffer.append(state)

    def get_sequence(self, as_batch: bool = False) -> torch.Tensor:
        """返回当前滑动窗口中的状态序列张量。

        LSTM 要求输入形状为 (batch, seq_len, input_size)。
        本方法默认返回 (seq_len, state_dim)，设 as_batch=True 可得 (1, seq_len, state_dim)。

        Args:
            as_batch: 是否在最前面添加 batch 维度。

        Returns:
            torch.Tensor:
              - as_batch=False → shape (seq_len, state_dim) = (3, 18)
              - as_batch=True  → shape (1, seq_len, state_dim) = (1, 3, 18)
        """
        # deque → numpy → tensor
        seq_np = np.stack(list(self._buffer), axis=0)  # (seq_len, state_dim)
        seq_tensor = torch.from_numpy(seq_np).to(self.device)

        if as_batch:
            seq_tensor = seq_tensor.unsqueeze(0)  # (1, seq_len, state_dim)

        return seq_tensor

    def get_numpy(self) -> np.ndarray:
        """返回 numpy 格式的序列，形状 (seq_len, state_dim)，用于 gym 观测空间。"""
        return np.stack(list(self._buffer), axis=0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_full(self) -> bool:
        """缓冲区是否已用真实数据填满（不含初始零向量）。"""
        # 简化判断：只有当所有数据都非全零时才认为满
        return len(self._buffer) == self.seq_len and not np.allclose(
            self._buffer[0], 0.0
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"StateBuffer(state_dim={self.state_dim}, seq_len={self.seq_len}, "
            f"current_len={len(self._buffer)}, device={self.device})"
        )


# ======================================================================
# 快速自测
# ======================================================================
if __name__ == "__main__":
    buf = StateBuffer(state_dim=18, seq_len=3)
    print(buf)

    # 模拟 5 步推入
    for step in range(5):
        fake_state = np.random.randn(18).astype(np.float32)
        fake_state[0] = 20.0 + step  # v_ego
        fake_state[1] = 1.0          # lane_id
        buf.push(fake_state)
        print(f"Step {step}: buffer len={len(buf)}, is_full={buf.is_full}")

    seq = buf.get_sequence(as_batch=True)
    print(f"\nOutput shape: {seq.shape}")   # 期望 (1, 3, 18)
    print(f"v_ego 列 (最近3步): {seq[0, :, 0].tolist()}")  # 应为 [22, 23, 24]
