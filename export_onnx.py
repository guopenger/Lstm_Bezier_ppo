#!/usr/bin/env python
"""
export_onnx.py — 导出训练好的 HierarchicalPolicy 为 ONNX 格式

用途: C++ / ROS2 部署
  - 输入: state_sequence (batch, 3, 18)
  - 输出: goal (batch,), offset (batch,)

使用方式:
  python export_onnx.py --checkpoint checkpoints/best_policy.pth --output policy.onnx
"""

import argparse
import os
import sys
import torch

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gym_carla.models.hierarchical_policy import HierarchicalPolicy
from gym_carla import config as cfg


def export(checkpoint_path: str, output_path: str, verify: bool = True):
    """Load checkpoint and export to ONNX.

    Args:
        checkpoint_path: Path to .pth checkpoint.
        output_path:     Output .onnx file path.
        verify:          Whether to verify exported model with random input.
    """
    # Load policy
    policy = HierarchicalPolicy(
        state_dim=cfg.STATE_DIM,
        seq_len=cfg.SEQ_LEN,
        hidden_dim=cfg.HIDDEN_DIM,
        num_goals=cfg.NUM_GOALS,
        num_offsets=cfg.NUM_OFFSETS,
    )

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        policy.load_state_dict(ckpt['policy_state_dict'])
        print(f"Loaded checkpoint: {checkpoint_path}")
        if 'episode' in ckpt:
            print(f"  Episode: {ckpt['episode']}")
        if 'avg_reward' in ckpt:
            print(f"  Avg reward: {ckpt['avg_reward']:.1f}")
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        print("  Exporting with random weights for testing purposes.")

    policy.eval()

    # Export
    policy.export_onnx(output_path)
    print(f"\nONNX model exported to: {output_path}")

    # Verify
    if verify:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(output_path)
            dummy_input = torch.randn(1, cfg.SEQ_LEN, cfg.STATE_DIM).numpy()

            # PyTorch inference
            with torch.no_grad():
                result = policy.forward(torch.from_numpy(dummy_input))
                pt_goal = result['goal_idx'].numpy()
                pt_offset = result['offset_idx'].numpy()

            # ONNX inference
            onnx_result = sess.run(None, {'state_sequence': dummy_input})
            onnx_goal, onnx_offset = onnx_result

            print(f"\nVerification:")
            print(f"  PyTorch:  goal={pt_goal}, offset={pt_offset}")
            print(f"  ONNX:     goal={onnx_goal}, offset={onnx_offset}")

            if (pt_goal == onnx_goal).all() and (pt_offset == onnx_offset).all():
                print("  ✓ Results match!")
            else:
                print("  ✗ Results differ (may be due to floating point)")
        except ImportError:
            print("\nNote: Install onnxruntime to verify: pip install onnxruntime")

    # Print model info
    file_size = os.path.getsize(output_path) / 1024
    params = policy.count_parameters()
    print(f"\nModel info:")
    print(f"  File size:     {file_size:.1f} KB")
    print(f"  Q1 params:     {params['q1']:,}")
    print(f"  Q2 params:     {params['q2']:,}")
    print(f"  Total params:  {params['total_online']:,}")
    print(f"  Input shape:   (batch, {cfg.SEQ_LEN}, {cfg.STATE_DIM})")
    print(f"  Output:        goal ∈ {{0,1,2}}, offset ∈ {{0,1,2}}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export hierarchical policy to ONNX')
    parser.add_argument('--checkpoint', '-c', type=str,
                        default='checkpoints/best_policy.pth',
                        help='Path to checkpoint .pth file')
    parser.add_argument('--output', '-o', type=str,
                        default='hierarchical_policy.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification with onnxruntime')
    args = parser.parse_args()

    export(args.checkpoint, args.output, verify=not args.no_verify)
