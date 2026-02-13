#!/usr/bin/env python
"""
train_hierarchical.py — 分层强化学习 DQN 训练主脚本

论文: A Trajectory Planning and Tracking Method Based on Deep Hierarchical RL

训练流程:
  1. env.reset() → obs (3, 18)
  2. HierarchicalPolicy 选择 (goal, offset) ← ε-greedy
  3. env.step([goal, offset]) → obs', reward, done, info
  4. 存入 ReplayBuffer
  5. 从 ReplayBuffer 采样 mini-batch 计算 DQN loss
  6. Q1, Q2 各自独立更新
  7. 定期同步 Target Network

使用方式:
  conda activate carla_rl
  # 先启动 CARLA 服务器 (CarlaUE4.exe)
  python train_hierarchical.py
"""

import glob
import os
import sys
import time
import random
import argparse
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("WARNING: tensorboard not installed, logging to console only.")
    print("  Install: pip install tensorboard==2.11.2")

# ==========================================
# 挂载 CARLA Python API (必须在 import gym / carla 之前)
# ==========================================
try:
    carla_root = r'D:\electron\CARLA_0.9.13\WindowsNoEditor'
    sys.path.append(glob.glob(carla_root + '/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    print("成功挂载 CARLA API！")
except IndexError:
    print("找不到 CARLA egg 文件，请检查 Python 版本是否为 3.7！")
    sys.exit(1)

import carla  # noqa: E402  — 验证 carla 可导入
import gym
import gym_carla
from gym_carla.models.hierarchical_policy import HierarchicalPolicy
from gym_carla import config as cfg


# ======================================================================
# Replay Buffer
# ======================================================================

class ReplayBuffer:
    """Experience replay buffer for DQN training.

    Stores transitions (state, goal, offset, reward, next_state, done).
    """

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, goal, offset, reward, next_state, done):
        """Store a transition.

        Args:
            state:      np.ndarray (seq_len, state_dim)
            goal:       int (0-2)
            offset:     int (0-2)
            reward:     float
            next_state: np.ndarray (seq_len, state_dim)
            done:       bool
        """
        self.buffer.append((state, goal, offset, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a random mini-batch.

        Returns:
            Tuple of batched tensors: (states, goals, offsets, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, goals, offsets, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))          # (B, seq_len, state_dim)
        goals = torch.LongTensor(goals)                       # (B,)
        offsets = torch.LongTensor(offsets)                    # (B,)
        rewards = torch.FloatTensor(rewards)                   # (B,)
        next_states = torch.FloatTensor(np.array(next_states)) # (B, seq_len, state_dim)
        dones = torch.FloatTensor(dones)                       # (B,)

        return states, goals, offsets, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ======================================================================
# Trainer
# ======================================================================

class HierarchicalDQNTrainer:
    """DQN trainer for hierarchical Q1 + Q2 networks.

    Each sub-network (Q1, Q2) has its own target network.
    Both are updated with the same transition but their own TD-target.
    """

    def __init__(
        self,
        policy: HierarchicalPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 100000,
        target_update_freq: int = 100,
        tau: float = 1.0,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau

        # Separate optimizers for Q1 and Q2
        self.optimizer_q1 = optim.Adam(policy.q1.parameters(), lr=lr)
        self.optimizer_q2 = optim.Adam(policy.q2.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.train_step = 0

    def store_transition(self, state, goal, offset, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, goal, offset, reward, next_state, done)

    def update(self) -> dict:
        """Perform one gradient update step on both Q1 and Q2.

        Returns:
            dict with 'loss_q1', 'loss_q2', or empty if buffer too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample mini-batch
        states, goals, offsets, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        goals = goals.to(self.device)
        offsets = offsets.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # --- Q1 Loss ---
        # Current Q1 values for selected goals
        goal_q_current, _ = self.policy.q1(states)
        goal_q_selected = goal_q_current.gather(1, goals.unsqueeze(1)).squeeze(1)

        # Target Q1 values (from target network)
        with torch.no_grad():
            goal_q_next, _ = self.policy.q1_target(next_states)
            goal_q_next_max = goal_q_next.max(dim=1)[0]
            goal_td_target = rewards + self.gamma * goal_q_next_max * (1 - dones)

        loss_q1 = self.loss_fn(goal_q_selected, goal_td_target)

        self.optimizer_q1.zero_grad()
        loss_q1.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.q1.parameters(), 10.0)
        self.optimizer_q1.step()

        # --- Q2 Loss ---
        # Get goal one-hot for Q2 input
        goal_onehot = torch.nn.functional.one_hot(
            goals, self.policy.num_goals).float()

        # Current Q2 values for selected offsets
        offset_q_current = self.policy.q2(states, goal_onehot)
        offset_q_selected = offset_q_current.gather(
            1, offsets.unsqueeze(1)).squeeze(1)

        # Target Q2 values
        with torch.no_grad():
            # Use target Q1 to get next goal for target Q2
            goal_q_next_target, raw_next = self.policy.q1_target(next_states)
            next_goals = goal_q_next_target.argmax(dim=1)
            next_goal_onehot = torch.nn.functional.one_hot(
                next_goals, self.policy.num_goals).float()

            offset_q_next = self.policy.q2_target(next_states, next_goal_onehot)
            offset_q_next_max = offset_q_next.max(dim=1)[0]
            offset_td_target = rewards + self.gamma * offset_q_next_max * (1 - dones)

        loss_q2 = self.loss_fn(offset_q_selected, offset_td_target)

        self.optimizer_q2.zero_grad()
        loss_q2.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.q2.parameters(), 10.0)
        self.optimizer_q2.step()

        # --- Target network update ---
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.policy.update_target(tau=self.tau)

        return {
            'loss_q1': loss_q1.item(),
            'loss_q2': loss_q2.item(),
        }


# ======================================================================
# Epsilon schedule
# ======================================================================

def get_epsilon(episode: int, total_episodes: int,
                eps_start: float = 1.0, eps_end: float = 0.05,
                eps_decay_frac: float = 0.5) -> float:
    """Linear epsilon decay schedule.

    Args:
        episode:        Current episode.
        total_episodes: Total episodes.
        eps_start:      Starting epsilon.
        eps_end:        Minimum epsilon.
        eps_decay_frac: Fraction of total episodes over which to decay.
    """
    decay_episodes = int(total_episodes * eps_decay_frac)
    if episode >= decay_episodes:
        return eps_end
    return eps_start - (eps_start - eps_end) * episode / decay_episodes


# ======================================================================
# Main training loop
# ======================================================================

def make_env_params():
    """Construct environment parameters for hierarchical mode."""
    return {
        # Display / rendering
        'display_size': 256,
        'max_past_step': 1,
        'number_of_vehicles': cfg.NUMBER_OF_VEHICLES,
        'number_of_walkers': cfg.NUMBER_OF_WALKERS,
        'dt': cfg.CARLA_DT,
        'task_mode': 'random',
        'max_time_episode': cfg.MAX_STEPS_PER_EPISODE,
        'max_waypt': 12,
        'obs_range': 32,
        'lidar_bin': 0.125,
        'd_behind': 12,
        'out_lane_thres': 2.0,
        'desired_speed': cfg.DESIRED_SPEED,
        'max_ego_spawn_times': 200,
        'display_route': True,
        'pixor': False,
        'pixor_size': 32,
        'discrete': False,
        'discrete_acc': [-3.0, 0.0, 3.0],
        'discrete_steer': [-0.2, 0.0, 0.2],
        'continuous_accel_range': [-3.0, 3.0],
        'continuous_steer_range': [-0.3, 0.3],
        'ego_vehicle_filter': 'vehicle.lincoln*',
        'port': cfg.CARLA_PORT,
        'town': cfg.CARLA_TOWN,
        # Hierarchical RL
        'hierarchical': True,
        'state_dim': cfg.STATE_DIM,
        'seq_len': cfg.SEQ_LEN,
        'lane_width': cfg.LANE_WIDTH,
    }


def train(resume_path: str = None):
    """Main training entry point."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    # Create environment
    print("Creating hierarchical CARLA environment...")
    params = make_env_params()
    env = gym.make('carla-v0', params=params)

    # Create policy
    policy = HierarchicalPolicy(
        state_dim=cfg.STATE_DIM,
        seq_len=cfg.SEQ_LEN,
        hidden_dim=cfg.HIDDEN_DIM,
        num_goals=cfg.NUM_GOALS,
        num_offsets=cfg.NUM_OFFSETS,
    )

    # Create trainer
    trainer = HierarchicalDQNTrainer(
        policy=policy,
        lr=cfg.LEARNING_RATE,
        gamma=cfg.GAMMA,
        batch_size=cfg.BATCH_SIZE,
        buffer_size=cfg.REPLAY_BUFFER_SIZE,
        target_update_freq=cfg.TARGET_UPDATE_FREQ,
        tau=1.0,   # Hard update
        device=device,
    )

    # Resume from checkpoint
    start_episode = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming training from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        trainer.optimizer_q1.load_state_dict(checkpoint['optimizer_q1'])
        trainer.optimizer_q2.load_state_dict(checkpoint['optimizer_q2'])
        start_episode = checkpoint['episode'] + 1
        if 'avg_reward' in checkpoint:
            print(f"  Last avg_reward: {checkpoint['avg_reward']:.2f}")

    # Training loop
    num_episodes = cfg.NUM_EPISODES
    max_steps = cfg.MAX_STEPS_PER_EPISODE

    # Logging
    episode_rewards = deque(maxlen=100)
    best_avg_reward = -float('inf')
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    # TensorBoard
    log_dir = os.path.join(os.path.dirname(__file__), 'runs',
                           time.strftime('DQN_%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir) if HAS_TENSORBOARD else None
    if writer:
        print(f"  TensorBoard log: {log_dir}")
        print(f"  启动命令: tensorboard --logdir={os.path.dirname(log_dir)}")

    print(f"\n{'='*60}")
    print(f"Training Config:")
    print(f"  Episodes:       {num_episodes}")
    print(f"  Max steps/ep:   {max_steps}")
    print(f"  Batch size:     {cfg.BATCH_SIZE}")
    print(f"  LR:             {cfg.LEARNING_RATE}")
    print(f"  Gamma:          {cfg.GAMMA}")
    print(f"  Buffer size:    {cfg.REPLAY_BUFFER_SIZE}")
    print(f"  Target update:  every {cfg.TARGET_UPDATE_FREQ} steps")
    params_count = policy.count_parameters()
    print(f"  Q1 params:      {params_count['q1']:,}")
    print(f"  Q2 params:      {params_count['q2']:,}")
    print(f"{'='*60}\n")

    for episode in range(start_episode, num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        epsilon = get_epsilon(episode, num_episodes)

        for step in range(max_steps):
            # State to tensor
            state_tensor = torch.FloatTensor(obs).to(device)

            # Select action
            goal, offset = policy.select_action(state_tensor, epsilon=epsilon)

            # Environment step
            next_obs, reward, done, info = env.step([goal, offset])

            # Store transition
            trainer.store_transition(obs, goal, offset, reward, next_obs, float(done))

            # Update networks
            losses = trainer.update()

            episode_reward += reward
            obs = next_obs

            if done:
                break

        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards)

        # TensorBoard logging (every episode)
        if writer:
            writer.add_scalar('reward/episode', episode_reward, episode)
            writer.add_scalar('reward/avg100', avg_reward, episode)
            writer.add_scalar('train/epsilon', epsilon, episode)
            writer.add_scalar('train/steps_per_episode', step + 1, episode)
            if losses:
                writer.add_scalar('loss/Q1', losses.get('loss_q1', 0), episode)
                writer.add_scalar('loss/Q2', losses.get('loss_q2', 0), episode)
            writer.add_scalar('train/buffer_size', len(trainer.replay_buffer), episode)

        # Console logging
        if episode % 10 == 0 or episode == num_episodes - 1:
            loss_str = ""
            if losses:
                loss_str = f", Q1_loss={losses.get('loss_q1', 0):.4f}, Q2_loss={losses.get('loss_q2', 0):.4f}"
            print(f"[Ep {episode:4d}/{num_episodes}] "
                  f"reward={episode_reward:7.1f}, avg100={avg_reward:7.1f}, "
                  f"eps={epsilon:.3f}, steps={step+1}{loss_str}")

        # Save best model
        if avg_reward > best_avg_reward and len(episode_rewards) >= 20:
            best_avg_reward = avg_reward
            save_path = os.path.join(save_dir, 'best_policy.pth')
            torch.save({
                'episode': episode,
                'policy_state_dict': policy.state_dict(),
                'optimizer_q1': trainer.optimizer_q1.state_dict(),
                'optimizer_q2': trainer.optimizer_q2.state_dict(),
                'avg_reward': avg_reward,
            }, save_path)
            print(f"  >> Saved best model (avg_reward={avg_reward:.1f})")

        # Periodic checkpoint
        if episode % 500 == 0 and episode > 0:
            ckpt_path = os.path.join(save_dir, f'policy_ep{episode}.pth')
            torch.save({
                'episode': episode,
                'policy_state_dict': policy.state_dict(),
                'optimizer_q1': trainer.optimizer_q1.state_dict(),
                'optimizer_q2': trainer.optimizer_q2.state_dict(),
                'avg_reward': avg_reward,
            }, ckpt_path)
            print(f"  >> Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(save_dir, 'final_policy.pth')
    torch.save({
        'episode': num_episodes,
        'policy_state_dict': policy.state_dict(),
        'avg_reward': avg_reward,
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")

    if writer:
        writer.close()
        print(f"TensorBoard logs saved to: {log_dir}")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    train(resume_path=args.resume)
