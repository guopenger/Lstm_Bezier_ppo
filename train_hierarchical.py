#!/usr/bin/env python
"""
train_hierarchical.py — 分层强化学习 PPO 训练主脚本

论文: A Trajectory Planning and Tracking Method Based on Deep Hierarchical RL
算法: Proximal Policy Optimization (PPO) — 论文 HRL-TRPO 的一阶近似

训练流程:
  1. 收集 rollout_steps 步的交互数据 (on-policy)
  2. 用 GAE 计算 advantage
  3. PPO clip 损失更新 Q1 (离散 Categorical) + Q2 (连续 Gaussian) + Critic
  4. 重复

动作空间 (论文 §2.1.3):
  - Q1: goal ∈ {0=左换道, 1=保持, 2=右换道} — 离散 Categorical (Eq.1)
  - Q2: p_off ∈ ℝ                         — 连续 Gaussian    (Eq.2)

使用方式:
  conda activate carla_rl
  # 先启动 CARLA 服务器 (CarlaUE4.exe -quality-level=Low)
  python train_hierarchical.py
  # 恢复训练:
  python train_hierarchical.py --resume checkpoints/best_policy.pth
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
import torch.nn.functional as F

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
# Rollout Buffer (替代 DQN 的 ReplayBuffer)
# ======================================================================

class RolloutBuffer:
    """PPO On-Policy Rollout Buffer。

    与 DQN ReplayBuffer 的关键区别:
      - 不做随机重采样，使用全部数据
      - 每次 PPO 更新后清空
      - 存储 log_prob 和 value 用于 importance sampling ratio 计算
    """

    def __init__(self):
        self.states = []
        self.goals = []
        self.offsets = []
        self.rewards = []
        self.dones = []
        self.log_probs_goal = []
        self.log_probs_offset = []
        self.values = []
        # 在 compute 后填充
        self.returns = None
        self.advantages = None

    def add(self, state, goal, offset, reward, done,
            log_prob_goal, log_prob_offset, value):
        """添加一步交互数据。"""
        self.states.append(state.copy() if isinstance(state, np.ndarray) else state)
        self.goals.append(goal)
        self.offsets.append(offset)
        self.rewards.append(reward)
        self.dones.append(float(done))
        self.log_probs_goal.append(log_prob_goal)
        self.log_probs_offset.append(log_prob_offset)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """用 GAE (Generalized Advantage Estimation) 计算 returns 和 advantages。

        GAE 公式:
          δ_t = r_t + γ V(s_{t+1}) (1-d_t) - V(s_t)
          A_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}
          returns_t = A_t + V(s_t)
        """
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values + [last_value], dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        self.advantages = advantages
        self.returns = advantages + np.array(self.values, dtype=np.float32)

    def get_mini_batches(self, num_mini_batches, device='cpu'):
        """将 rollout 数据随机分成 mini-batch 并转为 Tensor。

        Yields:
            dict of Tensors, one mini-batch at a time.
        """
        batch_size = len(self.states)
        mini_batch_size = max(batch_size // num_mini_batches, 1)
        indices = np.random.permutation(batch_size)

        # 一次性转为 Tensor
        all_states = torch.FloatTensor(np.array(self.states)).to(device)
        all_goals = torch.LongTensor(self.goals).to(device)
        all_offsets = torch.FloatTensor(self.offsets).to(device)
        all_old_lp_goal = torch.FloatTensor(self.log_probs_goal).to(device)
        all_old_lp_offset = torch.FloatTensor(self.log_probs_offset).to(device)
        all_returns = torch.FloatTensor(self.returns).to(device)
        all_advantages = torch.FloatTensor(self.advantages).to(device)

        # 归一化 advantages (PPO 标准做法)
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

        for start in range(0, batch_size, mini_batch_size):
            end = min(start + mini_batch_size, batch_size)
            idx = indices[start:end]
            yield {
                'states': all_states[idx],
                'goals': all_goals[idx],
                'offsets': all_offsets[idx],
                'old_log_probs_goal': all_old_lp_goal[idx],
                'old_log_probs_offset': all_old_lp_offset[idx],
                'returns': all_returns[idx],
                'advantages': all_advantages[idx],
            }

    def clear(self):
        """清空 buffer (每次 PPO 更新后调用)。"""
        self.__init__()

    def __len__(self):
        return len(self.states)


# ======================================================================
# PPO Trainer
# ======================================================================

class HierarchicalPPOTrainer:
    """PPO trainer for hierarchical Q1 (discrete) + Q2 (continuous) policy.

    损失函数:
      L = L_clip + c1 × L_value - c2 × H

      L_clip = -min(r(θ) × A, clip(r(θ), 1-ε, 1+ε) × A)
        其中 r(θ) = π_new(a|s) / π_old(a|s) = exp(log_prob_new - log_prob_old)
        联合 log_prob = log_prob_goal + log_prob_offset

      L_value = MSE(V(s), returns)
      H = H_goal + H_offset  (entropy bonus)
    """

    def __init__(self, policy, lr=3e-4, clip_epsilon=0.2,
                 ppo_epochs=4, num_mini_batches=4,
                 value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, device='cpu'):
        self.policy = policy.to(device)
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.num_mini_batches = num_mini_batches
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # 统一优化器 (Actor Q1 + Actor Q2 + Critic)
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

        self.rollout = RolloutBuffer()

    def collect_step(self, state, goal, offset, reward, done,
                     log_prob_goal, log_prob_offset, value):
        """存储一步数据到 rollout buffer。"""
        self.rollout.add(state, goal, offset, reward, done,
                         log_prob_goal, log_prob_offset, value)

    def update(self, last_value):
        """执行一次完整的 PPO 更新。

        Args:
            last_value: V(s_T+1) — rollout 最后一步之后的状态价值。

        Returns:
            dict: 训练统计 {policy_loss, value_loss, entropy, clip_fraction}
        """
        gamma = cfg.GAMMA
        gae_lambda = cfg.GAE_LAMBDA

        # 1. 计算 GAE advantages 和 returns
        self.rollout.compute_returns_and_advantages(last_value, gamma, gae_lambda)

        # 2. 多轮 PPO epoch 更新
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        n_updates = 0

        for epoch in range(self.ppo_epochs):
            for batch in self.rollout.get_mini_batches(
                    self.num_mini_batches, self.device):

                # 评估当前策略下的 log_prob 和 value
                eval_result = self.policy.evaluate_actions(
                    batch['states'], batch['goals'], batch['offsets'])

                new_lp_goal = eval_result['log_prob_goal']
                new_lp_offset = eval_result['log_prob_offset']
                entropy_goal = eval_result['entropy_goal']
                entropy_offset = eval_result['entropy_offset']
                new_value = eval_result['value']

                # --- Policy Loss (联合 ratio) ---
                old_log_prob = batch['old_log_probs_goal'] + batch['old_log_probs_offset']
                new_log_prob = new_lp_goal + new_lp_offset

                log_ratio = new_log_prob - old_log_prob
                ratio = log_ratio.exp()

                advantages = batch['advantages']

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio,
                                    1.0 - self.clip_epsilon,
                                    1.0 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clip fraction (诊断指标)
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

                # --- Value Loss ---
                value_loss = F.mse_loss(new_value, batch['returns'])

                # --- Entropy Bonus ---
                entropy = entropy_goal.mean() + entropy_offset.mean()

                # --- Total Loss ---
                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_clip_frac += clip_frac.item()
                n_updates += 1

        # 3. 清空 rollout buffer
        self.rollout.clear()

        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
            'clip_fraction': total_clip_frac / max(n_updates, 1),
        }


# ======================================================================
# Environment setup
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


# ======================================================================
# Main PPO training loop
# ======================================================================

def train(resume_path=None):
    """PPO 训练主入口。

    与 DQN 的关键区别:
      - 基于 step 的 on-policy 数据收集 (非 episode)
      - 每收集 rollout_steps 步后做 PPO 更新
      - 无 replay buffer / target network
      - 使用 GAE + clip surrogate objective
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    # Create environment
    print("Creating hierarchical CARLA environment...")
    params = make_env_params()
    env = gym.make('carla-v0', params=params)

    # Create policy (Actor-Critic)
    policy = HierarchicalPolicy(
        state_dim=cfg.STATE_DIM,
        seq_len=cfg.SEQ_LEN,
        hidden_dim=cfg.HIDDEN_DIM,
        num_goals=cfg.NUM_GOALS,
        log_std_init=cfg.OFFSET_LOG_STD_INIT,
    )

    # Create PPO trainer
    trainer = HierarchicalPPOTrainer(
        policy=policy,
        lr=cfg.LEARNING_RATE,
        clip_epsilon=cfg.PPO_CLIP_EPSILON,
        ppo_epochs=cfg.PPO_EPOCHS,
        num_mini_batches=cfg.NUM_MINI_BATCHES,
        value_coef=cfg.VALUE_COEF,
        entropy_coef=cfg.ENTROPY_COEF,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        device=device,
    )

    # Resume from checkpoint
    start_iteration = 0
    total_steps = 0
    episode_num = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        policy.load_state_dict(ckpt['policy_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_iteration = ckpt.get('iteration', 0) + 1
        total_steps = ckpt.get('total_steps', 0)
        if 'avg_reward' in ckpt:
            print(f"  Last avg_reward: {ckpt['avg_reward']:.2f}")
        print(f"  Resuming at iteration {start_iteration}, total_steps={total_steps}")

    # Training config
    rollout_steps = cfg.ROLLOUT_STEPS
    num_iterations = cfg.NUM_ITERATIONS
    max_steps = cfg.MAX_STEPS_PER_EPISODE
    offset_range = cfg.OFFSET_RANGE

    # Logging
    episode_rewards = deque(maxlen=100)
    best_avg_reward = -float('inf')
    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    # TensorBoard
    log_dir = os.path.join(os.path.dirname(__file__), 'runs',
                           time.strftime('PPO_%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir) if HAS_TENSORBOARD else None
    if writer:
        print(f"  TensorBoard log: {log_dir}")
        print(f"  启动命令: tensorboard --logdir={os.path.dirname(log_dir)}")

    # Print config
    params_count = policy.count_parameters()
    print(f"\n{'='*60}")
    print(f"PPO Training Config:")
    print(f"  Iterations:     {num_iterations}")
    print(f"  Rollout steps:  {rollout_steps}")
    print(f"  PPO epochs:     {cfg.PPO_EPOCHS}")
    print(f"  Mini-batches:   {cfg.NUM_MINI_BATCHES}")
    print(f"  Clip epsilon:   {cfg.PPO_CLIP_EPSILON}")
    print(f"  LR:             {cfg.LEARNING_RATE}")
    print(f"  Gamma:          {cfg.GAMMA}")
    print(f"  GAE lambda:     {cfg.GAE_LAMBDA}")
    print(f"  Value coef:     {cfg.VALUE_COEF}")
    print(f"  Entropy coef:   {cfg.ENTROPY_COEF}")
    print(f"  Offset range:   +/-{offset_range}m (continuous)")
    print(f"  Q1 Actor:       {params_count['q1_actor']:,} params")
    print(f"  Q2 Actor:       {params_count['q2_actor']:,} params")
    print(f"  Critic:         {params_count['critic']:,} params")
    print(f"  Total:          {params_count['total']:,} params")
    print(f"  Est. total steps: ~{num_iterations * rollout_steps:,}")
    print(f"{'='*60}\n")

    # ==========================================
    # Training loop (step-based, PPO on-policy)
    # ==========================================
    episode_reward = 0.0
    episode_steps = 0
    obs = env.reset()

    for iteration in range(start_iteration, num_iterations):
        iter_start = time.time()

        # --- Phase 1: Collect rollout ---
        policy.eval()
        for step in range(rollout_steps):
            state_tensor = torch.FloatTensor(obs).to(device)

            # 从策略分布中采样动作
            action_info = policy.select_action(state_tensor)
            goal = action_info['goal']
            p_off_raw = action_info['offset']

            # Clip offset 到合理范围
            p_off = float(np.clip(p_off_raw, -offset_range, offset_range))

            # 环境交互
            next_obs, reward, done, info = env.step([goal, p_off])

            # 存储到 rollout buffer (存储原始采样值以保持 log_prob 一致性)
            trainer.collect_step(
                state=obs,
                goal=goal,
                offset=p_off_raw,
                reward=reward,
                done=done,
                log_prob_goal=action_info['log_prob_goal'],
                log_prob_offset=action_info['log_prob_offset'],
                value=action_info['value'],
            )

            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            obs = next_obs

            if done or episode_steps >= max_steps:
                episode_num += 1
                episode_rewards.append(episode_reward)

                if writer:
                    writer.add_scalar('reward/episode', episode_reward, episode_num)
                    writer.add_scalar('train/steps_per_episode', episode_steps, episode_num)

                episode_reward = 0.0
                episode_steps = 0
                obs = env.reset()

        # --- Phase 2: Compute last value for GAE ---
        state_tensor = torch.FloatTensor(obs).to(device)
        last_value = policy.get_value(state_tensor)

        # --- Phase 3: PPO update ---
        policy.train()
        losses = trainer.update(last_value)

        iter_time = time.time() - iter_start
        avg_reward = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0

        # --- TensorBoard logging ---
        if writer:
            writer.add_scalar('reward/avg100', avg_reward, iteration)
            writer.add_scalar('loss/policy', losses['policy_loss'], iteration)
            writer.add_scalar('loss/value', losses['value_loss'], iteration)
            writer.add_scalar('loss/entropy', losses['entropy'], iteration)
            writer.add_scalar('train/clip_fraction', losses['clip_fraction'], iteration)
            writer.add_scalar('train/total_steps', total_steps, iteration)
            writer.add_scalar('train/episodes', episode_num, iteration)
            writer.add_scalar('train/q2_log_std', policy.q2.log_std.item(), iteration)
            writer.add_scalar('train/q2_std', policy.q2.log_std.exp().item(), iteration)

        # --- Console logging ---
        if iteration % 5 == 0 or iteration == num_iterations - 1:
            print(f"[Iter {iteration:4d}/{num_iterations}] "
                  f"ep={episode_num}, steps={total_steps}, "
                  f"avg_r={avg_reward:7.1f}, "
                  f"pi={losses['policy_loss']:.4f}, "
                  f"v={losses['value_loss']:.4f}, "
                  f"H={losses['entropy']:.3f}, "
                  f"clip={losses['clip_fraction']:.3f}, "
                  f"sigma={policy.q2.log_std.exp().item():.3f}, "
                  f"{iter_time:.1f}s")

        # --- Save best model ---
        if avg_reward > best_avg_reward and len(episode_rewards) >= 20:
            best_avg_reward = avg_reward
            save_path = os.path.join(save_dir, 'best_policy.pth')
            torch.save({
                'iteration': iteration,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'avg_reward': avg_reward,
                'total_steps': total_steps,
            }, save_path)
            print(f"  >> Saved best model (avg_reward={avg_reward:.1f})")

        # --- Periodic checkpoint ---
        if iteration % 50 == 0 and iteration > 0:
            ckpt_path = os.path.join(save_dir, f'policy_iter{iteration}.pth')
            torch.save({
                'iteration': iteration,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'avg_reward': avg_reward,
                'total_steps': total_steps,
            }, ckpt_path)
            print(f"  >> Checkpoint: {ckpt_path}")

    # Final save
    final_path = os.path.join(save_dir, 'final_policy.pth')
    torch.save({
        'iteration': num_iterations,
        'policy_state_dict': policy.state_dict(),
        'avg_reward': avg_reward,
        'total_steps': total_steps,
    }, final_path)
    print(f"\nTraining complete! Final model: {final_path}")
    print(f"  Total steps: {total_steps:,}, Episodes: {episode_num}")

    if writer:
        writer.close()
        print(f"TensorBoard logs: {log_dir}")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchical PPO Training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    train(resume_path=args.resume)
