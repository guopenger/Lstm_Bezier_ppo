#!/usr/bin/env python
"""
test_hierarchical.py — 分层强化学习策略测试脚本 (带 Pygame 可视化)

使用方式:
  conda activate carla_rl
  # 先启动 CARLA 服务器 (CarlaUE4.exe -quality-level=Low)
  python test_hierarchical.py                                       # 默认加载 best_policy.pth
  python test_hierarchical.py --ckpt checkpoints/final_policy.pth   # 指定 checkpoint
  python test_hierarchical.py --ckpt checkpoints/policy_iter250.pth --episodes 5

键盘控制:
  ESC / Q  — 退出
  R        — 立即重置当前 episode
  P        — 暂停 / 继续
"""

import glob
import os
import sys
import time
import argparse
import numpy as np

import torch
import pygame

# ==========================================
# 挂载 CARLA Python API
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

import carla  # noqa: E402
import gym
import gym_carla
from gym_carla.models.hierarchical_policy import HierarchicalPolicy
from gym_carla import config as cfg


# ======================================================================
# Pygame HUD — 在鸟瞰图旁边显示实时信息
# ======================================================================

class InfoHUD:
    """在 Pygame 窗口右侧绘制 HUD 信息面板。"""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        # 使用微软雅黑以支持中文显示, 回退到 SimHei / Consolas
        _fn = 'microsoftyahei, simhei, consolas'
        self.font_large = pygame.font.SysFont(_fn, 20, bold=True)
        self.font = pygame.font.SysFont(_fn, 16)
        self.font_small = pygame.font.SysFont(_fn, 13)
        self._info_lines = []

    def update(self, info_dict):
        """更新 HUD 显示内容。"""
        self._info_lines = []
        for key, value in info_dict.items():
            self._info_lines.append((key, value))

    def render(self, display, x_offset):
        """绘制 HUD 到指定位置。"""
        # 背景面板
        panel = pygame.Surface((self.width, self.height))
        panel.set_alpha(200)
        panel.fill((20, 20, 30))
        display.blit(panel, (x_offset, 0))

        y = 10
        # 标题
        title_surf = self.font_large.render('Hierarchical RL Test', True, (0, 255, 200))
        display.blit(title_surf, (x_offset + 10, y))
        y += 30

        # 分隔线
        pygame.draw.line(display, (80, 80, 100),
                         (x_offset + 5, y), (x_offset + self.width - 5, y), 1)
        y += 8

        for key, value in self._info_lines:
            if key == '---':
                # 分隔线
                pygame.draw.line(display, (60, 60, 80),
                                 (x_offset + 5, y + 3),
                                 (x_offset + self.width - 5, y + 3), 1)
                y += 10
                continue

            # 特殊着色
            color = (220, 220, 220)
            if key == 'Goal':
                goal_colors = {
                    '← 左换道': (255, 200, 50),
                    '→ 保持车道': (100, 255, 100),
                    '→ 右换道': (50, 200, 255),
                }
                color = goal_colors.get(str(value), color)
            elif key == 'Reward':
                val = float(value) if isinstance(value, (int, float)) else 0
                color = (100, 255, 100) if val >= 0 else (255, 100, 100)
            elif key == 'Collision':
                color = (255, 50, 50) if value == 'YES !!!' else (100, 255, 100)

            key_surf = self.font.render(f'{key}:', True, (150, 150, 170))
            val_surf = self.font.render(f' {value}', True, color)
            display.blit(key_surf, (x_offset + 10, y))
            display.blit(val_surf, (x_offset + 10 + key_surf.get_width(), y))
            y += 22

        # 底部按键提示
        y = self.height - 60
        pygame.draw.line(display, (60, 60, 80),
                         (x_offset + 5, y), (x_offset + self.width - 5, y), 1)
        y += 8
        hints = [
            'ESC/Q: 退出   R: 重置',
            'P: 暂停/继续',
        ]
        for hint in hints:
            hint_surf = self.font_small.render(hint, True, (120, 120, 140))
            display.blit(hint_surf, (x_offset + 10, y))
            y += 18


# ======================================================================
# 鸟瞰图渲染 — 在测试时渲染环境的 birdeye view
# ======================================================================

def render_birdeye(env_unwrapped, display, display_size):
    """手动触发鸟瞰图渲染 (用于 hierarchical 模式)。

    在 hierarchical 模式下 _get_obs 跳过了渲染，
    此函数从环境内部状态手动执行渲染。
    """
    br = env_unwrapped.birdeye_render
    if br is None:
        return

    # 更新渲染器数据
    br.vehicle_polygons = env_unwrapped.vehicle_polygons
    br.walker_polygons = env_unwrapped.walker_polygons
    br.waypoints = env_unwrapped.waypoints

    # 渲染鸟瞰图
    render_types = ['roadmap', 'actors']
    if env_unwrapped.display_route:
        render_types.append('waypoints')
    br.render(display, render_types)

    # 绘制轨迹 (如果有)
    if env_unwrapped._last_trajectory is not None:
        _draw_trajectory(display, br, env_unwrapped._last_trajectory,
                         env_unwrapped.ego, display_size)


def _draw_trajectory(display, birdeye_render, trajectory, ego, display_size):
    """在鸟瞰图上绘制规划的贝塞尔轨迹。

    鸟瞰图经过 rotozoom(yaw+90°) 旋转, 使 ego 车头始终朝上。
    因此必须将世界坐标差值旋转到 ego 车体坐标系后再映射到屏幕:
      forward (前方) → 屏幕 -Y (向上)
      right   (右侧) → 屏幕 +X (向右)
    """
    if len(trajectory) < 2:
        return

    ego_trans = ego.get_transform()
    ego_x = ego_trans.location.x
    ego_y = ego_trans.location.y
    ego_yaw = ego_trans.rotation.yaw * np.pi / 180.0
    cos_yaw = np.cos(ego_yaw)
    sin_yaw = np.sin(ego_yaw)

    # 从 BirdeyeRender.params 获取渲染参数
    ppm = birdeye_render.params['pixels_per_meter']
    pav = birdeye_render.params['pixels_ahead_vehicle']

    # Ego 在屏幕上的位置: 水平居中, 垂直偏下 (pixels_ahead_vehicle)
    ego_sx = display_size / 2.0
    ego_sy = display_size / 2.0 + pav

    points = []
    for wp in trajectory[::3]:  # 每隔3个点画一个
        # 世界坐标差值
        dx = wp[0] - ego_x
        dy = wp[1] - ego_y
        # 旋转到 ego 车体坐标系
        fwd   =  dx * cos_yaw + dy * sin_yaw   # 前方距离
        right = -dx * sin_yaw + dy * cos_yaw   # 右侧距离
        # 映射到屏幕: forward→上(-Y), right→右(+X)
        px = int(ego_sx + right * ppm)
        py = int(ego_sy - fwd * ppm)
        if 0 <= px < display_size and 0 <= py < display_size:
            points.append((px, py))

    if len(points) >= 2:
        pygame.draw.lines(display, (255, 100, 50), False, points, 2)
    for pt in points:
        pygame.draw.circle(display, (255, 200, 0), pt, 3)


# ======================================================================
# 环境参数 (测试专用 — 启用 pygame 显示)
# ======================================================================

def make_test_env_params(display_size=256):
    """测试环境参数: 与训练相同，但启用 pygame 显示。"""
    return {
        # 关键区别: display_size > 0 以启用 pygame 渲染
        'display_size': display_size,
        'max_past_step': 1,
        'number_of_vehicles': cfg.NUMBER_OF_VEHICLES,
        'number_of_walkers': cfg.NUMBER_OF_WALKERS,
        'dt': cfg.CARLA_DT,
        'task_mode': 'random',
        'max_time_episode': cfg.MAX_STEPS_PER_EPISODE,
        'max_waypt': 12,
        'obs_range': 48,  # 增大到 48 看到更远
        'lidar_bin': 0.125,
        'd_behind': 8,  # 减小让更多视野给前方
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
# 主测试循环
# ======================================================================

def test(ckpt_path, num_episodes=3, display_size=256):
    """加载训练好的策略，在 CARLA 中执行测试并通过 Pygame 可视化。"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test device: {device}")

    # ------------------------------------------------------------------
    # 1. 加载策略网络
    # ------------------------------------------------------------------
    print(f"\n加载 checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"错误: 文件不存在 — {ckpt_path}")
        sys.exit(1)

    policy = HierarchicalPolicy(
        state_dim=cfg.STATE_DIM,
        seq_len=cfg.SEQ_LEN,
        hidden_dim=cfg.HIDDEN_DIM,
        num_goals=cfg.NUM_GOALS,
        log_std_init=cfg.OFFSET_LOG_STD_INIT,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.to(device)
    policy.eval()

    info_str = (f"  Iteration:  {ckpt.get('iteration', '?')}\n"
                f"  Avg reward: {ckpt.get('avg_reward', '?')}\n"
                f"  Total steps:{ckpt.get('total_steps', '?')}")
    print(info_str)

    # ------------------------------------------------------------------
    # 2. 创建环境 (display_size > 0 会初始化 pygame)
    # ------------------------------------------------------------------
    print("\n创建 CARLA 测试环境 (display_size=%d)..." % display_size)
    params = make_test_env_params(display_size=display_size)
    env = gym.make('carla-v0', params=params)
    env_unwrapped = env.unwrapped  # 获取底层 CarlaEnv

    # ------------------------------------------------------------------
    # 3. 设置 Pygame 窗口 (真实俯视 + 抽象鸟瞰 + HUD)
    # ------------------------------------------------------------------
    # 窗口布局: 真实俯视(256) + 抽象鸟瞰(256) + HUD(350)
    hud_width = 350
    cam_width = display_size
    birdeye_width = display_size
    screen_w = cam_width + birdeye_width + hud_width
    screen_h = display_size
    screen = pygame.display.set_mode(
        (screen_w, screen_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption(
        f'Hierarchical RL Test — {os.path.basename(ckpt_path)}')

    # 替换环境中的 display 引用 (让渲染画到我们的窗口)
    env_unwrapped.display = screen

    hud = InfoHUD(hud_width, screen_h)
    clock = pygame.time.Clock()

    # ------------------------------------------------------------------
    # 4. 测试循环
    # ------------------------------------------------------------------
    goal_names = ['← 左换道', '→ 保持车道', '→ 右换道']
    offset_range = cfg.OFFSET_RANGE
    max_steps = cfg.MAX_STEPS_PER_EPISODE

    all_rewards = []
    all_steps = []
    running = True
    paused = False

    print(f"\n{'='*50}")
    print(f"开始测试: {num_episodes} 个 episode")
    print(f"按 ESC/Q 退出, R 重置, P 暂停")
    print(f"{'='*50}\n")

    ep = 0
    while ep < num_episodes and running:
        obs = env.reset()
        episode_reward = 0.0
        step = 0
        done = False
        ep_start = time.time()
        force_reset = False

        print(f"--- Episode {ep + 1}/{num_episodes} 开始 ---")

        while not done and step < max_steps and running and not force_reset:
            # Pygame 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_r:
                        force_reset = True
                        print("  >> 手动重置")
                    elif event.key == pygame.K_p:
                        paused = not paused
                        print("  >> " + ("暂停" if paused else "继续"))

            if paused:
                # 暂停时只刷新显示
                hud.update({'状态': '已暂停 (按 P 继续)'})
                hud.render(screen, display_size)
                pygame.display.flip()
                clock.tick(10)
                continue

            # 策略推理 (确定性模式)
            state_tensor = torch.FloatTensor(obs).to(device)
            with torch.no_grad():
                action_info = policy.select_action(state_tensor, deterministic=True)

            goal = action_info['goal']
            p_off = float(np.clip(action_info['offset'], -offset_range, offset_range))

            # 环境交互
            obs, reward, done, info = env.step([goal, p_off])

            episode_reward += reward
            step += 1

            # 获取车辆状态
            v = env_unwrapped.ego.get_velocity()
            speed_ms = np.sqrt(v.x**2 + v.y**2)
            speed_kmh = speed_ms * 3.6
            collision = len(env_unwrapped.collision_hist) > 0

            # 跟踪误差
            trk_err = info.get('tracking_error', {})
            lat_err = trk_err.get('lateral_error', 0.0)
            head_err = trk_err.get('heading_error', 0.0)

            # 渲染所有视图
            screen.fill((20, 20, 30))
            
            # 1. 真实第三人称俯视图（左侧）
            if hasattr(env_unwrapped, 'overhead_img') and env_unwrapped.overhead_img is not None:
                from skimage.transform import resize as sk_resize
                oh_img = env_unwrapped.overhead_img
                oh_resized = (sk_resize(oh_img, (display_size, display_size), anti_aliasing=True) * 255).astype(np.uint8)
                oh_surface = pygame.Surface((display_size, display_size))
                pygame.surfarray.blit_array(oh_surface, np.transpose(oh_resized, (1, 0, 2)))
                screen.blit(oh_surface, (0, 0))
            else:
                font = pygame.font.SysFont('microsoftyahei', 16)
                text = font.render('第三人称视图未启用', True, (150, 150, 150))
                text_rect = text.get_rect(center=(display_size//2, display_size//2))
                screen.blit(text, text_rect)

            # 2. 抽象鸟瞰图（中间）
            # 创建临时 surface 用于鸟瞰图渲染
            birdeye_surface = pygame.Surface((display_size, display_size))
            env_unwrapped.display = birdeye_surface
            render_birdeye(env_unwrapped, birdeye_surface, display_size)
            screen.blit(birdeye_surface, (display_size, 0))
            env_unwrapped.display = screen  # 恢复原始 display

            # 获取额外分析信息
            from gym_carla.envs.misc import get_lane_dis, get_pos
            ego_wp_info = env_unwrapped.carla_map.get_waypoint(
                env_unwrapped.ego.get_transform().location)
            lane_id_str = f'{ego_wp_info.lane_id}' if ego_wp_info else '?'
            lane_w_str = f'{ego_wp_info.lane_width:.1f}m' if ego_wp_info else '?'
            
            # 到中心线距离
            ex, ey = get_pos(env_unwrapped.ego)
            ld, _ = get_lane_dis(env_unwrapped.waypoints, ex, ey)

            # 更新 HUD
            hud.update({
                'Episode': f'{ep + 1} / {num_episodes}',
                'Step': f'{step} / {max_steps}',
                '---1': '',
                'Goal': goal_names[goal],
                'Offset': f'{p_off:+.3f} m',
                '---2': '',
                'Speed': f'{speed_kmh:.1f} km/h',
                'Target': f'{cfg.DESIRED_SPEED * 3.6:.1f} km/h',
                '---3': '',
                'Reward': f'{reward:.2f}',
                'Cum Reward': f'{episode_reward:.1f}',
                '---4': '',
                'Lane ID': lane_id_str,
                'Lane Width': lane_w_str,
                'Lane Offset': f'{ld:+.2f} m',
                'Lat Error': f'{lat_err:.3f} m',
                'Head Error': f'{np.degrees(head_err):.1f}°',
                '---5': '',
                'Collision': 'YES !!!' if collision else 'No',
            })
            hud.render(screen, display_size * 2)

            pygame.display.flip()
            clock.tick(20)  # ~20 FPS

        if not running:
            break

        ep_time = time.time() - ep_start
        all_rewards.append(episode_reward)
        all_steps.append(step)

        term_reason = ('碰撞' if len(env_unwrapped.collision_hist) > 0
                       else '超时' if step >= max_steps
                       else '手动重置' if force_reset
                       else '出界')
        print(f"  Episode {ep + 1}: reward={episode_reward:.1f}, "
              f"steps={step}, time={ep_time:.1f}s, "
              f"终止原因={term_reason}")

        ep += 1

    # ------------------------------------------------------------------
    # 5. 汇总统计
    # ------------------------------------------------------------------
    env.close()
    pygame.quit()

    if all_rewards:
        print(f"\n{'='*50}")
        print(f"测试结果汇总 ({len(all_rewards)} episodes)")
        print(f"{'='*50}")
        print(f"  平均奖励:   {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
        print(f"  最大奖励:   {np.max(all_rewards):.1f}")
        print(f"  最小奖励:   {np.min(all_rewards):.1f}")
        print(f"  平均步数:   {np.mean(all_steps):.0f}")
        print(f"  Checkpoint: {ckpt_path}")
        for i, (r, s) in enumerate(zip(all_rewards, all_steps)):
            print(f"    Ep {i+1}: reward={r:.1f}, steps={s}")
    else:
        print("\n未完成任何 episode。")


# ======================================================================
# Entry point
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hierarchical RL 策略测试 (Pygame 可视化)')
    parser.add_argument(
        '--ckpt', type=str,
        default=os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_policy.pth'),
        help='Checkpoint 文件路径 (默认: checkpoints/best_policy.pth)')
    parser.add_argument(
        '--episodes', type=int, default=3,
        help='测试 episode 数量 (默认: 3)')
    parser.add_argument(
        '--display-size', type=int, default=450,
        help='Pygame 鸟瞰图尺寸 (默认: 450)')


    args = parser.parse_args()
    test(ckpt_path=args.ckpt,
         num_episodes=args.episodes,
         display_size=args.display_size)
