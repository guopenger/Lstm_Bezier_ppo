#!/usr/bin/env python
"""
bezier_fitting.py — 在 Frenet 坐标系下进行 5 阶贝塞尔曲线轨迹拟合

论文依据 (§2.2.2):
  1. 在 Frenet (s, d) 坐标系下构造贝塞尔曲线控制点
  2. d(s) 用 5 阶贝塞尔多项式参数化，保证:
     - 起点切线沿 s 轴 (平滑接入当前行驶方向)
     - 终点切线沿 s 轴 (平滑汇入目标车道)
  3. 采样后通过 FrenetTransform 转回笛卡尔坐标

与上层 RL 的接口:
  - Q1 输出 Goal ∈ {0: 左换道, 1: 保持, 2: 右换道} — 离散 (Eq.1)
  - Q2 输出 p_off ∈ ℝ — 连续偏移值 (meters) (Eq.2)
  - 本模块根据 (Goal, p_off) 确定 Frenet 终点 d_f，生成轨迹
"""

import numpy as np
import math
from typing import List, Tuple, Optional

from gym_carla.planning.frenet_transform import FrenetTransform


class BezierFitting:
    """5 阶贝塞尔曲线轨迹生成器 (Frenet 坐标系)。

    根据论文 §2.2.2，轨迹在 Frenet 系下表示为 d = B(t)，其中:
      - t ∈ [0, 1] 为贝塞尔参数
      - s 从 s0 到 sf 线性映射
      - d 由 5 阶贝塞尔曲线决定横向偏移

    Attributes:
        frenet: FrenetTransform 实例，用于坐标转换。
        lane_width: 车道宽度 (m)，用于计算换道目标 d。
        plan_horizon: 规划距离 (m)，即 sf - s0。
        n_samples: 轨迹采样点数。
    """

    def __init__(
        self,
        frenet: FrenetTransform,      # Frenet坐标转换器（已构建好参考线）
        lane_width: float = 3.5,      # 车道宽度（CARLA默认3.5米）
        plan_horizon: float = 30.0,   # 规划距离（向前看30米）
        n_samples: int = 50,          # 轨迹采样点数（50个点）
    ):
        """
        Args:
            frenet:            FrenetTransform 实例 (参考线已构建)。
            lane_width:        标准车道宽度 (m)，CARLA 默认约 3.5m。
            plan_horizon:      纵向规划距离 (m)，即 Δs = sf - s0。
            n_samples:         沿轨迹的采样点数。
        """
        self.frenet = frenet
        self.lane_width = lane_width
        self.plan_horizon = plan_horizon
        self.n_samples = n_samples

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def generate_trajectory(
        self,
        ego_x: float,
        ego_y: float,
        ego_yaw: float,
        goal: int,
        offset: int,
        cf_dist: float = 50.0,
        ego_speed: float = 8.0,
        world=None,
        ego_vehicle=None,
    ) -> np.ndarray:
        """根据 RL 的 (Goal, Offset) 输出生成参考轨迹。

        流程:
          1. 将自车位置转换到 Frenet 坐标 (s0, d0)
          2. 根据 Goal 和 Offset 确定终点 (sf, df)
          3. 在 Frenet 系下构造 5 阶贝塞尔控制点
          4. 采样 → Frenet → Cartesian

        Args:
            ego_x, ego_y: 自车当前笛卡尔坐标 (m)。
            ego_yaw:      自车当前航向角 (rad)。
            goal:         Q1 输出 (int)。0=左换道, 1=保持, 2=右换道。
            offset:       Q2 输出 (float)。连续偏移值 p_off (meters)。

        Returns:
            np.ndarray: 形状 (n_samples, 2) 的轨迹点 [[x, y], ...]。
                        笛卡尔坐标，供 trajectory_tracker 跟踪。
        """
        # Step 1: 自车 → Frenet
        s0, d0 = self.frenet.cartesian_to_frenet(ego_x, ego_y, ego_yaw)

        # Step 2: 确定终点 Frenet 坐标（始终使用完整规划距离）
        sf = s0 + self.plan_horizon
        sf = min(sf, self.frenet.total_length - 0.1)
        
        if sf <= s0:
            # 参考线太短，退化为直行
            sf = s0 + 5.0

        df = self._compute_target_d(d0, goal, offset)

        # Step 3: 生成 Frenet 系下的贝塞尔轨迹
        s_arr, d_arr = self._bezier_frenet(s0, d0, sf, df)

        # Step 4: Frenet → Cartesian
        trajectory = self.frenet.frenet_to_cartesian_array(s_arr, d_arr)

        # Step 5: 智能碰撞检测截断
        if world is not None and ego_vehicle is not None:
            trajectory = self._smart_collision_truncation(
                trajectory=trajectory,
                world=world,
                ego_vehicle=ego_vehicle,
                ego_yaw=ego_yaw,
                ego_speed=ego_speed,
                min_points=2,  # 至少保留20个点，保证轨迹可用
            )
            
        return trajectory


    # ------------------------------------------------------------------
    # 检测障碍物截断贝塞尔曲线
    # ------------------------------------------------------------------
    def _smart_collision_truncation(
        self,
        trajectory: np.ndarray,
        world,
        ego_vehicle,
        ego_yaw: float,
        ego_speed: float,
        min_points: int = 2,
    ) -> np.ndarray:
        """
        智能碰撞检测截断：只在确定会碰撞时才截断轨迹
        
        设计原则：
        1. 使用自车坐标系统一计算（避免转弯时坐标系混乱）
        2. 使用 OBB 碰撞预测（考虑车辆尺寸）
        3. 动态预测（考虑障碍物速度）
        4. 保守策略（只截断真正会碰撞的情况）
        
        Args:
            trajectory: (N, 2) 轨迹点
            world: CARLA world
            ego_vehicle: CARLA ego vehicle
            ego_yaw: 自车航向角 (rad)
            ego_speed: 自车速度 (m/s)
            min_points: 最少保留的轨迹点数
            
        Returns:
            截断后的轨迹 (M, 2)，M >= min_points
        """
        if len(trajectory) < min_points:
            return trajectory
            
        # 获取自车信息
        ego_transform = ego_vehicle.get_transform()
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
        ego_bb = ego_vehicle.bounding_box
        ego_half_length = ego_bb.extent.x
        ego_half_width = ego_bb.extent.y
        
        # 自车坐标系旋转矩阵
        cos_yaw = math.cos(ego_yaw)
        sin_yaw = math.sin(ego_yaw)
        
        # 获取所有车辆
        vehicle_list = world.get_actors().filter('vehicle.*')
        
        # 计算每个轨迹点到达的预计时间
        # 假设沿轨迹匀速行驶
        cumulative_dist = np.zeros(len(trajectory))
        for i in range(1, len(trajectory)):
            dx = trajectory[i, 0] - trajectory[i-1, 0]
            dy = trajectory[i, 1] - trajectory[i-1, 1]
            cumulative_dist[i] = cumulative_dist[i-1] + math.sqrt(dx*dx + dy*dy)
        
        # 预计到达时间 (秒)
        arrival_time = cumulative_dist / max(ego_speed, 1.0)
        
        cutoff_index = len(trajectory)  # 默认不截断
        
        for vehicle in vehicle_list:
            if vehicle.id == ego_vehicle.id:
                continue
                
            v_transform = vehicle.get_transform()
            v_loc = v_transform.location
            v_bb = vehicle.bounding_box
            v_velocity = vehicle.get_velocity()
            v_yaw = math.radians(v_transform.rotation.yaw)
            
            # 障碍物速度分量
            v_vx = v_velocity.x
            v_vy = v_velocity.y
            v_speed = math.sqrt(v_vx**2 + v_vy**2)
            
            # 障碍物尺寸
            v_half_length = v_bb.extent.x
            v_half_width = v_bb.extent.y
            
            # 碰撞检测半径（两车的对角线之和的一半）
            collision_radius = math.sqrt(
                (ego_half_length + v_half_length)**2 + 
                (ego_half_width + v_half_width)**2
            ) * 0.3  # 0.7 是安全系数，稍微收紧一点
            
            # 逐点检查轨迹是否与障碍物碰撞
            for i in range(min_points, len(trajectory)):
                traj_x, traj_y = trajectory[i]
                t = arrival_time[i]
                
                # 预测障碍物在时刻 t 的位置
                pred_vx = v_loc.x + v_vx * t
                pred_vy = v_loc.y + v_vy * t
                
                # 计算轨迹点与预测障碍物位置的距离
                dist = math.sqrt((traj_x - pred_vx)**2 + (traj_y - pred_vy)**2)
                
                # 在自车坐标系下检查：障碍物是否在前方
                # 转换到自车坐标系
                dx = pred_vx - ego_x
                dy = pred_vy - ego_y
                local_x = dx * cos_yaw + dy * sin_yaw  # 前方为正
                local_y = -dx * sin_yaw + dy * cos_yaw  # 左方为正
                
                # 只考虑前方的障碍物（local_x > 0）
                if local_x < 0:
                    continue
                
                # 检查是否会碰撞
                if dist < collision_radius:
                    # 额外检查：障碍物是否真的在轨迹路径上
                    # 计算轨迹点到障碍物的横向距离
                    traj_dx = pred_vx - traj_x
                    traj_dy = pred_vy - traj_y
                    
                    # 轨迹方向（使用相邻点）
                    if i < len(trajectory) - 1:
                        fwd_x = trajectory[i+1, 0] - trajectory[i, 0]
                        fwd_y = trajectory[i+1, 1] - trajectory[i, 1]
                    else:
                        fwd_x = trajectory[i, 0] - trajectory[i-1, 0]
                        fwd_y = trajectory[i, 1] - trajectory[i-1, 1]
                    
                    fwd_len = math.sqrt(fwd_x**2 + fwd_y**2)
                    if fwd_len > 0.01:
                        fwd_x /= fwd_len
                        fwd_y /= fwd_len
                        
                        # 横向距离
                        lateral_dist = abs(traj_dx * (-fwd_y) + traj_dy * fwd_x)
                        
                        # 只有横向距离小于碰撞半径才认为是真正的碰撞
                        if lateral_dist < collision_radius:
                            # 检查是否是同向车辆（同向车不截断，交给跟车逻辑）
                            v_forward_x = math.cos(v_yaw)
                            v_forward_y = math.sin(v_yaw)
                            direction_dot = fwd_x * v_forward_x + fwd_y * v_forward_y
                            
                            if direction_dot > 0.7:
                                # 同向车辆，且速度差不大，不截断（交给跟车逻辑）
                                if v_speed > 0.5:
                                    continue
                            
                            # 找到碰撞点，更新截断位置
                            if i < cutoff_index:
                                cutoff_index = i
                                break  # 找到最近的碰撞点，检查下一个车辆
        
        # 截断轨迹（保留安全距离）
        if cutoff_index < len(trajectory):
            safe_cutoff = max(min_points, cutoff_index - 3)
            return trajectory[:safe_cutoff]
        
        return trajectory


    # ------------------------------------------------------------------
    # 轨迹终点 d 的计算
    # ------------------------------------------------------------------
    def _compute_target_d(self, d0: float, goal: int, offset: float) -> float:
        """根据 Q1(Goal) 和 Q2(p_off) 计算 Frenet 系终点横向偏移 df。

        CARLA 坐标系约定:
        - 车辆朝向 = forward vector (fw)
        - 左侧 = fw 逆时针旋转 90°
        - Frenet d: d < 0 = 左侧, d > 0 = 右侧

        目标 d 的计算 (论文 §2.1.3):
          Goal = 0 (左换道): d_goal = d0 - lane_width   (向左偏一个车道)
          Goal = 1 (保持):   d_goal = 0                 (回到车道中心)
          Goal = 2 (右换道): d_goal = d0 + lane_width   (向右偏一个车道)

        连续偏移 (论文 Eq.2):
          d_goal += p_off  (连续实数，正值向左，负值向右)

        Args:
            d0:     当前横向偏移 (m)。
            goal:   Q1 输出 {0, 1, 2}。
            offset: Q2 输出 p_off (float, meters)。连续偏移值。

        Returns:
            df: 目标横向偏移 (m)。
        """
        # 行为层 Goal → 大方向
        if goal == 0:  # 左换道
            d_goal = d0 - self.lane_width
        elif goal == 2:  # 右换道
            d_goal = d0 + self.lane_width
        else:  # goal == 1, 保持车道
            d_goal = 0.0

        # 轨迹层: 连续偏移 p_off (论文 Eq.2)
        # offset > 0 向右, offset < 0 向左，必须全链路统一！
        d_goal += float(offset)

        return d_goal

    # ------------------------------------------------------------------
    # 5 阶贝塞尔曲线 (Frenet 系)
    # ------------------------------------------------------------------

    def _bezier_frenet(
        self, s0: float, d0: float, sf: float, df: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """在 Frenet 坐标系下用 5 阶贝塞尔曲线生成 d(s) 轨迹。

        5 阶贝塞尔曲线有 6 个控制点 P0..P5。
        论文 §2.2.2 的关键约束:
          - 起点: d(0) = d0，d'(0) = 0 (切线沿 s 轴)
          - 终点: d(1) = df，d'(1) = 0 (切线沿 s 轴)
          - d''(0) = 0, d''(1) = 0 (曲率连续，保证平滑)

        由这些约束推导出控制点:
          P0_d = d0
          P1_d = d0            (保证 d'(0) = 0)
          P2_d = d0            (保证 d''(0) = 0)
          P3_d = df            (保证 d''(1) = 0)
          P4_d = df            (保证 d'(1) = 0)
          P5_d = df            (终点)

        s 方向线性映射: s(t) = s0 + t * (sf - s0)

        Args:
            s0, d0: 起点 Frenet 坐标。
            sf, df: 终点 Frenet 坐标。

        Returns:
            (s_arr, d_arr): 各 n_samples 个点的纵向弧长和横向偏移。
        """
        # 贝塞尔参数 t ∈ [0, 1] t = [0.00, 0.02, 0.04, ..., 0.98, 1.00] 50个点
        t = np.linspace(0.0, 1.0, self.n_samples)

        # 6 个控制点的 d 值 (根据边界条件推导)
        cp_d = np.array([d0, d0, d0, df, df, df])

        # 5 阶贝塞尔基函数 B_i^5(t)
        d_arr = self._bezier_curve(t, cp_d)

        # s 方向线性映射
        s_arr = s0 + t * (sf - s0)

        return s_arr, d_arr

    @staticmethod
    def _bezier_curve(t: np.ndarray, control_points: np.ndarray) -> np.ndarray:
        """计算 n 阶贝塞尔曲线值。

        B(t) = Σ_{i=0}^{n} C(n,i) * (1-t)^{n-i} * t^i * P_i

        Args:
            t:              参数数组 (M,)，值在 [0, 1]。
            control_points: 控制点数组 (n+1,)。

        Returns:
            曲线值数组 (M,)。
        """
        n = len(control_points) - 1
        result = np.zeros_like(t)

        for i in range(n + 1):
            # 二项式系数 C(n, i)
            binom = _binomial_coeff(n, i)
            # Bernstein 基多项式
            basis = binom * (t ** i) * ((1.0 - t) ** (n - i))
            result += basis * control_points[i]

        return result

    # ------------------------------------------------------------------
    # 调试 / 可视化辅助
    # ------------------------------------------------------------------

    def generate_trajectory_frenet(
        self, ego_x: float, ego_y: float, ego_yaw: float,
        goal: int, offset: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """同 generate_trajectory，但返回 Frenet 坐标 (用于调试/绘图)。

        Returns:
            (s_arr, d_arr): 轨迹在 Frenet 系下的坐标。
        """
        s0, d0 = self.frenet.cartesian_to_frenet(ego_x, ego_y, ego_yaw)
        sf = min(s0 + self.plan_horizon, self.frenet.total_length - 0.1)
        if sf <= s0:
            sf = s0 + 5.0
        df = self._compute_target_d(d0, goal, offset)
        return self._bezier_frenet(s0, d0, sf, df)

    def get_trajectory_curvature(self, trajectory: np.ndarray) -> np.ndarray:
        """计算笛卡尔轨迹的曲率序列 (用于奖励函数中的平滑性评估)。

        κ = |x'y'' - y'x''| / (x'^2 + y'^2)^{3/2}

        Args:
            trajectory: (N, 2) 轨迹点。

        Returns:
            (N-2,) 曲率数组。
        """
        dx = np.gradient(trajectory[:, 0])
        dy = np.gradient(trajectory[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        denom = (dx ** 2 + dy ** 2) ** 1.5
        denom = np.maximum(denom, 1e-6)  # 避免除零
        curvature = np.abs(dx * ddy - dy * ddx) / denom
        return curvature


# ======================================================================
# 工具函数
# ======================================================================

def _binomial_coeff(n: int, k: int) -> int:
    """计算二项式系数 C(n, k)。"""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    # 利用对称性
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


# ======================================================================
# 快速自测
# ======================================================================
if __name__ == "__main__":
    # 构建直线参考线 (沿 x 轴, 200m)
    waypoints = [[i * 5.0, 0.0, 0.0] for i in range(41)]  # 200m
    ft = FrenetTransform(ds=0.5)
    ft.build_reference_line(waypoints)

    planner = BezierFitting(
        frenet=ft,
        lane_width=3.5,
        plan_horizon=30.0,
        n_samples=50,
    )

    # 自车在 (10, 0)，朝 x 轴正方向行驶
    ego_x, ego_y, ego_yaw = 10.0, 0.0, 0.0

    print("=" * 60)
    print("Test: Bezier trajectory generation (continuous offset)")
    print("=" * 60)

    for goal_name, goal in [("左换道", 0), ("保持", 1), ("右换道", 2)]:
        traj = planner.generate_trajectory(ego_x, ego_y, ego_yaw, goal=goal, offset=0.0)
        s_arr, d_arr = planner.generate_trajectory_frenet(ego_x, ego_y, ego_yaw, goal=goal, offset=0.0)
        curv = planner.get_trajectory_curvature(traj)

        print(f"\n--- Goal: {goal_name} (offset=不偏) ---")
        print(f"  起点 Cartesian: ({traj[0, 0]:.2f}, {traj[0, 1]:.2f})")
        print(f"  终点 Cartesian: ({traj[-1, 0]:.2f}, {traj[-1, 1]:.2f})")
        print(f"  起点 Frenet:    s={s_arr[0]:.2f}, d={d_arr[0]:.2f}")
        print(f"  终点 Frenet:    s={s_arr[-1]:.2f}, d={d_arr[-1]:.2f}")
        print(f"  最大曲率:       {curv.max():.6f}")
        print(f"  轨迹点数:       {len(traj)}")

    # 测试 连续 offset 效果
    print("\n" + "=" * 60)
    print("Test: Continuous offset effect on lane-keeping")
    print("=" * 60)
    for off_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        traj = planner.generate_trajectory(ego_x, ego_y, ego_yaw, goal=1, offset=off_val)
        print(f"  保持+offset={off_val:+.1f}m: 终点 y={traj[-1, 1]:.3f} m")
