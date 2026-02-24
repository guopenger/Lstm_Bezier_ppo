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
        use_smooth: bool = False,
        cf_dist: float = 50.0,
        ego_speed: float = 8.0,
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

        # Step 2: 确定终点 Frenet 坐标
        # Step 2: 根据 goal 决策动态调整规划距离
        if goal == 1:
            if cf_dist < self.plan_horizon:
                # 前方有障碍物，缩短规划距离到障碍物后方
                safe_margin = 2.0  # 跟车距离 2 米
                adjusted_horizon = max(cf_dist - safe_margin, 5.0)
                sf = s0 + adjusted_horizon
                
                print(f"[Bezier] 保持车道 + 前方障碍物 {cf_dist:.1f}m → 跟车模式，规划 {adjusted_horizon:.1f}m")
            else:
                # 前方无障碍物，正常规划
                sf = s0 + self.plan_horizon
        else:
            # 前方换道，正常规划
            sf = s0 + self.plan_horizon

        # 限制 sf 不超过参考线总长度
        sf = min(sf, self.frenet.total_length - 0.1)
        if sf <= s0:
            # 参考线太短，退化为直行
            sf = s0 + 5.0

        df = self._compute_target_d(d0, goal, offset, use_smooth)

        # Step 3: 生成 Frenet 系下的贝塞尔轨迹
        s_arr, d_arr = self._bezier_frenet(s0, d0, sf, df)

        # Step 4: Frenet → Cartesian
        trajectory = self.frenet.frenet_to_cartesian_array(s_arr, d_arr)

        return trajectory

    # ------------------------------------------------------------------
    # 轨迹终点 d 的计算
    # ------------------------------------------------------------------

    def _compute_target_d(self, d0: float, goal: int, offset: float, use_smooth: bool = False) -> float:
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
            if use_smooth:
                # 【平滑模式】开局使用：让曲线贴在车上
                d_goal = d0 * 0.7
                if abs(d0) < 0.3:
                    d_goal = 0.0
            else:
                # 【正常模式】训练/测试使用：直接回中心
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
