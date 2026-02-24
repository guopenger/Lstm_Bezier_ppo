#!/usr/bin/env python
"""
trajectory_tracker.py — 轨迹跟踪控制器

训练阶段使用 PID + Pure Pursuit 混合跟踪策略:
  - 纵向控制: PID 控制速度 → throttle / brake
  - 横向控制: Pure Pursuit 几何跟踪 → steer

部署阶段可替换为 MPC / DMPC (C++ / ROS2)。

接口约定:
  输入: 参考轨迹 np.ndarray (N, 2) [[x, y], ...] — 来自 bezier_fitting
  输出: carla.VehicleControl 兼容的 (throttle, steer, brake) 元组

与 carla_env.py 的对接:
  原 env.step() 直接用 acc/steer 控车，改造后改为:
    control = tracker.compute_control(trajectory, ego_state)
    act = carla.VehicleControl(throttle=control[0], steer=control[1], brake=control[2])
"""

import math
import numpy as np
from typing import Tuple, Optional, NamedTuple


class EgoState(NamedTuple):
    """自车状态，从 CARLA 获取。"""
    x: float        # 位置 x (m)
    y: float        # 位置 y (m)
    yaw: float      # 航向角 (rad)
    speed: float    # 当前速度 (m/s)


class TrajectoryTracker:
    """Pure Pursuit + PID 轨迹跟踪控制器。

    纵向 (速度) 控制: 增量式 PID
      - 目标速度由轨迹曲率自适应调整 (弯道减速)
      - 输出 throttle ∈ [0, 1] 和 brake ∈ [0, 1]

    横向 (转向) 控制: Pure Pursuit 几何跟踪
      - 在轨迹上寻找前视点 (lookahead point)
      - 由几何关系计算转向角 δ = arctan(2L·sinα / L_d)
      - 输出 steer ∈ [-1, 1]

    Attributes:
        wheelbase:       轴距 (m)，Lincoln MKZ ≈ 2.85m。
        desired_speed:   期望巡航速度 (m/s)。
        lookahead_dist:  前视距离 (m)。
        max_steer:       最大转向角归一化值 (1.0 对应 CARLA 最大转角)。
    """

    def __init__(
        self,
        wheelbase: float = 2.85,
        desired_speed: float = 8.0,
        lookahead_dist: float = 5.0,
        min_lookahead: float = 2.0,
        max_lookahead: float = 15.0,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.05,
        max_steer: float = 1.0,
        max_throttle: float = 0.8,
        max_brake: float = 0.8,
        dt: float = 0.1,
        curvature_speed_factor: float = 3.0,
        min_speed: float = 2.0,
    ):
        """
        Args:
            wheelbase:              车辆轴距 (m)。
            desired_speed:          期望巡航速度 (m/s)。
            lookahead_dist:         基础前视距离 (m)，会随速度自适应。
            min_lookahead:          最小前视距离 (m)。
            max_lookahead:          最大前视距离 (m)。
            kp, ki, kd:             纵向 PID 增益。
            max_steer:              转向输出饱和值。
            max_throttle:           油门输出饱和值。
            max_brake:              制动输出饱和值。
            dt:                     控制周期 (s)，与 CARLA 同步步长一致。
            curvature_speed_factor: 曲率→速度衰减系数，越大弯道减速越多。
            min_speed:              曲率限速的最低速度 (m/s)。
        """
        # 车辆参数
        self.wheelbase = wheelbase
        self.desired_speed = desired_speed
        self.max_steer = max_steer
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.dt = dt

        # Pure Pursuit 参数
        self.lookahead_dist = lookahead_dist
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead

        # PID 参数
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._error_integral = 0.0
        self._prev_error = 0.0

        # 曲率自适应限速
        self.curvature_speed_factor = curvature_speed_factor
        self.min_speed = min_speed

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def compute_control(
        self,
        trajectory: np.ndarray,
        ego_state: EgoState,
        cf_dist: float = 50.0,        # 新增参数：前方障碍物距离
        cf_rel_speed: float = 0.0,    # 新增参数：前方车辆相对速度
    ) -> Tuple[float, float, float]:
        """计算跟踪参考轨迹所需的控制指令。

        Args:
            trajectory: 参考轨迹 (N, 2) [[x, y], ...]，来自 BezierFitting。
            ego_state:  自车当前状态 EgoState(x, y, yaw, speed)。

        Returns:
            (throttle, steer, brake):
                throttle ∈ [0, max_throttle]
                steer    ∈ [-max_steer, max_steer]  (左正右负，与 CARLA 一致需取反)
                brake    ∈ [0, max_brake]
        """
        if len(trajectory) < 2:
            return 0.0, 0.0, 1.0  # 轨迹太短，紧急制动

        # --- 横向控制: Pure Pursuit ---
        steer = self._pure_pursuit(trajectory, ego_state)

        # --- 纵向控制: PID + 曲率限速 ---
        target_speed = self._adaptive_target_speed(trajectory, ego_state, cf_dist, cf_rel_speed)
        throttle, brake = self._pid_speed_control(ego_state.speed, target_speed)

        return throttle, steer, brake

    def reset(self) -> None:
        """重置 PID 积分器 (每个 episode 开始时调用)。"""
        self._error_integral = 0.0
        self._prev_error = 0.0

    # ------------------------------------------------------------------
    # 横向控制: Pure Pursuit
    # ------------------------------------------------------------------

    def _pure_pursuit(self, trajectory: np.ndarray, ego: EgoState) -> float:
        """Pure Pursuit 横向跟踪算法。

        步骤:
          1. 速度自适应前视距离: L_d = base + k * v
          2. 在轨迹上找到距离自车最近的前视点
          3. 计算转向角: δ = arctan(2L·sinα / L_d)

        Returns:
            steer ∈ [-max_steer, max_steer]
        """
        # 速度自适应前视距离
        ld = np.clip(
            self.lookahead_dist + 0.5 * ego.speed,
            self.min_lookahead,
            self.max_lookahead,
        )

        # 找前视点: 轨迹上第一个距离 ≥ ld 的点
        lookahead_pt = self._find_lookahead_point(trajectory, ego, ld)

        if lookahead_pt is None:
            # 没找到前视点，用轨迹末端点
            lookahead_pt = trajectory[-1]

        # 计算前视点在自车坐标系下的位置
        dx = lookahead_pt[0] - ego.x
        dy = lookahead_pt[1] - ego.y

        # 转到自车坐标系 (前方为 x_local, 右方为 y_local)
        # CARLA 左手系: +Y = 右方, 正 local_y 表示目标在右侧
        local_x = dx * math.cos(ego.yaw) + dy * math.sin(ego.yaw)
        local_y = -dx * math.sin(ego.yaw) + dy * math.cos(ego.yaw)

        # 实际前视距离
        ld_actual = math.sqrt(local_x ** 2 + local_y ** 2)
        if ld_actual < 0.1:
            return 0.0

        # Pure Pursuit 公式: δ = arctan(2 * L * sin(α) / L_d)
        # 其中 sin(α) = local_y / L_d_actual
        steer = math.atan2(2.0 * self.wheelbase * local_y, ld_actual ** 2)

        # 归一化到 [-max_steer, max_steer]
        # CARLA 的 steer 范围 [-1, 1] 对应最大物理转角 ~70°
        max_physical_angle = math.radians(70.0)
        steer_normalized = steer / max_physical_angle
        steer_normalized = np.clip(steer_normalized, -self.max_steer, self.max_steer)

        return float(steer_normalized)

    def _find_lookahead_point(
        self, trajectory: np.ndarray, ego: EgoState, ld: float
    ) -> Optional[np.ndarray]:
        """在轨迹上找到前视点 (第一个距自车 ≥ ld 的点)。

        额外条件: 前视点必须在自车前方 (排除身后的点)。
        """
        for pt in trajectory:
            dx = pt[0] - ego.x
            dy = pt[1] - ego.y
            dist = math.sqrt(dx ** 2 + dy ** 2)

            if dist >= ld:
                # 检查是否在前方 (与航向方向的点积 > 0)
                forward_dot = dx * math.cos(ego.yaw) + dy * math.sin(ego.yaw)
                if forward_dot > 0:
                    return pt

        return None

    # ------------------------------------------------------------------
    # 纵向控制: PID + 曲率自适应限速
    # ------------------------------------------------------------------

    def _pid_speed_control(
        self, current_speed: float, target_speed: float
    ) -> Tuple[float, float]:
        """增量式 PID 速度控制器。

        Args:
            current_speed: 当前速度 (m/s)。
            target_speed:  目标速度 (m/s)。

        Returns:
            (throttle, brake): 分别 ∈ [0, max]。
        """
        error = target_speed - current_speed

        # PID
        self._error_integral += error * self.dt
        # 积分器抗饱和 (Anti-windup)
        self._error_integral = np.clip(self._error_integral, -10.0, 10.0)
        derivative = (error - self._prev_error) / max(self.dt, 1e-6)
        self._prev_error = error

        output = self.kp * error + self.ki * self._error_integral + self.kd * derivative

        if output >= 0:
            throttle = float(np.clip(output, 0.0, self.max_throttle))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-output, 0.0, self.max_brake))

        return throttle, brake

    def _adaptive_target_speed(
        self, trajectory: np.ndarray, ego: EgoState,
        cf_dist: float = 50.0,
        cf_rel_speed: float = 0.0,
    ) -> float:
        """根据轨迹曲率自适应调整目标速度 (弯道减速)。

        v_target = max(min_speed, desired_speed - factor * κ_max)

        Args:
            trajectory: 参考轨迹 (N, 2)。
            ego:        自车状态。

        Returns:
            target_speed (m/s)。
        """
        # 跟车模式逻辑
        if cf_dist < 15.0:
            # 前方有车，计算前车速度
            front_vehicle_speed = ego.speed + cf_rel_speed
            
            if front_vehicle_speed < 1.0:
                # 前车停止，ego 也要停止
                return 0.0
            elif cf_dist < 10.0:
                # 距离很近（<10m），严格跟随前车速度
                target_speed = front_vehicle_speed
                # 距离太近时额外减速
                if cf_dist < 5.0:
                    penalty = (5.0 - cf_dist) / 5.0 * 2.0  # 5m→0m, 减速 0→2 m/s
                    target_speed = max(0.0, target_speed - penalty)
                return target_speed
            else:
                # 距离适中（10-15m），允许稍快但不超过期望速度
                target_speed = min(front_vehicle_speed + 1.0, self.desired_speed)
                return target_speed

        if len(trajectory) < 3:
            return self.desired_speed

        # 计算轨迹前方一段的平均曲率
        curvature = self._compute_curvature(trajectory)
        if len(curvature) == 0:
            return self.desired_speed

        # 取前方 10 个点的最大曲率
        kappa_max = float(np.max(curvature[:min(10, len(curvature))]))

        # 曲率越大速度越低
        target = self.desired_speed - self.curvature_speed_factor * kappa_max
        target = max(target, self.min_speed)

        return target

    @staticmethod
    def _compute_curvature(trajectory: np.ndarray) -> np.ndarray:
        """计算轨迹的离散曲率序列。

        κ = |x'y'' - y'x''| / (x'² + y'²)^{3/2}

        Args:
            trajectory: (N, 2) 轨迹点。

        Returns:
            (N,) 曲率数组。
        """
        if len(trajectory) < 3:
            return np.array([0.0])

        dx = np.gradient(trajectory[:, 0])
        dy = np.gradient(trajectory[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        denom = (dx ** 2 + dy ** 2) ** 1.5
        denom = np.maximum(denom, 1e-6)
        curvature = np.abs(dx * ddy - dy * ddx) / denom
        return curvature

    # ------------------------------------------------------------------
    # 辅助: 计算当前跟踪误差 (用于奖励函数)
    # ------------------------------------------------------------------

    def compute_tracking_error(
        self, trajectory: np.ndarray, ego: EgoState
    ) -> dict:
        """计算当前跟踪误差，供奖励函数使用。

        Returns:
            dict:
              - 'lateral_error':  横向偏差 (m)，到轨迹最近点的距离。
              - 'heading_error':  航向偏差 (rad)，与轨迹切线方向的夹角。
              - 'nearest_idx':    最近轨迹点的索引。
        """
        if len(trajectory) < 2:
            return {'lateral_error': 0.0, 'heading_error': 0.0, 'nearest_idx': 0}

        # 找最近点
        dx = trajectory[:, 0] - ego.x
        dy = trajectory[:, 1] - ego.y
        dist = np.sqrt(dx ** 2 + dy ** 2)
        nearest_idx = int(np.argmin(dist))
        lateral_error = float(dist[nearest_idx])

        # 航向偏差: 轨迹切线方向 vs 自车航向
        if nearest_idx < len(trajectory) - 1:
            traj_dx = trajectory[nearest_idx + 1, 0] - trajectory[nearest_idx, 0]
            traj_dy = trajectory[nearest_idx + 1, 1] - trajectory[nearest_idx, 1]
        else:
            traj_dx = trajectory[nearest_idx, 0] - trajectory[nearest_idx - 1, 0]
            traj_dy = trajectory[nearest_idx, 1] - trajectory[nearest_idx - 1, 1]

        traj_yaw = math.atan2(traj_dy, traj_dx)
        heading_error = self._normalize_angle(ego.yaw - traj_yaw)

        return {
            'lateral_error': lateral_error,
            'heading_error': abs(heading_error),
            'nearest_idx': nearest_idx,
        }

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """将角度归一化到 [-π, π]。"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

# ======================================================================
# 快速自测
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Test: TrajectoryTracker (Pure Pursuit + PID)")
    print("=" * 60)

    tracker = TrajectoryTracker(
        wheelbase=2.85,
        desired_speed=8.0,
        lookahead_dist=5.0,
        dt=0.1,
    )

    # --- 测试 1: 直线轨迹 ---
    print("\n--- Test 1: Straight trajectory ---")
    traj_straight = np.column_stack([
        np.linspace(0, 50, 50),
        np.zeros(50),
    ])

    ego = EgoState(x=0.0, y=0.0, yaw=0.0, speed=5.0)
    throttle, steer, brake = tracker.compute_control(traj_straight, ego)
    print(f"  Ego at origin, facing +x, speed=5 m/s")
    print(f"  → throttle={throttle:.3f}, steer={steer:.4f}, brake={brake:.3f}")
    print(f"  Expected: throttle>0 (加速到8), steer≈0 (直行), brake=0")

    # --- 测试 2: 左换道轨迹 ---
    print("\n--- Test 2: Left lane change trajectory ---")
    t = np.linspace(0, 1, 50)
    traj_lc = np.column_stack([
        t * 30.0,                               # x: 0→30m
        3.5 * (10 * t**3 - 15 * t**4 + 6 * t**5),  # y: 0→3.5m (五次多项式平滑)
    ])

    ego2 = EgoState(x=0.0, y=0.0, yaw=0.0, speed=8.0)
    throttle2, steer2, brake2 = tracker.compute_control(traj_lc, ego2)
    print(f"  Ego at origin, speed=8 m/s, lane change to y=3.5m")
    print(f"  → throttle={throttle2:.3f}, steer={steer2:.4f}, brake={brake2:.3f}")
    print(f"  Expected: steer>0 (左转), throttle/brake视曲率")

    # --- 测试 3: 跟踪误差计算 ---
    print("\n--- Test 3: Tracking error ---")
    ego3 = EgoState(x=5.0, y=1.0, yaw=0.1, speed=6.0)
    err = tracker.compute_tracking_error(traj_straight, ego3)
    print(f"  Ego at (5, 1), yaw=0.1 rad, tracking straight line y=0")
    print(f"  → lateral_error={err['lateral_error']:.3f} m (expect≈1.0)")
    print(f"  → heading_error={err['heading_error']:.3f} rad (expect≈0.1)")

    # --- 测试 4: 多步模拟 ---
    print("\n--- Test 4: Multi-step simulation (simple kinematics) ---")
    tracker.reset()
    ego_sim = EgoState(x=0.0, y=0.5, yaw=0.05, speed=5.0)

    for step in range(5):
        ctrl = tracker.compute_control(traj_straight, ego_sim)
        err_sim = tracker.compute_tracking_error(traj_straight, ego_sim)

        if step < 3:
            print(f"  Step {step}: pos=({ego_sim.x:.1f}, {ego_sim.y:.2f}), "
                  f"speed={ego_sim.speed:.1f}, lat_err={err_sim['lateral_error']:.3f}, "
                  f"ctrl=({ctrl[0]:.2f}, {ctrl[1]:.3f}, {ctrl[2]:.2f})")

        # 简单运动学更新 (仅用于测试，实际由 CARLA 物理引擎驱动)
        ds = ego_sim.speed * 0.1
        new_yaw = ego_sim.yaw + ego_sim.speed * math.tan(ctrl[1] * math.radians(70)) / 2.85 * 0.1
        new_x = ego_sim.x + ds * math.cos(new_yaw)
        new_y = ego_sim.y + ds * math.sin(new_yaw)
        new_speed = ego_sim.speed + (ctrl[0] * 3.0 - ctrl[2] * 5.0) * 0.1
        new_speed = max(0.0, new_speed)
        ego_sim = EgoState(new_x, new_y, new_yaw, new_speed)

    print(f"  Final: pos=({ego_sim.x:.1f}, {ego_sim.y:.2f}), speed={ego_sim.speed:.1f}")
    print(f"  Expected: y→0 (converging to trajectory), speed→8")
