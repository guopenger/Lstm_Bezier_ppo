#!/usr/bin/env python
"""
frenet_transform.py — Frenet ⇌ Cartesian 坐标转换

论文依据 (§2.2.2):
  轨迹规划在 Frenet 坐标系下进行，然后转回笛卡尔坐标系。
  Frenet 坐标 (s, d):
    s — 沿参考线的弧长 (纵向位置)
    d — 到参考线的法向距离 (横向偏移)，左正右负

  Frenet → Cartesian 转换公式:
    x = x_ref(s) - d * sin(θ_ref(s))
    y = y_ref(s) + d * cos(θ_ref(s))

  其中 θ_ref(s) 是参考线在弧长 s 处的航向角。

与现有代码的接口:
  - route_planner.py 的 waypoints 格式: [[x, y, yaw_deg], ...]
  - misc.py 的 get_lane_dis() 已实现横向距离计算
  - 本模块基于 waypoints 构建稠密参考线，提供完整的 Frenet 转换
"""

import numpy as np
from typing import Tuple, Optional, List


class FrenetTransform:
    """基于参考线 (道路中心线) 的 Frenet ⇌ Cartesian 坐标变换器。

    参考线由 route_planner 提供的 waypoints 插值而成。
    使用最近点投影法实现 Cartesian → Frenet 转换。

    Attributes:
        ref_x, ref_y, ref_yaw: 参考线上各点的笛卡尔坐标和航向角 (rad)。
        ref_s: 参考线上各点的累计弧长。
    """

    def __init__(self, ds: float = 0.5):
        """
        Args:
            ds: 参考线插值间距 (米)。越小越精确，但计算量越大。
                训练阶段建议 0.5m，部署阶段可减至 0.1m。
        """
        self.ds = ds
        self.ref_x: np.ndarray = np.array([])
        self.ref_y: np.ndarray = np.array([])
        self.ref_yaw: np.ndarray = np.array([])  # rad
        self.ref_s: np.ndarray = np.array([])
        self._is_built = False

    # ------------------------------------------------------------------
    # 1. 构建参考线
    # ------------------------------------------------------------------

    def build_reference_line(self, waypoints: List[List[float]]) -> None:
        """从 route_planner 的 waypoint 列表构建稠密参考线。

        Args:
            waypoints: [[x0, y0, yaw_deg0], [x1, y1, yaw_deg1], ...]
                       来自 route_planner._get_waypoints() 的输出。
                       yaw 单位为 **度** (与 CARLA 一致)。

        构建结果存储在 self.ref_x, ref_y, ref_yaw, ref_s 中。
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints to build reference line.")

        wp = np.array(waypoints, dtype=np.float64)  # (N, 3)
        raw_x = wp[:, 0]
        raw_y = wp[:, 1]
        raw_yaw = np.deg2rad(wp[:, 2])  # 转弧度

        # 计算原始点间的弧长
        dx = np.diff(raw_x)
        dy = np.diff(raw_y)
        raw_ds = np.sqrt(dx ** 2 + dy ** 2)
        raw_s = np.concatenate([[0.0], np.cumsum(raw_ds)])

        total_length = raw_s[-1]
        if total_length < 1e-3:
            raise ValueError("Reference line too short (< 1mm).")

        # 等间距插值
        n_points = max(int(total_length / self.ds), 2)
        s_interp = np.linspace(0.0, total_length, n_points)

        self.ref_x = np.interp(s_interp, raw_s, raw_x)
        self.ref_y = np.interp(s_interp, raw_s, raw_y)
        self.ref_yaw = self._interp_yaw(s_interp, raw_s, raw_yaw)
        self.ref_s = s_interp
        self._is_built = True

    # ------------------------------------------------------------------
    # 2. Cartesian → Frenet
    # ------------------------------------------------------------------

    def cartesian_to_frenet(self, x: float, y: float,
                            yaw: Optional[float] = None) -> Tuple[float, float]:
        """将笛卡尔坐标 (x, y) 转换为 Frenet 坐标 (s, d)。

        使用最近点投影法:
          1. 在参考线上找到距 (x, y) 最近的点索引
          2. s = ref_s[idx] (可选: 用线性插值细化)
          3. d = 叉积确定带符号的横向距离 (左正右负)

        Args:
            x, y: 目标点的笛卡尔坐标。
            yaw:  目标点的航向角 (rad)，暂未使用，预留接口。

        Returns:
            (s, d): Frenet 坐标。
                s — 沿参考线弧长 (m)。
                d — 横向偏移 (m)，左正右负。
        """
        self._check_built()

        # 找最近参考点
        dx = self.ref_x - x
        dy = self.ref_y - y
        dist_sq = dx ** 2 + dy ** 2
        idx = int(np.argmin(dist_sq))

        # --- 用前后点线性插值精化 s ---
        s, d = self._project_to_segment(x, y, idx)
        return s, d

    # ------------------------------------------------------------------
    # 3. Frenet → Cartesian  (论文 §2.2.2 公式)
    # ------------------------------------------------------------------

    def frenet_to_cartesian(self, s: float, d: float) -> Tuple[float, float]:
        """将 Frenet 坐标 (s, d) 转换为笛卡尔坐标 (x, y)。

        论文公式:
            x = x_ref(s) - d * sin(θ_ref(s))
            y = y_ref(s) + d * cos(θ_ref(s))

        Args:
            s: 纵向弧长 (m)。
            d: 横向偏移 (m)，左正右负。

        Returns:
            (x, y): 笛卡尔坐标。
        """
        self._check_built()

        # 在参考线上插值得到 (x_ref, y_ref, θ_ref) at s
        x_ref = np.interp(s, self.ref_s, self.ref_x)
        y_ref = np.interp(s, self.ref_s, self.ref_y)
        yaw_ref = self._interp_yaw_single(s)

        # 论文 §2.2.2 转换公式
        x = x_ref - d * np.sin(yaw_ref)
        y = y_ref + d * np.cos(yaw_ref)
        return float(x), float(y)

    def frenet_to_cartesian_array(self, s_arr: np.ndarray,
                                  d_arr: np.ndarray) -> np.ndarray:
        """批量 Frenet → Cartesian 转换。

        Args:
            s_arr: 弧长数组 (N,)。
            d_arr: 横向偏移数组 (N,)。

        Returns:
            np.ndarray: shape (N, 2)，每行为 [x, y]。
        """
        self._check_built()

        x_ref = np.interp(s_arr, self.ref_s, self.ref_x)
        y_ref = np.interp(s_arr, self.ref_s, self.ref_y)
        yaw_ref = self._interp_yaw_array(s_arr)

        x = x_ref - d_arr * np.sin(yaw_ref)
        y = y_ref + d_arr * np.cos(yaw_ref)
        return np.column_stack([x, y])

    # ------------------------------------------------------------------
    # 4. 辅助方法: 获取参考线在某 s 处的航向角
    # ------------------------------------------------------------------

    def get_ref_yaw_at(self, s: float) -> float:
        """获取参考线在弧长 s 处的航向角 (rad)。"""
        self._check_built()
        return float(self._interp_yaw_single(s))

    def get_ref_point_at(self, s: float) -> Tuple[float, float, float]:
        """获取参考线在弧长 s 处的 (x, y, yaw_rad)。"""
        self._check_built()
        x = float(np.interp(s, self.ref_s, self.ref_x))
        y = float(np.interp(s, self.ref_s, self.ref_y))
        yaw = float(self._interp_yaw_single(s))
        return x, y, yaw

    @property
    def total_length(self) -> float:
        """参考线总长度 (m)。"""
        if not self._is_built:
            return 0.0
        return float(self.ref_s[-1])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_built(self):
        if not self._is_built:
            raise RuntimeError(
                "Reference line not built. Call build_reference_line() first."
            )

    def _project_to_segment(self, x: float, y: float, idx: int) -> Tuple[float, float]:
        """将点 (x, y) 投影到 idx 附近的参考线段上，返回精确的 (s, d)。"""
        n = len(self.ref_s)

        # 确定搜索范围: idx 前后各一段
        best_s = self.ref_s[idx]
        best_d = self._signed_lateral_distance(x, y, idx)

        for seg_start in [max(0, idx - 1), idx]:
            seg_end = seg_start + 1
            if seg_end >= n:
                continue

            # 参考线段向量
            seg_dx = self.ref_x[seg_end] - self.ref_x[seg_start]
            seg_dy = self.ref_y[seg_end] - self.ref_y[seg_start]
            seg_len_sq = seg_dx ** 2 + seg_dy ** 2
            if seg_len_sq < 1e-12:
                continue

            # 点到线段起点的向量
            vx = x - self.ref_x[seg_start]
            vy = y - self.ref_y[seg_start]

            # 投影参数 t ∈ [0, 1]
            t = (vx * seg_dx + vy * seg_dy) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)

            # 投影点
            proj_x = self.ref_x[seg_start] + t * seg_dx
            proj_y = self.ref_y[seg_start] + t * seg_dy

            dist = np.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)
            if dist < abs(best_d) + 1e-6:
                # 弧长
                seg_len = np.sqrt(seg_len_sq)
                s_candidate = self.ref_s[seg_start] + t * seg_len

                # 带符号横向距离 (叉积)
                cross = seg_dx * vy - seg_dy * vx
                sign = 1.0 if cross >= 0 else -1.0
                d_candidate = sign * dist

                best_s = s_candidate
                best_d = d_candidate

        return best_s, best_d

    def _signed_lateral_distance(self, x: float, y: float, idx: int) -> float:
        """用叉积计算 (x,y) 到参考线 idx 处的带符号距离。"""
        n = len(self.ref_s)
        # 参考线方向
        if idx < n - 1:
            fw_x = self.ref_x[idx + 1] - self.ref_x[idx]
            fw_y = self.ref_y[idx + 1] - self.ref_y[idx]
        else:
            fw_x = self.ref_x[idx] - self.ref_x[idx - 1]
            fw_y = self.ref_y[idx] - self.ref_y[idx - 1]

        # 指向目标点的向量
        vx = x - self.ref_x[idx]
        vy = y - self.ref_y[idx]

        # 叉积 → 符号
        cross = fw_x * vy - fw_y * vx
        dist = np.sqrt(vx ** 2 + vy ** 2)
        sign = 1.0 if cross >= 0 else -1.0
        return sign * dist

    # --- 航向角插值 (处理角度环绕问题) ---

    @staticmethod
    def _interp_yaw(s_query: np.ndarray, s_ref: np.ndarray,
                    yaw_ref: np.ndarray) -> np.ndarray:
        """对航向角做 unwrap 后插值，避免 ±π 跳变。"""
        yaw_unwrap = np.unwrap(yaw_ref)
        yaw_interp = np.interp(s_query, s_ref, yaw_unwrap)
        # 归一化到 [-π, π]
        return np.arctan2(np.sin(yaw_interp), np.cos(yaw_interp))

    def _interp_yaw_single(self, s: float) -> float:
        """单点航向角插值。"""
        yaw_unwrap = np.unwrap(self.ref_yaw)
        val = np.interp(s, self.ref_s, yaw_unwrap)
        return float(np.arctan2(np.sin(val), np.cos(val)))

    def _interp_yaw_array(self, s_arr: np.ndarray) -> np.ndarray:
        """数组航向角插值。"""
        yaw_unwrap = np.unwrap(self.ref_yaw)
        val = np.interp(s_arr, self.ref_s, yaw_unwrap)
        return np.arctan2(np.sin(val), np.cos(val))


# ======================================================================
# 快速自测: 直线参考线 + 圆弧参考线
# ======================================================================
if __name__ == "__main__":
    # --- 测试 1: 直线参考线 (沿 x 轴) ---
    print("=" * 50)
    print("Test 1: Straight reference line along X-axis")
    print("=" * 50)

    waypoints_straight = [[i * 5.0, 0.0, 0.0] for i in range(20)]  # 100m 直线
    ft = FrenetTransform(ds=0.5)
    ft.build_reference_line(waypoints_straight)
    print(f"Reference line length: {ft.total_length:.1f} m")

    # 测试点: (25, 3) → 应为 s≈25, d≈3 (在参考线左侧)
    s, d = ft.cartesian_to_frenet(25.0, 3.0)
    print(f"(25, 3) → Frenet: s={s:.2f}, d={d:.2f}")  # 期望 s≈25, d≈3

    # 反向: (s=25, d=3) → 应回到 (25, 3)
    x, y = ft.frenet_to_cartesian(25.0, 3.0)
    print(f"Frenet(25, 3) → Cartesian: x={x:.2f}, y={y:.2f}")  # 期望 (25, 3)

    # --- 测试 2: 圆弧参考线 ---
    print("\n" + "=" * 50)
    print("Test 2: Curved reference line (quarter circle R=50)")
    print("=" * 50)

    R = 50.0
    angles = np.linspace(0, np.pi / 2, 50)
    waypoints_curve = [
        [R * np.sin(a), R * (1 - np.cos(a)), np.degrees(a)]
        for a in angles
    ]
    ft2 = FrenetTransform(ds=0.5)
    ft2.build_reference_line(waypoints_curve)
    print(f"Reference line length: {ft2.total_length:.1f} m (expect ~{R * np.pi / 2:.1f})")

    # 测试: 圆弧中点的外侧 5m
    s_mid = ft2.total_length / 2
    x_c, y_c = ft2.frenet_to_cartesian(s_mid, 5.0)
    s_back, d_back = ft2.cartesian_to_frenet(x_c, y_c)
    print(f"Frenet({s_mid:.1f}, 5.0) → Cart({x_c:.2f}, {y_c:.2f}) → Frenet({s_back:.2f}, {d_back:.2f})")
