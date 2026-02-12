#!/usr/bin/env python
"""
zone_detector.py — 8 区域栅格感知模块

论文依据 (§2.1.2, Figure 2):
  将自车周围划分为 8 个矩形区域，每个区域提取最近车辆的:
    - Δv: 相对纵向速度 (m/s)，正值=对方更快，负值=自车更快
    - Δd: 相对距离 (m)

  区域布局 (自车局部坐标系, x轴=前方, y轴=左方):

              ┌───────┬───────┬───────┐
              │ Zone5 │ Zone6 │ Zone7 │   ← 前方 (x > side_th)
              │  LF   │  CF   │  RF   │
              ├───────┼───────┼───────┤
              │ Zone4 │ [EGO] │ Zone8 │   ← 侧方 (-side_th ≤ x ≤ side_th)
              │  LS   │       │  RS   │
              ├───────┼───────┼───────┤
              │ Zone3 │ Zone2 │ Zone1 │   ← 后方 (x < -side_th)
              │  LR   │  CR   │  RR   │
              └───────┴───────┴───────┘

  左右划分: y > lane_width/2 → 左; |y| ≤ lane_width/2 → 中; y < -lane_width/2 → 右

  输出状态向量 (18维):
    [0]     v_ego          自车速度 (m/s)
    [1]     lane_id        当前车道编号 (CARLA OpenDRIVE lane_id)
    [2:10]  Δv_1 ~ Δv_8   8 区域相对速度 (m/s)
    [10:18] Δd_1 ~ Δd_8   8 区域相对距离 (m)

与 misc.py 的坐标系一致性:
  本模块使用与 misc.get_local_pose() 相同的局部坐标变换:
    local_x =  dx * cos(ego_yaw) + dy * sin(ego_yaw)   → 前方为正
    local_y = -dx * sin(ego_yaw) + dy * cos(ego_yaw)   → 左方为正
"""

import math
import numpy as np

# 尝试导入 carla (运行时由调用脚本添加到 sys.path)
try:
    import carla
except ImportError:
    pass  # 允许在没有 CARLA 的环境中导入本模块 (用于单元测试)


class ZoneDetector:
    """8 区域周围车辆感知器。

    无状态设计：所有信息每次调用 detect() 时实时获取。

    Attributes:
        NUM_ZONES:        区域数量 (固定为 8)。
        lane_width:       车道宽度 (m)，用于左/中/右划分。
        front_dist:       前方最大感知距离 (m)。
        rear_dist:        后方最大感知距离 (m)。
        side_threshold:   侧方区域在纵向上的半宽 (m)。
        default_rel_speed: 区域内无车时的默认相对速度。
        default_rel_dist:  区域内无车时的默认距离。
    """

    NUM_ZONES = 8

    # Zone ID 常量 (与论文 Figure 2 对应)
    ZONE_RR = 1   # 右后
    ZONE_CR = 2   # 中后
    ZONE_LR = 3   # 左后
    ZONE_LS = 4   # 左侧
    ZONE_LF = 5   # 左前
    ZONE_CF = 6   # 中前
    ZONE_RF = 7   # 右前
    ZONE_RS = 8   # 右侧

    ZONE_NAMES = {
        1: 'RR (右后)', 2: 'CR (中后)', 3: 'LR (左后)', 4: 'LS (左侧)',
        5: 'LF (左前)', 6: 'CF (中前)', 7: 'RF (右前)', 8: 'RS (右侧)',
    }

    def __init__(
        self,
        lane_width: float = 3.5,
        front_dist: float = 50.0,
        rear_dist: float = 30.0,
        side_threshold: float = 5.0,
        default_rel_speed: float = 0.0,
        default_rel_dist: float = 100.0,
    ):
        """
        Args:
            lane_width:       车道宽度 (m)。
            front_dist:       前方感知距离上限 (m)。
            rear_dist:        后方感知距离上限 (m)。
            side_threshold:   侧方区域的纵向半宽 (m)。超过此距离算前/后。
            default_rel_speed: 区域内无车时默认 Δv (m/s)。
            default_rel_dist:  区域内无车时默认 Δd (m)。
        """
        self.lane_width = lane_width
        self.front_dist = front_dist
        self.rear_dist = rear_dist
        self.side_threshold = side_threshold
        self.default_rel_speed = default_rel_speed
        self.default_rel_dist = default_rel_dist
        self.half_lane = lane_width / 2.0

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def detect(self, world, ego_vehicle, carla_map) -> np.ndarray:
        """扫描周围车辆，返回 18 维状态向量。

        Args:
            world:       carla.World 实例。
            ego_vehicle: carla.Vehicle 自车。
            carla_map:   carla.Map 地图 (用于获取 lane_id)。

        Returns:
            np.ndarray: shape (18,), dtype float32。
              [v_ego, lane_id, Δv_1..Δv_8, Δd_1..Δd_8]
        """
        # --- 自车信息 ---
        ego_transform = ego_vehicle.get_transform()
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
        ego_yaw = math.radians(ego_transform.rotation.yaw)

        ego_vel = ego_vehicle.get_velocity()
        ego_speed = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2)  # m/s

        cos_yaw = math.cos(ego_yaw)
        sin_yaw = math.sin(ego_yaw)

        # 车道 ID
        ego_wp = carla_map.get_waypoint(ego_transform.location)
        lane_id = float(ego_wp.lane_id)

        # --- 初始化 8 个区域 ---
        zones = {}
        for z in range(1, self.NUM_ZONES + 1):
            zones[z] = {
                'dist': self.default_rel_dist,
                'rel_v': self.default_rel_speed,
            }

        # --- 扫描所有 NPC 车辆 ---
        vehicle_list = world.get_actors().filter('vehicle.*')
        for vehicle in vehicle_list:
            if vehicle.id == ego_vehicle.id:
                continue

            # 全局位置
            v_loc = vehicle.get_transform().location
            dx = v_loc.x - ego_x
            dy = v_loc.y - ego_y

            # 变换到 ego 局部坐标系 (与 misc.get_local_pose 一致)
            local_x = dx * cos_yaw + dy * sin_yaw      # 前方为正
            local_y = -dx * sin_yaw + dy * cos_yaw      # 左方为正

            # 粗略距离裁剪 (超出感知范围的直接跳过)
            if local_x > self.front_dist or local_x < -self.rear_dist:
                continue
            if abs(local_y) > self.lane_width * 2.5:  # 超过 2.5 个车道宽度
                continue

            # 欧氏距离
            dist = math.sqrt(dx ** 2 + dy ** 2)

            # 相对纵向速度: 对方沿 ego 前方向的速度分量 - 自车速度
            v_vel = vehicle.get_velocity()
            v_forward_speed = v_vel.x * cos_yaw + v_vel.y * sin_yaw
            rel_speed = v_forward_speed - ego_speed

            # 分类到区域
            zone_id = self._classify_zone(local_x, local_y)
            if zone_id is None:
                continue

            # 保留每个区域中最近的车辆
            if dist < zones[zone_id]['dist']:
                zones[zone_id]['dist'] = dist
                zones[zone_id]['rel_v'] = rel_speed

        # --- 组装 18 维状态向量 ---
        state = np.zeros(18, dtype=np.float32)
        state[0] = ego_speed
        state[1] = lane_id
        for z in range(1, self.NUM_ZONES + 1):
            state[1 + z] = zones[z]['rel_v']       # indices [2..9]  = Δv_1..Δv_8
            state[9 + z] = zones[z]['dist']         # indices [10..17] = Δd_1..Δd_8

        return state

    # ------------------------------------------------------------------
    # 区域分类
    # ------------------------------------------------------------------

    def _classify_zone(self, local_x: float, local_y: float):
        """根据 ego 局部坐标将车辆分类到 Zone 1-8。

        Args:
            local_x: 纵向位置 (前方为正)。
            local_y: 横向位置 (左方为正)。

        Returns:
            int (1-8) 或 None (不在任何区域)。
        """
        hlw = self.half_lane
        st = self.side_threshold

        # 纵向分类
        is_front = local_x > st
        is_rear = local_x < -st
        is_side = not is_front and not is_rear

        # 横向分类
        is_left = local_y > hlw
        is_right = local_y < -hlw
        is_center = not is_left and not is_right

        # 映射到 Zone ID
        if is_rear and is_right:
            return self.ZONE_RR       # Zone 1
        elif is_rear and is_center:
            return self.ZONE_CR       # Zone 2
        elif is_rear and is_left:
            return self.ZONE_LR       # Zone 3
        elif is_side and is_left:
            return self.ZONE_LS       # Zone 4
        elif is_front and is_left:
            return self.ZONE_LF       # Zone 5
        elif is_front and is_center:
            return self.ZONE_CF       # Zone 6
        elif is_front and is_right:
            return self.ZONE_RF       # Zone 7
        elif is_side and is_right:
            return self.ZONE_RS       # Zone 8
        else:
            return None  # 不应发生

    # ------------------------------------------------------------------
    # 调试辅助
    # ------------------------------------------------------------------

    def format_state(self, state: np.ndarray) -> str:
        """将 18 维状态向量格式化为可读字符串 (用于调试)。

        Args:
            state: 18 维状态向量。

        Returns:
            格式化字符串。
        """
        lines = [
            f"v_ego={state[0]:.2f} m/s, lane_id={state[1]:.0f}",
            "Zone | Δv (m/s) | Δd (m)",
            "-" * 35,
        ]
        for z in range(1, self.NUM_ZONES + 1):
            dv = state[1 + z]
            dd = state[9 + z]
            name = self.ZONE_NAMES[z]
            marker = " *" if dd < self.default_rel_dist else ""
            lines.append(f"  {z} {name:12s}  {dv:+7.2f}   {dd:7.1f}{marker}")
        return "\n".join(lines)


# ======================================================================
# 快速自测 (无需 CARLA，用 mock 数据)
# ======================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("ZoneDetector: Unit test with mock data")
    print("=" * 50)

    detector = ZoneDetector(lane_width=3.5)

    # 测试区域分类
    test_cases = [
        # (local_x, local_y, expected_zone)
        (20.0,  0.0,    6),   # 正前方 → CF
        (-15.0, 0.0,    2),   # 正后方 → CR
        (20.0,  3.0,    5),   # 左前 → LF
        (20.0, -3.0,    7),   # 右前 → RF
        (-15.0, 3.0,    3),   # 左后 → LR
        (-15.0, -3.0,   1),   # 右后 → RR
        (2.0,   3.0,    4),   # 左侧 → LS
        (2.0,  -3.0,    8),   # 右侧 → RS
        (0.0,   0.0,    None), # 自车位置附近 (中心区无区域)
    ]

    print("\n--- Zone classification tests ---")
    all_passed = True
    for lx, ly, expected in test_cases:
        result = detector._classify_zone(lx, ly)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        name = detector.ZONE_NAMES.get(result, "None")
        print(f"  {status} ({lx:+6.1f}, {ly:+6.1f}) → Zone {result} ({name}), expected {expected}")

    print(f"\n{'All tests passed!' if all_passed else 'SOME TESTS FAILED!'}")

    # 测试 format_state
    print("\n--- State vector formatting ---")
    fake_state = np.zeros(18, dtype=np.float32)
    fake_state[0] = 8.5          # v_ego
    fake_state[1] = -1           # lane_id
    fake_state[2 + 5] = -2.3    # Zone 6 (CF) rel speed: 对方比自车慢 2.3 m/s
    fake_state[10 + 5] = 25.0   # Zone 6 (CF) distance: 25m
    fake_state[2 + 4] = 1.5     # Zone 5 (LF) rel speed 
    fake_state[10 + 4] = 35.0   # Zone 5 (LF) distance
    print(detector.format_state(fake_state))
