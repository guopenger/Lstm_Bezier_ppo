#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import numpy as np
import pygame
import random
import time
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla

from gym_carla.envs.render import BirdeyeRender
from gym_carla.envs.route_planner import RoutePlanner
from gym_carla.envs.misc import *

# Hierarchical RL modules
from gym_carla.envs.zone_detector import ZoneDetector
from gym_carla.planning.state_buffer import StateBuffer
from gym_carla.planning.frenet_transform import FrenetTransform
from gym_carla.planning.bezier_fitting import BezierFitting
from gym_carla.control.trajectory_tracker import TrajectoryTracker, EgoState
from gym_carla import config as cfg
from gym_carla.reward import get_hierarchical_reward


class CarlaEnv(gym.Env):
  """An OpenAI gym wrapper for CARLA simulator."""

  def __init__(self, params):
    # parameters
    self.display_size = params['display_size']  # rendering screen size
    self.max_past_step = params['max_past_step']
    self.number_of_vehicles = params['number_of_vehicles']
    self.number_of_walkers = params['number_of_walkers']
    self.dt = params['dt']
    self.task_mode = params['task_mode']
    self.max_time_episode = params['max_time_episode']
    self.max_waypt = params['max_waypt']
    self.obs_range = params['obs_range']
    self.lidar_bin = params['lidar_bin']
    self.d_behind = params['d_behind']
    self.obs_size = int(self.obs_range/self.lidar_bin)
    self.out_lane_thres = params['out_lane_thres']
    self.desired_speed = params['desired_speed']
    self.max_ego_spawn_times = params['max_ego_spawn_times']
    self.display_route = params['display_route']
    self._prev_lane_id = None
    self._lane_change_detected = False   # 是否检测到了一次有效换道
    self._stable_counter = 0             # 在新车道上稳定行驶的步数
    self._cached_zone_state = None       # 缓存 zone_detector 结果，避免重复调用 
    if 'pixor' in params.keys():
      self.pixor = params['pixor']
      self.pixor_size = params['pixor_size']
    else:
      self.pixor = False

    # Hierarchical RL mode
    self.hierarchical = params.get('hierarchical', False)
    if self.hierarchical:
      self.state_dim = params.get('state_dim', cfg.STATE_DIM)
      self.seq_len = params.get('seq_len', cfg.SEQ_LEN)
      self.lane_width = params.get('lane_width', cfg.LANE_WIDTH)
      self.zone_detector = ZoneDetector(
        lane_width=self.lane_width,
        front_dist=params.get('zone_front_dist', cfg.ZONE_FRONT_DIST),
        rear_dist=params.get('zone_rear_dist', cfg.ZONE_REAR_DIST),
      )
      self.state_buffer = StateBuffer(
        state_dim=self.state_dim,
        seq_len=self.seq_len,
      )
      # Frenet + Bezier + Tracker
      self.frenet = FrenetTransform(ds=0.5)
      self.bezier = None  # built in reset() after reference line is ready
      self.tracker = TrajectoryTracker(
        wheelbase=cfg.WHEELBASE,
        desired_speed=params.get('desired_speed', cfg.DESIRED_SPEED),
        lookahead_dist=cfg.LOOKAHEAD_DIST,
        dt=params.get('dt', cfg.CONTROL_DT),
      )
      self._last_trajectory = None  # cache for reward computation
      self._last_tracking_error = None
      self.last_lane_speed = 0.0 

    # Destination
    if params['task_mode'] == 'roundabout':
      self.dests = [[4.46, -61.46, 0], [-49.53, -2.89, 0], [-6.48, 55.47, 0], [35.96, 3.33, 0]]
    else:
      self.dests = None

    # action and observation spaces
    self.discrete = params['discrete']
    self.discrete_act = [params['discrete_acc'], params['discrete_steer']] # acc, steer
    self.n_acc = len(self.discrete_act[0])
    self.n_steer = len(self.discrete_act[1])
    if self.discrete:
      self.action_space = spaces.Discrete(self.n_acc*self.n_steer)
    else:
      self.action_space = spaces.Box(np.array([params['continuous_accel_range'][0], 
      params['continuous_steer_range'][0]]), np.array([params['continuous_accel_range'][1],
      params['continuous_steer_range'][1]]), dtype=np.float32)  # acc, steer
    observation_space_dict = {
      'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
      'state': spaces.Box(np.array([-2, -1, -5, 0]), np.array([2, 1, 30, 1]), dtype=np.float32)
      }
    if self.pixor:
      observation_space_dict.update({
        'roadmap': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
        'vh_clas': spaces.Box(low=0, high=1, shape=(self.pixor_size, self.pixor_size, 1), dtype=np.float32),
        'vh_regr': spaces.Box(low=-5, high=5, shape=(self.pixor_size, self.pixor_size, 6), dtype=np.float32),
        'pixor_state': spaces.Box(np.array([-1000, -1000, -1, -1, -5]), np.array([1000, 1000, 1, 1, 20]), dtype=np.float32)
        })
    self.observation_space = spaces.Dict(observation_space_dict)

    # Override observation space for hierarchical RL
    if self.hierarchical:
      self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(self.seq_len, self.state_dim),
        dtype=np.float32)
      # Hybrid action: [goal ∈ {0,1,2} (discrete), p_off ∈ [-R,R] (continuous)]
      # 论文 Eq.1: A_g = [g_l, g_r, g_s], Eq.2: A_p = [p_off]
      self.action_space = spaces.Box(
        low=np.array([0, -cfg.OFFSET_RANGE], dtype=np.float32),
        high=np.array([2, cfg.OFFSET_RANGE], dtype=np.float32),
        dtype=np.float32)

    # Connect to carla server and get world object
    print('connecting to Carla server...')
    client = carla.Client('localhost', params['port'])
    client.set_timeout(60.0)          # 首次加载地图可能很慢，给足 60 秒
    self.client = client              # 保存引用

    # 检查当前地图是否已经是目标地图，避免 load_world 触发 Shader 重编译崩溃
    current_world = client.get_world()
    current_map_name = current_world.get_map().name.split('/')[-1]
    target_town = params['town']
    if current_map_name == target_town:
      print(f'当前地图已经是 {target_town}，跳过 load_world')
      self.world = current_world
    else:
      print(f'当前地图 {current_map_name} != {target_town}，正在切换...')
      self.world = client.load_world(target_town)

    self.carla_map = self.world.get_map()
    print('Carla server connected!')

    # Set weather
    self.world.set_weather(carla.WeatherParameters.ClearNoon)

    # Get spawn points
    self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
    self.walker_spawn_points = []
    for i in range(self.number_of_walkers):
      spawn_point = carla.Transform()
      loc = self.world.get_random_location_from_navigation()
      if (loc != None):
        spawn_point.location = loc
        self.walker_spawn_points.append(spawn_point)

    # Create the ego vehicle blueprint
    self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')

    # Collision sensor
    self.collision_hist = [] # The collision history
    self.collision_hist_l = 1 # collision history length
    self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

    # Lidar sensor
    self.lidar_data = None
    self.lidar_height = 2.1
    self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
    self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
    self.lidar_bp.set_attribute('channels', '32')
    self.lidar_bp.set_attribute('range', '5000')
    self.lidar_bp.set_attribute('points_per_second', '100000')
    self.lidar_bp.set_attribute('rotation_frequency', str(1.0 / self.dt))  # Match simulation tick rate

    # Camera sensor
    self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
    self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
    self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
    self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
    self.camera_bp.set_attribute('fov', '110')
    # Set the time in seconds between sensor captures
    self.camera_bp.set_attribute('sensor_tick', '0.02')

    # Set fixed simulation step for synchronous mode
    self.settings = self.world.get_settings()
    self.settings.fixed_delta_seconds = self.dt

    # Record the time of total steps and resetting steps
    self.reset_step = 0
    self.total_step = 0

    # Pre-initialize sensor references (reset() checks these before first use)
    self.collision_sensor = None
    self.lidar_sensor = None
    self.camera_sensor = None
    
    # BEV 功能已完全移除
    self.bev_producer = None
    self.bev_img = None
    
    # Initialize the renderer only if display_size > 0
    if self.display_size > 0:
      self._init_renderer()
    else:
      self.display = None
      self.birdeye_render = None

    # Get pixel grid points
    if self.pixor:
      x, y = np.meshgrid(np.arange(self.pixor_size), np.arange(self.pixor_size)) # make a canvas with coordinates
      x, y = x.flatten(), y.flatten()
      self.pixel_grid = np.vstack((x, y)).T

  def reset(self):
    # Clear sensor objects carefully to avoid WARNINGs
    if self.collision_sensor is not None and self.collision_sensor.is_alive:
      self.collision_sensor.destroy()
    self.collision_sensor = None

    self._prev_lane_id = None
    self._lane_change_detected = False
    self._stable_counter = 0
    self._cached_zone_state = None 

    if self.lidar_sensor is not None and self.lidar_sensor.is_alive:
      self.lidar_sensor.destroy()
    self.lidar_sensor = None

    if self.camera_sensor is not None and self.camera_sensor.is_alive:
      self.camera_sensor.destroy()
    self.camera_sensor = None

    # 清理高空俯视相机（测试可视化用）
    if hasattr(self, 'overhead_sensor') and self.overhead_sensor is not None:
      if self.overhead_sensor.is_alive:
        self.overhead_sensor.destroy()
      self.overhead_sensor = None

    # Delete sensors, vehicles and walkers
    self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker', 'walker.*'])

    # Disable sync mode
    self._set_synchronous_mode(False)

    # Spawn surrounding vehicles
    random.shuffle(self.vehicle_spawn_points)
    count = self.number_of_vehicles
    if count > 0:
      for spawn_point in self.vehicle_spawn_points:
        if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
        count -= 1

    # Spawn pedestrians
    random.shuffle(self.walker_spawn_points)
    count = self.number_of_walkers
    if count > 0:
      for spawn_point in self.walker_spawn_points:
        if self._try_spawn_random_walker_at(spawn_point):
          count -= 1
        if count <= 0:
          break
    while count > 0:
      if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
        count -= 1

    # Get actors polygon list
    self.vehicle_polygons = []
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    self.walker_polygons = []
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)

    # Spawn the ego vehicle
    ego_spawn_times = 0
    while True:
      if ego_spawn_times > self.max_ego_spawn_times:
        self.reset()

      if self.task_mode == 'random':
        transform = random.choice(self.vehicle_spawn_points)
      if self.task_mode == 'roundabout':
        self.start=[52.1+np.random.uniform(-5,5),-4.2, 178.66] # random
        # self.start=[52.1,-4.2, 178.66] # static
        transform = set_carla_transform(self.start)
      if self._try_spawn_ego_vehicle_at(transform):
        break
      else:
        ego_spawn_times += 1
        time.sleep(0.1)

    # Add collision sensor
    self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
    self.collision_hist = []
    def get_collision_hist(event):
      impulse = event.normal_impulse
      intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
      self.collision_hist.append(intensity)
      if len(self.collision_hist)>self.collision_hist_l:
        self.collision_hist.pop(0)
    self.collision_sensor.listen(get_collision_hist)

    # Add lidar sensor (skip in hierarchical mode)
    if not self.hierarchical:
      self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
      def get_lidar_data(data):
        self.lidar_data = data
      self.lidar_sensor.listen(get_lidar_data)

      # Add camera sensor
      self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
      def get_camera_img(data):
        array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
        array = np.reshape(array, (data.height, data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.camera_img = array
      self.camera_sensor.listen(get_camera_img)

    # 高空俯视相机（仅用于测试可视化，hierarchical 模式 + display_size > 0）
    if self.hierarchical and self.display_size > 0:
      self.overhead_img = np.zeros((256, 256, 3), dtype=np.uint8)
      overhead_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
      overhead_bp.set_attribute('image_size_x', '256')
      overhead_bp.set_attribute('image_size_y', '256')
      overhead_bp.set_attribute('fov', '90')
      # 后上方俯视: 后退 8m，高度 25m，俯角 70°
      overhead_transform = carla.Transform(
        carla.Location(x=-8.0, z=25.0),
        carla.Rotation(pitch=-70.0)
      )
      self.overhead_sensor = self.world.spawn_actor(
        overhead_bp, overhead_transform, attach_to=self.ego)
      
      def get_overhead_img(data):
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        array = np.reshape(array, (data.height, data.width, 4))
        self.overhead_img = array[:, :, :3][:, :, ::-1].copy()
      
      self.overhead_sensor.listen(get_overhead_img)
    else:
      self.overhead_sensor = None
      self.overhead_img = None

    # Update timesteps
    self.time_step=0
    self.reset_step+=1

    # Enable sync mode
    self.settings.synchronous_mode = True
    self.world.apply_settings(self.settings)

    # Tick multiple times to ensure sensors receive data
    for _ in range(5):
      self.world.tick()
      time.sleep(0.05)  # Small delay to allow sensor callbacks to process

    self.routeplanner = RoutePlanner(self.ego, self.max_waypt)
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    ego_trans = self.ego.get_transform()
    # 使用下一个 waypoint 的 yaw 避免方向突变
    next_yaw = self.waypoints[0][2] if len(self.waypoints) > 0 else ego_trans.rotation.yaw
    self.waypoints.insert(0, [
        ego_trans.location.x,
        ego_trans.location.y,
        next_yaw
    ])

    # Set ego information for render (only if display is enabled)
    if self.birdeye_render is not None:
      self.birdeye_render.set_hero(self.ego, self.ego.id)

    # Reset hierarchical state buffer and build reference line
    if self.hierarchical:
      self.state_buffer.reset()
      self.tracker.reset()
      self._last_trajectory = None
      self._last_tracking_error = None
      # Build Frenet reference line from initial waypoints
      self._build_reference_line()
      self.last_lane_speed = 0.0
      
    return self._get_obs()
  
  def step(self, action):
    # ========== Hierarchical RL mode ==========
    if self.hierarchical:
      return self._step_hierarchical(action)

    # ========== Original mode ==========
    # Calculate acceleration and steering
    if self.discrete:
      acc = self.discrete_act[0][action//self.n_steer]
      steer = self.discrete_act[1][action%self.n_steer]
    else:
      acc = action[0]
      steer = action[1]

    # Convert acceleration to throttle and brake
    # Add a small base throttle to encourage exploration
    if acc > -0.5:  # Allow some negative acc before braking
      # Map acc from [-0.5, 3] to throttle [0.2, 1.0] (base throttle of 0.2)
      throttle = np.clip(0.2 + (acc + 0.5) / 3.5 * 0.8, 0, 1)
      brake = 0
    else:
      throttle = 0
      brake = np.clip((-acc - 0.5) / 2.5, 0, 1)

    # Debug: Print action every 100 steps
    if self.total_step % 100 == 0:
      v = self.ego.get_velocity()
      speed = np.sqrt(v.x**2 + v.y**2)
      print(f"[Step {self.total_step}] acc={acc:.2f}, throttle={throttle:.2f}, brake={brake:.2f}, speed={speed:.2f}")

    # Apply control
    act = carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))
    self.ego.apply_control(act)

    self.world.tick()

    # Append actors polygon list
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    while len(self.vehicle_polygons) > self.max_past_step:
      self.vehicle_polygons.pop(0)
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)
    while len(self.walker_polygons) > self.max_past_step:
      self.walker_polygons.pop(0)

    # route planner
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    # state information
    info = {
      'waypoints': self.waypoints,
      'vehicle_front': self.vehicle_front
    }
    
    # Update timesteps
    self.time_step += 1
    self.total_step += 1

    return (self._get_obs(), self._get_reward(), self._terminal(), copy.deepcopy(info))

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def render(self, mode):
    pass

  def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
    """Create the blueprint for a specific actor type.

    Args:
      actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

    Returns:
      bp: the blueprint object of carla.
    """
    blueprints = self.world.get_blueprint_library().filter(actor_filter)
    blueprint_library = []
    for nw in number_of_wheels:
      blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
    bp = random.choice(blueprint_library)
    if bp.has_attribute('color'):
      if not color:
        color = random.choice(bp.get_attribute('color').recommended_values)
      bp.set_attribute('color', color)
    return bp

  def _init_renderer(self):
    """Initialize the birdeye view renderer.
    """
    pygame.init()
    self.display = pygame.display.set_mode(
    (self.display_size * 3, self.display_size),
    pygame.HWSURFACE | pygame.DOUBLEBUF)

    pixels_per_meter = self.display_size / self.obs_range
    pixels_ahead_vehicle = (self.obs_range/2 - self.d_behind) * pixels_per_meter
    birdeye_params = {
      'screen_size': [self.display_size, self.display_size],
      'pixels_per_meter': pixels_per_meter,
      'pixels_ahead_vehicle': pixels_ahead_vehicle
    }
    self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

  def _set_synchronous_mode(self, synchronous = True):
    """Set whether to use the synchronous mode.
    """
    self.settings.synchronous_mode = synchronous
    self.world.apply_settings(self.settings)

  def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
    """Try to spawn a surrounding vehicle at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
    blueprint.set_attribute('role_name', 'autopilot')
    vehicle = self.world.try_spawn_actor(blueprint, transform)
    if vehicle is not None:
      vehicle.set_autopilot()
      return True
    return False

  def _try_spawn_random_walker_at(self, transform):
    """Try to spawn a walker at specific transform with random bluprint.

    Args:
      transform: the carla transform object.

    Returns:
      Bool indicating whether the spawn is successful.
    """
    walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
    # set as not invencible
    if walker_bp.has_attribute('is_invincible'):
      walker_bp.set_attribute('is_invincible', 'false')
    walker_actor = self.world.try_spawn_actor(walker_bp, transform)

    if walker_actor is not None:
      walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
      walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
      # start walker
      walker_controller_actor.start()
      # set walk to random point
      walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
      # random max speed
      walker_controller_actor.set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)
      return True
    return False

  def _try_spawn_ego_vehicle_at(self, transform):
    """Try to spawn the ego vehicle at specific transform.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
    vehicle = None
    # Check if ego position overlaps with surrounding vehicles
    overlap = False
    for idx, poly in self.vehicle_polygons[-1].items():
      poly_center = np.mean(poly, axis=0)
      ego_center = np.array([transform.location.x, transform.location.y])
      dis = np.linalg.norm(poly_center - ego_center)
      if dis > 8:
        continue
      else:
        overlap = True
        break

    if not overlap:
      vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

    if vehicle is not None:
      self.ego=vehicle
      return True
      
    return False

  def _get_actor_polygons(self, filt):
    """Get the bounding box polygon of actors.

    Args:
      filt: the filter indicating what type of actors we'll look at.

    Returns:
      actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
    """
    actor_poly_dict={}
    for actor in self.world.get_actors().filter(filt):
      # Get x, y and yaw of the actor
      trans=actor.get_transform()
      x=trans.location.x
      y=trans.location.y
      yaw=trans.rotation.yaw/180*np.pi
      # Get length and width
      bb=actor.bounding_box
      l=bb.extent.x
      w=bb.extent.y
      # Get bounding box polygon in the actor's local coordinate
      poly_local=np.array([[l,w],[l,-w],[-l,-w],[-l,w]]).transpose()
      # Get rotation matrix to transform to global coordinate
      R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
      # Get global bounding box polygon
      poly=np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],4,axis=0)
      actor_poly_dict[actor.id]=poly
    return actor_poly_dict

  # ==================================================================
  # Hierarchical RL: step, reward, reference line
  # ==================================================================

  def _step_hierarchical(self, action):
    """Hierarchical mode step: action = [goal, offset] → Bezier → Tracker → Control.

    Args:
      action: array-like [goal, p_off].
        goal:  int ∈ {0, 1, 2} — 0=左换道, 1=保持, 2=右换道 (Eq.1)
        p_off: float — 连续横向偏移量 (meters) (Eq.2)
    """
    goal = int(action[0])
    offset = float(action[1])  # 连续偏移值
    # Get ego state
    ego_trans = self.ego.get_transform()
    ego_x = ego_trans.location.x
    ego_y = ego_trans.location.y
    ego_yaw = ego_trans.rotation.yaw / 180.0 * np.pi
    v = self.ego.get_velocity()
    ego_speed = np.sqrt(v.x**2 + v.y**2)
        
    # 提前获取前方障碍物信息
    self._cached_zone_state = self.zone_detector.detect(self.world, self.ego, self.carla_map) 
    cf_dist = self._cached_zone_state[15]  # Zone 6 (CF 中前) 距离 
    cf_rel_speed = self._cached_zone_state[7] # Zone 6 (CF 中前) 相对速度


    # Update reference line periodically
    if self.time_step % 10 == 0:
      self._build_reference_line()

    # Generate Bezier trajectory
    try:
      trajectory = self.bezier.generate_trajectory(
        ego_x, ego_y, ego_yaw, goal, offset, cf_dist=cf_dist, ego_speed=ego_speed, world=self.world, ego_vehicle=self.ego)
      self._last_trajectory = trajectory
    except Exception as e:
      # Fallback: straight ahead trajectory
      if self.total_step % 50 == 0:
        print(f"[Hierarchical] Bezier failed: {e}, using straight fallback")
      fwd = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
      trajectory = np.array([
        [ego_x + fwd[0] * d, ego_y + fwd[1] * d]
        for d in np.linspace(1, 30, 50)
      ])
      self._last_trajectory = trajectory

    # Trajectory tracking → vehicle control
    ego_state = EgoState(x=ego_x, y=ego_y, yaw=ego_yaw, speed=ego_speed)
    throttle, steer, brake = self.tracker.compute_control(trajectory, ego_state, cf_dist=cf_dist, cf_rel_speed=cf_rel_speed)

    # Compute tracking error for reward
    self._last_tracking_error = self.tracker.compute_tracking_error(
      trajectory, ego_state)

    # Debug log
    if self.total_step % 100 == 0:
      goal_names = ['左换道', '保持', '右换道']
      err = self._last_tracking_error
      print(f"[Step {self.total_step}] Goal={goal_names[goal]}, p_off={offset:+.3f}m, "
            f"speed={ego_speed:.1f}, throttle={throttle:.2f}, steer={steer:.3f}, "
            f"lat_err={err['lateral_error']:.2f}")

    # Apply control
    # Pure Pursuit 输出已符合 CARLA 约定 (正值=右转)，无需取反
    # 注: 原版非分层模式的 -steer 是因为 RL 策略用「左正」约定
    act = carla.VehicleControl(
      throttle=float(throttle), steer=float(steer), brake=float(brake))
    self.ego.apply_control(act)

    self.world.tick()

    # Update polygon lists
    vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    self.vehicle_polygons.append(vehicle_poly_dict)
    while len(self.vehicle_polygons) > self.max_past_step:
      self.vehicle_polygons.pop(0)
    walker_poly_dict = self._get_actor_polygons('walker.*')
    self.walker_polygons.append(walker_poly_dict)
    while len(self.walker_polygons) > self.max_past_step:
      self.walker_polygons.pop(0)

    # Route planner update
    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()

    # 开局前10步：插入车辆位置，让蓝色轨迹从脚下开始
    if self.time_step < 15:
        ego_trans = self.ego.get_transform()
        # 使用下一个 waypoint 的 yaw 避免方向突变
        next_yaw = self.waypoints[0][2] if len(self.waypoints) > 0 else ego_trans.rotation.yaw
        self.waypoints.insert(0, [
            ego_trans.location.x,
            ego_trans.location.y,
            next_yaw
        ])
    
    # === 被动全局轨迹跟随：蓝色轨迹顺应车辆换道行为 ===
    ego_wp = self.carla_map.get_waypoint(self.ego.get_transform().location)
    cur_lane_id = ego_wp.lane_id
    
    # 首次初始化
    if self._prev_lane_id is None:
        self._prev_lane_id = cur_lane_id
    
    # --- 阶段 1：检测 lane_id 变化 ---
    if not self._lane_change_detected:
        if cur_lane_id != self._prev_lane_id:
            # lane_id 变了，检查两个前提条件：
            # 条件 A：同向车道（lane_id 符号相同）
            # 说明：cur_lane_id * prev_lane_id > 0 表示符号相同
            #       如果任一为 0，乘积为 0，自动判定为非同向
            same_direction = (cur_lane_id * self._prev_lane_id > 0)
            
            # 条件 B：前方有障碍物
            has_obstacle = (cf_dist < 15.0)
            
            if same_direction and has_obstacle:
                # 两个条件都满足，启动稳定计数
                self._lane_change_detected = True
                self._stable_counter = 1  # 当前步算第 1 步
                print(f"[LaneFollow] 检测到换道: {self._prev_lane_id} → {cur_lane_id}, 开始计数")
            elif not same_direction:
                # 对向车道或 lane_id=0，完全忽略，不更新 _prev_lane_id
                pass
            else:
                # 同向但无障碍物（弯道漂移等），忽略本次 lane_id 变化
                self._prev_lane_id = cur_lane_id
    
    # --- 阶段 2：等待稳定 ---
    elif self._lane_change_detected:
        if cur_lane_id == self._prev_lane_id:
            # 车辆又回到了原来的车道，取消本次重建
            self._lane_change_detected = False
            self._stable_counter = 0
            print(f"[LaneFollow] 换道取消: 车辆回到原车道 {cur_lane_id}")
        elif cur_lane_id == 0 or cur_lane_id * self._prev_lane_id < 0:
            # lane_id=0（中心线）或对向车道（符号相反），取消并不更新
            self._lane_change_detected = False
            self._stable_counter = 0
            print(f"[LaneFollow] 换道取消: 检测到异常车道 {cur_lane_id}")
        else:
            # 还在新车道上，继续计数
            self._stable_counter += 1
            
            if self._stable_counter >= 10:
                # ✅ 稳定 10 步，执行重建！
                if self.routeplanner.rebuild_from_vehicle():
                    self.waypoints, _, self.vehicle_front = self.routeplanner.run_step()
                    self._build_reference_line()
                    print(f"[LaneFollow] 全局轨迹已更新到 lane_id={cur_lane_id}")
                
                # 重置所有状态
                self._prev_lane_id = cur_lane_id
                self._lane_change_detected = False
                self._stable_counter = 0

    # State information
    info = {
      'waypoints': self.waypoints,
      'vehicle_front': self.vehicle_front,
      'goal': goal,
      'offset': offset,
      'tracking_error': self._last_tracking_error,
    }

    # Update timesteps
    self.time_step += 1
    self.total_step += 1

    return (self._get_obs(), self._get_hierarchical_reward(goal, offset),
            self._terminal(), copy.deepcopy(info))

  def _build_reference_line(self):
    """Build Frenet reference line from current waypoints."""
    try:
      if len(self.waypoints) >= 2:
        self.frenet.build_reference_line(self.waypoints)
        self.bezier = BezierFitting(
          frenet=self.frenet,
          lane_width=self.lane_width,
          plan_horizon=cfg.PLAN_HORIZON,
          n_samples=cfg.BEZIER_SAMPLES,
        )
    except Exception as e:
      if self.total_step % 50 == 0:
        print(f"[Hierarchical] Build reference line failed: {e}")

  def _get_hierarchical_reward(self, goal, offset):
    """
    调用论文奖励函数
    
    Args:
        goal: Q1 action (0=left, 1=keep, 2=right)
        offset: Q2 action
    
    Returns:
        float: reward
    """
    # 复用 _step_hierarchical 中缓存的 zone_detector 结果，避免重复调用 
    front_state = (self._cached_zone_state if self._cached_zone_state is not None else self.zone_detector.detect(self.world, self.ego, self.carla_map)) 
    cf_rel_speed = front_state[7] # Zone 6 (CF 中前) 相对速度
    cf_dist = front_state[15]   # Zone 6 (CF 前方) 距离
    lf_dist = front_state[14]   # Zone 5 (LF 左前) 距离
    rf_dist = front_state[16]   # Zone 7 (RF 右前) 距离
    ls_dist = front_state[13]   # Zone 4 (LS 左侧) 距离
    rs_dist = front_state[17]   # Zone 8 (RS 右侧) 距离
    
    # 1.基础奖励
    reward, new_last_lane_speed = get_hierarchical_reward(
        ego=self.ego,
        collision_hist=self.collision_hist,
        desired_speed=self.desired_speed,
        goal=goal,
        last_lane_speed=self.last_lane_speed,
        cf_dist=cf_dist,
        cf_rel_speed=cf_rel_speed
    )
    self.last_lane_speed = new_last_lane_speed

    # 2.横向跟踪误差惩罚,跟踪黄色曲线的能力 (Q2 的核心学习信号)
    if self._last_tracking_error is not None:
        lat_err = self._last_tracking_error['lateral_error']
        reward -= 0.6 * lat_err  # 线性惩罚，比二次更稳定

    # 3.贝塞尔曲线偏离中心距离惩罚
    ego_x, ego_y = get_pos(self.ego)
    lane_dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
    
    # 检查 NaN（当车辆正好在 waypoint 上时可能出现）
    if np.isnan(lane_dis):
        lane_dis = 0.0
    
    if self._lane_change_detected:
        reward -= 0.2 * abs(lane_dis)  # 降低到原来的 1/3
    else:
        reward -= 1 * abs(lane_dis)  # 正常惩罚

    # 4.五面感知障碍物惩罚(连续信号, 碰撞前就开始惩罚)
      # 前方有车
    if cf_dist < 15.0:
        reward -= 1.0 * (15.0 - cf_dist) / 15.0   # 0 ~ 3.0，感知范围扩大到20m
    if cf_dist < 10.0:
        reward -= 2.0 * (10.0 - cf_dist) / 10.0   # 再叠加 0 ~ 3.0
    if cf_dist < 6.0:
        reward -= 3.0 * (6.0 - cf_dist) / 6.0

    # 5. 换道成本
    if goal != 1:
        if cf_dist > 20.0:
            reward -= 1.0   # 没必要换道
        else:
            reward -= 0.1   # 鼓励避障换道

    # 6.避障超出道路边界惩罚
    abs_lane_dis = abs(lane_dis)
    ego_wp = self.carla_map.get_waypoint(self.ego.get_transform().location)
    half_lane = ego_wp.lane_width / 2.0 if ego_wp else 1.75
    if self._lane_change_detected:
        if abs_lane_dis > half_lane:
            reward -= 0.4 * (abs_lane_dis - half_lane)
        if abs_lane_dis > half_lane + 1.0:
            reward -= 1.0
    else:
        if abs_lane_dis > half_lane:
            reward -= 1.0 * (abs_lane_dis - half_lane)
        if abs_lane_dis > half_lane + 1.0:
            reward -= 3.0

    # 7.换道可行性惩罚，不能往不存在/对向的车道换
    if goal == 0:
        left_lane = ego_wp.get_left_lane()
        if (left_lane is None or left_lane.lane_type != carla.LaneType.Driving
            or left_lane.lane_id * ego_wp.lane_id < 0):  # 符号不同=对向车道
            reward -= 3.0
    elif goal == 2:
        right_lane = ego_wp.get_right_lane()
        if (right_lane is None or right_lane.lane_type != carla.LaneType.Driving
            or right_lane.lane_id * ego_wp.lane_id < 0):  # 符号不同=对向车道
            reward -= 3.0
    
    # 8.动态速度调节，前方全堵时要减速等待
    v = self.ego.get_velocity()
    ego_speed = np.sqrt(v.x**2 + v.y**2)
    
    all_blocked_dist = 15.0
    all_blocked = (cf_dist < all_blocked_dist and
                   lf_dist < all_blocked_dist and
                   rf_dist < all_blocked_dist)

    if all_blocked:
        min_front_dist = min(cf_dist, lf_dist, rf_dist)
        safe_speed = max(0.0, min_front_dist / 15.0 * 4.0)

        if ego_speed <= safe_speed + 1.0:
            reward += 0.5    # 原来 2.0 → 0.5
        else:
            reward -= 0.3 * (ego_speed - safe_speed)  # 原来 0.5 → 0.3
    else:
        if ego_speed < 3.0 and cf_dist > 20.0:
            reward -= 0.3    # 原来 0.5 → 0.3


    return reward

  # ==================================================================
  def _get_obs(self):
    """Get the observations."""
    # Hierarchical mode: return (seq_len, state_dim) state sequence
    if self.hierarchical:
      return self._get_hierarchical_obs()

    ## Birdeye rendering (skip if display is disabled)
    if self.birdeye_render is not None:
      self.birdeye_render.vehicle_polygons = self.vehicle_polygons
      self.birdeye_render.walker_polygons = self.walker_polygons
      self.birdeye_render.waypoints = self.waypoints

      # birdeye view with roadmap and actors
      birdeye_render_types = ['roadmap', 'actors']
      if self.display_route:
        birdeye_render_types.append('waypoints')
      self.birdeye_render.render(self.display, birdeye_render_types)
      birdeye = pygame.surfarray.array3d(self.display)
      birdeye = birdeye[0:self.display_size, :, :]
      birdeye = display_to_rgb(birdeye, self.obs_size)
    else:
      # Create dummy birdeye image when display is disabled
      birdeye = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)

    # Roadmap (skip if display is disabled)
    if self.pixor:
      if self.birdeye_render is not None:
        roadmap_render_types = ['roadmap']
        if self.display_route:
          roadmap_render_types.append('waypoints')
        self.birdeye_render.render(self.display, roadmap_render_types)
        roadmap = pygame.surfarray.array3d(self.display)
        roadmap = roadmap[0:self.display_size, :, :]
        roadmap = display_to_rgb(roadmap, self.obs_size)
        # Add ego vehicle
        for i in range(self.obs_size):
          for j in range(self.obs_size):
            if abs(birdeye[i, j, 0] - 255)<20 and abs(birdeye[i, j, 1] - 0)<20 and abs(birdeye[i, j, 0] - 255)<20:
              roadmap[i, j, :] = birdeye[i, j, :]
      else:
        # Create dummy roadmap when display is disabled
        roadmap = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)

    # Display birdeye image (only if display is enabled)
    if self.display is not None:
      birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
      self.display.blit(birdeye_surface, (0, 0))

    ## Lidar image generation
    # Get point cloud data - CARLA 0.9.13+ uses raw_data buffer format
    if self.lidar_data is not None and len(self.lidar_data.raw_data) > 0:
      # Parse raw_data: each point has 4 float32 values (x, y, z, intensity)
      points = np.frombuffer(self.lidar_data.raw_data, dtype=np.float32)
      points = np.reshape(points, (int(points.shape[0] / 4), 4))
      # Extract x, y, z (note: z is negated for coordinate conversion)
      point_cloud = np.column_stack((points[:, 0], points[:, 1], -points[:, 2]))
    else:
      point_cloud = np.zeros((0, 3))
    # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin,
    # and z is set to be two bins.
    y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind+self.lidar_bin, self.lidar_bin)
    x_bins = np.arange(-self.obs_range/2, self.obs_range/2+self.lidar_bin, self.lidar_bin)
    z_bins = [-self.lidar_height-1, -self.lidar_height+0.25, 1]
    # Get lidar image according to the bins
    lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
    lidar[:,:,0] = np.array(lidar[:,:,0]>0, dtype=np.uint8)
    lidar[:,:,1] = np.array(lidar[:,:,1]>0, dtype=np.uint8)
    # Add the waypoints to lidar image
    if self.display_route:
      wayptimg = (birdeye[:,:,0] <= 10) * (birdeye[:,:,1] <= 10) * (birdeye[:,:,2] >= 240)
    else:
      wayptimg = birdeye[:,:,0] < 0  # Equal to a zero matrix
    wayptimg = np.expand_dims(wayptimg, axis=2)
    wayptimg = np.fliplr(np.rot90(wayptimg, 3))

    # Get the final lidar image
    lidar = np.concatenate((lidar, wayptimg), axis=2)
    lidar = np.flip(lidar, axis=1)
    lidar = np.rot90(lidar, 1)
    lidar = lidar * 255

    # Display lidar image (only if display is enabled)
    if self.display is not None:
      lidar_surface = rgb_to_display_surface(lidar, self.display_size)
      self.display.blit(lidar_surface, (self.display_size, 0))

    ## Display camera image
    if self.display is not None:
      camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
      camera_surface = rgb_to_display_surface(camera, self.display_size)
      self.display.blit(camera_surface, (self.display_size * 2, 0))

    # Display on pygame (only if display is enabled)
    if self.display is not None:
      pygame.display.flip()
      
      # Process pygame events to prevent window from freezing
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          pass

    # State observation
    ego_trans = self.ego.get_transform()
    ego_x = ego_trans.location.x
    ego_y = ego_trans.location.y
    ego_yaw = ego_trans.rotation.yaw/180*np.pi
    lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
    delta_yaw = np.arcsin(np.cross(w, 
      np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    state = np.array([lateral_dis, - delta_yaw, speed, self.vehicle_front])

    if self.pixor:
      ## Vehicle classification and regression maps (requires further normalization)
      vh_clas = np.zeros((self.pixor_size, self.pixor_size))
      vh_regr = np.zeros((self.pixor_size, self.pixor_size, 6))

      # Generate the PIXOR image. Note in CARLA it is using left-hand coordinate
      # Get the 6-dim geom parametrization in PIXOR, here we use pixel coordinate
      for actor in self.world.get_actors().filter('vehicle.*'):
        x, y, yaw, l, w = get_info(actor)
        x_local, y_local, yaw_local = get_local_pose((x, y, yaw), (ego_x, ego_y, ego_yaw))
        if actor.id != self.ego.id:
          if abs(y_local)<self.obs_range/2+1 and x_local<self.obs_range-self.d_behind+1 and x_local>-self.d_behind-1:
            x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel = get_pixel_info(
              local_info=(x_local, y_local, yaw_local, l, w),
              d_behind=self.d_behind, obs_range=self.obs_range, image_size=self.pixor_size)
            cos_t = np.cos(yaw_pixel)
            sin_t = np.sin(yaw_pixel)
            logw = np.log(w_pixel)
            logl = np.log(l_pixel)
            pixels = get_pixels_inside_vehicle(
              pixel_info=(x_pixel, y_pixel, yaw_pixel, l_pixel, w_pixel),
              pixel_grid=self.pixel_grid)
            for pixel in pixels:
              vh_clas[pixel[0], pixel[1]] = 1
              dx = x_pixel - pixel[0]
              dy = y_pixel - pixel[1]
              vh_regr[pixel[0], pixel[1], :] = np.array(
                [cos_t, sin_t, dx, dy, logw, logl])

      # Flip the image matrix so that the origin is at the left-bottom
      vh_clas = np.flip(vh_clas, axis=0)
      vh_regr = np.flip(vh_regr, axis=0)

      # Pixor state, [x, y, cos(yaw), sin(yaw), speed]
      pixor_state = [ego_x, ego_y, np.cos(ego_yaw), np.sin(ego_yaw), speed]

    obs = {
      'camera':camera.astype(np.uint8),
      'lidar':lidar.astype(np.uint8),
      'birdeye':birdeye.astype(np.uint8),
      'state': state,
    }

    if self.pixor:
      obs.update({
        'roadmap':roadmap.astype(np.uint8),
        'vh_clas':np.expand_dims(vh_clas, -1).astype(np.float32),
        'vh_regr':vh_regr.astype(np.float32),
        'pixor_state': pixor_state,
      })

    return obs

  def _get_hierarchical_obs(self):
    """Get hierarchical observation: 18-dim state × 3 timesteps.

    论文 §2.1.2: 状态空间 = [v_ego, lane_id, Δv×8, Δd×8] (18维)
    论文 §2.2.1: 输入过去 3 个采样时刻的状态序列

    Returns:
      np.ndarray: shape (seq_len, state_dim) = (3, 18), dtype float32.
    """
    # 使用 zone_detector 计算 18 维状态向量
    state_18d = self.zone_detector.detect(
      self.world, self.ego, self.carla_map)
    self.state_buffer.push(state_18d)

    # Debug: 首次调用时打印状态信息
    if self.time_step == 0 and self.reset_step <= 1:
      print(f"[Hierarchical Obs] state shape: {self.state_buffer.get_numpy().shape}")
      print(f"[Hierarchical Obs] state[0] (latest):")
      print(f"  v_ego={state_18d[0]:.2f} m/s, lane_id={state_18d[1]:.0f}")
      print(f"  Δv: {state_18d[2:10]}")
      print(f"  Δd: {state_18d[10:18]}")

    # Process pygame events to prevent window from freezing (only if pygame is initialized)
    if self.display is not None:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          pass

    return self.state_buffer.get_numpy().astype(np.float32)

  def _get_reward(self):
    """Calculate the step reward."""
    # reward for speed tracking
    v = self.ego.get_velocity()
    speed = np.sqrt(v.x**2 + v.y**2)
    r_speed = -abs(speed - self.desired_speed)
    
    # reward for collision
    r_collision = 0
    if len(self.collision_hist) > 0:
      r_collision = -1

    # reward for steering:
    r_steer = -self.ego.get_control().steer**2

    # reward for out of lane
    ego_x, ego_y = get_pos(self.ego)
    dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
    r_out = 0
    if abs(dis) > self.out_lane_thres:
      r_out = -1

    # longitudinal speed
    lspeed = np.array([v.x, v.y])
    lspeed_lon = np.dot(lspeed, w)

    # cost for too fast
    r_fast = 0
    if lspeed_lon > self.desired_speed:
      r_fast = -1

    # cost for lateral acceleration
    r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2

    r = 200*r_collision + 1*lspeed_lon + 10*r_fast + 1*r_out + r_steer*5 + 0.2*r_lat - 0.1

    return r

  def _terminal(self):
    """Calculate whether to terminate the current episode."""
    # Get ego state
    ego_x, ego_y = get_pos(self.ego)

    # If collides
    if len(self.collision_hist)>0: 
      return True

    # If reach maximum timestep
    if self.time_step>self.max_time_episode:
      return True

    # If at destination
    if self.dests is not None: # If at destination
      for dest in self.dests:
        if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2)<4:
          return True

    # If out of lane
    if self.hierarchical:
        # 使用 CARLA 原生 API 判断：只要在任何可行驶车道上就不算出界
        ego_loc = self.ego.get_transform().location
        ego_wp = self.carla_map.get_waypoint(ego_loc, project_to_road=True)
        if ego_wp is None or ego_wp.lane_type != carla.LaneType.Driving:
            return True
        # 宽松的安全检查：允许车辆偏离车道中心较远（1.5倍车道宽度）
        # 这样 3.5m 车道允许偏离 4.2m，4.5m 车道允许偏离 5.4m
        wp_loc = ego_wp.transform.location
        dist_to_center = ego_loc.distance(wp_loc)
        if dist_to_center > ego_wp.lane_width * 1.5:
            return True
    else:
        dis, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        if abs(dis) > self.out_lane_thres:
            return True

    return False

  def _clear_all_actors(self, actor_filters):
    """Clear specific actors (batch destroy to suppress C++ error messages)."""
    actors_to_destroy = []
    for actor_filter in actor_filters:
      for actor in self.world.get_actors().filter(actor_filter):
        if actor.is_alive:
          if actor.type_id == 'controller.ai.walker':
            actor.stop()
          actors_to_destroy.append(actor.id)
    if actors_to_destroy:
      self.client.apply_batch([carla.command.DestroyActor(aid) for aid in actors_to_destroy])
