import glob
import os
import sys
import time

# ==========================================
# 1. 暴力挂载 CARLA Python API (最重要的一步)
# ==========================================
try:
    # 指向你的 CARLA 0.9.13 WindowsNoEditor 目录
    carla_root = r'D:\electron\CARLA_0.9.13\WindowsNoEditor'
    
    # 找 egg 文件 (Python 3.7 版本)
    sys.path.append(glob.glob(carla_root + '/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    
    print("成功挂载 CARLA API！")
except IndexError:
    print("找不到 CARLA egg 文件，请检查 Python 版本是否为 3.7！")
    sys.exit()

import carla
import gym
from stable_baselines3 import PPO # 咱们用最稳的 PPO 算法
from gym_carla.envs.carla_env import CarlaEnv # 引用你刚才下载的本地包

# ==========================================
# 2. 配置环境参数
# ==========================================
params = {
    # === 基础连接参数 ===
    'port': 2000,
    'town': 'Town03',
    'task_mode': 'random',
    'max_time_episode': 1000,
    
    # === 车辆与渲染 ===
    'number_of_vehicles': 0,
    'number_of_walkers': 0,
    'display_size': 256,
    'max_past_step': 1,
    'dt': 0.1,
    'ego_vehicle_filter': 'vehicle.tesla.model3',
    
    # === 动作空间 (连续) ===
    'discrete': False,
    'discrete_acc': [-3.0, 0.0, 3.0], 
    'discrete_steer': [-0.2, 0.0, 0.2],
    'continuous_accel_range': [-3.0, 3.0],
    'continuous_steer_range': [-0.3, 0.3],
    
    # === 观测空间 (状态) ===
    'state': ['position', 'velocity', 'heading'],
    'observation_size': 256, # 图像大小
    'pixel_category': True,  # 使用语义分割图
    
    # === 【关键】补全所有可能缺失的参数，防止KeyError ===
    'max_ego_spawn_times': 200,  # [本次报错修复] 尝试生成自车的最大次数
    'max_waypt': 12,             # 导航点数量
    'obs_range': 32,             # 观测范围
    'lidar_bin': 0.125,          # Lidar参数 (占位)
    'd_behind': 12,              # 后方视野
    'out_lane_thres': 2.0,       # 偏离车道阈值
    'desired_speed': 8,          # 期望速度
    'display_route': True,       # [预防下一个报错] 是否在画面显示路径
    'pixor_size': 64,            # [预防] Pixor网络参数
    'pixor': False,              # [预防] 不使用Pixor

    # === 关键修改 ===
    'pixel_category': False,  # 【关掉】不需要语义分割图
    'pixel_depth': False,     # 【关掉】不需要深度图
    'ego_vehicle_filter': 'vehicle.tesla.model3',
    
    # 既然关了图像，observation_size 就不重要了，但为了防报错保持原样即可
    'observation_size': 256,
}

# ==========================================
# 3. 开始训练
# ==========================================
if __name__ == '__main__':
    print("正在连接 CARLA 服务器...")
    
    # 创建环境
    env = gym.make('carla-v0', params=params)
    
    # 创建模型 (PPO)
    # MlpPolicy 表示用全连接神经网络 (处理向量输入)
    # 如果要处理图像输入，这里改用 'CnnPolicy'
    model = PPO("MultiInputPolicy", env, verbose=1)
    
    print("开始训练！按 Ctrl+C 可以随时停止并保存。")
    try:
        # 训练 10000 步试试水
        model.learn(total_timesteps=10000)
        
        # 保存模型
        model.save("ppo_carla_model")
        print("模型已保存！")
        
    except Exception as e:
        print(f"训练出错: {e}")
    finally:
        env.close()