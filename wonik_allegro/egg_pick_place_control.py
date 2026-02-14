#!/usr/bin/env python3
"""
Wonik Allegro Hand - 鸡蛋抓取与放置仿真
自动执行：抓取鸡蛋 -> 移动到放置位置 -> 放下鸡蛋
"""

import mujoco
import numpy as np
from enum import Enum
import time

class PickPlaceState(Enum):
    """抓取放置状态机"""
    INITIAL = "initial"              # 初始状态
    OPEN_HAND = "open_hand"          # 张开手指
    MOVE_TO_EGG = "move_to_egg"      # 移动到鸡蛋位置
    GRASP = "grasp"                  # 抓取鸡蛋
    LIFT = "lift"                    # 抬起鸡蛋
    MOVE_TO_PLACE = "move_to_place"  # 移动到放置位置
    RELEASE = "release"              # 放下鸡蛋
    RESET = "reset"                  # 复位
    COMPLETE = "complete"            # 完成

class AllegroController:
    """Wonik Allegro手控制器"""
    
    def __init__(self, xml_path):
        """初始化控制器"""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 关节名称列表
        self.joint_names = [
            'ffj0', 'ffj1', 'ffj2', 'ffj3',  # 食指
            'mfj0', 'mfj1', 'mfj2', 'mfj3',  # 中指
            'rfj0', 'rfj1', 'rfj2', 'rfj3',  # 无名指
            'thj0', 'thj1', 'thj2', 'thj3'   # 拇指
        ]
        
        # 获取关节ID
        self.joint_ids = []
        for name in self.joint_names:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.joint_ids.append(joint_id)
            except:
                print(f"警告: 找不到关节 {name}")
        
        # 控制参数
        self.kp = 1.0     # 位置增益
        self.kd = 0.1     # 速度增益
        
        # 状态机
        self.state = PickPlaceState.INITIAL
        self.state_start_time = 0
        self.state_duration = 2.0  # 每个状态的默认持续时间
        
        # 目标关节角度
        self.target_qpos = np.zeros(len(self.joint_ids))
        self.current_qpos = np.zeros(len(self.joint_ids))
        
        # 鸡蛋位置
        self.egg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'egg')
        
        # 初始化关节位置
        self.reset_joint_positions()
    
    def reset_joint_positions(self):
        """重置关节到初始位置（张开手指）"""
        # 初始位置：张开手指准备抓取
        initial_positions = {
            'ffj0': 0.0,    # 食指基座
            'ffj1': 0.5,    # 食指近端
            'ffj2': 0.8,    # 食指中端
            'ffj3': 0.3,    # 食指远端
            'mfj0': 0.0,    # 中指基座
            'mfj1': 0.5,    # 中指近端
            'mfj2': 0.8,    # 中指中端
            'mfj3': 0.3,    # 中指远端
            'rfj0': 0.0,    # 无名指基座
            'rfj1': 0.5,    # 无名指近端
            'rfj2': 0.8,    # 无名指中端
            'rfj3': 0.3,    # 无名指远端
            'thj0': 0.5,    # 拇指基座（张开）
            'thj1': 0.3,    # 拇指近端
            'thj2': 0.5,    # 拇指中端
            'thj3': 0.3     # 拇指远端
        }
        
        for i, name in enumerate(self.joint_names):
            if name in initial_positions:
                self.target_qpos[i] = initial_positions[name]
                self.current_qpos[i] = initial_positions[name]
        
        self.set_joint_positions(self.target_qpos)
    
    def set_joint_positions(self, positions):
        """设置目标关节位置"""
        for i, joint_id in enumerate(self.joint_ids):
            if i < len(positions):
                self.data.joint(joint_id).qpos = positions[i]
    
    def get_joint_positions(self):
        """获取当前关节位置"""
        positions = []
        for joint_id in self.joint_ids:
            positions.append(self.data.joint(joint_id).qpos)
        return np.array(positions)
    
    def get_joint_velocities(self):
        """获取当前关节速度"""
        velocities = []
        for joint_id in self.joint_ids:
            velocities.append(self.data.joint(joint_id).qvel)
        return np.array(velocities)
    
    def grasp_pose(self):
        """生成抓取姿态"""
        # 闭合手指的抓取姿态
        grasp_positions = {
            'ffj0': 0.1,    # 食指基座（稍微向内）
            'ffj1': 0.8,    # 食指近端（弯曲）
            'ffj2': 1.2,    # 食指中端（弯曲）
            'ffj3': 0.8,    # 食指远端（弯曲）
            'mfj0': 0.0,    # 中指基座
            'mfj1': 0.9,    # 中指近端
            'mfj2': 1.3,    # 中指中端
            'mfj3': 0.9,    # 中指远端
            'rfj0': -0.1,   # 无名指基座（稍微向内）
            'rfj1': 0.8,    # 无名指近端
            'rfj2': 1.2,    # 无名指中端
            'rfj3': 0.8,    # 无名指远端
            'thj0': 0.8,    # 拇指基座（张开准备抓取）
            'thj1': 0.5,    # 拇指近端
            'thj2': 0.8,    # 拇指中端
            'thj3': 0.5     # 拇指远端
        }
        
        for i, name in enumerate(self.joint_names):
            if name in grasp_positions:
                self.target_qpos[i] = grasp_positions[name]
    
    def open_hand_pose(self):
        """生成张开手掌的姿态"""
        open_positions = {
            'ffj0': 0.0,
            'ffj1': 0.3,
            'ffj2': 0.5,
            'ffj3': 0.2,
            'mfj0': 0.0,
            'mfj1': 0.3,
            'mfj2': 0.5,
            'mfj3': 0.2,
            'rfj0': 0.0,
            'rfj1': 0.3,
            'rfj2': 0.5,
            'rfj3': 0.2,
            'thj0': 0.3,
            'thj1': 0.2,
            'thj2': 0.3,
            'thj3': 0.2
        }
        
        for i, name in enumerate(self.joint_names):
            if name in open_positions:
                self.target_qpos[i] = open_positions[name]
    
    def interpolate_pose(self, target_pose, alpha):
        """在当前姿态和目标姿态之间插值"""
        return self.current_qpos + alpha * (target_pose - self.current_qpos)
    
    def step(self):
        """执行一步控制"""
        # 更新当前关节位置
        self.current_qpos = self.get_joint_positions()
        
        # 计算控制力矩（简化的PD控制）
        q_error = self.target_qpos - self.current_qpos
        q_vel = self.get_joint_velocities()
        
        # 应用控制
        for i, joint_id in enumerate(self.joint_ids):
            torque = self.kp * q_error[i] - self.kd * q_vel[i]
            self.data.joint(joint_id).qfrc_applied = torque
        
        # 仿真一步
        mujoco.mj_step(self.model, self.data)
    
    def get_egg_position(self):
        """获取鸡蛋的当前位置"""
        return self.data.body(self.egg_body_id).xpos.copy()
    
    def get_egg_orientation(self):
        """获取鸡蛋的当前姿态（四元数）"""
        return self.data.body(self.egg_body_id).xquat.copy()


class PickPlaceSimulator:
    """抓取放置仿真器"""
    
    def __init__(self, xml_path):
        """初始化仿真器"""
        self.controller = AllegroController(xml_path)
        self.start_time = time.time()
        
        # 轨迹参数
        self.move_speed = 2.0  # 移动速度
    
    def get_state_duration(self, state):
        """获取每个状态的持续时间"""
        durations = {
            PickPlaceState.INITIAL: 1.0,
            PickPlaceState.OPEN_HAND: 1.5,
            PickPlaceState.MOVE_TO_EGG: 2.0,
            PickPlaceState.GRASP: 1.0,
            PickPlaceState.LIFT: 1.5,
            PickPlaceState.MOVE_TO_PLACE: 2.0,
            PickPlaceState.RELEASE: 1.0,
            PickPlaceState.RESET: 1.5,
            PickPlaceState.COMPLETE: float('inf')
        }
        return durations.get(state, 2.0)
    
    def run_state_machine(self):
        """运行状态机"""
        current_time = time.time() - self.start_time
        state_time = current_time - self.controller.state_start_time
        state_duration = self.get_state_duration(self.controller.state)
        progress = min(state_time / state_duration, 1.0) if state_duration != float('inf') else 1.0
        
        # 平滑过渡函数
        smooth_progress = self.smooth_step(progress)
        
        # 状态机逻辑
        if self.controller.state == PickPlaceState.INITIAL:
            print("状态: 初始化...")
            self.controller.reset_joint_positions()
            if progress >= 1.0:
                self.controller.state = PickPlaceState.OPEN_HAND
                self.controller.state_start_time = current_time
        
        elif self.controller.state == PickPlaceState.OPEN_HAND:
            print("状态: 张开手指...")
            self.controller.open_hand_pose()
            if progress >= 1.0:
                self.controller.state = PickPlaceState.MOVE_TO_EGG
                self.controller.state_start_time = current_time
        
        elif self.controller.state == PickPlaceState.MOVE_TO_EGG:
            print("状态: 移动到鸡蛋位置...")
            # 保持张开姿态，移动到鸡蛋上方
            self.controller.open_hand_pose()
            if progress >= 1.0:
                self.controller.state = PickPlaceState.GRASP
                self.controller.state_start_time = current_time
        
        elif self.controller.state == PickPlaceState.GRASP:
            print("状态: 抓取鸡蛋...")
            self.controller.grasp_pose()
            if progress >= 1.0:
                self.controller.state = PickPlaceState.LIFT
                self.controller.state_start_time = current_time
        
        elif self.controller.state == PickPlaceState.LIFT:
            print("状态: 抬起鸡蛋...")
            # 保持抓取姿态，稍微抬起
            self.controller.grasp_pose()
            if progress >= 1.0:
                self.controller.state = PickPlaceState.MOVE_TO_PLACE
                self.controller.state_start_time = current_time
        
        elif self.controller.state == PickPlaceState.MOVE_TO_PLACE:
            print("状态: 移动到放置位置...")
            # 保持抓取姿态，移动到放置位置
            self.controller.grasp_pose()
            if progress >= 1.0:
                self.controller.state = PickPlaceState.RELEASE
                self.controller.state_start_time = current_time
        
        elif self.controller.state == PickPlaceState.RELEASE:
            print("状态: 放下鸡蛋...")
            self.controller.open_hand_pose()
            if progress >= 1.0:
                self.controller.state = PickPlaceState.RESET
                self.controller.state_start_time = current_time
        
        elif self.controller.state == PickPlaceState.RESET:
            print("状态: 复位...")
            self.controller.open_hand_pose()
            if progress >= 1.0:
                self.controller.state = PickPlaceState.COMPLETE
                self.controller.state_start_time = current_time
        
        elif self.controller.state == PickPlaceState.COMPLETE:
            print("仿真完成！")
            # 可以选择重新开始
            if state_time > 3.0:  # 等待3秒后重新开始
                self.start_time = time.time()
                self.controller.state = PickPlaceState.INITIAL
                self.controller.state_start_time = 0
    
    def smooth_step(self, x):
        """平滑阶跃函数（用于平滑过渡）"""
        return x * x * (3 - 2 * x)
    
    def smooth_sigmoid(self, x):
        """平滑sigmoid函数"""
        return 1 / (1 + np.exp(-10 * (x - 0.5)))
    
    def run(self, viewer=None, max_steps=None):
        """运行仿真"""
        print("="*50)
        print("Wonik Allegro 鸡蛋抓取仿真")
        print("="*50)
        
        step_count = 0
        
        try:
            while True:
                # 运行状态机
                self.run_state_machine()
                
                # 执行控制
                self.controller.step()
                
                # 更新视图
                if viewer is not None:
                    viewer.sync()
                
                step_count += 1
                
                # 打印状态信息
                if step_count % 100 == 0:
                    egg_pos = self.controller.get_egg_position()
                    print(f"步骤 {step_count}: 鸡蛋位置 = {egg_pos}")
                
                # 检查是否达到最大步数
                if max_steps is not None and step_count >= max_steps:
                    break
                
                # 模拟控制频率
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n仿真被用户中断")


def main():
    """主函数"""
    # 场景文件路径
    scene_path = "wonik_allegro/egg_pick_place_scene.xml"
    
    # 创建仿真器
    simulator = PickPlaceSimulator(scene_path)
    
    # 创建MuJoCo视图器
    viewer = None
    try:
        print("尝试启动MuJoCo视图器...")
        viewer = mujoco.viewer.launch_passive(
            simulator.controller.model, 
            simulator.controller.data
        )
        print("视图器启动成功！")
    except Exception as e:
        print(f"无法启动视图器: {e}")
        print("将使用无视图模式运行...")
    
    # 运行仿真
    simulator.run(viewer=viewer, max_steps=10000)
    
    # 关闭视图器
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
