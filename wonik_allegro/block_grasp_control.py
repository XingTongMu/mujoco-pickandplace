#!/usr/bin/env python3
"""
Wonik Allegro 手抓取随机方块仿真
在桌面上抓取随机生成的小方块
"""

import mujoco
import mujoco.viewer as viewer
import numpy as np
import time
import random
from enum import Enum

class GraspState(Enum):
    """抓取状态机"""
    INITIALIZE = "initialize"
    SELECT_BLOCK = "select_block"
    APPROACH = "approach"
    GRASP = "grasp"
    LIFT = "lift"
    MOVE_TO_PLACE = "move_to_place"
    RELEASE = "release"
    RESET = "reset"
    COMPLETE = "complete"

class BlockGraspController:
    """方块抓取控制器"""
    
    # 关节名称映射
    JOINT_NAMES = [
        'ffj0', 'ffj1', 'ffj2', 'ffj3',  # Index finger
        'mfj0', 'mfj1', 'mfj2', 'mfj3',  # Middle finger
        'rfj0', 'rfj1', 'rfj2', 'rfj3',  # Ring finger
        'thj0', 'thj1', 'thj2', 'thj3'   # Thumb
    ]
    
    # 默认张开姿态
    DEFAULT_OPEN = {
        'ffj0': 0.0, 'ffj1': 0.4, 'ffj2': 0.6, 'ffj3': 0.2,
        'mfj0': 0.0, 'mfj1': 0.4, 'mfj2': 0.6, 'mfj3': 0.2,
        'rfj0': 0.0, 'rfj1': 0.4, 'rfj2': 0.6, 'rfj3': 0.2,
        'thj0': 0.4, 'thj1': 0.3, 'thj2': 0.4, 'thj3': 0.2
    }
    
    # 抓取姿态
    GRASP_POSE = {
        'ffj0': 0.1, 'ffj1': 0.8, 'ffj2': 1.2, 'ffj3': 0.8,
        'mfj0': 0.05, 'mfj1': 0.9, 'mfj2': 1.3, 'mfj3': 0.9,
        'rfj0': -0.05, 'rfj1': 0.8, 'rfj2': 1.2, 'rfj3': 0.8,
        'thj0': 0.6, 'thj1': 0.4, 'thj2': 0.6, 'thj3': 0.4
    }
    
    # 预抓取姿态
    PRE_GRASP_POSE = {
        'ffj0': 0.05, 'ffj1': 0.5, 'ffj2': 0.7, 'ffj3': 0.3,
        'mfj0': 0.02, 'mfj1': 0.5, 'mfj2': 0.7, 'mfj3': 0.3,
        'rfj0': -0.02, 'rfj1': 0.5, 'rfj2': 0.7, 'rfj3': 0.3,
        'thj0': 0.5, 'thj1': 0.35, 'thj2': 0.5, 'thj3': 0.3
    }
    
    def __init__(self, scene_path):
        """初始化控制器"""
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        
        # 控制参数
        self.kp = np.ones(16) * 2.0
        self.kd = np.ones(16) * 0.2
        
        # 状态管理
        self.state = GraspState.INITIALIZE
        self.state_start_time = time.time()
        self.selected_block = None
        self.block_positions = {}
        
        # 目标关节位置
        self.target_qpos = np.zeros(16)
        self.current_qpos = np.zeros(16)
        
        # 初始化
        self.initialize_blocks()
        self.reset_joints()
        
        # 获取body IDs
        self.get_body_ids()
    
    def get_body_ids(self):
        """获取所有方块的body ID"""
        self.block_body_ids = {}
        for i in range(1, 6):
            try:
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f'block{i}'
                )
                self.block_body_ids[f'block{i}'] = body_id
                print(f"✓ 找到方块 block{i}, ID: {body_id}")
            except:
                print(f"✗ 未找到方块 block{i}")
    
    def initialize_blocks(self):
        """初始化方块位置记录"""
        for i in range(1, 6):
            self.block_positions[f'block{i}'] = {
                'x': random.uniform(-0.18, 0.22),
                'y': random.uniform(-0.18, 0.18)
            }
        print("方块位置已随机初始化")
    
    def get_joint_id(self, name):
        """获取关节ID"""
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        except:
            return None
    
    def reset_joints(self):
        """重置关节到张开状态"""
        self.target_qpos = self.config_to_array(self.DEFAULT_OPEN)
        self.apply_joint_positions(self.target_qpos)
        print("关节已重置到张开状态")
    
    def config_to_array(self, config):
        """将配置字典转换为数组"""
        arr = np.zeros(16)
        for i, name in enumerate(self.JOINT_NAMES):
            if name in config:
                arr[i] = config[name]
        return arr
    
    def get_current_qpos(self):
        """获取当前关节位置"""
        positions = []
        for name in self.JOINT_NAMES:
            joint_id = self.get_joint_id(name)
            if joint_id is not None:
                positions.append(self.data.joint(joint_id).qpos)
            else:
                positions.append(0.0)
        return np.array(positions)
    
    def apply_joint_positions(self, positions):
        """应用关节位置"""
        for i, name in enumerate(self.JOINT_NAMES):
            joint_id = self.get_joint_id(name)
            if joint_id is not None and i < len(positions):
                self.data.joint(joint_id).qpos = positions[i]
    
    def set_pose(self, pose_config, duration=1.0):
        """设置目标姿态"""
        self.start_qpos = self.get_current_qpos()
        self.target_qpos = self.config_to_array(pose_config)
        self.pose_duration = duration
        self.pose_start_time = time.time()
    
    def interpolate_pose(self):
        """插值计算当前目标位置"""
        elapsed = time.time() - self.pose_start_time
        progress = min(elapsed / self.pose_duration, 1.0)
        
        # 使用平滑曲线
        smooth_progress = progress * progress * (3 - 2 * progress)
        
        return self.start_qpos + smooth_progress * (self.target_qpos - self.start_qpos)
    
    def compute_control(self):
        """计算PD控制"""
        current_qpos = self.get_current_qpos()
        current_qvel = self.get_joint_velocities()
        
        # 获取当前目标位置
        target = self.interpolate_pose()
        
        # PD控制
        q_error = target - current_qpos
        control = self.kp * q_error - self.kd * current_qvel
        
        # 应用控制
        for i, name in enumerate(self.JOINT_NAMES):
            joint_id = self.get_joint_id(name)
            if joint_id is not None and i < len(control):
                self.data.joint(joint_id).qfrc_applied = control[i]
        
        return q_error
    
    def get_joint_velocities(self):
        """获取关节速度"""
        velocities = []
        for name in self.JOINT_NAMES:
            joint_id = self.get_joint_id(name)
            if joint_id is not None:
                velocities.append(self.data.joint(joint_id).qvel)
            else:
                velocities.append(0.0)
        return np.array(velocities)
    
    def get_block_position(self, block_name):
        """获取方块位置"""
        if block_name in self.block_body_ids:
            body_id = self.block_body_ids[block_name]
            return self.data.body(body_id).xpos.copy()
        return None
    
    def select_random_block(self):
        """随机选择一个方块"""
        available_blocks = list(self.block_body_ids.keys())
        if available_blocks:
            selected = random.choice(available_blocks)
            print(f"\n选择目标方块: {selected}")
            pos = self.get_block_position(selected)
            if pos is not None:
                print(f"方块位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            return selected
        return None
    
    def step(self):
        """执行一步控制"""
        # 计算并应用控制
        self.compute_control()
        
        # MuJoCo仿真步进
        mujoco.mj_step(self.model, self.data)
    
    def run_state_machine(self):
        """运行状态机"""
        current_time = time.time()
        state_time = current_time - self.state_start_time
        
        # 根据状态执行不同动作
        if self.state == GraspState.INITIALIZE:
            print("\n=== 状态: 初始化 ===")
            self.reset_joints()
            self.selected_block = self.select_random_block()
            
            if state_time > 1.0:
                self.state = GraspState.APPROACH
                self.state_start_time = current_time
                self.set_pose(self.PRE_GRASP_POSE, 1.5)
        
        elif self.state == GraspState.APPROACH:
            if state_time > 1.5:
                print("\n=== 状态: 抓取 ===")
                self.state = GraspState.GRASP
                self.state_start_time = current_time
                self.set_pose(self.GRASP_POSE, 1.0)
        
        elif self.state == GraspState.GRASP:
            if state_time > 1.0:
                print("\n=== 状态: 抬起 ===")
                self.state = GraspState.LIFT
                self.state_start_time = current_time
        
        elif self.state == GraspState.LIFT:
            if state_time > 1.0:
                print("\n=== 状态: 移动到放置区 ===")
                self.state = GraspState.MOVE_TO_PLACE
                self.state_start_time = current_time
        
        elif self.state == GraspState.MOVE_TO_PLACE:
            if state_time > 1.5:
                print("\n=== 状态: 释放 ===")
                self.state = GraspState.RELEASE
                self.state_start_time = current_time
                self.set_pose(self.DEFAULT_OPEN, 1.0)
        
        elif self.state == GraspState.RELEASE:
            if state_time > 1.0:
                print("\n=== 状态: 复位 ===")
                self.state = GraspState.RESET
                self.state_start_time = current_time
                self.set_pose(self.DEFAULT_OPEN, 1.5)
        
        elif self.state == GraspState.RESET:
            if state_time > 1.5:
                print("\n=== 状态: 完成 ===")
                self.state = GraspState.COMPLETE
                self.state_start_time = current_time
        
        elif self.state == GraspState.COMPLETE:
            # 等待后重新开始
            if state_time > 2.0:
                print("\n" + "="*50)
                print("重新开始新的抓取任务...")
                print("="*50)
                self.state = GraspState.INITIALIZE
                self.state_start_time = current_time
                self.selected_block = self.select_random_block()


def main():
    """主函数"""
    print("="*60)
    print("Wonik Allegro 手抓取随机方块仿真")
    print("="*60)
    print()
    
    # 场景文件路径
    scene_path = "wonik_allegro/table_scene.xml"
    
    try:
        # 创建控制器
        print("初始化仿真...")
        controller = BlockGraspController(scene_path)
        print("✓ 初始化成功!")
        print()
        
        # 启动可视化
        print("启动可视化窗口...")
        print("提示: 按 Q 退出")
        print()
        
        with viewer.launch(controller.model, controller.data) as v:
            print("✓ 可视化窗口已打开!")
            print()
            
            step_count = 0
            last_print_time = time.time()
            
            while v.is_running():
                # 运行状态机
                controller.run_state_machine()
                
                # 执行控制
                controller.step()
                
                step_count += 1
                
                # 每秒打印一次状态
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    print(f"步骤 {step_count:5d} | "
                          f"状态: {controller.state.value:12s} | "
                          f"时间: {current_time - controller.state_start_time:.1f}s")
                    
                    # 显示所有方块位置
                    if step_count % 5 == 0:
                        for block_name in controller.block_body_ids.keys():
                            pos = controller.get_block_position(block_name)
                            if pos is not None:
                                print(f"         {block_name}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                    
                    last_print_time = current_time
            
            print()
            print("仿真结束")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
