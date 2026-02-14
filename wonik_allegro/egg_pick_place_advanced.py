#!/usr/bin/env python3
"""
Wonik Allegro Hand - 高级鸡蛋抓取与放置仿真
增强功能：力控制、轨迹规划、状态可视化
"""

import mujoco
import numpy as np
from enum import Enum
import time
from typing import Dict, List, Tuple, Optional
import threading


class GripperState(Enum):
    """夹爪状态"""
    OPEN = "open"
    PARTIAL = "partial"
    GRASP = "grasp"
    RELEASE = "release"


class MotionPhase(Enum):
    """运动阶段"""
    INITIALIZE = 0          # 初始化
    APPROACH = 1           # 接近目标
    PRE_GRASP = 2          # 预抓取位置
    GRASPING = 3          # 抓取
    LIFTING = 4           # 抬起
    TRANSPORTING = 5      # 移动
    PLACING = 6           # 放置
    RELEASING = 7         # 释放
    WITHDRAWING = 8       # 撤回
    RESETTING = 9         # 复位
    COMPLETE = 10         # 完成


class TrajectoryPlanner:
    """轨迹规划器"""
    
    def __init__(self):
        """初始化轨迹规划器"""
        self.velocity_limit = 2.0      # 速度限制
        self.acceleration_limit = 5.0  # 加速度限制
    
    def generate_trapezoidal_profile(
        self, 
        start: float, 
        end: float, 
        duration: float,
        current_time: float,
        start_time: float
    ) -> float:
        """生成梯形速度轨迹"""
        # 归一化时间 (0-1)
        if duration <= 0:
            return end
        
        t = (current_time - start_time) / duration
        t = np.clip(t, 0, 1)
        
        # 梯形速度曲线
        t_acc = 0.2  # 加速阶段比例
        t_dec = 0.2  # 减速阶段比例
        
        if t < t_acc:
            # 加速阶段
            progress = 0.5 * (t / t_acc) ** 2
        elif t < (1 - t_dec):
            # 匀速阶段
            progress = t_acc * 0.5 + (t - t_acc) / (1 - t_acc - t_dec) * (1 - t_acc - t_dec)
        else:
            # 减速阶段
            remaining = (t - (1 - t_dec)) / t_dec
            progress = 1 - 0.5 * (1 - remaining) ** 2
        
        return start + progress * (end - start)
    
    def smooth_interpolation(
        self, 
        start: np.ndarray, 
        end: np.ndarray, 
        progress: float
    ) -> np.ndarray:
        """平滑插值（使用S曲线）"""
        # Sigmoid函数形式的平滑
        smooth_progress = 1 / (1 + np.exp(-10 * (progress - 0.5)))
        return start + smooth_progress * (end - start)


class WonikAllegroController:
    """Wonik Allegro高级控制器"""
    
    # 关节名称到索引的映射
    JOINT_MAP = {
        'ffj0': 0, 'ffj1': 1, 'ffj2': 2, 'ffj3': 3,  # Index finger
        'mfj0': 4, 'mfj1': 5, 'mfj2': 6, 'mfj3': 7,  # Middle finger
        'rfj0': 8, 'rfj1': 9, 'rfj2': 10, 'rfj3': 11, # Ring finger
        'thj0': 12, 'thj1': 13, 'thj2': 14, 'thj3': 15 # Thumb
    }
    
    # 默认关节配置
    DEFAULT_OPEN = {
        'ffj0': 0.0, 'ffj1': 0.4, 'ffj2': 0.6, 'ffj3': 0.2,
        'mfj0': 0.0, 'mfj1': 0.4, 'mfj2': 0.6, 'mfj3': 0.2,
        'rfj0': 0.0, 'rfj1': 0.4, 'rfj2': 0.6, 'rfj3': 0.2,
        'thj0': 0.4, 'thj1': 0.3, 'thj2': 0.4, 'thj3': 0.2
    }
    
    DEFAULT_GRASP = {
        'ffj0': 0.15, 'ffj1': 0.9, 'ffj2': 1.3, 'ffj3': 1.0,
        'mfj0': 0.05, 'mfj1': 1.0, 'mfj2': 1.4, 'mfj3': 1.1,
        'rfj0': -0.05, 'rfj1': 0.9, 'rfj2': 1.3, 'rfj3': 1.0,
        'thj0': 0.7, 'thj1': 0.5, 'thj2': 0.8, 'thj3': 0.5
    }
    
    def __init__(self, xml_path: str):
        """初始化控制器"""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 控制器参数
        self.kp = np.ones(16) * 2.0   # 比例增益
        self.kd = np.ones(16) * 0.2  # 微分增益
        
        # 轨迹规划器
        self.planner = TrajectoryPlanner()
        
        # 状态管理
        self.phase =MotionPhase.INITIALIZE
        self.phase_start_time = time.time()
        self.current_gripper_state = GripperState.OPEN
        
        # 目标配置
        self.target_qpos = np.zeros(16)
        self.start_qpos = np.zeros(16)
        self.phase_start_qpos = np.zeros(16)
        
        # 获取鸡蛋body ID
        try:
            self.egg_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, 'egg'
            )
        except:
            self.egg_body_id = None
            print("警告: 未找到鸡蛋body")
        
        # 获取手掌body ID（用于位置控制）
        try:
            self.palm_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, 'palm'
            )
        except:
            self.palm_body_id = None
        
        # 初始化
        self.initialize_joints()
    
    def initialize_joints(self):
        """初始化关节到默认张开位置"""
        self.target_qpos = self._config_to_array(self.DEFAULT_OPEN)
        self.start_qpos = self.target_qpos.copy()
        self.phase_start_qpos = self.target_qpos.copy()
        
        # 设置初始位置
        for name, idx in self.JOINT_MAP.items():
            if name in self.DEFAULT_OPEN:
                joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                self.data.joint(joint_id).qpos = self.DEFAULT_OPEN[name]
    
    def _config_to_array(self, config: Dict[str, float]) -> np.ndarray:
        """将字典配置转换为数组"""
        arr = np.zeros(16)
        for name, value in config.items():
            if name in self.JOINT_MAP:
                arr[self.JOINT_MAP[name]] = value
        return arr
    
    def set_target_pose(self, config: Dict[str, float], duration: float = 1.0):
        """设置目标姿态"""
        self.start_qpos = self.get_current_qpos()
        self.target_qpos = self._config_to_array(config)
        self.planner_duration = duration
    
    def get_current_qpos(self) -> np.ndarray:
        """获取当前关节位置"""
        positions = []
        for name in self.JOINT_MAP.keys():
            try:
                joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                positions.append(self.data.joint(joint_id).qpos)
            except:
                positions.append(0.0)
        return np.array(positions)
    
    def get_current_qvel(self) -> np.ndarray:
        """获取当前关节速度"""
        velocities = []
        for name in self.JOINT_MAP.keys():
            try:
                joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                velocities.append(self.data.joint(joint_id).qvel)
            except:
                velocities.append(0.0)
        return np.array(velocities)
    
    def compute_control(self):
        """计算控制量（PD控制）"""
        current_qpos = self.get_current_qpos()
        current_qvel = self.get_current_qvel()
        
        # 计算位置误差
        q_error = self.target_qpos - current_qpos
        
        # PD控制律
        control = self.kp * q_error - self.kd * current_qvel
        
        # 应用控制
        for i, name in enumerate(self.JOINT_MAP.keys()):
            try:
                joint_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
                self.data.joint(joint_id).qfrc_applied = control[i]
            except:
                pass
        
        return q_error, current_qvel
    
    def get_egg_position(self) -> Optional[np.ndarray]:
        """获取鸡蛋位置"""
        if self.egg_body_id is None:
            return None
        return self.data.body(self.egg_body_id).xpos.copy()
    
    def get_egg_orientation(self) -> Optional[np.ndarray]:
        """获取鸡蛋姿态"""
        if self.egg_body_id is None:
            return None
        return self.data.body(self.egg_body_id).xquat.copy()
    
    def get_palm_position(self) -> Optional[np.ndarray]:
        """获取手掌位置"""
        if self.palm_body_id is None:
            return None
        return self.data.body(self.palm_body_id).xpos.copy()


class PickPlaceManager:
    """抓取放置管理器"""
    
    PHASE_DURATIONS = {
        MotionPhase.INITIALIZE: 1.0,
        MotionPhase.APPROACH: 2.0,
        MotionPhase.PRE_GRASP: 0.5,
        MotionPhase.GRASPING: 1.0,
        MotionPhase.LIFTING: 1.5,
        MotionPhase.TRANSPORTING: 2.0,
        MotionPhase.PLACING: 1.0,
        MotionPhase.RELEASING: 1.0,
        MotionPhase.WITHDRAWING: 1.5,
        MotionPhase.RESETTING: 1.5,
        MotionPhase.COMPLETE: float('inf')
    }
    
    PHASE_CONFIGURATIONS = {
        MotionPhase.INITIALIZE: WonikAllegroController.DEFAULT_OPEN,
        MotionPhase.APPROACH: WonikAllegroController.DEFAULT_OPEN,
        MotionPhase.PRE_GRASP: {
            'ffj0': 0.1, 'ffj1': 0.6, 'ffj2': 0.8, 'ffj3': 0.4,
            'mfj0': 0.05, 'mfj1': 0.6, 'mfj2': 0.8, 'mfj3': 0.4,
            'rfj0': -0.05, 'rfj1': 0.6, 'rfj2': 0.8, 'rfj3': 0.4,
            'thj0': 0.5, 'thj1': 0.4, 'thj2': 0.5, 'thj3': 0.3
        },
        MotionPhase.GRASPING: WonikAllegroController.DEFAULT_GRASP,
        MotionPhase.LIFTING: {
            'ffj0': 0.1, 'ffj1': 0.8, 'ffj2': 1.1, 'ffj3': 0.9,
            'mfj0': 0.05, 'mfj1': 0.8, 'mfj2': 1.2, 'mfj3': 1.0,
            'rfj0': -0.05, 'rfj1': 0.8, 'ffj2': 1.1, 'rfj3': 0.9,
            'thj0': 0.6, 'thj1': 0.5, 'thj2': 0.7, 'thj3': 0.4
        },
        MotionPhase.TRANSPORTING: WonikAllegroController.DEFAULT_GRASP,
        MotionPhase.PLACING: WonikAllegroController.DEFAULT_GRASP,
        MotionPhase.RELEASING: WonikAllegroController.DEFAULT_OPEN,
        MotionPhase.WITHDRAWING: WonikAllegroController.DEFAULT_OPEN,
        MotionPhase.RESETTING: WonikAllegroController.DEFAULT_OPEN,
        MotionPhase.COMPLETE: WonikAllegroController.DEFAULT_OPEN
    }
    
    def __init__(self, xml_path: str):
        """初始化管理器"""
        self.controller = WonikAllegroController(xml_path)
        self.start_time = time.time()
        self.simulation_speed = 1.0
        self.auto_reset = True
        self.is_paused = False
    
    def get_elapsed_time(self) -> float:
        """获取仿真经过时间"""
        return time.time() - self.start_time
    
    def get_phase_elapsed_time(self) -> float:
        """获取当前阶段经过时间"""
        return self.get_elapsed_time() - self.controller.phase_start_time
    
    def get_phase_progress(self) -> float:
        """获取当前阶段进度 (0-1)"""
        duration = self.PHASE_DURATIONS.get(
            self.controller.phase, 1.0
        )
        if duration == float('inf'):
            return 1.0
        return min(self.get_phase_elapsed_time() / duration, 1.0)
    
    def set_phase(self, new_phase: MotionPhase):
        """设置新阶段"""
        print(f"切换到阶段: {new_phase.name}")
        self.controller.phase = new_phase
        self.controller.phase_start_time = self.get_elapsed_time()
        
        # 获取目标配置
        if new_phase in self.PHASE_CONFIGURATIONS:
            config = self.PHASE_CONFIGURATIONS[new_phase]
            duration = self.PHASE_DURATIONS.get(new_phase, 1.0)
            self.controller.set_target_pose(config, duration)
    
    def update_phase(self):
        """更新阶段"""
        progress = self.get_phase_progress()
        
        if self.controller.phase == MotionPhase.INITIALIZE:
            if progress >= 1.0:
                self.set_phase(MotionPhase.APPROACH)
        
        elif self.controller.phase == MotionPhase.APPROACH:
            if progress >= 1.0:
                self.set_phase(MotionPhase.PRE_GRASP)
        
        elif self.controller.phase == MotionPhase.PRE_GRASP:
            if progress >= 1.0:
                self.set_phase(MotionPhase.GRASPING)
        
        elif self.controller.phase == MotionPhase.GRASPING:
            if progress >= 1.0:
                self.set_phase(MotionPhase.LIFTING)
        
        elif self.controller.phase == MotionPhase.LIFTING:
            if progress >= 1.0:
                self.set_phase(MotionPhase.TRANSPORTING)
        
        elif self.controller.phase == MotionPhase.TRANSPORTING:
            if progress >= 1.0:
                self.set_phase(MotionPhase.PLACING)
        
        elif self.controller.phase == MotionPhase.PLACING:
            if progress >= 1.0:
                self.set_phase(MotionPhase.RELEASING)
        
        elif self.controller.phase == MotionPhase.RELEASING:
            if progress >= 1.0:
                self.set_phase(MotionPhase.WITHDRAWING)
        
        elif self.controller.phase == MotionPhase.WITHDRAWING:
            if progress >= 1.0:
                self.set_phase(MotionPhase.RESETTING)
        
        elif self.controller.phase == MotionPhase.RESETTING:
            if progress >= 1.0:
                self.set_phase(MotionPhase.COMPLETE)
        
        elif self.controller.phase == MotionPhase.COMPLETE:
            # 自动重置
            if self.auto_reset and self.get_phase_elapsed_time() > 3.0:
                self.reset()
    
    def reset(self):
        """重置仿真"""
        print("重置仿真...")
        self.start_time = time.time()
        self.controller = WonikAllegroController(
            self.controller.model.fname
        )
        self.set_phase(MotionPhase.INITIALIZE)
    
    def step(self):
        """执行一步仿真"""
        if self.is_paused:
            return
        
        # 更新阶段
        self.update_phase()
        
        # 计算并应用控制
        self.controller.compute_control()
        
        # MuJoCo仿真步进
        mujoco.mj_step(self.controller.model, self.controller.data)
    
    def get_status(self) -> Dict:
        """获取当前状态"""
        egg_pos = self.controller.get_egg_position()
        palm_pos = self.controller.get_palm_position()
        
        return {
            'phase': self.controller.phase.name,
            'progress': self.get_phase_progress(),
            'egg_position': egg_pos.tolist() if egg_pos is not None else None,
            'palm_position': palm_pos.tolist() if palm_pos is not None else None,
            'elapsed_time': self.get_elapsed_time(),
            'is_paused': self.is_paused
        }


def create_viewer_config() -> str:
    """创建视图器配置脚本"""
    config_script = """
<script>
// 自定义视图器配置
document.addEventListener('keydown', function(e) {
    if (e.key === ' ') {
        // 空格键暂停/继续
        console.log('Toggle pause');
    } else if (e.key === 'r') {
        // R键重置
        console.log('Reset');
    } else if (e.key === '1') {
        // 1键视角1
        console.log('Camera 1');
    } else if (e.key === '2') {
        // 2键视角2
        console.log('Camera 2');
    }
});
</script>
"""
    return config_script


def main():
    """主函数"""
    print("=" * 60)
    print("Wonik Allegro Hand - 高级鸡蛋抓取放置仿真")
    print("=" * 60)
    print("功能说明:")
    print("  - 自动执行抓取和放置序列")
    print("  - 包含完整的运动阶段管理")
    print("  - 支持暂停、重置等交互控制")
    print("  - 实时显示状态信息")
    print()
    print("控制:")
    print("  - Space: 暂停/继续")
    print("  - R: 重置仿真")
    print("  - Q: 退出")
    print("=" * 60)
    
    # 场景文件路径
    scene_path = "wonik_allegro/egg_pick_place_scene.xml"
    
    # 创建仿真管理器
    manager = PickPlaceManager(scene_path)
    manager.auto_reset = True
    
    # 创建视图器
    viewer = None
    try:
        print("启动MuJoCo视图器...")
        viewer = mujoco.viewer.launch_passive(
            manager.controller.model,
            manager.controller.data
        )
        print("视图器启动成功!")
        print()
        
        # 设置视图器选项
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        
    except Exception as e:
        print(f"无法启动视图器: {e}")
        print("将使用无视图模式运行...")
        viewer = None
    
    # 仿真循环
    try:
        step_count = 0
        last_status_time = 0
        
        while True:
            # 执行仿真步
            manager.step()
            
            # 同步视图器
            if viewer is not None:
                viewer.sync()
            
            step_count += 1
            
            # 每秒打印一次状态
            current_time = manager.get_elapsed_time()
            if current_time - last_status_time >= 1.0:
                status = manager.get_status()
                print(f"步骤 {step_count:5d} | "
                      f"阶段: {status['phase']:12s} | "
                      f"进度: {status['progress']:.1%} | "
                      f"时间: {status['elapsed_time']:.1f}s")
                
                if status['egg_position'] is not None:
                    print(f"                 | "
                          f"鸡蛋位置: [{status['egg_position'][0]:.3f}, "
                          f"{status['egg_position'][1]:.3f}, "
                          f"{status['egg_position'][2]:.3f}]")
                
                last_status_time = current_time
            
            # 控制仿真速度
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n仿真被用户中断")
    except Exception as e:
        print(f"\n仿真出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        if viewer is not None:
            viewer.close()
        print("\n仿真结束")


if __name__ == "__main__":
    main()
