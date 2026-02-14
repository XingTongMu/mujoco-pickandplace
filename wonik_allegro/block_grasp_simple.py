#!/usr/bin/env python3
"""
Wonik Allegro 手自主抓取方块仿真 - 简化稳定版本
"""

import mujoco
import mujoco.viewer as viewer
import numpy as np
import time

# 场景文件路径
SCENE_PATH = "wonik_allegro/table_scene.xml"

# 关节名称
JOINT_NAMES = [
    'ffj0', 'ffj1', 'ffj2', 'ffj3',  # Index finger
    'mfj0', 'mfj1', 'mfj2', 'mfj3',  # Middle finger
    'rfj0', 'rfj1', 'rfj2', 'rfj3',  # Ring finger
    'thj0', 'thj1', 'thj2', 'thj3'   # Thumb
]

# 抓取姿态配置
GRASP_POSE = [0.1, 0.8, 1.2, 0.8,   # Index
              0.05, 0.9, 1.3, 0.9,   # Middle  
              -0.05, 0.8, 1.2, 0.8,  # Ring
              0.6, 0.4, 0.6, 0.4]    # Thumb

# 张开姿态配置
OPEN_POSE = [0.0, 0.4, 0.6, 0.2,     # Index
             0.0, 0.4, 0.6, 0.2,      # Middle
             0.0, 0.4, 0.6, 0.2,      # Ring
             0.4, 0.3, 0.4, 0.2]      # Thumb

# 中间姿态
MID_POSE = [0.05, 0.5, 0.7, 0.3,     # Index
            0.02, 0.5, 0.7, 0.3,     # Middle
            -0.02, 0.5, 0.7, 0.3,    # Ring
            0.5, 0.35, 0.5, 0.3]     # Thumb

def get_joint_id(model, name):
    """获取关节ID"""
    try:
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    except:
        return None

def set_joint_positions(model, data, positions):
    """直接设置关节位置"""
    for i, name in enumerate(JOINT_NAMES):
        joint_id = get_joint_id(model, name)
        if joint_id is not None and i < len(positions):
            data.joint(joint_id).qpos = positions[i]

def get_joint_positions(model, data):
    """获取当前关节位置"""
    positions = []
    for name in JOINT_NAMES:
        joint_id = get_joint_id(model, name)
        if joint_id is not None:
            positions.append(data.joint(joint_id).qpos)
        else:
            positions.append(0.0)
    return np.array(positions)

def smooth_interpolate(start, end, progress):
    """平滑插值"""
    # 使用余弦插值
    progress = (1 - np.cos(progress * np.pi)) / 2
    return start + progress * (end - start)

def get_block_position(model, data, block_num):
    """获取方块位置"""
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'block{block_num}')
        return data.body(body_id).xpos.copy()
    except:
        return None

def run_grasp_sequence(model, data, block_num, phase_duration=1.0):
    """执行完整的抓取序列"""
    start_time = time.time()
    
    # 阶段1: 从张开到预抓取
    print(f"\n🔄 开始抓取 block{block_num}")
    start_pose = np.array(OPEN_POSE)
    target_pose = np.array(MID_POSE)
    
    while True:
        elapsed = time.time() - start_time
        progress = min(elapsed / phase_duration, 1.0)
        
        # 平滑插值
        current_pose = smooth_interpolate(start_pose, target_pose, progress)
        set_joint_positions(model, data, current_pose)
        
        # 仿真步进
        mujoco.mj_step(model, data)
        
        if progress >= 1.0:
            break
        
        time.sleep(0.01)
    
    # 短暂停顿
    time.sleep(0.2)
    
    # 阶段2: 闭合手指抓取
    start_time = time.time()
    start_pose = np.array(MID_POSE)
    target_pose = np.array(GRASP_POSE)
    
    while True:
        elapsed = time.time() - start_time
        progress = min(elapsed / phase_duration, 1.0)
        
        current_pose = smooth_interpolate(start_pose, target_pose, progress)
        set_joint_positions(model, data, current_pose)
        
        mujoco.mj_step(model, data)
        
        if progress >= 1.0:
            break
        
        time.sleep(0.01)
    
    # 阶段3: 保持抓取
    print(f"✊ 抓取 block{block_num}，保持...")
    for _ in range(50):
        mujoco.mj_step(model, data)
        time.sleep(0.01)
    
    # 阶段4: 张开手指释放
    print(f"👐 释放 block{block_num}")
    start_time = time.time()
    start_pose = np.array(GRASP_POSE)
    target_pose = np.array(OPEN_POSE)
    
    while True:
        elapsed = time.time() - start_time
        progress = min(elapsed / phase_duration, 1.0)
        
        current_pose = smooth_interpolate(start_pose, target_pose, progress)
        set_joint_positions(model, data, current_pose)
        
        mujoco.mj_step(model, data)
        
        if progress >= 1.0:
            break
        
        time.sleep(0.01)
    
    # 短暂停顿
    time.sleep(0.3)

def main():
    """主函数"""
    print("="*60)
    print("🤖 Wonik Allegro 手自主抓取仿真")
    print("="*60)
    print()
    print("场景: 桌子上有5个彩色方块")
    print("功能: 自主抓取并放置方块")
    print()
    
    try:
        # 加载场景
        print("📂 加载场景文件...")
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        data = mujoco.MjData(model)
        print("✓ 场景加载成功!")
        print()
        
        # 初始化关节
        print("🔧 初始化关节位置...")
        set_joint_positions(model, data, OPEN_POSE)
        print("✓ 初始化完成!")
        print()
        
        # 启动可视化
        print("🖥️ 启动可视化窗口...")
        print("💡 提示: 按 Q 关闭窗口")
        print()
        
        with viewer.launch(model, data) as v:
            print("✓ 可视化窗口已打开!")
            print()
            print("="*60)
            print("🚀 开始自主抓取序列...")
            print("="*60)
            
            step_count = 0
            last_status_time = time.time()
            
            while v.is_running():
                # 遍历所有方块进行抓取
                for block_num in range(1, 6):
                    # 检查窗口是否还开着
                    if not v.is_running():
                        break
                    
                    # 执行抓取序列
                    run_grasp_sequence(model, data, block_num, phase_duration=0.8)
                    
                    # 显示方块位置
                    pos = get_block_position(model, data, block_num)
                    if pos is not None:
                        print(f"📍 block{block_num} 位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                
                print()
                print("="*60)
                print("🔄 完成一轮抓取，重新开始...")
                print("="*60)
                print()
                
                step_count += 1
                
                # 定期打印状态
                current_time = time.time()
                if current_time - last_status_time >= 5.0:
                    print(f"📊 运行轮次: {step_count}")
                    last_status_time = current_time
            
            print()
            print("👋 仿真结束")
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
