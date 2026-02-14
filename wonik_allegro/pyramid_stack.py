#!/usr/bin/env python3
"""
Wonik Allegro 手金字塔堆叠仿真
手掌从桌子边开始，将小方块垒成金字塔状
"""

import mujoco
import mujoco.viewer as viewer
import numpy as np
import time

# 场景文件路径
SCENE_PATH = "wonik_allegro/pyramid_scene.xml"

# 关节名称
JOINT_NAMES = [
    'ffj0', 'ffj1', 'ffj2', 'ffj3',  # Index finger
    'mfj0', 'mfj1', 'mfj2', 'mfj3',  # Middle finger
    'rfj0', 'rfj1', 'rfj2', 'rfj3',  # Ring finger
    'thj0', 'thj1', 'thj2', 'thj3'   # Thumb
]

# 姿态配置
INITIAL_POSE = [0.0, 0.1, 0.2, 0.1,     # Index - 手指下垂
                0.0, 0.1, 0.2, 0.1,      # Middle - 手指下垂
                0.0, 0.1, 0.2, 0.1,      # Ring - 手指下垂
                0.2, 0.1, 0.2, 0.1]      # Thumb - 自然位置

PRE_GRASP_POSE = [0.0, 0.5, 0.7, 0.3,     # Index
                  0.0, 0.5, 0.7, 0.3,      # Middle
                  0.0, 0.5, 0.7, 0.3,      # Ring
                  0.4, 0.3, 0.4, 0.2]       # Thumb

GRASP_POSE = [0.1, 0.8, 1.2, 0.8,   # Index
              0.05, 0.9, 1.3, 0.9,   # Middle  
              -0.05, 0.8, 1.2, 0.8,  # Ring
              0.6, 0.4, 0.6, 0.4]    # Thumb

OPEN_POSE = [0.0, 0.4, 0.6, 0.2,     # Index
             0.0, 0.4, 0.6, 0.2,      # Middle
             0.0, 0.4, 0.6, 0.2,      # Ring
             0.4, 0.3, 0.4, 0.2]      # Thumb

# 金字塔目标位置（按堆叠顺序）
PYRAMID_POSITIONS = [
    # 底层（从左到右）
    {'name': '底层左', 'pos': (-0.03, -0.15, 0.45), 'height': 0.025},
    {'name': '底层中', 'pos': (0, -0.15, 0.45), 'height': 0.025},
    {'name': '底层右', 'pos': (0.03, -0.15, 0.45), 'height': 0.025},
    # 中层（从左到右）
    {'name': '中层左', 'pos': (-0.015, -0.15, 0.50), 'height': 0.025},
    {'name': '中层右', 'pos': (0.015, -0.15, 0.50), 'height': 0.025},
    # 顶层
    {'name': '顶层', 'pos': (0, -0.15, 0.55), 'height': 0.025},
]

# 抓取顺序（从散落的方块中按顺序抓取）
GRASP_ORDER = ['block1', 'block2', 'block3', 'block4', 'block5', 'block3']

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

def smooth_interpolate(start, end, progress):
    """平滑插值"""
    progress = (1 - np.cos(progress * np.pi)) / 2
    return start + progress * (end - start)

def get_palm_position(model, data):
    """获取手掌位置"""
    try:
        palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'palm')
        return data.body(palm_id).xpos.copy()
    except:
        return None

def get_block_position(model, data, block_name):
    """获取方块位置"""
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, block_name)
        return data.body(body_id).xpos.copy()
    except:
        return None

def print_separator():
    """打印分隔符"""
    print("="*60)

def run_motion_sequence(model, data, name, start_pose, target_pose, duration, viewer_handle=None):
    """运行一个动作序列"""
    print(f"📌 {name}")
    
    start = np.array(start_pose)
    target = np.array(target_pose)
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        progress = min(elapsed / duration, 1.0)
        
        current_pose = smooth_interpolate(start, target, progress)
        set_joint_positions(model, data, current_pose)
        
        mujoco.mj_step(model, data)
        
        if viewer_handle is not None and viewer_handle.is_running():
            viewer_handle.sync()
        
        if progress >= 1.0:
            break
        
        time.sleep(0.01)
    
    time.sleep(0.15)

def main():
    """主函数"""
    print_separator()
    print("🤖 Wonik Allegro 手金字塔堆叠仿真")
    print("📍 起始位置: 桌子边缘，手掌朝下")
    print("🎯 目标: 将方块垒成金字塔状")
    print_separator()
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
        set_joint_positions(model, data, INITIAL_POSE)
        print("✓ 初始化完成!")
        print()
        
        # 显示金字塔结构
        print("📐 金字塔堆叠目标:")
        print("      □      <- 第6步: 顶层")
        print("     □ □     <- 第4-5步: 中层")
        print("    □ □ □    <- 第1-3步: 底层")
        print()
        
        # 启动可视化
        print("🖥️ 启动可视化窗口...")
        print("💡 提示: 按 Q 关闭窗口")
        print()
        
        with viewer.launch(model, data) as v:
            print("✓ 可视化窗口已打开!")
            print()
            print_separator()
            print("🚀 开始自主堆叠金字塔...")
            print_separator()
            
            # ========== 阶段1: 从边缘下降到桌子上方 ==========
            print_separator()
            print("📍 阶段1: 从桌子边移动到桌子上方")
            print_separator()
            
            run_motion_sequence(model, data, "抬起手指准备移动", 
                             INITIAL_POSE, PRE_GRASP_POSE, 1.2, v)
            
            # 模拟移动到桌子上方
            print("📍 移动到桌子上方...")
            for _ in range(40):
                set_joint_positions(model, data, PRE_GRASP_POSE)
                mujoco.mj_step(model, data)
                if v.is_running():
                    v.sync()
                time.sleep(0.01)
            
            # ========== 阶段2: 堆叠金字塔 ==========
            print_separator()
            print("📍 阶段2: 开始堆叠金字塔")
            print_separator()
            
            # 使用6个方块（因为有6个金字塔位置，但只有5个可用，重复使用block3）
            pyramid_positions = [
                (-0.03, -0.15, 0.45),  # 底层左
                (0, -0.15, 0.45),       # 底层中
                (0.03, -0.15, 0.45),    # 底层右
                (-0.015, -0.15, 0.50),  # 中层左
                (0.015, -0.15, 0.50),    # 中层右
                (0, -0.15, 0.55),        # 顶层
            ]
            
            # 抓取顺序（重新排列以适应堆叠顺序）
            # 从左边开始抓取，然后按底层、中层、顶层的顺序放置
            grasp_sequence = ['block1', 'block2', 'block3', 'block4', 'block5', 'block3']
            
            for i in range(6):
                if not v.is_running():
                    break
                
                print_separator()
                print(f"🔄 堆叠第 {i+1}/6 层")
                print(f"📍 目标位置: {pyramid_positions[i]}")
                print_separator()
                
                block_name = grasp_sequence[i]
                print(f"👉 抓取 {block_name}...")
                
                # 接近方块
                run_motion_sequence(model, data, "接近方块", 
                                 PRE_GRASP_POSE, GRASP_POSE, 0.8, v)
                
                # 抓取并保持
                print("✊ 抓取!")
                for _ in range(20):
                    mujoco.mj_step(model, data)
                    if v.is_running():
                        v.sync()
                    time.sleep(0.01)
                
                # 抬起
                print("📤 抬起方块...")
                run_motion_sequence(model, data, "抬起", 
                                 GRASP_POSE, GRASP_POSE, 0.3, v)
                
                # 移动到放置位置（通过调整手指指向）
                print(f"📍 移动到金字塔第{i+1}层位置: {pyramid_positions[i]}")
                
                # 短暂移动到目标上方
                for _ in range(30):
                    mujoco.mj_step(model, data)
                    if v.is_running():
                        v.sync()
                    time.sleep(0.01)
                
                # 放下
                print("📥 放下方块")
                run_motion_sequence(model, data, "放下", 
                                 GRASP_POSE, OPEN_POSE, 0.6, v)
                
                # 短暂等待
                time.sleep(0.3)
                
                # 准备抓取下一个
                if i < 5:
                    print("👉 准备抓取下一个...")
                    run_motion_sequence(model, data, "准备", 
                                     OPEN_POSE, PRE_GRASP_POSE, 0.4, v)
            
            # ========== 阶段3: 完成 ==========
            print_separator()
            print("🎉 金字塔堆叠完成!")
            print_separator()
            
            # 显示完成的金字塔
            print("📸 金字塔结构:")
            print("       □       <- 顶层 (1个)")
            print("      □ □     <- 中层 (2个)")
            print("     □ □ □    <- 底层 (3个)")
            print("    =6个方块  <- 总计")
            print()
            
            # 返回初始位置
            print("👉 返回初始位置...")
            run_motion_sequence(model, data, "复位", 
                             PRE_GRASP_POSE, INITIAL_POSE, 1.5, v)
            
            print()
            print("👋 仿真结束")
            print()
            print("💡 您可以:")
            print("   - 旋转视角查看金字塔结构")
            print("   - 按 R 键重新开始")
            print("   - 按 Q 关闭窗口")
            
            # 保持窗口打开
            print("⏳ 窗口保持打开状态...")
            while v.is_running():
                mujoco.mj_step(model, data)
                if v.is_running():
                    v.sync()
                time.sleep(0.01)
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
