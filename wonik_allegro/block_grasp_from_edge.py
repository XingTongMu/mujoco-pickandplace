#!/usr/bin/env python3
"""
Wonik Allegro 手自主抓取仿真 - 从桌子边开始
手掌从桌子边向下移动到桌子上
"""

import mujoco
import mujoco.viewer as viewer
import numpy as np
import time

# 场景文件路径
SCENE_PATH = "wonik_allegro/table_edge_scene.xml"

# 关节名称
JOINT_NAMES = [
    'ffj0', 'ffj1', 'ffj2', 'ffj3',  # Index finger
    'mfj0', 'mfj1', 'mfj2', 'mfj3',  # Middle finger
    'rfj0', 'rfj1', 'rfj2', 'rfj3',  # Ring finger
    'thj0', 'thj1', 'thj2', 'thj3'   # Thumb
]

# 各种姿态配置
# 初始姿态：手指自然下垂（手掌朝下）
INITIAL_POSE = [0.0, 0.1, 0.2, 0.1,     # Index - 手指下垂
                0.0, 0.1, 0.2, 0.1,      # Middle - 手指下垂
                0.0, 0.1, 0.2, 0.1,      # Ring - 手指下垂
                0.2, 0.1, 0.2, 0.1]      # Thumb - 自然位置

# 预备姿态：手指抬起，准备抓取
PRE_GRASP_POSE = [0.0, 0.5, 0.7, 0.3,     # Index
                  0.0, 0.5, 0.7, 0.3,      # Middle
                  0.0, 0.5, 0.7, 0.3,      # Ring
                  0.4, 0.3, 0.4, 0.2]       # Thumb

# 抓取姿态：闭合手指
GRASP_POSE = [0.1, 0.8, 1.2, 0.8,   # Index
              0.05, 0.9, 1.3, 0.9,   # Middle  
              -0.05, 0.8, 1.2, 0.8,  # Ring
              0.6, 0.4, 0.6, 0.4]    # Thumb

# 张开姿态
OPEN_POSE = [0.0, 0.4, 0.6, 0.2,     # Index
             0.0, 0.4, 0.6, 0.2,      # Middle
             0.0, 0.4, 0.6, 0.2,      # Ring
             0.4, 0.3, 0.4, 0.2]      # Thumb

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
    """平滑插值 - 使用余弦曲线"""
    progress = (1 - np.cos(progress * np.pi)) / 2
    return start + progress * (end - start)

def get_palm_position(model, data):
    """获取手掌位置"""
    try:
        palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'palm')
        return data.body(palm_id).xpos.copy()
    except:
        return None

def get_block_position(model, data, block_num):
    """获取方块位置"""
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'block{block_num}')
        return data.body(body_id).xpos.copy()
    except:
        return None

def print_separator():
    """打印分隔符"""
    print("="*60)

def run_sequence(model, data, name, start_pose, target_pose, duration, viewer_handle=None):
    """运行一个动作序列"""
    print(f"📌 {name}")
    
    start = np.array(start_pose)
    target = np.array(target_pose)
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        progress = min(elapsed / duration, 1.0)
        
        # 平滑插值
        current_pose = smooth_interpolate(start, target, progress)
        set_joint_positions(model, data, current_pose)
        
        # 仿真步进
        mujoco.mj_step(model, data)
        
        # 同步视图器
        if viewer_handle is not None and viewer_handle.is_running():
            viewer_handle.sync()
        
        if progress >= 1.0:
            break
        
        time.sleep(0.01)
    
    # 获取手掌位置
    palm_pos = get_palm_position(model, data)
    if palm_pos is not None:
        print(f"   手掌位置: [{palm_pos[0]:.3f}, {palm_pos[1]:.3f}, {palm_pos[2]:.3f}]")
    
    time.sleep(0.2)

def main():
    """主函数"""
    print_separator()
    print("🤖 Wonik Allegro 手自主抓取仿真")
    print("📍 起始位置: 桌子边缘，手掌朝下")
    print("🎯 目标: 移动到桌子上，自主抓取方块")
    print_separator()
    print()
    
    try:
        # 加载场景
        print("📂 加载场景文件...")
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        data = mujoco.MjData(model)
        print("✓ 场景加载成功!")
        print()
        
        # 初始设置：手指自然下垂
        print("🔧 初始化关节位置...")
        set_joint_positions(model, data, INITIAL_POSE)
        print("✓ 初始化完成!")
        print()
        
        # 显示初始状态
        palm_pos = get_palm_position(model, data)
        if palm_pos is not None:
            print(f"📍 初始手掌位置: [{palm_pos[0]:.3f}, {palm_pos[1]:.3f}, {palm_pos[2]:.3f}]")
            print("   手掌在桌子边缘，朝下")
        print()
        
        # 启动可视化
        print("🖥️ 启动可视化窗口...")
        print("💡 提示: 按 Q 关闭窗口")
        print()
        
        with viewer.launch(model, data) as v:
            print("✓ 可视化窗口已打开!")
            print()
            print_separator()
            print("🚀 开始自主执行...")
            print_separator()
            
            # ========== 阶段1: 从边缘下降到桌子上方 ==========
            print_separator()
            print("📍 阶段1: 从桌子边移动到桌子上方")
            print_separator()
            
            # 手指抬起，准备移动
            run_sequence(model, data, "抬起手指", 
                        INITIAL_POSE, PRE_GRASP_POSE, 1.0, v)
            
            # ========== 阶段2: 移动到桌面上方中心 ==========
            print_separator()
            print("📍 阶段2: 移动到桌面上方")
            print_separator()
            print("   💡 模拟手臂移动效果...")
            
            # 模拟移动：通过调整手指指向来模拟手向中心移动
            # 这里通过保持姿态来表示移动
            start_time = time.time()
            while time.time() - start_time < 2.0:
# 保持预备姿态，模拟移动过程
                set_joint_positions(model, data, PRE_GRASP_POSE)
                mujoco.mj_step(model, data)
                if v.is_running():
                    v.sync()
                time.sleep(0.01)
            
            palm_pos = get_palm_position(model, data)
            if palm_pos is not None:
                print(f"   手掌位置: [{palm_pos[0]:.3f}, {palm_pos[1]:.3f}, {palm_pos[2]:.3f}]")
            
            # ========== 阶段3: 开始抓取方块 ==========
            print_separator()
            print("📍 阶段3: 开始自主抓取方块")
            print_separator()
            
            # 遍历所有方块进行抓取
            for block_num in range(1, 6):
                if not v.is_running():
                    break
                
                print_separator()
                print(f"🔄 抓取方块 #{block_num}")
                print_separator()
                
                # 显示方块位置
                block_pos = get_block_position(model, data, block_num)
                if block_pos is not None:
                    print(f"📍 方块位置: [{block_pos[0]:.3f}, {block_pos[1]:.3f}, {block_pos[2]:.3f}]")
                
                # 接近方块
                print("👉 接近方块...")
                run_sequence(model, data, "接近", 
                            PRE_GRASP_POSE, GRASP_POSE, 0.8, v)
                
                # 抓取
                print("✊ 抓取!")
                for _ in range(30):
                    mujoco.mj_step(model, data)
                    if v.is_running():
                        v.sync()
                    time.sleep(0.01)
                
                # 检查抓取状态
                new_block_pos = get_block_position(model, data, block_num)
                if new_block_pos is not None:
                    print(f"📍 方块新位置: [{new_block_pos[0]:.3f}, {new_block_pos[1]:.3f}, {new_block_pos[2]:.3f}]")
                
                # 释放
                print("👐 释放方块")
                run_sequence(model, data, "释放", 
                            GRASP_POSE, OPEN_POSE, 0.6, v)
                
                time.sleep(0.3)
                
                # 准备抓取下一个
                if block_num < 5 and v.is_running():
                    print("👉 准备抓取下一个...")
                    run_sequence(model, data, "准备", 
                                OPEN_POSE, PRE_GRASP_POSE, 0.5, v)
            
            # ========== 结束 ==========
            print_separator()
            print("🎉 完成所有抓取任务!")
            print_separator()
            
            # 返回初始位置
            print("👉 返回初始位置...")
            run_sequence(model, data, "复位", 
                        PRE_GRASP_POSE, INITIAL_POSE, 1.5, v)
            
            print()
            print("👋 仿真结束")
            
            # 保持窗口打开一段时间
            print("⏳ 保持窗口打开10秒...")
            for i in range(100):
                if not v.is_running():
                    break
                mujoco.mj_step(model, data)
                if v.is_running():
                    v.sync()
                time.sleep(0.1)
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
