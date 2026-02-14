import mujoco as mj
import mujoco.viewer
import numpy as np
import time
import os
import random
from scene_gen import SceneGenerator
from grasp_db import GraspDB

def run_experiment_loop(num_episodes=5):
    # 初始化
    db = GraspDB()
    gen = SceneGenerator()
    
    print(f"Starting {num_episodes} random grasp experiments...")
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode+1}/{num_episodes} ---")
        
        # 1. 生成场景
        scene_info = gen.generate_random_scene("temp_scene.xml")
        xml_path = os.path.join(os.getcwd(), "temp_scene.xml")
        print(f"Target: {scene_info['type']}, Size: {scene_info['size']}")
        
        # 2. 加载模型
        try:
            model = mj.MjModel.from_xml_path(xml_path)
            data = mj.MjData(model)
        except Exception as e:
            print(f"Failed to load model: {e}")
            continue
            
        # 获取ID
        suction_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "suction")
        site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "attachment_site")
        obj_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "target_object")
        
        # 简单的IK函数
        def get_ik_delta(target_pos):
            site_pos = data.site_xpos[site_id]
            jac = np.zeros((6, model.nv))
            mj.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            jac_arm = jac[:3, :7] 
            error = target_pos - site_pos
            dq = jac_arm.T @ np.linalg.inv(jac_arm @ jac_arm.T + 0.01 * np.eye(3)) @ error * 0.1
            return dq, np.linalg.norm(error)

        # 规划参数
        obj_pos = scene_info['pos']
        # 计算吸取高度：物体Z中心 + 物体顶部偏移 + 吸盘长度偏移(0.19)
        # 简易计算：物体Z位置已经是中心高度
        # 不同形状的顶部偏移不同:
        # box: sz
        # sphere: r
        # cylinder: h
        # capsule: h+r (如果是竖着)
        
        top_offset = 0
        otype = scene_info['type']
        osize = scene_info['size']
        if otype == "box": top_offset = osize[2]
        elif otype == "sphere": top_offset = osize[0]
        elif otype == "cylinder": top_offset = osize[1]
        elif otype == "capsule": top_offset = osize[1] + osize[0] # 竖放
        
        # 目标Z = 物体中心Z + 顶部偏移 + 吸盘偏移(0.19) - 压入量(0.005)
        PICK_Z = obj_pos[2] + top_offset + 0.19 - 0.005
        SAFE_Z = 0.5
        
        # 动作序列
        phases = [
            {"target": np.array([obj_pos[0], obj_pos[1], SAFE_Z]), "suction": 0.0, "desc": "Approach", "tol": 0.02, "steps": 500},
            {"target": np.array([obj_pos[0], obj_pos[1], PICK_Z]), "suction": 0.0, "desc": "Descend", "tol": 0.005, "steps": 800},
            {"target": np.array([obj_pos[0], obj_pos[1], PICK_Z]), "suction": 30.0, "desc": "Suction", "tol": 0.01, "steps": 400},
            {"target": np.array([obj_pos[0], obj_pos[1], SAFE_Z]), "suction": 30.0, "desc": "Lift", "tol": 0.02, "steps": 800},
        ]
        
        # 运行仿真
        # 为了演示效果，我们开启 Viewer，但这会阻塞，需要自动关闭或快速播放
        # 这里我们使用 passive viewer 但设置自动退出条件
        
        success = False
        final_h = 0.0
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            current_phase = 0
            step_cnt = 0
            
            while viewer.is_running():
                if current_phase >= len(phases):
                    # 实验结束，检查结果
                    obj_curr_pos = data.xpos[obj_body_id]
                    final_h = obj_curr_pos[2]
                    # 判定标准：物体被抬起到 0.3m 以上
                    if final_h > 0.3:
                        success = True
                    break
                
                phase = phases[current_phase]
                target = phase["target"]
                suction = phase["suction"]
                
                dq, err = get_ik_delta(target)
                data.ctrl[:7] = data.qpos[:7] + dq
                data.ctrl[suction_id] = suction
                
                mj.mj_step(model, data)
                viewer.sync()
                
                step_cnt += 1
                
                is_move = phase["desc"] not in ["Suction"]
                if is_move:
                    if err < phase["tol"]:
                        current_phase += 1
                        step_cnt = 0
                else:
                    if step_cnt > phase["steps"]:
                        current_phase += 1
                        step_cnt = 0
                        
                if step_cnt > phase["steps"] + 500: # Timeout
                    current_phase += 1
                    step_cnt = 0
                    
                # 稍微快一点，不sleep太久
                # time.sleep(0.001) 
        
        # 记录结果
        print(f"Result: {'Success' if success else 'Fail'} (Height: {final_h:.3f}m)")
        db.log_experiment(
            scene_info['type'], 
            scene_info['size'], 
            scene_info['pos'], 
            scene_info['color'], 
            success, 
            final_h
        )
        
        time.sleep(0.5) # 间隔
        
    # 打印最终统计
    total, succ_cnt, rate = db.get_stats()
    print(f"\n=== Final Stats ===")
    print(f"Total Experiments: {total}")
    print(f"Successes: {succ_cnt}")
    print(f"Success Rate: {rate:.1f}%")

if __name__ == "__main__":
    run_experiment_loop(5)
