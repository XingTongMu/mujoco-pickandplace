import mujoco as mj
import mujoco.viewer
import numpy as np
import time
import os

XML_PATH = os.path.join(os.getcwd(), "block_stack.xml")

def run():
    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)
    
    # 获取ID
    suction_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "suction")
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "attachment_site")
    
    # 简单的IK函数
    def get_ik_delta(target_pos, current_q):
        site_pos = data.site_xpos[site_id]
        jac = np.zeros((6, model.nv))
        mj.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        jac_arm = jac[:3, :7] 
        
        error = target_pos - site_pos
        dq = jac_arm.T @ np.linalg.inv(jac_arm @ jac_arm.T + 0.01 * np.eye(3)) @ error * 0.1
        return dq, np.linalg.norm(error)

    print("Launching stack viewer...")
    
    # 关键高度
    SAFE_Z = 0.45
    PICK_Z = 0.238     # 地面方块吸取高度
    STACK_BASE_X = 0.6
    STACK_BASE_Y = 0.0
    
    # 3个方块的初始位置 (近似)
    blocks = [
        {"pos": np.array([0.45, -0.1, 0.025]), "color": "Red"},
        {"pos": np.array([0.45, 0.0, 0.025]), "color": "Green"},
        {"pos": np.array([0.45, 0.1, 0.025]), "color": "Blue"},
    ]
    
    # 生成动作序列
    phases = []
    
    # 方块高度 = 0.05
    # 堆叠目标高度: 
    # 第1层: 地面 (z=0.025 center) -> 放置高度 ~ 0.24 (吸盘底)
    # 第2层: z=0.075 center -> 放置高度 = 0.24 + 0.05 = 0.29
    # 第3层: z=0.125 center -> 放置高度 = 0.29 + 0.05 = 0.34
    
    current_stack_z = 0.24
    
    for i, block in enumerate(blocks):
        pick_pos = np.array([block["pos"][0], block["pos"][1], PICK_Z])
        place_pos = np.array([STACK_BASE_X, STACK_BASE_Y, current_stack_z + 0.005]) # 稍微高一点点避免硬撞
        
        phases.extend([
            # 1. 移动到方块上方
            {"target": np.array([pick_pos[0], pick_pos[1], SAFE_Z]), "suction": 0.0, "desc": f"Move to {block['color']}", "tol": 0.02, "steps": 800},
            # 2. 下降
            {"target": pick_pos, "suction": 0.0, "desc": f"Descend to {block['color']}", "tol": 0.005, "steps": 800},
            # 3. 吸取
            {"target": pick_pos, "suction": 30.0, "desc": f"Suction {block['color']}", "tol": 0.005, "steps": 400},
            # 4. 抬起
            {"target": np.array([pick_pos[0], pick_pos[1], SAFE_Z]), "suction": 30.0, "desc": "Lift", "tol": 0.02, "steps": 800},
            # 5. 移动到堆叠点上方
            {"target": np.array([place_pos[0], place_pos[1], SAFE_Z]), "suction": 30.0, "desc": "Move to Stack", "tol": 0.02, "steps": 1000},
            # 6. 下降放置
            {"target": place_pos, "suction": 30.0, "desc": f"Place Layer {i+1}", "tol": 0.01, "steps": 1000},
            # 7. 松开
            {"target": place_pos, "suction": 0.0, "desc": "Release", "tol": 0.01, "steps": 500},
            # 8. 抬起 (准备下一个)
            {"target": np.array([place_pos[0], place_pos[1], SAFE_Z]), "suction": 0.0, "desc": "Retract", "tol": 0.02, "steps": 800},
        ])
        
        current_stack_z += 0.05 # 下一层增加方块高度
    
    # 增加一个回到Home的动作
    phases.append({"target": np.array([0.5, 0, 0.5]), "suction": 0.0, "desc": "Home", "tol": 0.05, "steps": 1000})

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        current_phase_idx = 0
        step_counter = 0
        
        while viewer.is_running():
            step_start = time.time()
            
            if current_phase_idx < len(phases):
                phase = phases[current_phase_idx]
                target_pos = phase["target"]
                target_suction = phase["suction"]
                
                dq, err_norm = get_ik_delta(target_pos, data.qpos[:7])
                
                data.ctrl[:7] = data.qpos[:7] + dq
                data.ctrl[suction_id] = target_suction
                
                mj.mj_step(model, data)
                step_counter += 1
                
                is_move_phase = phase["desc"] not in ["Suction", "Release"] and "Suction" not in phase["desc"]
                
                if is_move_phase:
                    if err_norm < phase["tol"]:
                        # print(f"Finished: {phase['desc']}")
                        current_phase_idx += 1
                        step_counter = 0
                else:
                    if step_counter >= phase["steps"]:
                        # print(f"Finished: {phase['desc']}")
                        current_phase_idx += 1
                        step_counter = 0
                        
                if step_counter > phase["steps"] + 800:
                     print(f"Timeout: {phase['desc']}")
                     current_phase_idx += 1
                     step_counter = 0
            else:
                mj.mj_step(model, data)

            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    run()
