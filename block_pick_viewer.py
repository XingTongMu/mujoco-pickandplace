import mujoco as mj
import mujoco.viewer
import numpy as np
import time
import os

XML_PATH = os.path.join(os.getcwd(), "block_pick.xml")

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
        jac_arm = jac[:3, :7] # 只看位置, 前7个关节
        
        error = target_pos - site_pos
        # 阻尼最小二乘
        dq = jac_arm.T @ np.linalg.inv(jac_arm @ jac_arm.T + 0.01 * np.eye(3)) @ error * 0.1
        return dq, np.linalg.norm(error)

    print("Launching viewer...")
    
    # 计算目标抓取高度
    # 块中心高度 = 0.025 (方块半高)
    # 吸盘接触点偏移 = 0.18 (杆) + 0.01 (接触垫中心到接触面) = 0.19 
    # attachment_site 到 gripper_root = 0 (aligned)
    # gripper_root 到 suction_cup = 0.18
    # suction_cup 到 suction_point = 0.01
    # Total Offset = 0.18 + 0.01 = 0.19
    # 目标 attachment_site Z = 0.05 (块顶) + 0.19 = 0.24
    
    # 实际上我们希望压紧一点点，所以目标设为 0.235
    PICK_Z = 0.238
    SAFE_Z = 0.45
    PLACE_Z = 0.24 # 放置时稍高
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        # 动作序列定义
        # (目标位置, 吸力值, 描述, 精度阈值, 最大步数)
        # 吸力值: 0=off, 30=max
        phases = [
            # 1. 移动到方块上方
            {"target": np.array([0.5, 0, SAFE_Z]), "suction": 0.0, "desc": "Approaching", "tol": 0.01, "steps": 1000},
            # 2. 下降接触
            {"target": np.array([0.5, 0, PICK_Z]), "suction": 0.0, "desc": "Descending", "tol": 0.005, "steps": 1000},
            # 3. 开启吸盘 (保持位置)
            {"target": np.array([0.5, 0, PICK_Z]), "suction": 30.0, "desc": "Suction On", "tol": 0.005, "steps": 500}, 
            # 4. 抬起
            {"target": np.array([0.5, 0, SAFE_Z]), "suction": 30.0, "desc": "Lifting", "tol": 0.02, "steps": 1000},
            # 5. 移动到目标
            {"target": np.array([0.5, 0.4, SAFE_Z]), "suction": 30.0, "desc": "Moving", "tol": 0.02, "steps": 1500},
            # 6. 下降放置
            {"target": np.array([0.5, 0.4, PLACE_Z]), "suction": 30.0, "desc": "Placing", "tol": 0.01, "steps": 1000},
            # 7. 关闭吸盘
            {"target": np.array([0.5, 0.4, PLACE_Z]), "suction": 0.0, "desc": "Suction Off", "tol": 0.01, "steps": 500},
            # 8. 回升
            {"target": np.array([0.5, 0.4, SAFE_Z]), "suction": 0.0, "desc": "Retracting", "tol": 0.02, "steps": 1000},
        ]
        
        current_phase_idx = 0
        step_counter = 0
        
        while viewer.is_running():
            step_start = time.time()
            
            if current_phase_idx < len(phases):
                phase = phases[current_phase_idx]
                target_pos = phase["target"]
                target_suction = phase["suction"]
                
                # 计算IK
                dq, err_norm = get_ik_delta(target_pos, data.qpos[:7])
                
                data.ctrl[:7] = data.qpos[:7] + dq
                data.ctrl[suction_id] = target_suction
                
                mj.mj_step(model, data)
                
                step_counter += 1
                
                # 检查阶段完成条件
                is_move_phase = phase["desc"] not in ["Suction On", "Suction Off"]
                
                if is_move_phase:
                    if err_norm < phase["tol"]:
                        print(f"Finished: {phase['desc']}")
                        current_phase_idx += 1
                        step_counter = 0
                else:
                    if step_counter >= phase["steps"]:
                        print(f"Finished: {phase['desc']}")
                        current_phase_idx += 1
                        step_counter = 0
                        
                # 超时强制切换
                if step_counter > phase["steps"] + 500:
                     print(f"Timeout: {phase['desc']}")
                     current_phase_idx += 1
                     step_counter = 0
            else:
                # 完成所有阶段，保持最后状态
                mj.mj_step(model, data)

            # 同步视图
            viewer.sync()
            
            # 简单的实时控制
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    run()
