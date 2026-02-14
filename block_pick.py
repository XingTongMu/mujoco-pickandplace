import mujoco as mj
import numpy as np
import time
import os

XML_PATH = os.path.join(os.getcwd(), "block_pick.xml")

def run():
    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)
    
    # 获取ID
    actuator_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"fr3_joint{i+1}") for i in range(7)]
    grasp_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "grasp")
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "attachment_site")
    block_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "block")
    
    # 简单的IK函数 (Damped Least Squares)
    def solve_ik(target_pos, target_quat=None):
        site_pos = data.site_xpos[site_id]
        site_quat = data.site_xmat[site_id] # 注意这是旋转矩阵展平
        
        # 位置误差
        err_pos = target_pos - site_pos
        
        # 旋转误差 (简化，仅位置控制可忽略，但为了抓取姿态需要控制)
        # 这里为了简单，我们主要控制位置，并保持末端向下
        # 目标姿态：末端垂直向下 (gripper z轴与世界z轴对齐或相反)
        # FR3的 attachment_site 定义需要检查，通常 z轴是法向
        
        error = np.zeros(6)
        error[:3] = err_pos
        
        # 简单的姿态保持：让末端尽量保持当前姿态或垂直向下
        # 假设我们只做位置控制，姿态由初始位姿自然演变或轻微修正
        # 为了更稳定，我们添加姿态约束：目标是末端z轴朝下
        
        jac = np.zeros((6, model.nv))
        mj.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        
        # 只使用手臂关节 (前7个DoF)
        jac_arm = jac[:, :7]
        
        # 阻尼最小二乘
        lambda_val = 0.01
        delta_q = jac_arm.T @ np.linalg.inv(jac_arm @ jac_arm.T + lambda_val**2 * np.eye(6)) @ error
        
        return delta_q

    # 预设路径点
    # FR3 Home: qpos=0 0 0 -1.57 0 1.57 -0.78
    home_qpos = np.array([0, -0.78, 0, -2.35, 0, 1.57, 0.78]) # 修改Home让它更舒展
    
    # 仿真循环
    print("Starting simulation...")
    
    # 1. 移动到方块上方
    print("Approaching block...")
    target_pos = np.array([0.5, 0, 0.4])
    for i in range(1000):
        # 简单的PD控制或直接设置qpos (如果IK解算每步都做)
        # 这里我们用差分IK更新目标qpos
        
        # 计算当前末端位置
        current_site_pos = data.site_xpos[site_id]
        
        # 计算雅可比
        jac = np.zeros((6, model.nv))
        mj.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        jac_arm = jac[:3, :7] # 只看位置
        
        error = target_pos - current_site_pos
        if np.linalg.norm(error) < 0.01:
            break
            
        dq = jac_arm.T @ np.linalg.inv(jac_arm @ jac_arm.T + 0.01 * np.eye(3)) @ error * 0.1
        
        # 更新控制量
        current_q = data.qpos[:7]
        data.ctrl[:7] = current_q + dq
        data.ctrl[grasp_id] = 0.0 # 张开
        
        mj.mj_step(model, data)
        
    # 2. 下降
    print("Descending...")
    target_pos = np.array([0.5, 0, 0.15]) # 方块高度0.05，中心在0.025，抓取点需要在中心偏上
    for i in range(1000):
        current_site_pos = data.site_xpos[site_id]
        jac = np.zeros((6, model.nv))
        mj.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        jac_arm = jac[:3, :7]
        error = target_pos - current_site_pos
        if np.linalg.norm(error) < 0.005:
            break
        dq = jac_arm.T @ np.linalg.inv(jac_arm @ jac_arm.T + 0.01 * np.eye(3)) @ error * 0.05
        data.ctrl[:7] = data.qpos[:7] + dq
        mj.mj_step(model, data)
        
    # 3. 抓取
    print("Grasping...")
    for i in range(500):
        data.ctrl[grasp_id] = 0.2 # 闭合
        # 保持手臂位置
        mj.mj_step(model, data)
        
    # 4. 抬起
    print("Lifting...")
    target_pos = np.array([0.5, 0, 0.4])
    for i in range(1000):
        current_site_pos = data.site_xpos[site_id]
        jac = np.zeros((6, model.nv))
        mj.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        jac_arm = jac[:3, :7]
        error = target_pos - current_site_pos
        if np.linalg.norm(error) < 0.01:
            break
        dq = jac_arm.T @ np.linalg.inv(jac_arm @ jac_arm.T + 0.01 * np.eye(3)) @ error * 0.05
        data.ctrl[:7] = data.qpos[:7] + dq
        data.ctrl[grasp_id] = 0.25 # 保持闭合
        mj.mj_step(model, data)
        
    # 5. 移动到目标
    print("Moving to target...")
    target_pos = np.array([0.5, 0.4, 0.4])
    for i in range(1500):
        current_site_pos = data.site_xpos[site_id]
        jac = np.zeros((6, model.nv))
        mj.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        jac_arm = jac[:3, :7]
        error = target_pos - current_site_pos
        if np.linalg.norm(error) < 0.01:
            break
        dq = jac_arm.T @ np.linalg.inv(jac_arm @ jac_arm.T + 0.01 * np.eye(3)) @ error * 0.05
        data.ctrl[:7] = data.qpos[:7] + dq
        data.ctrl[grasp_id] = 0.25
        mj.mj_step(model, data)

    # 6. 下降放置
    print("Placing...")
    target_pos = np.array([0.5, 0.4, 0.15])
    for i in range(1000):
        current_site_pos = data.site_xpos[site_id]
        jac = np.zeros((6, model.nv))
        mj.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        jac_arm = jac[:3, :7]
        error = target_pos - current_site_pos
        if np.linalg.norm(error) < 0.01:
            break
        dq = jac_arm.T @ np.linalg.inv(jac_arm @ jac_arm.T + 0.01 * np.eye(3)) @ error * 0.05
        data.ctrl[:7] = data.qpos[:7] + dq
        data.ctrl[grasp_id] = 0.25
        mj.mj_step(model, data)
        
    # 7. 松开
    print("Releasing...")
    for i in range(500):
        data.ctrl[grasp_id] = 0.0
        mj.mj_step(model, data)
        
    # 8. 回升
    print("Retracting...")
    target_pos = np.array([0.5, 0.4, 0.4])
    for i in range(1000):
        current_site_pos = data.site_xpos[site_id]
        jac = np.zeros((6, model.nv))
        mj.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        jac_arm = jac[:3, :7]
        error = target_pos - current_site_pos
        dq = jac_arm.T @ np.linalg.inv(jac_arm @ jac_arm.T + 0.01 * np.eye(3)) @ error * 0.05
        data.ctrl[:7] = data.qpos[:7] + dq
        data.ctrl[grasp_id] = 0.0
        mj.mj_step(model, data)
        
    print("Done!")

if __name__ == "__main__":
    run()
