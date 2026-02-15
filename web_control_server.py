import mujoco as mj
import mujoco.viewer
import numpy as np
import threading
import time
import os
from flask import Flask, render_template
from flask_socketio import SocketIO
import logging

# 配置日志
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__, template_folder='.')
socketio = SocketIO(app, cors_allowed_origins='*')

# 全局变量，用于线程间通信
control_data = {
    'x': 0.0,
    'y': 0.0,
    'z': 0.0,
    'grasp': 0.0,
    'active': False,
    'reset': False
}
data_lock = threading.Lock()

# 场景文件
XML_PATH = os.path.join(os.getcwd(), "block_pick.xml") # 复用方块抓取场景

# Flask 路由
@app.route('/')
def index():
    return render_template('index.html')

# SocketIO 事件
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('control_state')
def handle_control_state(data):
    with data_lock:
        control_data['active'] = data['active']
    print(f"Control active: {data['active']}")

@socketio.on('reset_sim')
def handle_reset():
    with data_lock:
        control_data['reset'] = True
    print("Reset requested")

@socketio.on('hand_data')
def handle_hand_data(data):
    # data: {x, y, z, grasp}
    # x, y 是相对值 (-0.5 ~ 0.5)
    with data_lock:
        control_data['x'] = data['x']
        control_data['y'] = data['y']
        control_data['z'] = data['z']
        control_data['grasp'] = data['grasp']

# MuJoCo 仿真线程
def run_mujoco_thread():
    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)
    
    # 获取ID
    suction_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "suction")
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "attachment_site")
    
    # IK 参数
    # 初始 Home 位置
    HOME_POS = np.array([0.5, 0.0, 0.4])
    current_target = HOME_POS.copy()
    
    # 映射参数
    # 手的 X (左右) -> 机器人 Y (左右)
    # 手的 Y (上下) -> 机器人 X (前后)
    # 范围缩放
    SCALE_X = 0.8 # 手移动1单位 -> 机器人移动0.8m
    SCALE_Y = 0.8
    
    # 中心偏移
    CENTER_X = 0.5 # 机器人X中心
    CENTER_Y = 0.0 # 机器人Y中心
    
    # 简单的IK函数
    def get_ik_delta(target_pos):
        site_pos = data.site_xpos[site_id]
        jac = np.zeros((6, model.nv))
        mj.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        jac_arm = jac[:3, :7] 
        error = target_pos - site_pos
        dq = jac_arm.T @ np.linalg.inv(jac_arm @ jac_arm.T + 0.01 * np.eye(3)) @ error * 0.1
        return dq
    
    print("Launching MuJoCo viewer...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # 读取控制数据
            with data_lock:
                active = control_data['active']
                hand_x = control_data['x']
                hand_y = control_data['y']
                hand_grasp = control_data['grasp']
                reset_req = control_data['reset']
                
                if reset_req:
                    mj.mj_resetData(model, data)
                    control_data['reset'] = False
                    current_target = HOME_POS.copy()
            
            if active:
                # 映射坐标
                # hand_x: -0.5(左) ~ 0.5(右) -> robot Y: 0.4 ~ -0.4
                # hand_y: -0.5(下) ~ 0.5(上) -> robot X: 0.3 ~ 0.7
                
                # 这里的 hand_y 实际上是 (0.5 - wrist.y)，wrist.y=0是顶，所以hand_y=0.5是顶
                # 我们希望手向上移(屏幕上方)，机器人向远处理(X增大)
                # 或者手向前推(屏幕上方)，机器人向前
                
                target_y = hand_x * SCALE_Y + CENTER_Y # 左右对应Y
                target_x = hand_y * SCALE_X + CENTER_X # 上下对应X
                
                # Z轴控制：
                # hand_z: 0.0 (低) ~ 1.0 (高)
                # 映射到: 0.2 (抓取高度) ~ 0.5 (悬停高度)
                hand_z = control_data['z']
                # 限制范围
                if hand_z < 0: hand_z = 0
                if hand_z > 1: hand_z = 1
                
                target_z = 0.23 + hand_z * 0.3 # 0.23 ~ 0.53
                
                # 平滑滤波
                current_target[0] = current_target[0] * 0.9 + target_x * 0.1
                current_target[1] = current_target[1] * 0.9 + target_y * 0.1
                current_target[2] = current_target[2] * 0.9 + target_z * 0.1
                
                # 应用 IK
                dq = get_ik_delta(current_target)
                data.ctrl[:7] = data.qpos[:7] + dq
                
                # 吸盘控制
                # grasp > 0.5 -> 开启吸盘
                if hand_grasp > 0.5:
                    data.ctrl[suction_id] = 30.0
                else:
                    data.ctrl[suction_id] = 0.0
                    
            else:
                # 保持 Home 或当前位置
                # 简单的阻尼保持
                pass

            mj.mj_step(model, data)
            viewer.sync()
            
            # 帧率控制
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

def run_server():
    print("Starting Flask server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    # 启动 MuJoCo 线程
    sim_thread = threading.Thread(target=run_mujoco_thread)
    sim_thread.daemon = True
    sim_thread.start()
    
    # 启动 Web 服务器
    run_server()
