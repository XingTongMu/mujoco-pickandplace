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

# 全局变量
control_data = {
    'fingers': [0.0] * 5, 
    'wrist': {'x': 0.0, 'y': 0.0, 'z': 0.5}, # 默认位置
    'active': False,
    'reset': False
}
data_lock = threading.Lock()

# 场景文件
XML_PATH = r"D:\python\mujoco-3.4.0-windows-x86_64\mujoco_menagerie-main\mujoco_menagerie-main\wonik_allegro\egg_pick_place_scene.xml"

# Flask 路由
@app.route('/')
def index():
    return render_template('index.html')

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

@socketio.on('finger_data')
def handle_finger_data(data):
    with data_lock:
        control_data['fingers'] = data['fingers']

@socketio.on('wrist_data')
def handle_wrist_data(data):
    # data: {x, y, z}
    with data_lock:
        control_data['wrist'] = data

def run_mujoco_thread():
    try:
        model = mj.MjModel.from_xml_path(XML_PATH)
        data = mj.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 获取 mocap body ID
    mocap_id = model.body('hand_mocap').mocapid[0]
    
    # 初始位置
    HOME_POS = np.array([0.3, 0.0, 0.28])
    current_pos = HOME_POS.copy()

    # Allegro Hand 关节映射
    # 每个手指4个关节: 0(base), 1(proximal), 2(medial), 3(distal)
    # 食指 (ff), 中指 (mf), 无名指 (rf), 拇指 (th)
    # 无名指实际上也可能被用来模拟小指动作，或者 Allegro 只有4指
    # Allegro Hand 只有 4 个手指：食指、中指、无名指、拇指。没有小指。
    
    # 关节名称前缀
    fingers_map = ['ff', 'mf', 'rf', 'th'] # Index, Middle, Ring, Thumb
    
    # 关节ID缓存
    joint_ids = {}
    for f in fingers_map:
        for i in range(4):
            name = f"{f}j{i}" # e.g. ffj0
            # actuator name usually matches joint name or has prefix
            # In XML: <position name="ffa0" joint="ffj0"...>
            act_name = f"{f}a{i}"
            try:
                # 使用 actuator id 控制
                aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, act_name)
                joint_ids[f"{f}_{i}"] = aid
            except:
                print(f"Warning: Actuator {act_name} not found")

    print("Launching Allegro Hand viewer...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            with data_lock:
                active = control_data['active']
                fingers_flex = control_data['fingers'] # [thumb, index, middle, ring, pinky]
                wrist_pos = control_data['wrist']
                reset_req = control_data['reset']
                
                if reset_req:
                    mj.mj_resetData(model, data)
                    control_data['reset'] = False
                    current_pos[:] = HOME_POS[:]
            
            if active:
                # 移动手掌 (Mocap)
                # 映射:
                # X: Wrist Y (Left-Right) -> World Y
                # Y: Wrist X (Up-Down) -> World X (Back-Forth)
                # Z: Wrist Z (Scale) -> World Z (Up-Down)
                
                SCALE_X = 0.5
                SCALE_Y = 0.5
                SCALE_Z = 0.3
                
                target_x = HOME_POS[0] + wrist_pos['y'] * SCALE_X
                target_y = HOME_POS[1] + wrist_pos['x'] * SCALE_Y
                target_z = 0.15 + wrist_pos['z'] * SCALE_Z # 0.15 ~ 0.45
                
                # 平滑
                current_pos[0] = current_pos[0] * 0.9 + target_x * 0.1
                current_pos[1] = current_pos[1] * 0.9 + target_y * 0.1
                current_pos[2] = current_pos[2] * 0.9 + target_z * 0.1
                
                data.mocap_pos[mocap_id] = current_pos
                
                # 映射手指逻辑：
                # MediaPipe: Thumb(0), Index(1), Middle(2), Ring(3), Pinky(4)
                # Allegro: Index(ff), Middle(mf), Ring(rf), Thumb(th)
                
                # 1. Index Finger (ff) <- Index(1)
                idx_val = fingers_flex[1]
                data.ctrl[joint_ids['ff_1']] = idx_val * 1.6 # Proximal
                data.ctrl[joint_ids['ff_2']] = idx_val * 1.7 # Medial
                data.ctrl[joint_ids['ff_3']] = idx_val * 1.6 # Distal
                
                # 2. Middle Finger (mf) <- Middle(2)
                mid_val = fingers_flex[2]
                data.ctrl[joint_ids['mf_1']] = mid_val * 1.6
                data.ctrl[joint_ids['mf_2']] = mid_val * 1.7
                data.ctrl[joint_ids['mf_3']] = mid_val * 1.6
                
                # 3. Ring Finger (rf) <- Ring(3)
                rng_val = fingers_flex[3]
                data.ctrl[joint_ids['rf_1']] = rng_val * 1.6
                data.ctrl[joint_ids['rf_2']] = rng_val * 1.7
                data.ctrl[joint_ids['rf_3']] = rng_val * 1.6
                
                # 4. Thumb (th) <- Thumb(0)
                th_val = fingers_flex[0]
                # 拇指动作比较复杂，简化为向掌心弯曲
                data.ctrl[joint_ids['th_1']] = th_val * 1.1 
                data.ctrl[joint_ids['th_2']] = th_val * 1.6
                data.ctrl[joint_ids['th_3']] = th_val * 1.7
                # 拇指旋转 (Opposite)
                data.ctrl[joint_ids['th_0']] = 1.3 if th_val > 0.5 else 0.3
            else:
                # 非激活状态保持位置
                data.mocap_pos[mocap_id] = HOME_POS
                
            mj.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

def run_server():
    print("Starting Allegro Control Server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    sim_thread = threading.Thread(target=run_mujoco_thread)
    sim_thread.daemon = True
    sim_thread.start()
    run_server()
