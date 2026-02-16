import json
import os
import threading
import time
from pathlib import Path

import mujoco as mj
import mujoco.viewer
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO

from photo_to_scene import generate_scene_from_photo


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

TEMPLATE_XML = BASE_DIR / "photo_template.xml"
GENERATED_XML = BASE_DIR / "generated_photo_scene.xml"

app = Flask(__name__, template_folder=str(BASE_DIR))
socketio = SocketIO(app, cors_allowed_origins="*")

state_lock = threading.Lock()
state = {
    "active": False,
    "x": 0.0,
    "y": 0.0,
    "z": 0.5,
    "grasp": 0.0,
    "reset": False,
    "reload": False,
    "xml_path": str(TEMPLATE_XML),
    "pick_request": None,
}


@app.route("/")
def index():
    return render_template("photo_app.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "photo" not in request.files:
        return jsonify({"ok": False, "error": "缺少 photo 文件"}), 400

    f = request.files["photo"]
    if not f.filename:
        return jsonify({"ok": False, "error": "文件名为空"}), 400

    img_path = UPLOAD_DIR / "latest.jpg"
    f.save(str(img_path))

    meters_per_px = float(request.form.get("meters_per_px", "0.0015"))
    center_x = float(request.form.get("center_x", "0.5"))
    center_y = float(request.form.get("center_y", "0.0"))
    table_z = float(request.form.get("table_z", "0.0"))

    objs, out_xml = generate_scene_from_photo(
        template_xml_path=TEMPLATE_XML,
        image_path=img_path,
        output_xml_path=GENERATED_XML,
        meters_per_px=meters_per_px,
        world_center_xy=(center_x, center_y),
        table_z=table_z,
    )

    meta = [
        {
            "name": o.name,
            "geom_type": o.geom_type,
            "size": list(o.size),
            "pos": list(o.pos),
            "rgba": list(o.rgba),
            "mass": o.mass,
        }
        for o in objs
    ]

    (UPLOAD_DIR / "latest_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    with state_lock:
        state["xml_path"] = str(out_xml)
        state["reload"] = True

    return jsonify({"ok": True, "objects": meta, "xml": str(out_xml)})


@socketio.on("control_state")
def on_control_state(data):
    with state_lock:
        state["active"] = bool(data.get("active", False))


@socketio.on("reset_sim")
def on_reset():
    with state_lock:
        state["reset"] = True


@socketio.on("hand_data")
def on_hand_data(data):
    with state_lock:
        state["x"] = float(data.get("x", 0.0))
        state["y"] = float(data.get("y", 0.0))
        state["z"] = float(data.get("z", 0.5))
        state["grasp"] = float(data.get("grasp", 0.0))


@socketio.on("pick_object")
def on_pick_object(data):
    name = data.get("name")
    if not isinstance(name, str) or not name:
        return
    with state_lock:
        state["pick_request"] = name


def _run_sim_loop():
    current_xml_path = Path(state["xml_path"])

    while True:
        with state_lock:
            current_xml_path = Path(state["xml_path"])
            state["reload"] = False

        model = mj.MjModel.from_xml_path(str(current_xml_path))
        data = mj.MjData(model)

        suction_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "suction")
        ee_site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "attachment_site")

        home_pos = np.array([0.5, 0.0, 0.45], dtype=np.float64)
        current_target = home_pos.copy()

        def ik_delta(target_pos: np.ndarray) -> np.ndarray:
            site_pos = data.site_xpos[ee_site_id].copy()
            jac = np.zeros((6, model.nv), dtype=np.float64)
            mj.mj_jacSite(model, data, jac[:3], jac[3:], ee_site_id)
            j = jac[:3, :7]
            e = target_pos - site_pos
            dq = j.T @ np.linalg.inv(j @ j.T + 0.01 * np.eye(3)) @ e * 0.1
            return dq

        def get_site_pos_by_name(site_name: str) -> np.ndarray | None:
            try:
                sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
            except Exception:
                return None
            if sid < 0:
                return None
            return data.site_xpos[sid].copy()

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                with state_lock:
                    active = bool(state["active"])
                    hx = float(state["x"])
                    hy = float(state["y"])
                    hz = float(state["z"])
                    grasp = float(state["grasp"])
                    reset_req = bool(state["reset"])
                    reload_req = bool(state["reload"])
                    pick_req = state["pick_request"]
                    state["pick_request"] = None

                    if reset_req:
                        mj.mj_resetData(model, data)
                        state["reset"] = False
                        current_target[:] = home_pos

                if reload_req:
                    break

                if pick_req:
                    pick_site = f"{pick_req}_site"
                    pos = get_site_pos_by_name(pick_site)
                    if pos is not None:
                        current_target[:] = pos
                        current_target[2] = max(current_target[2] + 0.12, 0.20)

                if active:
                    hz = _clamp(hz, 0.0, 1.0)
                    tx = home_pos[0] + hy * 0.6
                    ty = home_pos[1] + hx * 0.6
                    tz = 0.18 + hz * 0.35

                    current_target[0] = current_target[0] * 0.9 + tx * 0.1
                    current_target[1] = current_target[1] * 0.9 + ty * 0.1
                    current_target[2] = current_target[2] * 0.9 + tz * 0.1

                    dq = ik_delta(current_target)
                    data.ctrl[:7] = data.qpos[:7] + dq
                    data.ctrl[suction_id] = 30.0 if grasp > 0.5 else 0.0
                else:
                    data.ctrl[:7] = data.qpos[:7]
                    data.ctrl[suction_id] = 0.0

                mj.mj_step(model, data)
                viewer.sync()

                dt = model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def main():
    sim_thread = threading.Thread(target=_run_sim_loop, daemon=True)
    sim_thread.start()
    socketio.run(app, host="0.0.0.0", port=5002, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
