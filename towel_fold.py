import os
import time
import mujoco as mj
import numpy as np

XML_PATH = os.path.join(os.getcwd(), "towel_fold.xml")

def name_to_actuator_id(model, name):
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)

def set_ctrl(model, data, name, value):
    idx = name_to_actuator_id(model, name)
    if idx >= 0:
        data.ctrl[idx] = value

def step(model, data, steps):
    for _ in range(steps):
        mj.mj_step(model, data)

def run():
    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    def set_fr3(j1,j2,j3,j4,j5,j6,j7):
        set_ctrl(model, data, "fr3_joint1", j1)
        set_ctrl(model, data, "fr3_joint2", j2)
        set_ctrl(model, data, "fr3_joint3", j3)
        set_ctrl(model, data, "fr3_joint4", j4)
        set_ctrl(model, data, "fr3_joint5", j5)
        set_ctrl(model, data, "fr3_joint6", j6)
        set_ctrl(model, data, "fr3_joint7", j7)

    set_fr3(0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853)
    set_ctrl(model, data, "grasp", 0.0)
    step(model, data, 2000)

    set_fr3(-0.4, -0.6, 0.2, -1.8, 0.3, 1.3, -0.6)
    step(model, data, 2000)

    set_ctrl(model, data, "grasp", 0.25)
    step(model, data, 1500)

    set_fr3(-0.2, -0.7, 0.3, -1.6, 0.4, 1.2, -0.5)
    step(model, data, 2500)

    set_ctrl(model, data, "grasp", 0.02)
    step(model, data, 1000)

    set_fr3(0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853)
    step(model, data, 1000)
    towel_body_ids = []
    for i in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("towel_"):
            towel_body_ids.append(i)
    if towel_body_ids:
        pos = data.xpos[towel_body_ids]
        moved = np.mean((pos[:, 1] > 0.0) & (pos[:, 2] > 0.05))
        print(f"fold_metric={moved:.2f}")
    else:
        names = []
        for i in range(model.nbody):
            names.append(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i))
        print("sample_body_names=", names[:10])

if __name__ == "__main__":
    run()
    run()
