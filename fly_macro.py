import os
import math
import mujoco as mj

XML_PATH = os.path.join(os.getcwd(), "fly_scene.xml")

def id_act(model, name):
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)

def run():
    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)
    t = 0.0
    left = id_act(model, "flap_left")
    right = id_act(model, "flap_right")
    ax = id_act(model, "ax")
    ay = id_act(model, "ay")
    az = id_act(model, "az")

    for step in range(8000):
        flap = math.sin(2.0 * math.pi * 10.0 * t)
        if left >= 0:
            data.ctrl[left] = flap
        if right >= 0:
            data.ctrl[right] = -flap
        if step < 2000:
            if az >= 0:
                data.ctrl[az] = 0.3
        elif step < 5000:
            if ax >= 0:
                data.ctrl[ax] = 0.2
            if az >= 0:
                data.ctrl[az] = 0.2
        else:
            if az >= 0:
                data.ctrl[az] = 0.05
            if ax >= 0:
                data.ctrl[ax] = 0.0
            if ay >= 0:
                data.ctrl[ay] = 0.0
        mj.mj_step(model, data)
        t += model.opt.timestep

if __name__ == "__main__":
    run()
