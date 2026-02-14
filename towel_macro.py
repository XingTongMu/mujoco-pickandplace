import os
import mujoco as mj

XML_PATH = os.path.join(os.getcwd(), "towel_fold.xml")
SEQUENCE = [
    ("home_fold", 1500),
    ("approach_corner", 2000),
    ("grasp_close", 1500),
    ("drag_fold", 2500),
    ("release", 1000),
    ("lift_home", 1500),
]

def run():
    model = mj.MjModel.from_xml_path(XML_PATH)
    data = mj.MjData(model)
    for name, steps in SEQUENCE:
        kid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, name)
        if kid >= 0:
            mj.mj_resetDataKeyframe(model, data, kid)
        for _ in range(steps):
            mj.mj_step(model, data)

if __name__ == "__main__":
    run()
