import os
import mujoco as mj

XML_PATH = os.path.join(os.getcwd(), "clothes_fold.xml")
SEQUENCE = [
    ("home_fold", 1200),
    ("approach_left", 1800),
    ("fold_left", 2200),
    ("approach_right", 1800),
    ("fold_right", 2200),
    ("approach_bottom", 1800),
    ("fold_bottom", 2200),
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
