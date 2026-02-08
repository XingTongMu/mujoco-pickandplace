# -*- coding: utf-8 -*-
import time
import threading
from threading import Thread
import tkinter as tk
from tkinter import ttk, messagebox

import glfw
import mujoco
import numpy as np


class Demo:
    qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    K = np.array([900.0, 900.0, 900.0, 40.0, 40.0, 40.0])
    height, width = 480, 640
    fps = 30

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path("world.xml")
        self.data = mujoco.MjData(self.model)

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = 0
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)

        self.run = True
        self.stop_flag = threading.Event()

        # open gripper + home joints
        self.gripper(True)
        for i in range(1, 8):
            self.data.joint(f"panda_joint{i}").qpos = self.qpos0[i - 1]
        mujoco.mj_forward(self.model, self.data)

        # hold targets (keep pose fixed when idle)
        hand = self.data.body("panda_hand")
        self.target_pos = hand.xpos.copy()
        self.target_quat = hand.xquat.copy()
        self._hold_running = True

    # ---------------- controller ----------------
    def gripper(self, open=True):
        self.data.actuator("pos_panda_finger_joint1").ctrl = 0.04 if open else 0.0
        self.data.actuator("pos_panda_finger_joint2").ctrl = 0.04 if open else 0.0

    def control(self, xpos_d, xquat_d):
        xpos = self.data.body("panda_hand").xpos
        xquat = self.data.body("panda_hand").xquat
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)

        error = np.zeros(6)
        error[:3] = xpos_d - xpos
        res = np.zeros(3)
        mujoco.mju_subQuat(res, xquat, xquat_d)
        mujoco.mju_rotVecQuat(res, res, xquat)
        error[3:] = -res

        J = np.concatenate((jacp, jacr))
        v = J @ self.data.qvel
        for i in range(1, 8):
            dofadr = self.model.joint(f"panda_joint{i}").dofadr
            self.data.actuator(f"panda_joint{i}").ctrl = self.data.joint(
                f"panda_joint{i}"
            ).qfrc_bias
            self.data.actuator(f"panda_joint{i}").ctrl += (
                J[:, dofadr].T @ np.diag(self.K) @ error
            )
            self.data.actuator(f"panda_joint{i}").ctrl -= (
                J[:, dofadr].T @ np.diag(2 * np.sqrt(self.K)) @ v
            )

    # --------------- hold loop ----------------
    def _hold_loop(self):
        while self.run and self._hold_running:
            self.control(self.target_pos, self.target_quat)
            mujoco.mj_step(self.model, self.data)
            time.sleep(1 / 500)

    # ---------------- math helpers ----------------
    def _quat_err(self, q, r):
        dot = abs(float(np.dot(q, r)))
        dot = max(min(dot, 1.0), -1.0)
        return 2.0 * np.arccos(dot)

    def _reach_pose(self, pos_goal, quat_goal, pos_tol=0.003, ang_tol=0.03, timeout=2.0):
        t0 = time.time()
        self.target_pos = pos_goal.copy()
        self.target_quat = quat_goal.copy()
        while time.time() - t0 < timeout and not self.stop_flag.is_set():
            hand = self.data.body("panda_hand")
            p_err = np.linalg.norm(pos_goal - hand.xpos)
            a_err = self._quat_err(quat_goal, hand.xquat)
            if p_err < pos_tol and a_err < ang_tol:
                return True
            time.sleep(1 / 400)
        return False

    # --------------- motion helpers ------------------
    def _move_linear(self, target_pos, xquat_ref, duration_s):
        start = self.data.body("panda_hand").xpos.copy()
        steps = max(1, int(duration_s * 400))
        self.target_quat = xquat_ref
        for k in range(steps):
            if self.stop_flag.is_set():
                return
            a = (k + 1) / steps
            self.target_pos = (1.0 - a) * start + a * target_pos
            time.sleep(1 / 400)

    def wait(self, seconds):
        t0 = time.time()
        while time.time() - t0 < seconds:
            if self.stop_flag.is_set():
                return
            time.sleep(1 / 400)

    def move_via_hover(self, xy, z_target, z_safe=None, t_up=0.6, t_xy=1.0, t_down=0.8):
        """
        先抬到安全高度 z_safe -> 水平移到 xy -> 垂直下到 z_target
        """
        hand = self.data.body("panda_hand")
        xquat_ref = hand.xquat.copy()

        if z_safe is None:
            # 自动安全高度：当前/目标取最大 + 8cm
            z_safe = float(max(hand.xpos[2], z_target) + 0.08)

        # 1) up
        p1 = np.array([hand.xpos[0], hand.xpos[1], z_safe], dtype=float)
        self._move_linear(p1, xquat_ref, t_up)
        self._reach_pose(p1, xquat_ref, 0.003, 0.03, 2.0)

        # 2) horizontal at safe height
        p2 = np.array([xy[0], xy[1], z_safe], dtype=float)
        self._move_linear(p2, xquat_ref, t_xy)
        self._reach_pose(p2, xquat_ref, 0.003, 0.03, 2.0)

        # 3) down vertically
        p3 = np.array([xy[0], xy[1], z_target], dtype=float)
        self._move_linear(p3, xquat_ref, t_down)
        self._reach_pose(p3, xquat_ref, 0.0025, 0.03, 2.0)

    # --------------- object helpers ------------------
    def _find_body_and_top_z(self, names=("box1", "box2", "box3", "box4"), allow_auto=True):
        """
        Return (name, xy, z_top). Named lookup first; optionally auto-detect a small object.
        """
        for nm in names:
            try:
                b = self.data.body(nm)
                xy = b.xpos[:2].copy()
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nm)
                z_top = float(b.xpos[2])
                for g in range(self.model.ngeom):
                    if self.model.geom_bodyid[g] == bid and self.model.geom_size.shape[1] >= 3:
                        z_top = float(b.xpos[2] + self.model.geom_size[g][2])
                        break
                return nm, xy, z_top
            except KeyError:
                pass

        if not allow_auto:
            raise KeyError("Target body not found. Tried: " + ", ".join(names))

        # Auto-detect: smallest non-robot box/cylinder body
        skip_keywords = ("panda", "floor", "world", "table", "ground", "base")
        candidate = None
        best_score = 1e9
        for bid in range(self.model.nbody):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if any(k in nm.lower() for k in skip_keywords):
                continue
            # first geom on this body
            g_idx = None
            for g in range(self.model.ngeom):
                if self.model.geom_bodyid[g] == bid:
                    g_idx = g
                    break
            if g_idx is None:
                continue
            gtype = int(self.model.geom_type[g_idx])
            size = np.array(self.model.geom_size[g_idx])
            if gtype not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER):
                continue
            s = float(np.linalg.norm(size))
            if s < best_score:
                best_score = s
                xy = self.data.xpos[bid][:2].copy()
                z_top = float(self.data.xpos[bid][2] + (size[2] if size.size >= 3 else 0.0))
                candidate = (nm, xy, z_top)

        if candidate:
            return candidate

        raise KeyError("Could not auto-detect a target body; please enter its exact name in the GUI.")

    def _get_body_top_and_halfz(self, body_name: str):
        """
        Return (xy, z_top, half_z) for the first geom attached to body_name.
        """
        b = self.data.body(body_name)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        g_idx = None
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == bid:
                g_idx = g
                break
        if g_idx is None:
            raise KeyError(f"Body '{body_name}' has no geom attached.")

        half_z = float(self.model.geom_size[g_idx][2])
        z_top = float(b.xpos[2] + half_z)
        xy = b.xpos[:2].copy()
        return xy, z_top, half_z

    # --------------- actions ------------------
    def pick_only(self, down_time=1.2, settle_s=0.30, up_time=1.0, target_hint="box1", hover_h=0.15):
        """
        先到高处 -> 水平对齐 -> 垂直下探抓取 -> 抬起
        hover_h: 抓取时在物体顶面上方的悬停高度
        """
        self.stop_flag.clear()

        nm = target_hint.strip() if target_hint else ""
        names = (nm,) if nm else ()
        _, xy, z_top = self._find_body_and_top_z(names or ("box1", "box2", "box3", "box4"), allow_auto=True)

        hand = self.data.body("panda_hand")
        xquat_ref = hand.xquat.copy()

        hover_z = float(z_top + abs(hover_h))
        pregrasp_z = float(z_top + 0.005)  # 5mm above top
        grasp_z = float(z_top - 0.01)      # 1cm below top (tune if needed)

        # open & settle
        self.gripper(True)
        self.wait(0.20)

        # go to hover via safe height (UP -> XY -> DOWN)
        self.move_via_hover(xy=xy, z_target=hover_z, z_safe=None, t_up=0.6, t_xy=max(0.8, up_time), t_down=0.6)
        if self.stop_flag.is_set():
            return
        self.wait(0.10)

        # vertical down to pregrasp then grasp (pure vertical moves)
        self._move_linear(np.array([xy[0], xy[1], pregrasp_z], dtype=float), xquat_ref, 0.6)
        self._reach_pose(np.array([xy[0], xy[1], pregrasp_z], dtype=float), xquat_ref, 0.002, 0.03, 2.0)

        self._move_linear(np.array([xy[0], xy[1], grasp_z], dtype=float), xquat_ref, max(0.6, down_time))
        self._reach_pose(np.array([xy[0], xy[1], grasp_z], dtype=float), xquat_ref, 0.002, 0.03, 2.0)

        # close & settle
        self.wait(settle_s)
        self.gripper(False)
        self.wait(0.20)

        # lift straight up back to hover height
        self._move_linear(np.array([xy[0], xy[1], hover_z], dtype=float), xquat_ref, max(0.8, up_time))
        self._reach_pose(np.array([xy[0], xy[1], hover_z], dtype=float), xquat_ref, 0.003, 0.03, 2.0)

    def place_only_xy(self, place_xy, z_target, down_time=1.0, up_time=1.0, hover_h=0.06, settle_contact=0.30, settle_release=0.20):
        """
        把手上抓着的东西放到指定 (x,y,z_target)。
        轨迹：先高处到位 -> 垂直下 -> 稳定 -> 松手 -> 稳定 -> 抬起
        """
        self.stop_flag.clear()
        hand = self.data.body("panda_hand")
        xquat_ref = hand.xquat.copy()

        hover_z = float(z_target + abs(hover_h))

        # go to hover (UP->XY->DOWN)
        self.move_via_hover(xy=place_xy, z_target=hover_z, z_safe=None, t_up=0.6, t_xy=max(0.8, up_time), t_down=0.6)
        if self.stop_flag.is_set():
            return
        self.wait(0.10)

        # vertical down to target z
        self._move_linear(np.array([place_xy[0], place_xy[1], z_target], dtype=float), xquat_ref, max(0.8, down_time))
        self._reach_pose(np.array([place_xy[0], place_xy[1], z_target], dtype=float), xquat_ref, 0.0025, 0.03, 2.0)

        self.wait(settle_contact)
        self.gripper(True)
        self.wait(settle_release)

        # lift back to hover
        self._move_linear(np.array([place_xy[0], place_xy[1], hover_z], dtype=float), xquat_ref, max(0.8, up_time))
        self._reach_pose(np.array([place_xy[0], place_xy[1], hover_z], dtype=float), xquat_ref, 0.003, 0.03, 2.0)

    def stack_box_on(self, pick_name: str, place_name: str,
                     hover_h=0.15, clearance=0.0015,
                     down_time=1.2, up_time=1.0,
                     settle_contact=0.30, settle_release=0.20):
        """
        Pick 'pick_name' then place it centered on top of 'place_name'.
        路径：全程走 “先高处到位 -> 再垂直下探”。
        """
        self.stop_flag.clear()

        # 1) pick
        self.pick_only(
            down_time=down_time,
            settle_s=0.25,
            up_time=up_time,
            target_hint=pick_name,
            hover_h=hover_h
        )
        if self.stop_flag.is_set():
            return

        # 2) compute place z: top(place) + half(pick) + clearance
        place_xy, place_z_top, _ = self._get_body_top_and_halfz(place_name)
        _, _, pick_halfz = self._get_body_top_and_halfz(pick_name)

        z_target = float(place_z_top + pick_halfz + clearance)

        # 3) place
        self.place_only_xy(
            place_xy=(float(place_xy[0]), float(place_xy[1])),
            z_target=z_target,
            down_time=down_time,
            up_time=up_time,
            hover_h=0.06,
            settle_contact=settle_contact,
            settle_release=settle_release
        )

    # ---------------- viewer ----------------
    def render(self) -> None:
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 8)
        window = glfw.create_window(self.width, self.height, "Panda Demo", None, None)
        glfw.make_context_current(window)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        while not glfw.window_should_close(window):
            w, h = glfw.get_framebuffer_size(window)
            viewport.width, viewport.height = w, h
            mujoco.mjv_updateScene(
                self.model, self.data, opt, pert, self.cam,
                mujoco.mjtCatBit.mjCAT_ALL, self.scene
            )
            mujoco.mjr_render(viewport, self.scene, self.context)
            time.sleep(1.0 / self.fps)
            glfw.swap_buffers(window)
            glfw.poll_events()

        self.run = False
        self._hold_running = False
        self.stop_flag.set()
        glfw.terminate()

    def start(self) -> None:
        Thread(target=self._hold_loop, daemon=True).start()
        self.render()


# ---------------- GUI ----------------
def launch_gui(demo: Demo):
    root = tk.Tk()
    root.title("Panda: Stack Boxes")
    root.geometry("760x260")

    pick_name = tk.StringVar(value="box1")
    place_name = tk.StringVar(value="box2")

    ttk.Label(root, text="Pick body:").grid(row=0, column=0, padx=6, pady=6, sticky="e")
    ttk.Entry(root, width=14, textvariable=pick_name).grid(row=0, column=1, padx=4, pady=6, sticky="w")

    ttk.Label(root, text="Place on body:").grid(row=0, column=2, padx=6, pady=6, sticky="e")
    ttk.Entry(root, width=14, textvariable=place_name).grid(row=0, column=3, padx=4, pady=6, sticky="w")

    ttk.Label(root, text="Valid: box1 box2 box3 box4").grid(row=0, column=4, padx=10, pady=6, sticky="w")

    # params
    down_t = tk.StringVar(value="1.2")
    up_t = tk.StringVar(value="1.0")
    settle = tk.StringVar(value="0.30")
    hover_pick = tk.StringVar(value="0.15")   # hover above picked object top
    clearance = tk.StringVar(value="0.0015")  # 1.5mm

    row = 1
    ttk.Label(root, text="Down time (s):").grid(row=row, column=0, sticky="e", padx=6)
    ttk.Entry(root, width=8, textvariable=down_t).grid(row=row, column=1, padx=4, sticky="w")

    ttk.Label(root, text="Up time (s):").grid(row=row, column=2, sticky="e", padx=6)
    ttk.Entry(root, width=8, textvariable=up_t).grid(row=row, column=3, padx=4, sticky="w")

    ttk.Label(root, text="Settle (s):").grid(row=row, column=4, sticky="e", padx=6)
    ttk.Entry(root, width=8, textvariable=settle).grid(row=row, column=5, padx=4, sticky="w")

    row += 1
    ttk.Label(root, text="Pick hover (m):").grid(row=row, column=0, sticky="e", padx=6)
    ttk.Entry(root, width=8, textvariable=hover_pick).grid(row=row, column=1, padx=4, sticky="w")

    ttk.Label(root, text="Clearance (m):").grid(row=row, column=2, sticky="e", padx=6)
    ttk.Entry(root, width=8, textvariable=clearance).grid(row=row, column=3, padx=4, sticky="w")
    ttk.Label(root, text="(1~3mm recommended)").grid(row=row, column=4, columnspan=2, padx=6, sticky="w")

    def _safe(s: str) -> str:
        return (s or "").strip()

    def stop_now():
        demo.stop_flag.set()

    def reset_stop():
        demo.stop_flag.clear()

    def pick_only_now():
        try:
            reset_stop()
            demo.pick_only(
                down_time=float(down_t.get()),
                settle_s=float(settle.get()),
                up_time=float(up_t.get()),
                target_hint=_safe(pick_name.get()),
                hover_h=float(hover_pick.get()),
            )
        except ValueError:
            messagebox.showerror("Invalid", "Please enter numeric values.")
        except KeyError as e:
            messagebox.showerror("Not found", str(e))

    def stack_now():
        try:
            reset_stop()
            demo.stack_box_on(
                pick_name=_safe(pick_name.get()),
                place_name=_safe(place_name.get()),
                hover_h=float(hover_pick.get()),
                clearance=float(clearance.get()),
                down_time=float(down_t.get()),
                up_time=float(up_t.get()),
                settle_contact=float(settle.get()),
                settle_release=0.20,
            )
        except ValueError:
            messagebox.showerror("Invalid", "Please enter numeric values.")
        except KeyError as e:
            messagebox.showerror("Not found", str(e))

    def place_only_on_body_now():
        """
        仅把当前抓着的物体放到 place_name 的顶面中心（假设抓着的是 box 级别大小）。
        如果你没抓着东西，效果就是夹爪到位然后张开。
        """
        try:
            reset_stop()
            target = _safe(place_name.get())
            place_xy, place_z_top, _ = demo._get_body_top_and_halfz(target)

            # 假设抓着的物体半高与 XML 一样是 0.03（如果你改了尺寸，这里也改）
            pick_halfz = 0.03
            z_target = float(place_z_top + pick_halfz + float(clearance.get()))

            demo.place_only_xy(
                place_xy=(float(place_xy[0]), float(place_xy[1])),
                z_target=z_target,
                down_time=float(down_t.get()),
                up_time=float(up_t.get()),
                hover_h=0.06,
                settle_contact=float(settle.get()),
                settle_release=0.20,
            )
        except ValueError:
            messagebox.showerror("Invalid", "Please enter numeric values.")
        except KeyError as e:
            messagebox.showerror("Not found", str(e))

    row += 1
    ttk.Button(root, text="Pick only", command=pick_only_now)\
        .grid(row=row, column=0, padx=8, pady=12, sticky="w")

    ttk.Button(root, text="Place only (on body)", command=place_only_on_body_now)\
        .grid(row=row, column=1, padx=8, pady=12, sticky="w")

    ttk.Button(root, text="STACK (Pick -> Place on body)", command=stack_now)\
        .grid(row=row, column=2, columnspan=2, padx=8, pady=12, sticky="w")

    ttk.Button(root, text="STOP", command=stop_now)\
        .grid(row=row, column=4, padx=8, pady=12, sticky="w")

    row += 1
    ttk.Label(
        root,
        text="Tip: STACK uses 'go high -> move XY -> go down'. If sliding occurs, increase friction or clearance slightly."
    ).grid(row=row, column=0, columnspan=6, padx=8, pady=6, sticky="w")

    root.mainloop()


if __name__ == "__main__":
    demo = Demo()
    Thread(target=launch_gui, args=(demo,), daemon=True).start()
    demo.start()
