# MuJoCo Robot Manipulation & Simulation Demo

这是一个基于 MuJoCo 的机器人仿真项目，展示了 Franka Emika FR3 机械臂在多个场景下的应用，包括柔性物体操作（折叠毛巾/衣物）、刚体抓取与堆叠、仿生飞行以及自动化数据采集实验。

## 📦 项目功能

### 1. 刚体操作 (Pick & Place / Stacking)
- **场景**: `block_pick.xml`, `block_stack.xml`
- **功能**:
  - **吸盘抓取**: 使用模拟真空吸盘抓取立方体。
  - **自动堆叠**: 机械臂自动识别并依次抓取红、绿、蓝三个方块，垂直堆叠在一起。
- **演示脚本**:
  - `block_pick_viewer.py`: 单一方块抓取放置演示。
  - `block_stack_viewer.py`: 多方块自动堆叠演示。

### 2. 柔性物体操作 (Deformable Objects)
- **场景**: `towel_fold.xml`, `clothes_fold.xml`
- **功能**:
  - **毛巾折叠**: 模拟柔性网格（Grid）毛巾的抓取与折叠。
  - **衣物操作**: 简单的衣物模型交互。
- **演示脚本**:
  - `towel_macro.py`: 自动执行毛巾折叠序列。
  - `clothes_macro.py`: 自动执行衣物折叠序列。

### 3. 仿生飞行 (Biomimetic Flight)
- **场景**: `fly_scene.xml`
- **功能**:
  - 模拟苍蝇的机翼拍打与飞行动力学。
  - 控制机翼频率与推力实现悬停与前进。
- **演示脚本**:
  - `fly_macro.py`: 自动演示起飞、巡航与降落。

### 4. 自动化实验与数据采集 (Randomized Experiments)
- **功能**:
  - 随机生成不同形状（球体、立方体、圆柱、胶囊）、尺寸和位置的物体。
  - 自动执行抓取测试并记录结果（成功率、物体属性）。
  - 数据存储于 SQLite 数据库中。
- **核心脚本**:
  - `random_grasp_app.py`: 主程序，循环生成场景并运行实验。
  - `scene_gen.py`: 基于模板生成随机场景 XML。
  - `grasp_db.py`: 数据库管理模块。

### 5. 摄像头手势控制 (Camera Hand Control)
- **功能**:
  - 利用单目摄像头（Webcam）捕捉手部动作。
  - **实时映射**: 手腕位置控制机械臂/机械手移动，手掌大小控制高度，手指动作控制抓取/弯曲。
  - **支持模型**:
    1. **Block Pick (方块抓取)**: 控制 Franka 机械臂移动与吸盘抓取。
    2. **Allegro Hand (灵巧手)**: 控制 Wonik Allegro Hand 的4指弯曲与手掌空间移动。
- **运行方式**:
  ```bash
  # 1. 安装依赖
  pip install flask flask-socketio eventlet mediapipe
  
  # 2. 启动方块抓取控制
  python web_control_server.py
  
  # 3. 启动灵巧手控制
  python web_control_allegro.py
  ```
  启动后访问浏览器: `http://localhost:5000`

## 🚀 如何运行

### 环境依赖
- Python 3.8+
- MuJoCo (`pip install mujoco`)
- NumPy (`pip install numpy`)
- Flask, MediaPipe (仅手势控制需要)

### 运行示例

**1. 启动交互式堆叠演示**
```bash
python block_stack_viewer.py
```

**2. 运行随机抓取实验**
```bash
python random_grasp_app.py
```
*实验数据将保存在 `grasp_experiments.db` 中。*

**3. 查看 MuJoCo 场景**
你可以直接使用 MuJoCo 自带的模拟器查看 XML 场景：
```bash
# Windows
bin/simulate.exe block_stack.xml
```

## 📂 文件结构
- `*.xml`: MuJoCo 场景描述文件
- `*_viewer.py`: 基于 `mujoco.viewer` 的交互式 Python 演示脚本
- `*_macro.py`: 用于控制动作序列的宏脚本
- `assets/`: 3D 模型与贴图资源
- `mujoco_menagerie-main/`: 第三方机器人模型库 (Franka FR3)

## 📝 备注
- 本项目使用了 MuJoCo 的 `adhesion` 执行器来模拟真空吸盘。
- 柔性物体仿真依赖于 MuJoCo 的 `flexcomp` 功能。
