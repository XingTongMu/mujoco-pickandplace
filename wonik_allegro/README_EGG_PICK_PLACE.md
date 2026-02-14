# Wonik Allegro 鸡蛋抓取放置仿真

## 项目概述

本项目实现了一个基于MuJoCo的Wonik Allegro灵巧手自动抓取和放置鸡蛋的仿真系统。

## 文件说明

### 1. `egg_pick_place_scene.xml`
完整的MuJoCo仿真场景文件，包含：
- **Wonik Allegro右手**：16自由度灵巧手模型（4个手指，每个3个关节）
- **鸡蛋模型**：椭球体形状，可进行物理交互
- **环境设置**：包含桌子、地面和放置平台
- **材质和视觉**：完整的材质定义和碰撞几何体
- **传感器**：指尖触觉传感器
- **执行器**：16个位置控制执行器

### 2. `egg_pick_place_control.py`
基础控制程序，包含：
- **状态机**：8个状态的抓取放置序列
- **关节控制**：PD控制器实现平滑运动
- **姿态规划**：张开和闭合姿态的预设配置
- **实时监控**：鸡蛋位置和状态显示

### 3. `egg_pick_place_advanced.py`
高级控制程序，包含：
- **高级轨迹规划**：梯形速度曲线和S曲线平滑
- **10个运动阶段**：初始化→接近→预抓取→抓取→抬起→运输→放置→释放→撤回→复位
- **力控制基础**：PD控制器的力控制框架
- **状态管理**：详细的阶段进度跟踪
- **自动循环**：仿真完成后自动重置

## 环境要求

- Python 3.8+
- MuJoCo 3.0+
- NumPy

安装依赖：
```bash
pip install mujoco numpy
```

## 运行仿真

### 基础版本
```bash
python egg_pick_place_control.py
```

### 高级版本
```bash
python egg_pick_place_advanced.py
```

## 仿真序列

### 基础版状态机
1. **INITIAL** - 初始化，重置关节位置
2. **OPEN_HAND** - 张开手指
3. **MOVE_TO_EGG** - 移动到鸡蛋上方
4. **GRASP** - 闭合手指抓取鸡蛋
5. **LIFT** - 抬起鸡蛋
6. **MOVE_TO_PLACE** - 移动到放置位置
7. **RELEASE** - 张开手指放下鸡蛋
8. **RESET** - 复位
9. **COMPLETE** - 完成（自动循环）

### 高级版运动阶段
1. **INITIALIZE** - 初始化
2. **APPROACH** - 接近目标
3. **PRE_GRASP** - 预抓取位置调整
4. **GRASPING** - 执行抓取
5. **LIFTING** - 抬起物体
6. **TRANSPORTING** - 运输到目标位置
7. **PLACING** - 放置物体
8. **RELEASING** - 释放物体
9. **WITHDRAWING** - 撤回手臂
10. **RESETTING** - 复位
11. **COMPLETE** - 完成

## 控制说明

### 基础版
- 仿真自动运行，显示实时状态信息

### 高级版
- **空格键**: 暂停/继续
- **R键**: 重置仿真
- **Q键**: 退出仿真

## 物理参数

### 鸡蛋参数
- **质量**: 0.06 kg (60g)
- **尺寸**: 椭球体 (0.025 × 0.018 × 0.035 m)
- **材质**: 蛋壳色

### Wonik Allegro参数
- **关节数**: 16个
- **手指数**: 4个 (食指、中指、无名指、拇指)
- **关节范围**: 各手指不同，详见XML文件
- **控制模式**: 位置控制

## 扩展功能

### 1. 修改抓取姿态
在`egg_pick_place_control.py`中修改`grasp_pose()`方法：
```python
def grasp_pose(self):
    # 修改这里的关节角度配置
    grasp_positions = {
        'ffj0': 0.1,
        # ... 其他关节
    }
```

### 2. 调整仿真速度
在高级版中修改`PHASE_DURATIONS`：
```python
PHASE_DURATIONS = {
    MotionPhase.GRASPING: 2.0,  # 增加抓取时间
    # ... 其他阶段
}
```

### 3. 添加力控制
在高级控制器中修改`compute_control()`方法添加力控制逻辑。

## 故障排除

### 视图器无法启动
确保已安装MuJoCo视图器依赖：
```bash
pip install glfw
```

### 仿真不稳定
- 减小控制增益 (`kp`, `kd`)
- 增加仿真步长
- 检查碰撞几何体是否合理

### 鸡蛋掉落
- 调整抓取姿态使手指更贴合鸡蛋
- 增加手指闭合角度
- 检查桌面摩擦系数

## 作者
Matrix Agent

## 许可证
MIT License
