# AirArm

普通摄像头驱动仿真机械臂的交互项目。

摄像头手部识别与仿真机械臂联动：

```sh
uv run --locked python main.py --mode teleop
```

窗口左侧显示手部，右侧显示机械臂。启动保持静止；单手进入画面后按空格启用，慢慢移动手掌即可控制机械臂。空格暂停/重新对齐，`I` 切换关键点编号，`Q/Esc` 退出。丢手、多手或输入过期导致的暂停，在正常单手检测恢复后会自动继续；空格造成的手动暂停只能再次按空格解除。

从项目根目录运行现有手部识别：

```sh
uv run --locked python main.py
```

运行机械臂预设目标演示：

```sh
uv run --locked python main.py --mode arm
```

键盘控制：

```sh
uv run --locked python main.py --mode arm --control keyboard
```

`W/S` 调整世界 X，`A/D` 调整世界 Y，`H` 返回初始目标，空格保持/恢复，`Q/Esc` 退出。

无窗口验证与测试：

```sh
uv run --locked python main.py --mode arm --headless --report reports/day3-preset.json
uv run --locked python -m unittest discover -s tests -v
```

机械臂资源使用 Mink 配套 UR5e 场景。测试使用真实模型与物理仿真；控制测试不需要摄像头、图形窗口或 GPU。
