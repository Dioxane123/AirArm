"""在 IK 参考状态和 MuJoCo 实际状态之间建立执行器控制链路。"""

from dataclasses import dataclass
from pathlib import Path

import mink
import mujoco
import numpy as np


DEFAULT_SCENE = (
    Path(__file__).resolve().parents[1]
    / "third_party/mink/examples/universal_robots_ur5e/scene.xml"
)
JOINT_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3",
)


class ArmControlError(RuntimeError):
    """不能继续产生有效控制命令；调用方应停止当前任务并报告错误。"""


@dataclass(frozen=True)
class ArmState:
    sim_time: float
    target_xyz: tuple[float, ...]
    actual_xyz: tuple[float, ...]
    q_reference: tuple[float, ...]
    q_actual: tuple[float, ...]
    position_error_m: float
    orientation_error_rad: float
    ik_position_error_m: float
    max_actual_joint_speed: float
    paused: bool


class ArmController:
    """只支持六关节 UR5e 位置执行器模型，控制目标位于初始 XY 平面。"""

    def __init__(
        self, scene: Path = DEFAULT_SCENE, *, control_dt: float = 0.01,
        max_joint_speed: float = 0.6, workspace_half_size: float = 0.06,
    ):
        scene = Path(scene)
        if not scene.is_file():
            raise FileNotFoundError(f"UR5e scene missing: {scene}. See docs/DAY1_SETUP.md for model setup.")
        for name, value in (
            ("control_dt", control_dt), ("max_joint_speed", max_joint_speed),
            ("workspace_half_size", workspace_half_size),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")

        self.scene = scene.resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.data = mujoco.MjData(self.model)
        ratio = control_dt / self.model.opt.timestep
        self.physics_steps = round(ratio)
        if self.physics_steps < 1 or not np.isclose(ratio, self.physics_steps, rtol=0, atol=1e-9):
            raise ValueError("control_dt must be an integer multiple of the model physics timestep.")
        self.control_dt = float(control_dt)
        self.max_joint_speed = float(max_joint_speed)

        # 按名称建立映射，不假定 qpos 的顺序一定与 actuator 的顺序相同。
        joint_ids = np.array([self.model.joint(name).id for name in JOINT_NAMES]) # 关节id
        self.actuator_ids = np.array([self.model.actuator(name).id for name in JOINT_NAMES]) # 执行器id
        self.qpos_ids = self.model.jnt_qposadr[joint_ids].copy() # 通用位置，adr: 地址
        self.dof_ids = self.model.jnt_dofadr[joint_ids].copy() # 自由度
        for joint_id, actuator_id in zip(joint_ids, self.actuator_ids):
            if (
                self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE # 检查是否为Hinge,仅有一个旋转自由度
                or self.model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT # 检查执行器类型是否为直接作用于关节
                or self.model.actuator_trnid[actuator_id, 0] != joint_id # 检查执行器是否作用于该关节
                or not np.allclose(self.model.actuator_gear[actuator_id], [1, 0, 0, 0, 0, 0]) # 检查执行器齿轮传动比是否为1
                or self.model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_NONE # 检查执行器动力学类型是否有额外的动力学特性
                or self.model.actuator_gaintype[actuator_id] != mujoco.mjtGain.mjGAIN_FIXED # 检查执行器增益类型是否为固定增益
                or self.model.actuator_biastype[actuator_id] != mujoco.mjtBias.mjBIAS_AFFINE # 检查执行器偏置类型是否为仿射
                or self.model.actuator_gainprm[actuator_id, 0] <= 0 # 检查执行器增益参数是否为正数
                or not np.isclose(
                    self.model.actuator_biasprm[actuator_id, 1],
                    -self.model.actuator_gainprm[actuator_id, 0],
                ) # 检查执行器偏置参数是否与增益参数成反比
                or not self.model.actuator_ctrllimited[actuator_id] # 检查执行器是否有限制控制输入
                or not self.model.jnt_limited[joint_id] # 检查关节是否有限制运动范围
            ):
                raise ValueError("Expected named hinge joints with direct position actuators and limits.")
        self.lower = np.maximum(
            self.model.jnt_range[joint_ids, 0],
            self.model.actuator_ctrlrange[self.actuator_ids, 0],
        )
        self.upper = np.minimum(
            self.model.jnt_range[joint_ids, 1],
            self.model.actuator_ctrlrange[self.actuator_ids, 1],
        )
        self.site_id = self.model.site("attachment_site").id
        self.marker_id = int(self.model.body("target").mocapid[0])
        if self.marker_id < 0:
            raise ValueError("The scene must contain a mocap body named 'target'.")

        # 仅在初始化时设置真实关节位置。正常运行只写 ctrl 并执行 mj_step。
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)
        self.home_xyz = self.data.site_xpos[self.site_id].copy()
        self.home_rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        self.workspace_low = self.home_xyz[:2] - workspace_half_size
        self.workspace_high = self.home_xyz[:2] + workspace_half_size
        self.target_xyz = self.home_xyz.copy()
        self.paused = False

        # configuration 有自己的 MjData；它保存 IK 参考，不是物理仿真的 data。
        self.configuration = mink.Configuration(self.model, self.data.qpos.copy())
        self.end_effector_task = mink.FrameTask(
            "attachment_site", "site", position_cost=1.0, orientation_cost=1.0,
            lm_damping=1e-3,
        )
        self.posture_task = mink.PostureTask(self.model, cost=1e-4)
        self.posture_task.set_target_from_configuration(self.configuration)
        self.tasks = [self.end_effector_task, self.posture_task]
        self.limits = [
            mink.ConfigurationLimit(self.model),
            mink.VelocityLimit(self.model, {name: max_joint_speed for name in JOINT_NAMES}),
        ]
        # 小工作区域位于障碍上方；此版本不是完整的碰撞规避或全局规划器。
        self._write_reference(self.configuration.q)
        self.set_target(self.home_xyz[:2])

    def _write_reference(self, q_reference: np.ndarray) -> None:
        values = np.asarray(q_reference)[self.qpos_ids]
        if (
            values.shape != (6,) or not np.all(np.isfinite(values))
            or np.any(values < self.lower - 1e-9) or np.any(values > self.upper + 1e-9)
        ):
            raise ArmControlError("Joint reference is non-finite or outside actuator/joint limits.")
        self.data.ctrl[self.actuator_ids] = np.clip(values, self.lower, self.upper)

    def set_target(self, xy) -> None:
        """设置世界坐标 XY（米）；无效请求被拒绝，之前的目标保持不变。"""
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise ValueError("Target must contain two finite world coordinates: x and y.")
        if np.any(xy < self.workspace_low - 1e-12) or np.any(xy > self.workspace_high + 1e-12):
            raise ValueError("Target is outside the small verified XY workspace.")
        self.target_xyz = np.array([xy[0], xy[1], self.home_xyz[2]])
        pose = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(self.home_rotation), self.target_xyz,
        )
        self.end_effector_task.set_target(pose)
        # mocap 只移动显示目标，不会直接移动机器人关节。
        self.data.mocap_pos[self.marker_id] = self.target_xyz
        self.data.mocap_quat[self.marker_id] = mink.SO3.from_matrix(self.home_rotation).wxyz

    def nudge(self, dx: float, dy: float) -> None:
        """键盘每次移动一个小步，超出活动区域的部分被截断。"""
        delta = np.asarray([dx, dy], dtype=float)
        if not np.all(np.isfinite(delta)):
            raise ValueError("Target increment must be finite.")
        self.set_target(np.clip(self.target_xyz[:2] + delta, self.workspace_low, self.workspace_high))

    def pause(self) -> None:
        """停止生成新参考，保持当前实际关节附近；物理仿真仍继续。"""
        self.paused = True
        self.configuration.update(self.data.qpos.copy())
        self._write_reference(self.configuration.q)

    def resume(self) -> None:
        """从当前实际状态继续逼近现有目标，避免继续使用旧的 IK 参考。"""
        self.configuration.update(self.data.qpos.copy())
        self._write_reference(self.configuration.q)
        self.paused = False

    def compute_control(self) -> None:
        """一个控制周期只求解一次；此函数不推进物理，也不改写 data.qpos。"""
        if self.paused:
            return
        previous = self.configuration.q.copy()
        try:
            velocity = np.asarray(mink.solve_ik(
                self.configuration, self.tasks, self.control_dt,
                solver="daqp", limits=self.limits, safety_break=True,
            ))
            if velocity.shape != (self.model.nv,) or not np.all(np.isfinite(velocity)):
                raise ArmControlError("IK returned an invalid joint velocity.")
            if np.max(np.abs(velocity[self.dof_ids])) > self.max_joint_speed + 1e-3:
                raise ArmControlError("IK exceeded the configured joint reference speed.")
            # QP 解有数值容差；最后再精确限制发送参考的速度。
            velocity = np.clip(velocity, -self.max_joint_speed, self.max_joint_speed)
            self.configuration.integrate_inplace(velocity, self.control_dt)
            self._write_reference(self.configuration.q)
        except (mink.MinkError, ValueError, np.linalg.LinAlgError, ArmControlError) as error:
            self.configuration.update(previous)
            self.pause()
            raise ArmControlError(f"IK failed; entered hold mode: {error}") from error

    def step(self) -> ArmState:
        """计算参考 → 写执行器目标 → 多个物理步 → 读取最新实际状态。"""
        self.compute_control()
        before = self.data.time
        mujoco.mj_step(self.model, self.data, nstep=self.physics_steps)
        if (
            not np.all(np.isfinite(self.data.qpos))
            or not np.all(np.isfinite(self.data.qvel))
            or not np.isclose(self.data.time, before + self.control_dt, rtol=0, atol=1e-8)
        ):
            raise ArmControlError("Physics state became invalid or the simulator reset unexpectedly.")
        # mj_step 后，重新计算派生量，避免读取到积分前的 site_xpos。
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self) -> ArmState:
        actual = self.data.site_xpos[self.site_id].copy()
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
        rotation_error = mink.SO3.from_matrix(self.home_rotation.T @ rotation).log()
        ik_position = self.configuration.get_transform_frame_to_world("attachment_site", "site").translation()
        return ArmState(
            sim_time=float(self.data.time),
            target_xyz=tuple(float(v) for v in self.target_xyz),
            actual_xyz=tuple(float(v) for v in actual),
            q_reference=tuple(float(v) for v in self.data.ctrl[self.actuator_ids]),
            q_actual=tuple(float(v) for v in self.data.qpos[self.qpos_ids]),
            position_error_m=float(np.linalg.norm(self.target_xyz - actual)),
            orientation_error_rad=float(np.linalg.norm(rotation_error)),
            ik_position_error_m=float(np.linalg.norm(self.target_xyz - ik_position)),
            max_actual_joint_speed=float(np.max(np.abs(self.data.qvel[self.dof_ids]))),
            paused=self.paused,
        )

    def at_target(self) -> bool:
        state = self.state()
        return (
            not self.paused and state.position_error_m <= 0.015
            and state.orientation_error_rad <= 0.04
            and state.ik_position_error_m <= 0.001
            and state.max_actual_joint_speed <= 0.05
        )
