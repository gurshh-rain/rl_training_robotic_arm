from collections import deque
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class ArmPushupEnv(gym.Env):
    FRAME_SKIP = 20
    MAX_STEPS = 625
    MAX_TARGET_STEP = 0.04
    POLICY_CONTRACT = "direct_joint_targets_v1"
    COMMAND_DELAY_STEPS = (0, 3)
    SERVO_SPEED_RANGE = (1.5, 2.5)
    SERVO_DEADBAND_RANGE = (0.005, 0.02)
    SERVO_BACKLASH_RANGE = (0.005, 0.03)
    SERVO_POSITION_BIAS = np.radians(2.0)
    SERVO_POSITION_NOISE = np.radians(0.5)
    SERVO_STRENGTH_RANGE = (0.75, 1.25)
    IMU_ACCEL_BIAS = 0.03
    IMU_ACCEL_NOISE = 0.01
    IMU_GYRO_BIAS = np.radians(1.5)
    IMU_GYRO_NOISE = np.radians(0.5)
    BODY_MASS_RANGE = (0.9, 1.1)
    FLOOR_FRICTION_RANGE = (0.7, 1.3)
    RESET_HEIGHT = 0.12
    UPRIGHT_RESET_HEIGHT = 0.08
    RESET_SETTLE_STEPS = 800
    TIP_HEIGHT = 0.008
    TARGET_LOW = np.array([-np.pi / 2.0, -np.pi / 2.0])
    TARGET_HIGH = np.array([np.pi / 2.0, np.pi / 2.0])
    STABLE_SUCCESS_STEPS = 50
    # Near upright, each full-slew target reversal costs this much per joint.
    # Gated by upright**4 so fast recovery motions stay nearly free.
    UPRIGHT_TARGET_RATE_PENALTY = 1.0
    GOAL_BASE_QUAT = np.array([np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0])
    FALL_SIDES = ("+x", "-x", "+z", "-z")
    FALL_AXES = {
        "+x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([1.0, 0.0, 0.0]),
        "+z": np.array([0.0, 0.0, 1.0]),
        "-z": np.array([0.0, 0.0, 1.0]),
    }
    FALL_ANGLES = {
        "+x": np.pi / 2.0,
        "-x": -np.pi / 2.0,
        "+z": np.pi / 2.0,
        "-z": -np.pi / 2.0,
    }

    def __init__(self, render_mode=None, domain_randomization=True, upright_reset_probability=0.0):
        super().__init__()

        xml_path = Path(__file__).resolve().parent / "models" / "attachment_arm.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.dt = self.model.opt.timestep * self.FRAME_SKIP

        self.base_body_id = self.model.body("base").id
        self.lower_body_id = self.model.body("lower_arm").id
        self.upper_body_id = self.model.body("upper_arm").id
        self.tip_geom_id = self.model.geom("ee_tip_geom").id
        self.floor_geom_id = self.model.geom("floor").id
        self.base_geom_id = self.model.geom("base_collision").id

        self.base_joint_id = self.model.joint("base_free").id
        self.roll_joint_id = self.model.joint("lower_link").id
        self.pitch_joint_id = self.model.joint("upper_link").id
        self.roll_servo_id = self.model.actuator("lower_link_servo").id
        self.pitch_servo_id = self.model.actuator("upper_link_servo").id

        self.base_qpos_adr = self.model.jnt_qposadr[self.base_joint_id]
        self.joint_qpos_adrs = np.array([
            self.model.jnt_qposadr[self.roll_joint_id],
            self.model.jnt_qposadr[self.pitch_joint_id],
        ])
        self.joint_qvel_adrs = np.array([
            self.model.jnt_dofadr[self.roll_joint_id],
            self.model.jnt_dofadr[self.pitch_joint_id],
        ])
        self.servo_ids = np.array([self.roll_servo_id, self.pitch_servo_id])
        self.randomized_body_ids = np.array([
            self.base_body_id,
            self.lower_body_id,
            self.upper_body_id,
        ])
        self.target_range = self.TARGET_HIGH - self.TARGET_LOW
        self.domain_randomization = domain_randomization
        if not 0.0 <= upright_reset_probability <= 1.0:
            raise ValueError("upright_reset_probability must be in [0, 1]")
        self.upright_reset_probability = upright_reset_probability
        self.default_body_mass = self.model.body_mass.copy()
        self.default_body_inertia = self.model.body_inertia.copy()
        self.default_floor_friction = self.model.geom_friction[self.floor_geom_id].copy()
        self.default_force_range = self.model.actuator_forcerange[self.servo_ids].copy()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32)

        self.fall_side = "+x"
        self.servo_targets = np.zeros(2, dtype=np.float64)
        self.actuator_targets = np.zeros(2, dtype=np.float64)
        self.command_queue = deque()
        self.servo_speed = np.full(2, 2.0)
        self.servo_deadband = np.zeros(2)
        self.servo_backlash = np.zeros(2)
        self.servo_position_bias = np.zeros(2)
        self.command_delay_steps = 0
        self.imu_accel_bias = np.zeros(3)
        self.imu_gyro_bias = np.zeros(3)
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_upright = 0.0
        self.upright_steps = 0
        self.step_count = 0

    def _randomize_dynamics(self):
        self.model.body_mass[:] = self.default_body_mass
        self.model.body_inertia[:] = self.default_body_inertia
        self.model.geom_friction[self.floor_geom_id] = self.default_floor_friction
        self.model.actuator_forcerange[self.servo_ids] = self.default_force_range

        if self.domain_randomization:
            mass_scales = self.np_random.uniform(
                *self.BODY_MASS_RANGE,
                size=len(self.randomized_body_ids),
            )
            self.model.body_mass[self.randomized_body_ids] *= mass_scales
            self.model.body_inertia[self.randomized_body_ids] *= mass_scales[:, None]
            friction_scale = self.np_random.uniform(*self.FLOOR_FRICTION_RANGE)
            self.model.geom_friction[self.floor_geom_id] *= friction_scale
            strength = self.np_random.uniform(
                *self.SERVO_STRENGTH_RANGE,
                size=len(self.servo_ids),
            )
            self.model.actuator_forcerange[self.servo_ids] *= strength[:, None]
            self.command_delay_steps = int(self.np_random.integers(
                self.COMMAND_DELAY_STEPS[0],
                self.COMMAND_DELAY_STEPS[1] + 1,
            ))
            self.servo_speed = self.np_random.uniform(
                *self.SERVO_SPEED_RANGE,
                size=2,
            )
            self.servo_deadband = self.np_random.uniform(
                *self.SERVO_DEADBAND_RANGE,
                size=2,
            )
            self.servo_backlash = self.np_random.uniform(
                *self.SERVO_BACKLASH_RANGE,
                size=2,
            )
            self.servo_position_bias = self.np_random.uniform(
                -self.SERVO_POSITION_BIAS,
                self.SERVO_POSITION_BIAS,
                size=2,
            )
            self.imu_accel_bias = self.np_random.uniform(
                -self.IMU_ACCEL_BIAS,
                self.IMU_ACCEL_BIAS,
                size=3,
            )
            self.imu_gyro_bias = self.np_random.uniform(
                -self.IMU_GYRO_BIAS,
                self.IMU_GYRO_BIAS,
                size=3,
            )
        else:
            self.command_delay_steps = 0
            self.servo_speed.fill(2.0)
            self.servo_deadband.fill(0.0)
            self.servo_backlash.fill(0.0)
            self.servo_position_bias.fill(0.0)
            self.imu_accel_bias.fill(0.0)
            self.imu_gyro_bias.fill(0.0)

        mujoco.mj_setConst(self.model, self.data)

    @staticmethod
    def _quat_multiply(left, right):
        lw, lx, ly, lz = left
        rw, rx, ry, rz = right
        return np.array([
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ])

    def _base_upright(self):
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        return float(np.clip(-rotation[2, 1], 0.0, 1.0))

    def _action_to_targets(self, action):
        return self.TARGET_LOW + 0.5 * (action + 1.0) * self.target_range

    def _targets_to_action(self, targets):
        return 2.0 * (targets - self.TARGET_LOW) / self.target_range - 1.0

    def _tip_contacts_floor(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if {contact.geom1, contact.geom2} == {self.floor_geom_id, self.tip_geom_id}:
                return True
        return False

    def _get_obs(self):
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        gravity_in_base = rotation.T @ np.array([0.0, 0.0, 1.0])
        angular_velocity = rotation.T @ self.data.cvel[self.base_body_id][0:3]
        if self.domain_randomization:
            gravity_in_base += self.imu_accel_bias + self.np_random.normal(
                0.0,
                self.IMU_ACCEL_NOISE,
                size=3,
            )
            gravity_in_base /= np.linalg.norm(gravity_in_base)
            angular_velocity += self.imu_gyro_bias + self.np_random.normal(
                0.0,
                self.IMU_GYRO_NOISE,
                size=3,
            )
        side_one_hot = np.zeros(len(self.FALL_SIDES), dtype=np.float32)
        side_one_hot[self.FALL_SIDES.index(self.fall_side)] = 1.0
        return np.concatenate([
            gravity_in_base,
            angular_velocity,
            self.servo_targets,
            [min(self.step_count / self.MAX_STEPS, 1.0)],
            side_one_hot,
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._randomize_dynamics()
        mujoco.mj_resetData(self.model, self.data)

        requested_side = options.get("fall_side") if options else None
        bearing = options.get("fall_bearing_deg") if options else None
        if bearing is not None and (requested_side is not None or not np.isfinite(bearing)):
            raise ValueError("specify a finite fall_bearing_deg or fall_side, not both")
        if requested_side not in (None, *self.FALL_SIDES):
            raise ValueError(f"fall_side must be one of {self.FALL_SIDES}")
        self.fall_side = requested_side or self.np_random.choice(self.FALL_SIDES)

        phi = self.np_random.uniform(0.0, 2.0 * np.pi)
        world_yaw = np.array([np.cos(phi / 2.0), 0.0, 0.0, np.sin(phi / 2.0)])
        start_upright = bearing is None and requested_side is None and self.np_random.random() < self.upright_reset_probability
        if start_upright:
            base_height = self.UPRIGHT_RESET_HEIGHT
            quat = self._quat_multiply(world_yaw, self.GOAL_BASE_QUAT)
            start_pose = self.np_random.uniform(-0.15, 0.15, size=2)
        else:
            fall_angle = self.FALL_ANGLES[self.fall_side]
            fall_axis = self.FALL_AXES[self.fall_side]
            if bearing is not None:
                theta = np.deg2rad(bearing)
                fall_axis = np.array([np.cos(theta), 0.0, np.sin(theta)])
                fall_angle = np.pi / 2.0
            local_fall = np.concatenate((
                [np.cos(fall_angle / 2.0)],
                np.sin(fall_angle / 2.0) * fall_axis,
            ))
            base_height = self.RESET_HEIGHT
            quat = self._quat_multiply(
                world_yaw,
                self._quat_multiply(self.GOAL_BASE_QUAT, local_fall),
            )
            start_pose = np.zeros(2)

        self.data.qpos[self.base_qpos_adr : self.base_qpos_adr + 3] = [0.0, 0.0, base_height]
        self.data.qpos[self.base_qpos_adr + 3 : self.base_qpos_adr + 7] = quat
        self.data.qpos[self.joint_qpos_adrs] = start_pose
        self.servo_targets[:] = start_pose
        self.actuator_targets[:] = start_pose
        self.command_queue = deque(
            start_pose.copy() for _ in range(self.command_delay_steps)
        )
        self.data.ctrl[self.servo_ids] = np.clip(
            self.actuator_targets + self.servo_position_bias,
            self.TARGET_LOW,
            self.TARGET_HIGH,
        )
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.RESET_SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)

        self.step_count = 0
        self.upright_steps = 0
        self.previous_action.fill(0.0)
        self.previous_upright = self._base_upright()
        obs = self._get_obs()
        if bearing is not None:
            # Hardware knows the resting pose, not the axis of the initial drop.
            x, _, z = obs[:3]
            if abs(z) >= abs(x):
                self.fall_side = "+x" if z >= 0 else "-x"
            else:
                self.fall_side = "+z" if x < 0 else "-z"
            obs[9:] = 0
            obs[9 + self.FALL_SIDES.index(self.fall_side)] = 1
        return obs, {"fall_side": self.fall_side}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("action must contain two finite normalized joint targets")
        action = np.clip(action, -1.0, 1.0)
        desired_targets = self._action_to_targets(action)
        target_change = np.clip(
            desired_targets - self.servo_targets,
            -self.MAX_TARGET_STEP,
            self.MAX_TARGET_STEP,
        )
        self.servo_targets += target_change
        self.command_queue.append(self.servo_targets.copy())
        delayed_targets = self.command_queue.popleft()
        raw_error = delayed_targets - self.actuator_targets
        actuator_error = np.sign(raw_error) * np.maximum(np.abs(raw_error) - self.servo_backlash, 0.0)
        actuator_error[np.abs(raw_error) <= self.servo_deadband] = 0.0
        maximum_servo_change = self.servo_speed * self.dt
        self.actuator_targets += np.clip(
            actuator_error,
            -maximum_servo_change,
            maximum_servo_change,
        )
        position_noise = (
            self.np_random.normal(0.0, self.SERVO_POSITION_NOISE, size=2)
            if self.domain_randomization
            else np.zeros(2)
        )
        self.data.ctrl[self.servo_ids] = np.clip(
            self.actuator_targets + self.servo_position_bias + position_noise,
            self.TARGET_LOW,
            self.TARGET_HIGH,
        )

        tip_contact = False
        for _ in range(self.FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
            tip_contact |= self._tip_contacts_floor()

        self.step_count += 1
        obs = self._get_obs()
        upright = self._base_upright()
        progress = upright - self.previous_upright
        self.previous_upright = upright

        joint_positions = self.data.qpos[self.joint_qpos_adrs]
        home_score = float(np.exp(-2.0 * np.square(joint_positions).sum()))
        upper_rotation = self.data.xmat[self.upper_body_id].reshape(3, 3)
        arm_vertical = float(np.clip(upper_rotation[2, 2], 0.0, 1.0))
        angular_speed = np.linalg.norm(self.data.cvel[self.base_body_id][:3])
        base_contact = any(
            {c.geom1, c.geom2} == {self.floor_geom_id, self.base_geom_id} and c.dist < 0.002
            for c in self.data.contact[:self.data.ncon]
        )
        stable_upright = (
            upright > 0.95 and arm_vertical > 0.98 and base_contact and
            np.max(np.abs(joint_positions)) < 0.2 and angular_speed < 0.5
        )
        self.upright_steps = self.upright_steps + 1 if stable_upright else 0
        pose_score = upright * arm_vertical * home_score

        reward = upright + 10.0 * progress
        reward += 3.0 * upright * arm_vertical + 2.0 * upright * home_score
        reward += 5.0 * float(stable_upright)
        reward += 0.1 * (1.0 - upright) * float(tip_contact)
        base_height = self.data.xpos[self.base_body_id][2]
        reward -= 5.0 * max(base_height - 0.25, 0.0)
        reward -= 0.01 * min(angular_speed ** 2, 100.0)
        reward -= 0.005 * np.square(action).sum()
        reward -= 0.01 * np.square(action - self.previous_action).sum()
        target_rate = target_change / self.MAX_TARGET_STEP
        reward -= self.UPRIGHT_TARGET_RATE_PENALTY * upright ** 4 * np.square(target_rate).sum()
        self.previous_action = action.astype(np.float32)

        terminated = False
        truncated = self.step_count >= self.MAX_STEPS
        if base_height > 0.6:
            terminated = True
            reward -= 50.0
        if not (np.isfinite(obs).all() and np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()):
            terminated = True
            reward -= 100.0

        return obs, float(reward), terminated, truncated, {
            "upright": upright,
            "tip_contact": tip_contact,
            "is_success": self.upright_steps >= self.STABLE_SUCCESS_STEPS,
            "arm_home": arm_vertical > 0.98,
            "arm_vertical": arm_vertical,
            "base_contact": base_contact,
            "pose_score": pose_score,
            "upright_steps": self.upright_steps,
            "stable_upright": stable_upright,
            "target_rate": float(np.abs(target_rate).mean()),
            "fall_side": self.fall_side,
        }
