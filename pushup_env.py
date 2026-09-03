from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class ArmPushupEnv(gym.Env):
    FRAME_SKIP = 5
    MAX_STEPS = 2500
    MAX_TARGET_STEP = 0.01
    HOME_TARGET_STEP = 0.002
    HOME_TRANSITION_STEP = 400
    RESIDUAL_ACTION_SCALE = 0.2
    RESET_HEIGHT = 0.12
    UPRIGHT_RESET_HEIGHT = 0.08
    UPRIGHT_RESET_PROBABILITY = 0.0
    RESET_SETTLE_STEPS = 800
    TIP_HEIGHT = 0.008
    TARGET_LOW = np.array([-np.pi / 2.0, -np.pi / 2.0])
    TARGET_HIGH = np.array([np.pi / 2.0, np.pi / 2.0])
    STABLE_SUCCESS_STEPS = 200
    HOME_ACTION = np.array([0.0, 0.0], dtype=np.float32)
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
    STANDING_ACTIONS = {
        "+x": np.array([1.0, 0.0], dtype=np.float32),
        "-x": np.array([-1.0, 0.0], dtype=np.float32),
        "+z": np.array([0.5, 1.0], dtype=np.float32),
        "-z": np.array([0.5, -1.0], dtype=np.float32),
    }
    REFERENCE_ACTIONS = {
        "+x": np.array([[1.0, 0.0]], dtype=np.float32),
        "-x": np.array([[-1.0, 0.0]], dtype=np.float32),
        "+z": np.array([[0.5, 1.0]], dtype=np.float32),
        "-z": np.array([[0.5, -1.0]], dtype=np.float32),
    }
    REFERENCE_STAGE_STEPS = 400

    def __init__(self, render_mode=None):
        super().__init__()

        xml_path = Path(__file__).resolve().parent / "models" / "attachment_arm.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.dt = self.model.opt.timestep * self.FRAME_SKIP

        self.base_body_id = self.model.body("base").id
        self.upper_body_id = self.model.body("upper_arm").id
        self.tip_geom_id = self.model.geom("ee_tip_geom").id
        self.floor_geom_id = self.model.geom("floor").id

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
        self.target_range = self.TARGET_HIGH - self.TARGET_LOW

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float32)

        self.fall_side = "+x"
        self.servo_targets = np.zeros(2, dtype=np.float64)
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_upright = 0.0
        self.upright_steps = 0
        self.step_count = 0

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

    def _reference_action(self):
        if self.step_count >= self.HOME_TRANSITION_STEP:
            return self.HOME_ACTION
        actions = self.REFERENCE_ACTIONS[self.fall_side]
        stage = min(self.step_count // self.REFERENCE_STAGE_STEPS, len(actions) - 1)
        return actions[stage]

    def _get_obs(self):
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        gravity_in_base = rotation.T @ np.array([0.0, 0.0, 1.0])
        linear_velocity = rotation.T @ self.data.cvel[self.base_body_id][3:6]
        angular_velocity = rotation.T @ self.data.cvel[self.base_body_id][0:3]
        return np.concatenate([
            gravity_in_base,
            [self.data.xpos[self.base_body_id][2]],
            linear_velocity,
            angular_velocity,
            self.data.qpos[self.joint_qpos_adrs],
            self.data.qvel[self.joint_qvel_adrs],
            self._targets_to_action(self.servo_targets),
            [min(self.step_count / self.MAX_STEPS, 1.0)],
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        requested_side = options.get("fall_side") if options else None
        if requested_side not in (None, *self.FALL_SIDES):
            raise ValueError(f"fall_side must be one of {self.FALL_SIDES}")
        self.fall_side = requested_side or self.np_random.choice(self.FALL_SIDES)

        phi = self.np_random.uniform(0.0, 2.0 * np.pi)
        world_yaw = np.array([np.cos(phi / 2.0), 0.0, 0.0, np.sin(phi / 2.0)])
        start_upright = requested_side is None and self.np_random.random() < self.UPRIGHT_RESET_PROBABILITY
        if start_upright:
            base_height = self.UPRIGHT_RESET_HEIGHT
            quat = self._quat_multiply(world_yaw, self.GOAL_BASE_QUAT)
            start_pose = self._action_to_targets(self.STANDING_ACTIONS[self.fall_side])
        else:
            fall_angle = self.FALL_ANGLES[self.fall_side]
            fall_axis = self.FALL_AXES[self.fall_side]
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
        self.data.ctrl[self.servo_ids] = self.servo_targets
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.RESET_SETTLE_STEPS):
            mujoco.mj_step(self.model, self.data)

        self.step_count = 0
        self.upright_steps = 0
        self.previous_action.fill(0.0)
        self.previous_upright = self._base_upright()
        return self._get_obs(), {"fall_side": self.fall_side}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        residual_scale = (
            0.0
            if self.step_count >= self.HOME_TRANSITION_STEP
            else self.RESIDUAL_ACTION_SCALE
        )
        control_action = np.clip(
            self._reference_action() + residual_scale * action,
            -1.0,
            1.0,
        )
        desired_targets = self._action_to_targets(control_action)
        max_target_step = (
            self.HOME_TARGET_STEP
            if self.step_count >= self.HOME_TRANSITION_STEP
            else self.MAX_TARGET_STEP
        )
        target_change = np.clip(
            desired_targets - self.servo_targets,
            -max_target_step,
            max_target_step,
        )
        self.servo_targets += target_change
        self.data.ctrl[self.servo_ids] = self.servo_targets

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
        home_weight = np.clip((upright - 0.75) / 0.2, 0.0, 1.0)
        target_action = (
            self.HOME_ACTION
            if self.step_count >= self.HOME_TRANSITION_STEP
            else self.STANDING_ACTIONS[self.fall_side]
        )
        target_pose = self._action_to_targets(target_action)
        pose_score = 0.5 * np.cos(joint_positions - target_pose).sum()
        upper_rotation = self.data.xmat[self.upper_body_id].reshape(3, 3)
        arm_vertical = float(np.clip(upper_rotation[2, 2], 0.0, 1.0))
        stable_upright = upright > 0.95 and arm_vertical > 0.98
        self.upright_steps = self.upright_steps + 1 if stable_upright else 0
        tip_height = self.data.geom_xpos[self.tip_geom_id][2]
        reach_floor = np.exp(-20.0 * abs(tip_height - self.TIP_HEIGHT))
        reference_error = np.square(action).sum()

        reward = 6.0 * upright + 100.0 * progress
        reward += (1.0 - home_weight) * (0.5 * reach_floor + float(tip_contact))
        reward += home_weight * (20.0 * pose_score)
        reward += (1.0 - home_weight) * (
            4.0 * np.exp(-2.0 * reference_error) - 0.25 * reference_error
        )
        reward += 15.0 * float(stable_upright)
        reward -= 20.0 * max(obs[3] - 0.25, 0.0)
        reward -= 0.05 * np.square(obs[7:10]).sum()
        reward -= 0.005 * np.square(action).sum()
        reward -= 0.01 * np.square(action - self.previous_action).sum()
        self.previous_action = action.astype(np.float32)

        terminated = False
        truncated = self.step_count >= self.MAX_STEPS
        if obs[3] > 0.6:
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
            "upright_steps": self.upright_steps,
            "fall_side": self.fall_side,
        }
