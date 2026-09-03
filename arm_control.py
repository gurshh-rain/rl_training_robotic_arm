import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


xml_path = Path(__file__).resolve().parent / "models" / "two_dof_arm.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

servo_ids = np.array([
    model.actuator("shoulder_roll_servo").id,
    model.actuator("shoulder_pitch_servo").id,
])
targets = np.array([0.0, 0.0])

mujoco.mj_resetData(model, data)
data.ctrl[servo_ids] = targets
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        data.ctrl[servo_ids] = targets
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(max(0.0, model.opt.timestep - (time.time() - step_start)))
