"""Joint PD with explicit gains for the foot-trajectory experiment."""
import numpy as np
from independent_probe import Robot


class ProbeRobot(Robot):
    def __init__(self, model, data, index, y, kp, kd, lane_gain=0.):
        super().__init__(model, data, index, y)
        self.kp, self.kd = kp, kd
        self.lane, self.lane_gain = y, lane_gain

    def control(self, model, data, gait_time=None):
        contacts = [any(c.geom1 == f or c.geom2 == f for c in data.contact[:data.ncon]) for f in self.feet]
        rot = data.xmat[self.body].reshape(3, 3)
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], np.hypot(rot[2, 1], rot[2, 2]))
        target = self.gait.targets(data.time if gait_time is None else gait_time, contacts, roll, pitch)
        if self.lane_gain:
            # Ideal local lateral odometry; hip sign reverses for the opposing heading.
            target[::3] += np.clip(self.lane_gain * (data.qpos[self.base+1] - self.lane) * rot[1,1], -.15, .15)
        data.ctrl[self.act] = np.clip(self.kp * (target-data.qpos[self.qpos]) - self.kd * data.qvel[self.dof],
                                     model.actuator_ctrlrange[self.act, 0], model.actuator_ctrlrange[self.act, 1])
