"""B3: Two independent Go2 controllers on a shared compliant bridge."""
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from gait import Gait
from scene import build_scene, ROOT


class Robot:
    def __init__(self, model, data, index, y):
        self.name = f'r{index}_'
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.name + 'base')
        self.joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.name + f'{leg}_{joint}_joint')
                       for leg in ('FL', 'FR', 'RL', 'RR') for joint in ('hip', 'thigh', 'calf')]
        self.qpos = model.jnt_qposadr[self.joints]
        self.dof = model.jnt_dofadr[self.joints]
        self.act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.name + f'{leg}_{joint}')
                    for leg in ('FL', 'FR', 'RL', 'RR') for joint in ('hip', 'thigh', 'calf')]
        self.feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, self.name + leg)
                     for leg in ('FL', 'FR', 'RL', 'RR')]
        free_joint = model.body_jntadr[self.body]
        self.base = model.jnt_qposadr[free_joint]
        data.qpos[self.base:self.base+7] = [-1.3, y, .27, 1, 0, 0, 0]
        data.qpos[self.qpos] = [0, .9, -1.8] * 4
        self.gait = Gait(data.qpos[self.qpos])
        self.contacts = 0
        self.peak_tilt = 0.
        self.fell = False
        self.arrival = None

    def control(self, model, data, gait_time=None):
        foot_contacts = [any(c.geom1 == f or c.geom2 == f for c in data.contact[:data.ncon]) for f in self.feet]
        rot = data.xmat[self.body].reshape(3, 3)
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], np.hypot(rot[2, 1], rot[2, 2]))
        target = self.gait.targets(data.time if gait_time is None else gait_time, foot_contacts, roll, pitch)
        data.ctrl[self.act] = np.clip(35 * (target-data.qpos[self.qpos]) - 1.5 * data.qvel[self.dof],
                                     model.actuator_ctrlrange[self.act, 0], model.actuator_ctrlrange[self.act, 1])

    def measure(self, data, deck):
        self.contacts += sum((c.geom1 == deck and c.geom2 in self.feet) or
                             (c.geom2 == deck and c.geom1 in self.feet) for c in data.contact[:data.ncon])
        self.peak_tilt = max(self.peak_tilt, float(np.arccos(np.clip(data.xmat[self.body, 8], -1, 1))))
        self.fell |= bool(data.qpos[self.base+2] < .12 or self.peak_tilt > 1.)
        if self.arrival is None and data.qpos[self.base] > 1.1:
            self.arrival = float(data.time)


def run():
    model = build_scene(2)
    data = mujoco.MjData(model)
    robots = [Robot(model, data, i, y) for i, y in enumerate((.4, -.4))]
    mujoco.mj_forward(model, data)
    deck = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'deck')
    hinge = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')]
    frames, overlap_steps = [], 0
    start = time.perf_counter()
    while data.time < 160:
        for robot in robots:
            robot.control(model, data, 0. if data.qpos[robot.base] >= 1.8 else data.time + model.opt.timestep)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError(f'Invalid physics at {data.time:.3f} s')
        for robot in robots:
            robot.measure(data, deck)
        on_deck = [any((c.geom1 == deck and c.geom2 in robot.feet) or
                       (c.geom2 == deck and c.geom1 in robot.feet) for c in data.contact[:data.ncon]) for robot in robots]
        overlap_steps += all(on_deck)
        if round(data.time / model.opt.timestep) % 10 == 0:
            frames.append((float(data.time), data.qpos.copy(), data.qvel.copy(), float(data.qpos[hinge]), *on_deck))
        if any(r.fell for r in robots) or all(r.arrival is not None for r in robots):
            break
    out = ROOT / 'outputs/b3'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'independent.npz', time=np.array([x[0] for x in frames]),
                        qpos=np.array([x[1] for x in frames]), qvel=np.array([x[2] for x in frames]),
                        deck_angle=np.array([x[3] for x in frames]), foot_deck=np.array([x[4:] for x in frames]))
    result = {'simulation_seconds': float(data.time), 'wall_seconds': time.perf_counter()-start,
              'shared_contact_steps': overlap_steps,
              'peak_deck_angle_rad': float(np.max(np.abs([x[3] for x in frames]))),
              'robots': [{'name': r.name, 'final_position_m': data.qpos[r.base:r.base+3].tolist(),
                          'fell': r.fell, 'arrival_time_s': r.arrival, 'peak_tilt_rad': r.peak_tilt,
                          'foot_deck_contact_samples': r.contacts} for r in robots]}
    (out / 'independent.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if any(r.fell or r.arrival is None or r.contacts == 0 for r in robots) or not overlap_steps:
        raise SystemExit('B3 baseline did not complete; see independent.json')


if __name__ == '__main__':
    run()
