"""B4 experiment: local entry decisions, explicit time-stamped messages."""
import json
import sys
import time
import mujoco
import numpy as np
from scene import build_scene, ROOT
from independent_probe import Robot
from coordination import Bus, Message, Negotiator


def run(disturbance=False):
    model = build_scene(2)
    data = mujoco.MjData(model)
    robots = [Robot(model, data, i, y) for i, y in enumerate((.4, -.4))]
    policies = [Negotiator(i) for i in range(2)]
    gait_time = [0., 0.]
    bus = Bus()
    mujoco.mj_forward(model, data)
    deck = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'deck')
    hinge = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')]
    frames, decisions = [], []
    angular_peak = [0., 0.]
    hinge_dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')]
    start = time.perf_counter()
    while data.time < 160:
        if round(data.time / model.opt.timestep) % 100 == 0:
            messages = [Message(i, float(data.time), p.phase) for i, p in enumerate(policies)]
            positions = {i: data.qpos[r.base:r.base+3].copy() for i, r in enumerate(robots)}
            inbox = bus.deliver(float(data.time), messages, positions)
            for i, (robot, policy) in enumerate(zip(robots, policies)):
                if policy.update(float(data.qpos[robot.base]), robot.fell, inbox[i], float(data.time), angular_peak[i]):
                    decisions.append({'time_s': float(data.time), 'robot': i, 'phase': policy.phase,
                                      'messages': [m.__dict__ for m in inbox[i]], 'local_peak_angular_rate_rad_s': angular_peak[i]})
                angular_peak[i] = 0.
        for i, (robot, policy) in enumerate(zip(robots, policies)):
            moving = policy.phase == 'CROSS' or (policy.phase == 'EXIT' and data.qpos[robot.base] < 1.8)
            if moving:
                gait_time[i] += model.opt.timestep
            robot.control(model, data, gait_time[i] if moving else 0.)
        data.qfrc_applied[hinge_dof] = 500. if disturbance and 20. <= data.time < 20.1 else 0.
        mujoco.mj_step(model, data)
        for i, robot in enumerate(robots):
            base_dof = model.jnt_dofadr[model.body_jntadr[robot.body]]
            angular_peak[i] = max(angular_peak[i], float(np.linalg.norm(data.qvel[base_dof+3:base_dof+6])))
        if not np.isfinite(data.qpos).all():
            raise RuntimeError(f'Invalid physics at {data.time:.3f} s')
        for robot in robots:
            robot.measure(data, deck)
        if round(data.time / model.opt.timestep) % 10 == 0:
            frames.append((float(data.time), data.qpos.copy(), data.qvel.copy(), float(data.qpos[hinge])))
        if any(r.fell for r in robots) or all(r.arrival is not None for r in robots):
            break
    out = ROOT / ('outputs/b5' if disturbance else 'outputs/b4')
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'coordinated.npz', time=np.array([f[0] for f in frames]),
                        qpos=np.array([f[1] for f in frames]), qvel=np.array([f[2] for f in frames]),
                        deck_angle=np.array([f[3] for f in frames]))
    result = {'simulation_seconds': float(data.time), 'wall_seconds': time.perf_counter()-start,
              'disturbance': {'time_s': 20., 'duration_s': .1, 'hinge_torque_nm': 500.} if disturbance else None,
              'peak_deck_angle_rad': float(np.max(np.abs([f[3] for f in frames]))),
              'peer_influenced': [p.peer_influenced for p in policies], 'decisions': decisions,
              'message_events': bus.events,
              'robots': [{'name': r.name, 'final_position_m': data.qpos[r.base:r.base+3].tolist(),
                          'fell': r.fell, 'arrival_time_s': r.arrival, 'peak_tilt_rad': r.peak_tilt,
                          'foot_deck_contact_samples': r.contacts} for r in robots]}
    (out / 'coordinated.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'message_events'}, indent=2))
    if any(r.fell or r.arrival is None or r.contacts == 0 for r in robots) or not any(p.peer_influenced for p in policies):
        raise SystemExit('B4 experiment did not complete; see coordinated.json')


if __name__ == '__main__':
    run(disturbance="--disturbance" in sys.argv)
