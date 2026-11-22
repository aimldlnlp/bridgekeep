"""Run reproducible BRIDGEKEEP experiments; all recorded states come from MuJoCo."""
import argparse
import json
import time
from pathlib import Path
import mujoco
import numpy as np
from coordination import AdaptiveNegotiator, Bus, Message, Negotiator
from independent_probe import Robot
from scene import ROOT, build_scene
from orientation_metrics import angles

DT_MESSAGE = .2
DEADLINE = 160.
SEED = 2026


def tangential_speed(model, data, contact, foot, deck):
    foot_jac = np.zeros((3, model.nv))
    deck_jac = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, foot_jac, None, contact.pos, model.geom_bodyid[foot])
    mujoco.mj_jac(model, data, deck_jac, None, contact.pos, model.geom_bodyid[deck])
    velocity = (foot_jac - deck_jac) @ data.qvel
    normal = contact.frame[:3]
    return float(np.linalg.norm(velocity - np.dot(velocity, normal) * normal))


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def run(scenario, lateral_offset=0., output_root=None):
    if scenario not in ('s1', 's2', 's3', 's4', 's5', 's6'):
        raise ValueError(f'Unknown scenario: {scenario}')
    coordinated = scenario in ('s3', 's4', 's5', 's6')
    count = 1 if scenario == 's1' else 2
    model = build_scene(count)
    data = mujoco.MjData(model)
    ys = (lateral_offset,) if count == 1 else (.4 + lateral_offset, -.4 + lateral_offset)
    robots = [Robot(model, data, i, y) for i, y in enumerate(ys)]
    policies = [(AdaptiveNegotiator if scenario == 's5' else Negotiator)(i) for i in range(count)] if coordinated else []
    bus = Bus()
    gait_times = [0.] * count
    angular_peak = [0.] * count
    deck = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'deck')
    hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')
    hinge_pos, hinge_dof = int(model.jnt_qposadr[hinge]), int(model.jnt_dofadr[hinge])
    mujoco.mj_forward(model, data)
    frames, decisions = [], []
    contact_counts = np.zeros((count, 4), dtype=np.int64)
    max_contact_force = np.zeros((count, 4))
    started = time.perf_counter()
    step = 0

    def capture():
        contacts = np.zeros((count, 4), dtype=bool)
        slip = np.zeros((count, 4))
        for c in data.contact[:data.ncon]:
            for i, robot in enumerate(robots):
                for foot, geom in enumerate(robot.feet):
                    if (c.geom1 == deck and c.geom2 == geom) or (c.geom2 == deck and c.geom1 == geom):
                        contacts[i, foot] = True
                        slip[i, foot] = max(slip[i, foot], tangential_speed(model, data, c, geom, deck))
        phases = [p.phase for p in policies] if coordinated else [
            'EXIT' if r.arrival is not None else 'CROSS' for r in robots]
        frames.append((float(data.time), data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), contacts, phases,
                       data.actuator_force.copy(), slip))

    capture()
    while data.time < DEADLINE:
        if coordinated and step % 100 == 0:
            on_deck = [any((c.geom1 == deck and c.geom2 in robot.feet) or
                           (c.geom2 == deck and c.geom1 in robot.feet)
                           for c in data.contact[:data.ncon]) for robot in robots]
            messages = [Message(i, float(data.time), p.phase, angular_peak[i],
                                1.5 * getattr(p, 'pace', 1.) if p.phase in ('CROSS', 'ADAPT') else 0.,
                                p.phase in ('REQUEST', 'WAIT'), robots[i].fell,
                                float(data.qpos[hinge_pos]) if scenario == 's5' and on_deck[i] else None,
                                float(data.qvel[hinge_dof]) if scenario == 's5' and on_deck[i] else None)
                        for i, p in enumerate(policies)]
            positions = {i: data.qpos[r.base:r.base+3].copy() for i, r in enumerate(robots)}
            inbox = bus.deliver(float(data.time), messages, positions)
            for i, (robot, policy) in enumerate(zip(robots, policies)):
                observation = ((messages[i].deck_angle_rad, messages[i].deck_rate_rad_s, on_deck[i])
                               if scenario == 's5' else ())
                if policy.update(float(data.qpos[robot.base]), robot.fell, inbox[i], float(data.time), angular_peak[i], *observation):
                    decisions.append({'time_s': float(data.time), 'robot': i, 'phase': policy.phase,
                                      'messages': [m.__dict__ for m in inbox[i]],
                                      'local_peak_angular_rate_rad_s': angular_peak[i],
                                      'deck_angle_rad': messages[i].deck_angle_rad,
                                      'deck_rate_rad_s': messages[i].deck_rate_rad_s,
                                      'pace': getattr(policy, 'pace', 1.)})
                angular_peak[i] = 0.
        for i, robot in enumerate(robots):
            if coordinated:
                phase = policies[i].phase
                exit_clearance = 1.5 if scenario in ('s5', 's6') else 1.8
                moving = phase in ('CROSS', 'ADAPT') or (phase == 'EXIT' and data.qpos[robot.base] < exit_clearance)
                if moving:
                    gait_times[i] += model.opt.timestep * getattr(policies[i], 'pace', 1.)
                robot.control(model, data, gait_times[i] if moving else 0.)
            else:
                robot.control(model, data, 0. if data.qpos[robot.base] >= 1.8 else data.time + model.opt.timestep)
        data.qfrc_applied[hinge_dof] = 500. if scenario == 's4' and 20. <= data.time < 20.1 else 0.
        mujoco.mj_step(model, data)
        step += 1
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f'Invalid physics at {data.time:.3f} s')
        for i, robot in enumerate(robots):
            robot.measure(data, deck)
            base_dof = model.jnt_dofadr[model.body_jntadr[robot.body]]
            angular_peak[i] = max(angular_peak[i], float(np.linalg.norm(data.qvel[base_dof+3:base_dof+6])))
        for ci, c in enumerate(data.contact[:data.ncon]):
            for i, robot in enumerate(robots):
                for foot, geom in enumerate(robot.feet):
                    if (c.geom1 == deck and c.geom2 == geom) or (c.geom2 == deck and c.geom1 == geom):
                        contact_counts[i, foot] += 1
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, ci, force)
                        max_contact_force[i, foot] = max(max_contact_force[i, foot], float(force[0]))
        if step % 10 == 0:
            capture()
        if any(r.fell for r in robots) or all(r.arrival is not None for r in robots):
            break
    if frames[-1][0] < data.time:
        capture()
    warnings = {mujoco.mjtWarning(i).name: int(data.warning[i].number)
                for i in range(int(mujoco.mjtWarning.mjNWARNING)) if data.warning[i].number}
    if warnings:
        raise RuntimeError(f'MuJoCo warnings: {warnings}')
    out = (Path(output_root) if output_root is not None else ROOT / 'outputs') / scenario
    out.mkdir(parents=True, exist_ok=True)
    t = np.array([f[0] for f in frames])
    qpos = np.array([f[1] for f in frames])
    qvel = np.array([f[2] for f in frames])
    controls = np.array([f[3] for f in frames])
    contacts = np.array([f[4] for f in frames])
    phases = np.array([f[5] for f in frames])
    actuator_force = np.array([f[6] for f in frames])
    slip = np.array([f[7] for f in frames])
    np.savez_compressed(out / 'recording.npz', time=t, qpos=qpos, qvel=qvel,
                        ctrl=controls, actuator_force_nm=actuator_force, foot_deck=contacts, phases=phases,
                        foot_deck_tangential_speed_m_s=slip)
    stop = 'fall' if any(r.fell for r in robots) else 'arrived' if all(r.arrival is not None for r in robots) else 'timeout'
    elapsed = time.perf_counter() - started
    config = {'scenario': scenario, 'seed': SEED, 'robot_count': count, 'initial_y_m': list(ys),
              'initial_x_m': -1.3, 'timestep_s': model.opt.timestep, 'sample_interval_s': .02,
              'deadline_s': DEADLINE, 'bridge_stiffness_nm_rad': 10000., 'bridge_damping_nms_rad': 80.,
              'bridge_hinge_range_rad': [-.15, .15],
              'exit_surface_height_m': -.015, 'coordination': coordinated,
              'lateral_offset_m': lateral_offset,
              'message_period_s': DT_MESSAGE, 'message_range_m': 4., 'angular_rate_safe_stop_rad_s': 1.5,
              'gait_frequency_hz': 1.5, 'joint_pd_kp': 35., 'joint_pd_kd': 1.5,
              'adaptive_deck_sensor': 'ideal hinge angle and rate available only on foot-deck contact' if scenario == 's5' else None,
              'exit_clearance_x_m': 1.5 if scenario in ('s5', 's6') else 1.8,
              'adaptive_thresholds': {'angle_rad': .012, 'rate_rad_s': .08, 'pace_factor': .85} if scenario == 's5' else None,
              'disturbance': {'start_s': 20., 'duration_s': .1, 'torque_nm': 500.} if scenario == 's4' else None}
    result = {'stop_reason': stop, 'simulation_seconds': float(data.time), 'wall_seconds': elapsed,
              'simulation_speed': float(data.time / elapsed),
              'robots': [{'name': r.name, 'final_position_m': data.qpos[r.base:r.base+3].tolist(),
                          'fell': r.fell, 'arrival_time_s': r.arrival, 'peak_tilt_rad': r.peak_tilt,
                          'foot_deck_contact_samples': contact_counts[i].tolist(),
                          'maximum_foot_deck_normal_force_n': max_contact_force[i].tolist()}
                         for i, r in enumerate(robots)],
              'decisions': decisions, 'message_events': bus.events, 'warnings': warnings}
    arrivals = [r.arrival for r in robots]
    metrics = {'window_s': [float(t[0]), float(t[-1])],
               'bridge_angular_velocity_rms_rad_s': float(np.sqrt(np.mean(qvel[:, hinge_dof]**2))),
               'peak_absolute_bridge_angle_rad': float(np.max(np.abs(qpos[:, hinge_pos]))),
               'team_crossing_time_s': max(arrivals) if all(a is not None for a in arrivals) else None,
               'successful_robots': sum(a is not None and not r.fell for a, r in zip(arrivals, robots)),
               'foot_deck_tangential_speed_peak_m_s': [float(np.max(slip[:, i])) if np.any(contacts[:, i]) else None
                                                       for i in range(count)],
               'foot_deck_samples_above_0_05_m_s': np.sum(slip > .05, axis=(0, 2)).tolist(),
               'slip_proxy_definition': 'Contact-point tangential speed relative to deck; 50 Hz per-foot samples; 0.05 m/s exploratory threshold'}
    metrics['body_orientation_50hz'] = []
    if scenario in ('s5', 's6'):
        metrics['wait_time_s'] = [float(np.sum(np.diff(t) * (phases[:-1, i] == 'WAIT'))) for i in range(count)]
        metrics['adapt_time_s'] = [float(np.sum(np.diff(t) * (phases[:-1, i] == 'ADAPT'))) for i in range(count)]
    for robot in robots:
        roll, pitch = angles(qpos[:, robot.base+3:robot.base+7])
        metrics['body_orientation_50hz'].append({
            'peak_absolute_roll_rad': float(np.max(np.abs(roll))),
            'peak_absolute_pitch_rad': float(np.max(np.abs(pitch)))})
    dump(out / 'config.json', config)
    dump(out / 'episode.json', result)
    dump(out / 'metrics.json', metrics)
    print(json.dumps({'scenario': scenario, 'stop_reason': stop, 'metrics': metrics,
                      'wall_seconds': elapsed, 'robots': result['robots'], 'decisions': decisions}, indent=2))
    return stop


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario', choices=('s1', 's2', 's3', 's4', 's5', 's6'))
    parser.add_argument('--lateral-offset', type=float, default=0.)
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    if run(args.scenario, args.lateral_offset, args.output_root) != 'arrived':
        raise SystemExit(f'{args.scenario} did not complete; inspect its episode.json')
