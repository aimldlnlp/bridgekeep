"""Opposed Go2 crossing on a passive seesaw. Existing scenarios stay unchanged."""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
import mujoco
import numpy as np
from scene import ROOT, build_scene
from independent_probe import Robot
from coordination import Bus
from gait import Gait
from run import tangential_speed
from foot_control import ProbeRobot
from foot_trajectory import FootTrajectory


class LiftGait(Gait):
    def __init__(self, home, lift):
        super().__init__(home)
        self.lift = lift

    def targets(self, time_s, contacts, roll, pitch):
        target = super().targets(time_s, contacts, roll, pitch)
        if time_s >= 1.:
            for leg, phase in enumerate((0, np.pi, np.pi, 0)):
                target[3 * leg + 2] += self.lift * max(0., np.sin(2 * np.pi * 1.5 * (time_s - 1.) + phase))
        return target


@dataclass(frozen=True)
class ReadyMessage:
    sender: int
    time_s: float
    phase: str
    deck_angle_rad: float | None = None
    deck_rate_rad_s: float | None = None
    progress_m: float | None = None
    on_deck: bool = False


def observed_pace(moving, observations):
    return .85 if moving and any(abs(m.deck_angle_rad) > .02 or abs(m.deck_rate_rad_s) > .08
                                 for m in observations if m.deck_angle_rad is not None and m.deck_rate_rad_s is not None) else 1.


def progress_pace(moving, progress, inbox, slow=.98):
    return slow if moving and any(m.on_deck and m.progress_m is not None and progress > m.progress_m + .1 for m in inbox) else 1.


def cleared_landing(model, data, robot, direction, landing, deck, margin=0.):
    """All feet clear the deck edge and at least one supports the target landing."""
    if not all(direction * data.geom_xpos[foot, 0] - model.geom_size[foot, 0] > 2. + margin for foot in robot.feet):
        return False
    contacts = [(int(c.geom1), int(c.geom2)) for c in data.contact[:data.ncon]]
    if any(deck in pair and any(foot in pair for foot in robot.feet) for pair in contacts):
        return False
    return any(landing in pair and any(foot in pair for foot in robot.feet) for pair in contacts)


def run(mode, output, delay=0., disturbance=False, lateral_offset=0., lift=0., foot_gait=False, adapt=False, half_width=1., stiffness=10000., staggered=False, balance_progress=False, exit_margin=0., lane_gain=0., transition_lip=False, hinged_lip=False):
    model = build_scene(2, counterbalance=True, half_width=half_width, bridge_stiffness=stiffness, transition_lip=transition_lip, hinged_lip=hinged_lip)
    data = mujoco.MjData(model)
    robots = [ProbeRobot(model, data, i, y, 140., 3., lane_gain) if foot_gait else Robot(model, data, i, y)
              for i, y in enumerate((.45 + lateral_offset, -.45 + lateral_offset))]
    if foot_gait:
        for robot in robots:
            robot.gait = FootTrajectory(robot.gait.home, stride=.05, height=.1)
    if lift:
        for robot in robots:
            robot.gait = LiftGait(robot.gait.home, lift)
    data.qpos[robots[0].base:robots[0].base+7] = [-2.35, .45 + lateral_offset, .27, 1, 0, 0, 0]
    data.qpos[robots[1].base:robots[1].base+7] = [2.35, -.45 + lateral_offset, .27, 0, 0, 0, 1]
    mujoco.mj_forward(model, data)
    deck = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'deck')
    landings = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ('exit', 'approach')]
    hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')
    q, v = int(model.jnt_qposadr[hinge]), int(model.jnt_dofadr[hinge])
    clocks = [0., 0.]
    bus = Bus(max_range_m=8.)
    moving = [False, False]
    pace = [1., 1.]
    decisions = []
    arrived = [None, None]
    first_contact = [None, None]
    robot_collisions = 0
    robot_contact_steps = 0
    lips = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ('left_lip','right_lip')} if transition_lip or hinged_lip else set()
    lip_contact_steps = [0, 0]; lip_peak_n = [0., 0.]; lip_impulse_ns = [0., 0.]
    lip_deck_impulse_ns = 0.; lip_deck_peak_n = 0.
    lip_deck_moment_ns = 0.; lip_deck_abs_moment_ns = 0.; lip_deck_first20_moment_ns = 0.
    first_lip_contact = [None, None]
    deck_impulse_ns = [0., 0.]; lip_first20_ns = [0., 0.]; deck_first20_ns = [0., 0.]
    geom_owners = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        while body and all(body != r.body for r in robots):
            body = int(model.body_parentid[body])
        geom_owners.append(next((i for i, r in enumerate(robots) if body == r.body), None))
    records = []
    started = time.perf_counter()
    # The reference leaves B standing on its landing throughout the run.
    for step in range(80000):
        if mode == 'coordinated' and step % 100 == 0:
            ready = [True, data.time >= delay]
            on_deck = [any((c.geom1 == deck and c.geom2 in r.feet) or (c.geom2 == deck and c.geom1 in r.feet)
                           for c in data.contact[:data.ncon]) for r in robots]
            progress = [(1 if i == 0 else -1) * float(data.qpos[r.base]) for i, r in enumerate(robots)]
            messages = [ReadyMessage(i, float(data.time), 'READY' if ready[i] else 'WAIT',
                                     float(data.qpos[q]) if adapt and on_deck[i] else None,
                                     float(data.qvel[v]) if adapt and on_deck[i] else None,
                                     progress[i] if balance_progress else None, on_deck[i]) for i in range(2)]
            inbox = bus.deliver(float(data.time), messages,
                                {i: data.qpos[r.base:r.base+3].copy() for i, r in enumerate(robots)})
            for i in range(2):
                if not moving[i] and ready[i] and any(m.phase == 'READY' or (staggered and i == 0) for m in inbox[i]):
                    moving[i] = True
                    decisions.append({'time_s': float(data.time), 'robot': i, 'decision': 'ENTER',
                                      'peer_message': inbox[i][0].phase})
                if adapt or balance_progress:
                    observations = [m for m in [messages[i], *inbox[i]] if m.deck_angle_rad is not None]
                    new_pace = min(observed_pace(moving[i], observations) if adapt else 1.,
                                   progress_pace(moving[i], progress[i], inbox[i]) if balance_progress else 1.)
                    if new_pace != pace[i]:
                        pace[i] = new_pace
                        decisions.append({'time_s': float(data.time), 'robot': i, 'decision': 'PACE', 'pace': new_pace})
        for i, robot in enumerate(robots):
            active = arrived[i] is None and (moving[i] if mode == 'coordinated' else
                                              i == 0 or (mode == 'pair' and data.time >= delay))
            if active:
                clocks[i] += model.opt.timestep * pace[i]
            robot.control(model, data, clocks[i] if active else 0.)
        data.qfrc_applied[v] = 100. if disturbance and 20. <= data.time < 20.1 else 0.
        pre_time, pre_qpos, pre_qvel = float(data.time), data.qpos.copy(), data.qvel.copy()
        mujoco.mj_step(model, data)
        lip_step_force = [0., 0.]
        deck_step_force = [0., 0.]
        for ci,c in enumerate(data.contact[:data.ncon]):
            is_lip = c.geom1 in lips or c.geom2 in lips
            is_deck = deck in (c.geom1,c.geom2)
            if not is_lip and not is_deck:
                continue
            support = lips if is_lip else {deck}
            other = int(c.geom2 if c.geom1 in support else c.geom1)
            if is_lip and other == deck:
                force = np.zeros(6); mujoco.mj_contactForce(model,data,ci,force)
                lip_deck_peak_n = max(lip_deck_peak_n, float(force[0]))
                lip_deck_impulse_ns += float(force[0]) * model.opt.timestep
                world_force = c.frame.reshape(3,3).T @ force[:3]
                if c.geom1 == deck:
                    world_force = -world_force
                moment = float(np.cross(c.pos - data.xanchor[hinge], world_force) @ data.xaxis[hinge])
                lip_deck_moment_ns += moment * model.opt.timestep
                lip_deck_abs_moment_ns += abs(moment) * model.opt.timestep
                if pre_time < 20.:
                    lip_deck_first20_moment_ns += moment * model.opt.timestep
            owner = geom_owners[other]
            if owner is not None:
                force = np.zeros(6); mujoco.mj_contactForce(model,data,ci,force)
                (lip_step_force if is_lip else deck_step_force)[owner] += float(force[0])
                if is_lip and first_lip_contact[owner] is None:
                    first_lip_contact[owner] = pre_time
        for i in range(2):
            lip_contact_steps[i] += lip_step_force[i] > 0.
            lip_peak_n[i] = max(lip_peak_n[i],lip_step_force[i])
            lip_impulse_ns[i] += lip_step_force[i] * model.opt.timestep
            deck_impulse_ns[i] += deck_step_force[i] * model.opt.timestep
            if pre_time < 20.:
                lip_first20_ns[i] += lip_step_force[i] * model.opt.timestep
                deck_first20_ns[i] += deck_step_force[i] * model.opt.timestep
        if any(geom_owners[int(c.geom1)] is not None and geom_owners[int(c.geom2)] is not None and
               geom_owners[int(c.geom1)] != geom_owners[int(c.geom2)] for c in data.contact[:data.ncon]):
            robot_contact_steps += 1
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f'Invalid physics at {data.time:.3f} s')
        for i, robot in enumerate(robots):
            robot.measure(data, deck)
            clear = cleared_landing(model, data, robot, (1, -1)[i], landings[i], deck, exit_margin)
            if arrived[i] is None and clear:
                arrived[i] = float(data.time)
            elif arrived[i] is not None and not clear:
                arrived[i] = None
        if step % 10 == 0 or any(robot.fell for robot in robots) or all(t is not None for t in arrived) or step == 79999:
            torque = np.zeros(2)
            feet = np.zeros((2, 4), dtype=bool)
            slip = np.zeros((2, 4))
            other_deck_contacts = 0
            for ci, c in enumerate(data.contact[:data.ncon]):
                owners = (geom_owners[int(c.geom1)], geom_owners[int(c.geom2)])
                if owners[0] is not None and owners[1] is not None and owners[0] != owners[1]:
                    robot_collisions += 1
                if deck not in (c.geom1, c.geom2):
                    continue
                other = c.geom2 if c.geom1 == deck else c.geom1
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, force)
                # Contact force is on geom2, expressed in the contact frame.
                world_force = c.frame.reshape(3, 3).T @ force[:3]
                if c.geom1 == deck:
                    world_force = -world_force
                moment = np.cross(c.pos - data.xanchor[hinge], world_force) @ data.xaxis[hinge]
                owner = next((i for i, robot in enumerate(robots)
                              if other in robot.feet), None)
                if owner is None:
                    other_deck_contacts += 1
                    continue
                torque[owner] += moment
                feet[owner, robots[owner].feet.index(other)] = True
                leg = robots[owner].feet.index(other)
                slip[owner, leg] = max(slip[owner, leg], tangential_speed(model, data, c, other, deck))
                if first_contact[owner] is None:
                    first_contact[owner] = pre_time
            records.append((pre_time, pre_qpos, pre_qvel, torque, feet,
                            data.ctrl.copy(), other_deck_contacts, slip))
        if any(robot.fell for robot in robots) or all(t is not None for t in arrived):
            break
    warnings = {mujoco.mjtWarning(i).name: int(data.warning[i].number)
                for i in range(int(mujoco.mjtWarning.mjNWARNING)) if data.warning[i].number}
    if warnings:
        raise RuntimeError(f'MuJoCo warnings: {warnings}')
    mujoco.mj_forward(model, data)
    terminal_clearance = [cleared_landing(model, data, robot, (1, -1)[i], landings[i], deck, exit_margin)
                          for i, robot in enumerate(robots)]
    path = Path(output) / mode
    path.mkdir(parents=True, exist_ok=True)
    t, qpos, qvel, torque, contacts, controls, other, slip = map(np.array, zip(*records))
    np.savez_compressed(path / 'recording.npz', time=t, qpos=qpos, qvel=qvel,
                        contact_torque_nm=torque, foot_deck=contacts, ctrl=controls,
                        other_deck_contacts=other, foot_deck_tangential_speed_m_s=slip,
                        terminal_time=float(data.time), terminal_qpos=data.qpos.copy(), terminal_qvel=data.qvel.copy())
    result = {'mode': mode, 'delay_s': delay, 'timestep_s': model.opt.timestep,
              'bridge_length_m': 4., 'bridge_half_width_m': half_width, 'bridge_mass_kg': 20., 'stiffness_nm_rad': stiffness,
              'disturbance': {'start_s': 20., 'duration_s': .1, 'torque_nm': 100.} if disturbance else None,
              'damping_nms_rad': 80., 'initial_x_m': [-2.35, 2.35],
              'lateral_offset_m': lateral_offset,
              'exit_margin_m': exit_margin,
              'lane_gain': lane_gain,
              'lane_sensor': 'ideal local lateral position' if lane_gain else None,
              'transition_lip': transition_lip,
              'hinged_lip': hinged_lip,
              'lip_deck_normal_impulse_ns': lip_deck_impulse_ns,
              'lip_deck_peak_normal_force_n': lip_deck_peak_n,
              'lip_deck_signed_moment_impulse_nms': lip_deck_moment_ns,
              'lip_deck_absolute_moment_impulse_nms': lip_deck_abs_moment_ns,
              'lip_deck_first20_signed_moment_impulse_nms': lip_deck_first20_moment_ns,
              'first_robot_lip_contact_s': first_lip_contact,
              'lip_contact_steps': lip_contact_steps,
              'lip_peak_total_normal_force_n': lip_peak_n,
              'lip_normal_impulse_ns': lip_impulse_ns,
              'deck_normal_impulse_ns': deck_impulse_ns,
              'lip_first20_normal_impulse_ns': lip_first20_ns,
              'deck_first20_normal_impulse_ns': deck_first20_ns,
              'terminal_landing_clear': terminal_clearance,
              'extra_knee_lift_rad': lift,
              'controller': 'foot trajectory, PD 140/3' if foot_gait else 'baseline PD 35/1.5',
              'adaptive_pacing': adapt,
              'staggered_entry': staggered,
              'balance_progress': balance_progress,
              'progress_slow_factor': .98 if balance_progress else None,
              'adaptive_sensor': 'ideal deck angle/rate only while touching deck' if adapt else None,
              'simulation_seconds': float(data.time), 'wall_seconds': time.perf_counter()-started,
              'arrival_s': arrived, 'fell': [r.fell for r in robots],
              'first_foot_deck_contact_s': first_contact,
              'peak_angle_rad': float(np.max(np.abs(qpos[:, q]))),
              'rms_rate_rad_s': float(np.sqrt(np.mean(qvel[:, v] ** 2))),
              'peak_tilt_rad': [r.peak_tilt for r in robots],
              'simultaneous_contact_samples': int(np.sum(contacts.any(axis=2).all(axis=1))),
              'opposing_torque_samples': int(np.sum(torque[:, 0] * torque[:, 1] < 0)),
              'nonfoot_deck_contact_samples': int(np.sum(other > 0)),
              'nonfoot_deck_contact_definition': 'Any non-foot contact, including hinged plate-deck contacts',
              'peak_foot_deck_tangential_speed_m_s': [float(np.max(slip[:, i])) if np.any(contacts[:, i]) else None for i in range(2)],
              'actuator_saturation_samples': int(np.count_nonzero(np.isclose(controls, model.actuator_ctrlrange[:, 0], atol=1e-8, rtol=0) | np.isclose(controls, model.actuator_ctrlrange[:, 1], atol=1e-8, rtol=0))),
              'sampled_robot_robot_contacts': robot_collisions,
              'robot_contact_steps': robot_contact_steps,
              'decisions': decisions, 'message_events': bus.events,
              'warnings': warnings,
              'stop_reason': ('fall' if any(r.fell for r in robots) else
                              'robot_contact' if robot_contact_steps else
                              'landing_incomplete' if all(a is not None for a in arrived) and not all(terminal_clearance) else
                              'arrived' if all(a is not None for a in arrived) else 'timeout')}
    (path / 'episode.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'message_events'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('reference', 'pair', 'coordinated'))
    parser.add_argument('--delay', type=float, default=0.)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/counterbalance')
    parser.add_argument('--disturbance', action='store_true')
    parser.add_argument('--lateral-offset', type=float, default=0.)
    parser.add_argument('--lift', type=float, default=0.)
    parser.add_argument('--foot-gait', action='store_true')
    parser.add_argument('--adapt', action='store_true')
    parser.add_argument('--staggered', action='store_true')
    parser.add_argument('--balance-progress', action='store_true')
    parser.add_argument('--half-width', type=float, default=1.)
    parser.add_argument('--stiffness', type=float, default=10000.)
    parser.add_argument('--exit-margin', type=float, default=0.)
    parser.add_argument('--lane-gain', type=float, default=0.)
    parser.add_argument('--transition-lip', action='store_true')
    parser.add_argument('--hinged-lip', action='store_true')
    args = parser.parse_args()
    if args.lift and args.foot_gait:
        parser.error('--lift and --foot-gait cannot be combined')
    if args.adapt and args.mode != 'coordinated':
        parser.error('--adapt requires coordinated mode')
    if args.staggered and args.mode != 'coordinated':
        parser.error('--staggered requires coordinated mode')
    if args.balance_progress and args.mode != 'coordinated':
        parser.error('--balance-progress requires coordinated mode')
    if not 0. <= args.exit_margin <= .3:
        parser.error('--exit-margin must be between 0 and 0.3 m')
    if not 0. <= args.lane_gain <= 1. or (args.lane_gain and not args.foot_gait):
        parser.error('--lane-gain requires --foot-gait and must be between 0 and 1')
    result = run(args.mode, args.output, args.delay, args.disturbance, args.lateral_offset, args.lift, args.foot_gait, args.adapt, args.half_width, args.stiffness, args.staggered, args.balance_progress, args.exit_margin, args.lane_gain, args.transition_lip, args.hinged_lip)
    if result['stop_reason'] != 'arrived':
        raise SystemExit(f"{args.mode} ended with {result['stop_reason']}; inspect episode.json")
