"""Single Go2 foot-trajectory feasibility probe; no playback or root animation."""
import argparse
import json
import time
from pathlib import Path
import mujoco
import numpy as np
from scene import ROOT, build_scene
from foot_control import ProbeRobot
from foot_trajectory import BlendedTrajectory
from counterbalance import cleared_landing
from run import tangential_speed


def run(surface, output, lateral=0., stride=.025, height=.025, frequency=1., kp=35., kd=1.5, pause_at=None, half_width=1., lane_gain=0., transition_lip=False, stiffness=10000., hinged_lip=False):
    model = build_scene(1, counterbalance=True, half_width=half_width, transition_lip=transition_lip, bridge_stiffness=stiffness, hinged_lip=hinged_lip)
    data = mujoco.MjData(model)
    robot = ProbeRobot(model, data, 0, lateral, kp, kd, lane_gain)
    robot.gait = BlendedTrajectory(robot.gait.home, stride=stride, height=height, frequency=frequency)
    data.qpos[robot.base:robot.base+3] = [-3.3 if surface == 'ground' else -2.35, lateral, .27]
    mujoco.mj_forward(model, data)
    deck = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'deck')
    landing = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'exit')
    initial = float(data.qpos[robot.base]); records = []; start = time.perf_counter(); arrived = False; gait_time = 0.
    hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')
    lips = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ('left_lip','right_lip')} if transition_lip or hinged_lip else set()
    lip_steps = 0; lip_peak_n = 0.; lip_deck_steps = 0; lip_deck_impulse = 0.
    robot_geoms = set()
    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        while body and body != robot.body:
            body = int(model.body_parentid[body])
        if body == robot.body:
            robot_geoms.add(g)
    for step in range(80000):
        elapsed = data.time - pause_at if pause_at is not None else -1.
        robot.gait.blend = (1. - elapsed if elapsed < 1. else 0. if elapsed < 3. else elapsed - 3.) if 0. <= elapsed < 4. else 1.
        paused = 1. <= elapsed < 3.
        robot.control(model, data, gait_time)
        if not paused:
            gait_time += model.opt.timestep
        mujoco.mj_step(model, data)
        touching_lip = False; touching_lip_deck = False
        for ci,c in enumerate(data.contact[:data.ncon]):
            if c.geom1 in lips or c.geom2 in lips:
                other = int(c.geom2 if c.geom1 in lips else c.geom1)
                if other in robot_geoms or other == deck:
                    force = np.zeros(6); mujoco.mj_contactForce(model,data,ci,force)
                    if other in robot_geoms:
                        touching_lip = True
                        lip_peak_n = max(lip_peak_n, float(force[0]))
                    if other == deck:
                        touching_lip_deck = True
                        lip_deck_impulse += float(force[0]) * model.opt.timestep
        lip_steps += touching_lip
        lip_deck_steps += touching_lip_deck
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f'Invalid physics at {data.time:.3f} s')
        robot.measure(data, deck)
        arrived = bool(data.qpos[robot.base] > -2.5) if surface == 'ground' else cleared_landing(model, data, robot, 1, landing, deck)
        if step % 10 == 0 or arrived or robot.fell or step == 79999:
            slip = np.zeros(4)
            for c in data.contact[:data.ncon]:
                for leg, foot in enumerate(robot.feet):
                    if {int(c.geom1), int(c.geom2)} == {deck, foot}:
                        slip[leg] = max(slip[leg], tangential_speed(model, data, c, foot, deck))
            records.append((float(data.time), data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.geom_xpos[robot.feet].copy(), slip))
        if arrived or robot.fell:
            break
    warnings = {mujoco.mjtWarning(i).name: int(data.warning[i].number) for i in range(int(mujoco.mjtWarning.mjNWARNING)) if data.warning[i].number}
    if warnings:
        raise RuntimeError(f'MuJoCo warnings: {warnings}')
    path = Path(output) / surface; path.mkdir(parents=True, exist_ok=True)
    t,qpos,qvel,ctrl,feet,slip = map(np.array, zip(*records))
    np.savez_compressed(path / 'recording.npz', time=t,qpos=qpos,qvel=qvel,ctrl=ctrl,foot_position_m=feet,foot_deck_tangential_speed_m_s=slip)
    report = {'surface':surface,'lateral_offset_m':lateral,'half_width_m':half_width,'lane_gain':lane_gain,'stride_m':stride,'height_m':height,'frequency_hz':frequency,'kp':kp,'kd':kd,'pause_at_s':pause_at,'controller':'planar foot trajectory and joint PD',
              'transition_lip':transition_lip,'hinged_lip':hinged_lip,'stiffness_nm_rad':stiffness,'lip_contact_steps':lip_steps,'lip_peak_normal_force_n':lip_peak_n,
              'lip_deck_contact_steps':lip_deck_steps,'lip_deck_normal_impulse_ns':lip_deck_impulse,
              'status':'fall' if robot.fell else 'arrived' if arrived else 'timeout',
              'simulation_seconds':float(data.time),'wall_seconds':time.perf_counter()-start,
              'displacement_m':float(data.qpos[robot.base]-initial),'peak_tilt_rad':robot.peak_tilt,
              'peak_bridge_angle_rad':float(np.max(np.abs(qpos[:,model.jnt_qposadr[hinge]]))),
              'peak_foot_deck_tangential_speed_m_s':float(np.max(slip)) if robot.contacts else None,
              'actuator_saturation_samples':int(np.count_nonzero(np.isclose(ctrl,model.actuator_ctrlrange[:,0],atol=1e-8,rtol=0)|np.isclose(ctrl,model.actuator_ctrlrange[:,1],atol=1e-8,rtol=0))),
              'foot_deck_contact_steps':robot.contacts,'warnings':warnings,
              'note':'ground arrival is a base-position target; bridge arrival includes foot clearance'}
    (path/'episode.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('surface',choices=('ground','bridge'));p.add_argument('--lateral-offset',type=float,default=0.);p.add_argument('--stride',type=float,default=.025);p.add_argument('--height',type=float,default=.025);p.add_argument('--frequency',type=float,default=1.);p.add_argument('--kp',type=float,default=35.);p.add_argument('--kd',type=float,default=1.5);p.add_argument('--pause-at',type=float);p.add_argument('--half-width',type=float,default=1.);p.add_argument('--lane-gain',type=float,default=0.);p.add_argument('--transition-lip',action='store_true');p.add_argument('--stiffness',type=float,default=10000.);p.add_argument('--output',type=Path,default=ROOT/'outputs/foot_trajectory')
    p.add_argument('--hinged-lip',action='store_true')
    args=p.parse_args()
    if not 0. <= args.lane_gain <= 1.: p.error('--lane-gain must be between 0 and 1')
    result=run(args.surface,args.output,args.lateral_offset,args.stride,args.height,args.frequency,args.kp,args.kd,args.pause_at,args.half_width,args.lane_gain,args.transition_lip,args.stiffness,args.hinged_lip)
    if result['status'] != 'arrived': raise SystemExit(f"{args.surface}: {result['status']}")
