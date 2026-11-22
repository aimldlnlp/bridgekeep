"""Physical Go2 deck-crossing experiment; records failures without replay tricks."""
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from gait import Gait

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'vendor/unitree_go2'


def run(compliant):
    # Surface collision masks allow robot contact but prevent deck/approach self-contact.
    joint = '<joint name="bridge_hinge" type="hinge" axis="0 1 0" stiffness="10000" damping="80"/>' if compliant else ''
    xml = f'''<mujoco><include file="go2.xml"/><worldbody>
      <geom name="approach" type="box" pos="-2.35 0 -.05" size="1.65 1 .05" contype="2" conaffinity="1"/>
      <geom name="exit" type="box" pos="2.35 0 -.065" size="1.65 1 .05" contype="2" conaffinity="1"/>
      <body name="bridge" pos="0 0 -.05">{joint}
        <geom name="deck" type="box" size=".7 1 .05" mass="20" contype="2" conaffinity="1"/>
      </body></worldbody></mujoco>'''
    scene = MODEL_DIR / '_bridgekeep_probe.xml'
    try:
        scene.write_text(xml)
        model = mujoco.MjModel.from_xml_path(str(scene))
    finally:
        scene.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[0] = -1.3  # Initial episode placement only.
    home = data.qpos[7:19].copy()
    feet_list = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ('FL','FR','RL','RR')]
    feet = set(feet_list)
    gait = Gait(home)
    deck = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'deck')
    hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge') if compliant else -1
    hinge_qpos = model.jnt_qposadr[hinge] if compliant else None
    contacts = 0
    trace = []
    peak_angle = 0.
    max_tilt = 0.
    min_height = float(data.qpos[2])
    start = time.perf_counter()
    while data.time < 100:
        local_contacts = [any(c.geom1 == foot or c.geom2 == foot for c in data.contact[:data.ncon]) for foot in feet_list]
        rotation = data.xmat[1].reshape(3, 3)
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        pitch = float(np.arctan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2])))
        target = gait.targets(data.time, local_contacts, roll, pitch)
        data.ctrl[:] = np.clip(35 * (target-data.qpos[7:19]) - 1.5 * data.qvel[6:18],
                               model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError(f'Invalid physics at {data.time:.3f} s')
        contacts += sum((c.geom1 == deck and c.geom2 in feet) or
                        (c.geom2 == deck and c.geom1 in feet) for c in data.contact[:data.ncon])
        if compliant:
            peak_angle = max(peak_angle, abs(float(data.qpos[hinge_qpos])))
        min_height = min(min_height, float(data.qpos[2]))
        max_tilt = max(max_tilt, float(np.arccos(np.clip(data.xmat[1, 8], -1, 1))))
        if int(round(data.time / model.opt.timestep)) % 10 == 0:
            trace.append((float(data.time), *data.qpos[:3], float(data.qpos[hinge_qpos]) if compliant else 0., contacts))
        if data.qpos[2] < .12 or max_tilt > 1 or data.qpos[0] > 1.1:
            break
    if not trace or trace[-1][0] < data.time:
        trace.append((float(data.time), *data.qpos[:3], float(data.qpos[hinge_qpos]) if compliant else 0., contacts))
    out = ROOT / 'outputs/b2'
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / ('compliant_trace.npz' if compliant else 'rigid_trace.npz'), samples=np.array(trace), columns=np.array(['time_s', 'x_m', 'y_m', 'z_m', 'deck_angle_rad', 'cumulative_foot_deck_contacts']))
    fall = bool(data.qpos[2] < .12 or max_tilt > 1)
    crossed = bool(data.qpos[0] > 1.1 and not fall and contacts)
    return {'surface': 'compliant' if compliant else 'rigid', 'crossed': crossed,
            'stop_reason': 'fall' if fall else 'arrived' if crossed else 'timeout',
            'simulated_seconds': float(data.time), 'wall_seconds': time.perf_counter()-start,
            'final_position_m': data.qpos[:3].tolist(), 'minimum_height_m': min_height,
            'maximum_tilt_rad': max_tilt, 'foot_deck_contact_samples': contacts,
            'peak_bridge_angle_rad': peak_angle if compliant else None}


if __name__ == '__main__':
    out = ROOT / 'outputs/b2'
    out.mkdir(parents=True, exist_ok=True)
    results = [run(False), run(True)]
    (out / 'deck_probes.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    if not all(r['crossed'] for r in results):
        raise SystemExit('Crossing gate failed; see deck_probes.json')
