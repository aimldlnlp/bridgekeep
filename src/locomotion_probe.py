"""Go2 rigid-ground standing and open-loop gait feasibility experiment."""
import json
from pathlib import Path
import time
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(walk, duration=12):
    model = mujoco.MjModel.from_xml_path(str(ROOT / 'vendor/unitree_go2/scene.xml'))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    home = data.qpos[7:].copy()
    foot_ids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ('FL', 'FR', 'RL', 'RR')}
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    foot_contacts = 0
    minimum_height = float(data.qpos[2])
    max_tilt = 0.0
    start = time.perf_counter()
    while data.time < duration:
        target = home.copy()
        if walk and data.time >= 1:
            for leg, phase in enumerate((0, np.pi, np.pi, 0)):
                swing = np.sin(2 * np.pi * 1.5 * (data.time - 1) + phase)
                target[3 * leg + 1] += .3 * swing
                target[3 * leg + 2] += .5 * max(0, swing)
        data.ctrl[:] = np.clip(35 * (target - data.qpos[7:]) - 1.5 * data.qvel[6:],
                               model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError(f'Invalid physics at {data.time:.3f} s')
        minimum_height = min(minimum_height, float(data.qpos[2]))
        max_tilt = max(max_tilt, float(np.arccos(np.clip(data.xmat[1, 8], -1, 1))))
        foot_contacts += sum((c.geom1 == floor and c.geom2 in foot_ids) or
                             (c.geom2 == floor and c.geom1 in foot_ids) for c in data.contact[:data.ncon])
        if data.qpos[2] < .12 or max_tilt > 1.0:
            break
    return {'mode': 'walk' if walk else 'stand', 'simulation_seconds': float(data.time),
            'wall_seconds': time.perf_counter() - start, 'displacement_m': data.qpos[:3].tolist(),
            'minimum_height_m': minimum_height, 'maximum_tilt_rad': max_tilt,
            'foot_floor_contact_samples': foot_contacts,
            'fall': bool(data.qpos[2] < .12 or max_tilt > 1.0)}


if __name__ == '__main__':
    output = ROOT / 'outputs/b2'
    output.mkdir(parents=True, exist_ok=True)
    results = [run(False), run(True)]
    (output / 'rigid_ground.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    if any(r['fall'] or r['foot_floor_contact_samples'] == 0 for r in results):
        raise SystemExit('Standing or walking experiment failed; inspect results.')
