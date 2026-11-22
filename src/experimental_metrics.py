"""Offline, paired-window bridge measurements for the current experiments."""
import json
import mujoco
import numpy as np
from scene import build_scene, ROOT


def main():
    model = build_scene(2)
    hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')
    dof = model.jnt_dofadr[hinge]
    scenarios = {'independent': 'b3/independent', 'coordinated': 'b4/coordinated',
                 'disturbance': 'b5/coordinated'}
    result = {}
    for name, path in scenarios.items():
        recording = np.load(ROOT / f'outputs/{path}.npz')
        episode = json.loads((ROOT / f'outputs/{path}.json').read_text())
        t, angle = recording['time'], recording['deck_angle']
        velocity = recording['qvel'][:, dof]
        if len(t) == 0 or not np.isfinite(velocity).all() or np.any(np.diff(t) <= 0):
            raise ValueError(f'Invalid recording: {name}')
        arrivals = [robot['arrival_time_s'] for robot in episode['robots']]
        result[name] = {'window_s': [float(t[0]), float(t[-1])],
                        'bridge_angular_velocity_rms_rad_s': float(np.sqrt(np.mean(velocity**2))),
                        'peak_absolute_bridge_angle_rad': float(np.max(np.abs(angle))),
                        'arrivals_s': arrivals,
                        'team_crossing_time_s': max(arrivals) if all(a is not None for a in arrivals) else None,
                        'falls': [robot['fell'] for robot in episode['robots']]}
    out = ROOT / 'outputs/experimental_comparison.json'
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
