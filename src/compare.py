"""Paired comparison on an identical sampled interval, plus crossing times."""
import json
import mujoco
import numpy as np
from scene import ROOT, build_scene


def compare():
    model = build_scene(2)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')
    qpos_id, dof_id = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    recordings = [np.load(ROOT / f'outputs/{name}/recording.npz') for name in ('s2', 's3')]
    episodes = [json.loads((ROOT / f'outputs/{name}/episode.json').read_text()) for name in ('s2', 's3')]
    # Exclude irregular terminal frames and compare precisely equal sample times.
    common = np.intersect1d(np.round(recordings[0]['time'], 6), np.round(recordings[1]['time'], 6))
    if len(common) < 2 or not np.allclose(np.diff(common), .02, atol=1e-6):
        raise ValueError('Recordings lack a common 50 Hz window')
    result = {'common_window_s': [float(common[0]), float(common[-1])],
              'signal': 'RMS bridge hinge angular velocity; rad/s', 'scenarios': {}}
    for name, rec, ep in zip(('s2', 's3'), recordings, episodes):
        indices = np.searchsorted(np.round(rec['time'], 6), common)
        v = rec['qvel'][indices, dof_id]
        angle = rec['qpos'][indices, qpos_id]
        arrivals = [r['arrival_time_s'] for r in ep['robots']]
        result['scenarios'][name] = {'bridge_angular_velocity_rms_rad_s': float(np.sqrt(np.mean(v*v))),
                                     'peak_angle_rad': float(np.max(np.abs(angle))),
                                     'team_crossing_time_s': max(arrivals) if all(a is not None for a in arrivals) else None,
                                     'arrival_times_s': arrivals, 'stop_reason': ep['stop_reason']}
    out = ROOT / 'outputs/showcase'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    compare()
