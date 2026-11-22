"""Compare matched initial states through the end of the shorter episode."""
import json
from pathlib import Path
import mujoco
import numpy as np
from scene import ROOT, build_scene


def main():
    root = ROOT / 'outputs/counterbalance_delay2'
    names = ('pair', 'coordinated')
    recordings = [np.load(root / name / 'recording.npz') for name in names]
    episodes = [json.loads((root / name / 'episode.json').read_text()) for name in names]
    if not np.array_equal(recordings[0]['qpos'][0], recordings[1]['qpos'][0]):
        raise ValueError('Initial positions differ')
    if not np.array_equal(recordings[0]['qvel'][0], recordings[1]['qvel'][0]):
        raise ValueError('Initial velocities differ')
    model = build_scene(2, counterbalance=True)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')
    q, v = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    common = np.intersect1d(np.round(recordings[0]['time'], 6), np.round(recordings[1]['time'], 6))
    if len(common) < 2:
        raise ValueError('No common sampled window')
    result = {'window_s': [float(common[0]), float(common[-1])],
              'limitation': 'Common window ends at failure of independent run; coordinated run continues. Conditions are deterministic and exactly symmetric.',
              'scenarios': {}}
    for name, rec, ep in zip(names, recordings, episodes):
        indices = np.searchsorted(np.round(rec['time'], 6), common)
        result['scenarios'][name] = {'peak_angle_rad': float(np.max(np.abs(rec['qpos'][indices, q]))),
            'rms_angular_rate_rad_s': float(np.sqrt(np.mean(rec['qvel'][indices, v] ** 2))),
            'stop_reason': ep['stop_reason'], 'arrivals_s': ep['arrival_s'], 'falls': ep['fell'],
            'sampled_robot_robot_contacts': ep['sampled_robot_robot_contacts']}
    (root / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
