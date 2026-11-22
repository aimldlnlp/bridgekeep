"""Calculate body roll and pitch from each recorded base quaternion."""
import json
import mujoco
import numpy as np
from scene import ROOT, build_scene


def angles(quaternions):
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1., 1.))
    return roll, pitch


def main():
    for name in ('s1', 's2', 's3', 's4'):
        path = ROOT / 'outputs' / name
        data = np.load(path / 'recording.npz')
        config = json.loads((path / 'config.json').read_text())
        model = build_scene(config['robot_count'])
        body_metrics = []
        for i in range(config['robot_count']):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'r{i}_base')
            pos = model.jnt_qposadr[model.body_jntadr[body]]
            roll, pitch = angles(data['qpos'][:, pos+3:pos+7])
            body_metrics.append({'peak_absolute_roll_rad': float(np.max(np.abs(roll))),
                                 'peak_absolute_pitch_rad': float(np.max(np.abs(pitch)))})
        metrics_path = path / 'metrics.json'
        metrics = json.loads(metrics_path.read_text())
        metrics['body_orientation_50hz'] = body_metrics
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(name, json.dumps(body_metrics))


if __name__ == '__main__':
    main()
