"""Compare the recorded episodes on their shared sampled time window."""
import json
import mujoco
import numpy as np
from scene import ROOT, build_scene


def main():
    names = ('s2', 's3', 's5', 's6')
    recordings = {name: np.load(ROOT / 'outputs' / name / 'recording.npz') for name in names}
    episodes = {name: json.loads((ROOT / 'outputs' / name / 'episode.json').read_text()) for name in names}
    model = build_scene(2)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')
    q, v = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    for name in names[1:]:
        np.testing.assert_array_equal(recordings['s2']['qpos'][0], recordings[name]['qpos'][0])
        np.testing.assert_array_equal(recordings['s2']['qvel'][0], recordings[name]['qvel'][0])
    times = [np.round(recordings[name]['time'], 6) for name in names]
    common = times[0]
    for time in times[1:]:
        common = np.intersect1d(common, time)
    if len(common) < 2 or not np.allclose(np.diff(common), .02, atol=1e-6):
        raise ValueError('Recordings lack a shared 50 Hz window')
    result = {'window_s': [float(common[0]), float(common[-1])],
              'limitation': 'Shared-window values cover the nominal deterministic run only. S5 and S6 share exit clearance; S2 and S3 use different clearance. S3/S5 robot B enters after this window.',
              'scenarios': {}}
    for name, time in zip(names, times):
        rec, ep = recordings[name], episodes[name]
        metrics = json.loads((ROOT / 'outputs' / name / 'metrics.json').read_text())
        idx = np.searchsorted(time, common)
        rate, angle = rec['qvel'][idx, v], rec['qpos'][idx, q]
        arrivals = [r['arrival_time_s'] for r in ep['robots']]
        result['scenarios'][name] = {
            'rms_angular_rate_rad_s': float(np.sqrt(np.mean(rate**2))),
            'peak_angle_rad': float(np.max(np.abs(angle))),
            'successful_robots': sum(r['arrival_time_s'] is not None and not r['fell'] for r in ep['robots']),
            'team_crossing_s': max(arrivals) if all(a is not None for a in arrivals) and not any(r['fell'] for r in ep['robots']) else None,
            'falls': [r['fell'] for r in ep['robots']],
            'peak_tilt_rad': [r['peak_tilt_rad'] for r in ep['robots']],
            'contact_samples': [sum(r['foot_deck_contact_samples']) for r in ep['robots']],
            'foot_deck_tangential_speed_peak_m_s': metrics['foot_deck_tangential_speed_peak_m_s'],
            'wait_time_s': [float(np.sum(np.diff(rec['time']) * (rec['phases'][:-1, i] == 'WAIT'))) for i in range(2)]}
    configs = {name: json.loads((ROOT / 'outputs' / name / 'config.json').read_text()) for name in ('s5', 's6')}
    for key in ('seed', 'robot_count', 'initial_x_m', 'initial_y_m', 'bridge_stiffness_nm_rad',
                'bridge_damping_nms_rad', 'bridge_hinge_range_rad', 'exit_clearance_x_m', 'timestep_s'):
        if configs['s5'][key] != configs['s6'][key]:
            raise ValueError(f'Paired scenario mismatch: {key}')
    pair_time = np.intersect1d(times[2], times[3])
    result['adaptive_pair'] = {'window_s': [float(pair_time[0]), float(pair_time[-1])], 'scenarios': {}}
    for name, time in zip(names[2:], times[2:]):
        rec = recordings[name]
        idx = np.searchsorted(time, pair_time)
        result['adaptive_pair']['scenarios'][name] = {
            'rms_angular_rate_rad_s': float(np.sqrt(np.mean(rec['qvel'][idx, v]**2))),
            'peak_angle_rad': float(np.max(np.abs(rec['qpos'][idx, q])))}
    result['paired_trials'] = []
    for offset in (0., -.05, .05):
        root = ROOT / 'outputs' if offset == 0. else ROOT / 'outputs' / 'trials' / str(offset) / 'outputs'
        pair = {}
        recs = []
        for name in ('s5', 's6'):
            rec = np.load(root / name / 'recording.npz')
            recs.append(rec)
            ep = json.loads((root / name / 'episode.json').read_text())
            pair[name] = {'stop_reason': ep['stop_reason'], 'falls': [r['fell'] for r in ep['robots']],
                          'arrivals_s': [r['arrival_time_s'] for r in ep['robots']]}
        np.testing.assert_array_equal(recs[0]['qpos'][0], recs[1]['qpos'][0])
        np.testing.assert_array_equal(recs[0]['qvel'][0], recs[1]['qvel'][0])
        result['paired_trials'].append({'lateral_offset_m': offset, 'scenarios': pair})
    path = ROOT / 'outputs' / 'adaptive_comparison.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
