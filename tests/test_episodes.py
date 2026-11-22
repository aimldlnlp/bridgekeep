import json
from pathlib import Path
import sys
import unittest
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from scene import build_scene
from compare import compare
from independent_probe import Robot
from coordination import Message, Negotiator
from orientation_metrics import angles


class EpisodeTests(unittest.TestCase):
    def test_adaptive_comparison_rms_matches_recordings(self):
        report = json.loads((ROOT / 'outputs/adaptive_comparison.json').read_text())
        model = build_scene(2)
        hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')
        for name in ('s5', 's6'):
            rec = np.load(ROOT / 'outputs' / name / 'recording.npz')
            times = np.round(rec['time'], 6)
            lo, hi = report['adaptive_pair']['window_s']
            samples = rec['qvel'][(times >= lo) & (times <= hi), model.jnt_dofadr[hinge]]
            self.assertAlmostEqual(float(np.sqrt(np.mean(samples**2))),
                                   report['adaptive_pair']['scenarios'][name]['rms_angular_rate_rad_s'])

    def test_adaptive_evidence_and_matched_control(self):
        configs = [json.loads((ROOT / 'outputs' / name / 'config.json').read_text()) for name in ('s5', 's6')]
        for key in ('seed', 'robot_count', 'initial_y_m', 'initial_x_m', 'timestep_s',
                    'bridge_stiffness_nm_rad', 'bridge_damping_nms_rad', 'exit_clearance_x_m'):
            self.assertEqual(configs[0][key], configs[1][key], key)
        a, b = [np.load(ROOT / 'outputs' / name / 'recording.npz') for name in ('s5', 's6')]
        np.testing.assert_array_equal(a['qpos'][0], b['qpos'][0])
        np.testing.assert_array_equal(a['qvel'][0], b['qvel'][0])
        episode = json.loads((ROOT / 'outputs/s5/episode.json').read_text())
        self.assertTrue(any(d['phase'] == 'ADAPT' and d['pace'] < 1. for d in episode['decisions']))
        delivered = [event[-1] for event in episode['message_events'] if event[0] == 'delivered']
        self.assertTrue(any(m['deck_angle_rad'] is not None and m['deck_rate_rad_s'] is not None for m in delivered))

    def test_hidden_state_does_not_change_local_control(self):
        model = build_scene(2)
        data = mujoco.MjData(model)
        first, second = [Robot(model, data, i, y) for i, y in enumerate((.4, -.4))]
        mujoco.mj_forward(model, data)
        first.control(model, data, 1.5)
        baseline = data.ctrl[first.act].copy()
        hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')
        data.qpos[model.jnt_qposadr[hinge]] = .1
        data.qpos[second.base] += 2.
        data.qpos[second.qpos] += .2
        data.qvel[second.dof] += 1.
        # Keep the first robot's measured contacts and attitude fixed for this counterfactual.
        first.control(model, data, 1.5)
        np.testing.assert_array_equal(data.ctrl[first.act], baseline)
        message = [Message(0, 1., 'CROSS')]
        before, after = Negotiator(1), Negotiator(1)
        before.update(-1.3, False, message, 1., .2)
        after.update(-1.3, False, message, 1., .2)
        self.assertEqual(before.phase, after.phase)

    def test_paired_initial_conditions_and_physics(self):
        configs = [json.loads((ROOT / 'outputs' / name / 'config.json').read_text()) for name in ('s2', 's3')]
        for key in ('seed', 'robot_count', 'initial_y_m', 'initial_x_m', 'timestep_s',
                    'bridge_stiffness_nm_rad', 'bridge_damping_nms_rad', 'bridge_hinge_range_rad',
                    'exit_surface_height_m'):
            self.assertEqual(configs[0][key], configs[1][key], key)
        a, b = [np.load(ROOT / 'outputs' / name / 'recording.npz') for name in ('s2', 's3')]
        np.testing.assert_array_equal(a['qpos'][0], b['qpos'][0])
        np.testing.assert_array_equal(a['qvel'][0], b['qvel'][0])
        self.assertTrue(np.any(a['foot_deck'][:, 0].any(axis=1) & a['foot_deck'][:, 1].any(axis=1)))
        self.assertEqual(compare()['common_window_s'], [0., 39.28])

    def test_recorded_motion_and_replay_contact(self):
        for name in ('s1', 's2', 's3', 's4'):
            rec = np.load(ROOT / 'outputs' / name / 'recording.npz')
            ep = json.loads((ROOT / 'outputs' / name / 'episode.json').read_text())
            metrics = json.loads((ROOT / 'outputs' / name / 'metrics.json').read_text())
            model = build_scene(len(ep['robots']))
            data = mujoco.MjData(model)
            hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')
            velocity = rec['qvel'][:, model.jnt_dofadr[hinge]]
            angle = rec['qpos'][:, model.jnt_qposadr[hinge]]
            self.assertAlmostEqual(metrics['bridge_angular_velocity_rms_rad_s'], float(np.sqrt(np.mean(velocity**2))))
            self.assertAlmostEqual(metrics['peak_absolute_bridge_angle_rad'], float(np.max(np.abs(angle))))
            self.assertTrue(np.all(np.diff(rec['time']) > 0))
            self.assertTrue(np.isfinite(rec['qpos']).all())
            self.assertEqual(rec['actuator_force_nm'].shape, rec['ctrl'].shape)
            self.assertTrue(np.isfinite(rec['actuator_force_nm']).all())
            self.assertGreater(np.max(np.abs(rec['actuator_force_nm'])), 0.)
            slip = rec['foot_deck_tangential_speed_m_s']
            self.assertEqual(slip.shape, rec['foot_deck'].shape)
            self.assertTrue(np.isfinite(slip).all())
            self.assertTrue((slip >= 0.).all())
            self.assertTrue((slip[~rec['foot_deck']] == 0.).all())
            self.assertGreater(np.max(slip), 0.)
            for i in range(len(ep['robots'])):
                expected = float(np.max(slip[:, i])) if rec['foot_deck'][:, i].any() else None
                self.assertEqual(metrics['foot_deck_tangential_speed_peak_m_s'][i], expected)
            self.assertEqual(ep['warnings'], {})
            for i in range(len(ep['robots'])):
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'r{i}_base')
                adr = model.jnt_qposadr[model.body_jntadr[body]]
                roll, pitch = angles(rec['qpos'][:, adr+3:adr+7])
                self.assertAlmostEqual(metrics['body_orientation_50hz'][i]['peak_absolute_roll_rad'], np.max(np.abs(roll)))
                self.assertAlmostEqual(metrics['body_orientation_50hz'][i]['peak_absolute_pitch_rad'], np.max(np.abs(pitch)))
                displacement = np.linalg.norm(np.diff(rec['qpos'][:, adr:adr+3], axis=0), axis=1)
                self.assertLess(np.max(displacement), .05, name)
            # Replay the recorded qpos and compare actual foot/deck contact booleans.
            deck = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'deck')
            for index in np.linspace(0, len(rec['time'])-1, 20, dtype=int):
                data.qpos[:] = rec['qpos'][index]
                data.qvel[:] = rec['qvel'][index]
                mujoco.mj_forward(model, data)
                for i in range(len(ep['robots'])):
                    for k, leg in enumerate(('FL','FR','RL','RR')):
                        foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'r{i}_{leg}')
                        present = any({c.geom1, c.geom2} == {deck, foot} for c in data.contact[:data.ncon])
                        self.assertEqual(present, rec['foot_deck'][index, i, k], (name,index,i,leg))

    def test_peer_and_disturbance_decisions(self):
        s3 = json.loads((ROOT / 'outputs/s3/episode.json').read_text())
        s4 = json.loads((ROOT / 'outputs/s4/episode.json').read_text())
        messages = [event[-1] for event in s3['message_events'] if event[0] == 'delivered']
        self.assertTrue(all({'time_s', 'phase', 'vibration_rad_s', 'gait_frequency_hz',
                             'entry_request', 'fallen'} <= set(message) for message in messages))
        self.assertTrue(any(message['vibration_rad_s'] > 0 for message in messages))
        self.assertTrue(any(x['robot'] == 1 and x['phase'] == 'WAIT' and x['messages'] for x in s3['decisions']))
        self.assertTrue(any(x['robot'] == 1 and x['phase'] == 'CROSS' and
                            any(m['phase'] == 'EXIT' for m in x['messages']) for x in s3['decisions']))
        self.assertTrue(any(x['robot'] == 0 and x['phase'] == 'SAFE_STOP' and 20 <= x['time_s'] <= 20.4
                            and x['local_peak_angular_rate_rad_s'] > 1.5 for x in s4['decisions']))


if __name__ == '__main__':
    unittest.main()
