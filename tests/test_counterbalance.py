import sys
import unittest
import json
import tempfile
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from scene import build_scene
from counterbalance import LiftGait, ReadyMessage, cleared_landing, observed_pace, progress_pace
from independent_probe import Robot
from compare_counterbalance_pacing import main as compare_pacing


class CounterbalanceSceneTests(unittest.TestCase):
    def test_hinged_lips_collide_with_deck(self):
        model=build_scene(2,counterbalance=True,hinged_lip=True)
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        deck=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'deck')
        for name in ('left_lip','right_lip'):
            lip=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
            hinge=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name+'_hinge')
            self.assertGreaterEqual(hinge,0)
            self.assertTrue(model.jnt_limited[hinge])
            self.assertTrue((model.geom_contype[lip]&model.geom_conaffinity[deck]) or (model.geom_contype[deck]&model.geom_conaffinity[lip]))
            contacts=[c for c in data.contact[:data.ncon] if {int(c.geom1),int(c.geom2)}=={lip,deck}]
            self.assertTrue(contacts)
            self.assertGreaterEqual(min(c.dist for c in contacts),-1e-8)
    def test_transition_lips_collide_with_feet_not_deck(self):
        model=build_scene(2,counterbalance=True,transition_lip=True)
        deck=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'deck')
        foot=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'r0_FL')
        for name in ('left_lip','right_lip'):
            lip=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
            self.assertGreaterEqual(lip,0)
            self.assertFalse((model.geom_contype[lip]&model.geom_conaffinity[deck]) or (model.geom_contype[deck]&model.geom_conaffinity[lip]))
            self.assertTrue((model.geom_contype[lip]&model.geom_conaffinity[foot]) or (model.geom_contype[foot]&model.geom_conaffinity[lip]))
    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'outputs/asym_lane_fair_nopace/coordinated/recording.npz').exists() and (Path(__file__).resolve().parents[1] / 'outputs/asym_lane_recheck/coordinated/recording.npz').exists(), 'Generate pacing comparison recordings first')
    def test_pacing_comparison_uses_common_timestamps(self):
        root=Path(__file__).resolve().parents[1]/'outputs'
        control=root/'asym_lane_fair_nopace/coordinated'
        treatment=root/'asym_lane_recheck/coordinated'
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            out=Path(directory)/'comparison.json'
            compare_pacing(control,treatment,out)
            result=json.loads(out.read_text())
        rec=np.load(control/'recording.npz');model=build_scene(2,counterbalance=True)
        hinge=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'bridge_hinge')
        times=np.round(rec['time'],6);lo,hi=result['common_window_s']
        values=rec['qvel'][(times>=lo)&(times<=hi),model.jnt_dofadr[hinge]]
        self.assertAlmostEqual(float(np.sqrt(np.mean(values**2))),result['scenarios']['no_progress_pacing']['bridge_rate_rms_rad_s'])
    def test_contact_torque_sign_matches_contact_jacobian(self):
        xml = '''<mujoco><option timestep="0.002"/><worldbody>
        <body name="deck"><joint name="hinge" type="hinge" axis="0 1 0" stiffness="10000" damping="80"/>
        <geom name="deck_geom" type="box" size="1 0.5 0.05" mass="20"/></body>
        <body pos="0.5 0 0.2"><freejoint/><geom name="load" type="sphere" size="0.05" mass="1"/></body>
        </worldbody></mujoco>'''
        model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
        for _ in range(1000):
            mujoco.mj_step(model,data)
        deck=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'deck_geom')
        hinge=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'hinge')
        contacts=[(i,c) for i,c in enumerate(data.contact[:data.ncon]) if deck in (c.geom1,c.geom2)]
        self.assertTrue(contacts)
        i,c=contacts[0];force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
        world=c.frame.reshape(3,3).T@force[:3]
        if c.geom1==deck: world=-world
        torque=np.cross(c.pos-data.xanchor[hinge],world)@data.xaxis[hinge]
        jac=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jac,None,c.pos,model.geom_bodyid[deck])
        self.assertGreater(torque,0.)
        self.assertAlmostEqual(torque,float(jac[:,model.jnt_dofadr[hinge]]@world),places=7)
    def test_progress_pacing_uses_delivered_peer_status(self):
        peer = ReadyMessage(1, 1., 'READY', progress_m=-.5, on_deck=True)
        self.assertEqual(progress_pace(True, -.3, [peer]), .98)
        self.assertEqual(progress_pace(True, -.45, [peer]), 1.)
        self.assertEqual(progress_pace(True, -.3, []), 1.)
        self.assertEqual(progress_pace(False, -.3, [peer]), 1.)

    def test_counterbalance_geometry_is_configurable_without_changing_legacy(self):
        model = build_scene(2, counterbalance=True, half_width=1.4, bridge_stiffness=6000.)
        deck = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'deck')
        hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')
        self.assertAlmostEqual(model.geom_size[deck, 1], 1.4)
        self.assertAlmostEqual(model.jnt_stiffness[hinge], 6000.)
        with self.assertRaises(ValueError):
            build_scene(2, counterbalance=True, half_width=0.)
    def test_pre_step_state_replays_recorded_contacts(self):
        model = build_scene(1, counterbalance=True)
        data, replay = mujoco.MjData(model), mujoco.MjData(model)
        robot = Robot(model, data, 0, 0.)
        deck = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'deck')
        mujoco.mj_forward(model, data)
        seen = 0
        for step in range(600):
            robot.control(model, data)
            qpos, qvel = data.qpos.copy(), data.qvel.copy()
            mujoco.mj_step(model, data)
            if step % 10:
                continue
            pairs = {frozenset((int(c.geom1), int(c.geom2))) for c in data.contact[:data.ncon]}
            actual = [frozenset((deck, foot)) in pairs for foot in robot.feet]
            replay.qpos[:], replay.qvel[:] = qpos, qvel
            mujoco.mj_forward(model, replay)
            replay_pairs = {frozenset((int(c.geom1), int(c.geom2))) for c in replay.contact[:replay.ncon]}
            self.assertEqual(actual, [frozenset((deck, foot)) in replay_pairs for foot in robot.feet])
            seen += sum(actual)
        self.assertGreater(seen, 0)
    def test_pace_uses_observations_and_ignores_missing_values(self):
        quiet = ReadyMessage(0, 1., 'READY', .01, .02)
        busy = ReadyMessage(1, 1., 'READY', .03, .02)
        missing = ReadyMessage(1, 1., 'READY')
        self.assertEqual(observed_pace(True, [quiet, missing]), 1.)
        self.assertEqual(observed_pace(True, [quiet, busy]), .85)
        self.assertEqual(observed_pace(False, [busy]), 1.)
    def test_landing_requires_clear_feet_and_target_contact(self):
        model = SimpleNamespace(geom_size=np.array([[.02], [.02], [.02], [.02]]))
        robot = SimpleNamespace(feet=[0, 1, 2, 3])
        data = SimpleNamespace(geom_xpos=np.array([[2.1, 0, 0]] * 4),
                               contact=[SimpleNamespace(geom1=0, geom2=4)], ncon=1)
        self.assertTrue(cleared_landing(model, data, robot, 1, 4, 5))
        data.geom_xpos[0, 0] = 2.01
        self.assertFalse(cleared_landing(model, data, robot, 1, 4, 5))
        data.geom_xpos[0, 0] = 2.1
        data.contact[0].geom2 = 5
        self.assertFalse(cleared_landing(model, data, robot, 1, 4, 5))
        data.geom_xpos[:, 0] = -2.1
        data.contact[0].geom2 = 4
        self.assertTrue(cleared_landing(model, data, robot, -1, 4, 5))

    def test_lift_gait_changes_only_swing_knee_targets(self):
        home = np.array([0., .9, -1.8] * 4)
        baseline = LiftGait(home, 0.).targets(1.1, [True] * 4, 0., 0.)
        lifted = LiftGait(home, .05).targets(1.1, [True] * 4, 0., 0.)
        self.assertTrue(np.all(lifted[[2, 5, 8, 11]] >= baseline[[2, 5, 8, 11]]))
        np.testing.assert_array_equal(lifted[[0, 1, 3, 4, 6, 7, 9, 10]], baseline[[0, 1, 3, 4, 6, 7, 9, 10]])
    def test_seesaw_and_legacy_dimensions_and_support(self):
        old = build_scene(2)
        new = build_scene(2, visual=True, counterbalance=True)
        for model, half_length in ((old, .7), (new, 2.)):
            deck = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'deck')
            hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')
            self.assertAlmostEqual(model.geom_size[deck, 0], half_length)
            self.assertTrue(model.jnt_limited[hinge])
        floor = mujoco.mj_name2id(new, mujoco.mjtObj.mjOBJ_GEOM, 'visual_floor')
        self.assertEqual(new.geom_contype[floor], 0)
        self.assertEqual(new.geom_conaffinity[floor], 0)
        self.assertTrue(np.isfinite(new.body_mass).all())

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / 'outputs/counterbalance_delay2/pair/recording.npz').exists() and (Path(__file__).resolve().parents[1] / 'outputs/counterbalance_delay2/coordinated/recording.npz').exists(), 'Generate paired recordings first')
    def test_recorded_opposing_contact_and_paired_initial_state(self):
        root = Path(__file__).resolve().parents[1] / 'outputs/counterbalance_delay2'
        pair = np.load(root / 'pair/recording.npz')
        coordinated = np.load(root / 'coordinated/recording.npz')
        np.testing.assert_array_equal(pair['qpos'][0], coordinated['qpos'][0])
        np.testing.assert_array_equal(pair['qvel'][0], coordinated['qvel'][0])
        torque = coordinated['contact_torque_nm']
        both = coordinated['foot_deck'].any(axis=2).all(axis=1)
        self.assertTrue(np.any(both & (torque[:, 0] * torque[:, 1] < 0)))
        episode = json.loads((root / 'coordinated/episode.json').read_text())
        self.assertEqual(episode['fell'], [False, False])
        self.assertEqual(len(episode['decisions']), 2)


if __name__ == '__main__':
    unittest.main()
