import sys
import unittest
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from verify_milestones import bridge
from scene import build_scene
from run import tangential_speed


class BridgeTests(unittest.TestCase):
    def test_visual_geometry_preserves_dynamics(self):
        normal, visual = build_scene(2), build_scene(2, visual=True)
        self.assertEqual((normal.nq, normal.nv, normal.nu), (visual.nq, visual.nv, visual.nu))
        np.testing.assert_array_equal(normal.body_mass, visual.body_mass)
        np.testing.assert_array_equal(normal.dof_M0, visual.dof_M0)
        a, b = mujoco.MjData(normal), mujoco.MjData(visual)
        for _ in range(100):
            mujoco.mj_step(normal, a)
            mujoco.mj_step(visual, b)
        np.testing.assert_array_equal(a.qpos, b.qpos)
        np.testing.assert_array_equal(a.qvel, b.qvel)

    def test_tangential_speed_excludes_normal_motion(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom name="floor" type="plane" size="1 1 .1"/>
          <body pos="0 0 .09"><freejoint/><geom name="foot" type="sphere" size=".1" mass="1"/></body>
          </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        data.qvel[:3] = [.3, .4, .2]
        mujoco.mj_forward(model, data)
        self.assertGreater(data.ncon, 0)
        foot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'foot')
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.assertAlmostEqual(tangential_speed(model, data, data.contact[0], foot, floor), .5)

    def test_main_bridge_has_finite_limits(self):
        model = build_scene(1)
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')
        self.assertTrue(model.jnt_limited[joint])
        np.testing.assert_allclose(model.jnt_range[joint], [-.15, .15])

    def test_main_bridge_free_decay(self):
        model = build_scene(1)
        data = mujoco.MjData(model)
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')
        position, velocity = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
        base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'r0_base')
        base_position = model.jnt_qposadr[model.body_jntadr[base]]
        data.qpos[base_position:base_position+7] = [-2., 0., .4, 1., 0., 0., 0.]
        angles = []
        for step in range(5000):
            data.qfrc_applied[velocity] = 100. if step < 50 else 0.
            mujoco.mj_step(model, data)
            angles.append(data.qpos[position])
        angles = np.asarray(angles)
        self.assertTrue(np.isfinite(angles).all())
        self.assertGreater(np.max(np.abs(angles[:1000])), .001)
        self.assertLess(np.max(np.abs(angles[-500:])), .00001)

    def test_equilibrium_and_physical_response(self):
        _, equilibrium, _, _ = bridge()
        _, decay, _, _ = bridge(impulse=True)
        _, load, contacts, _ = bridge(load=True)
        self.assertLess(np.max(np.abs(equilibrium)), 1e-10)
        self.assertGreater(np.max(np.abs(decay)), .001)
        self.assertLess(np.max(np.abs(decay[-500:])), np.max(np.abs(decay[:1000])))
        self.assertGreater(contacts, 0)
        self.assertGreater(np.max(np.abs(load)), .001)

    def test_recorded_compliant_crossing(self):
        path = Path(__file__).resolve().parents[1] / 'outputs/b2/compliant_trace.npz'
        if not path.exists():
            self.skipTest('Run python src/crossing_probe.py first')
        samples = np.load(path)['samples']
        self.assertTrue(np.all(np.diff(samples[:, 0]) > 0))
        self.assertLess(np.max(np.linalg.norm(np.diff(samples[:, 1:4], axis=0), axis=1)), .02)
        self.assertGreater(samples[-1, 1], 1.1)
        self.assertGreater(samples[-1, 5], 0)
        self.assertEqual(np.max(np.abs(samples[samples[:, 5] == 0, 4])), 0)
        self.assertGreater(np.max(np.abs(samples[:, 4])), .001)


if __name__ == '__main__':
    unittest.main()
