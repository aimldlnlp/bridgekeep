import sys
import unittest
from pathlib import Path
import math
import numpy as np
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from foot_trajectory import LINK, leg_ik, FootTrajectory, BlendedTrajectory
from scene import build_scene
from independent_probe import Robot
from foot_control import ProbeRobot


class FootTrajectoryTests(unittest.TestCase):
    def test_lane_feedback_reverses_with_heading(self):
        model=build_scene(1,counterbalance=True);data=mujoco.MjData(model)
        robot=ProbeRobot(model,data,0,0.,140.,3.,lane_gain=.3)
        data.qpos[robot.base+1]=.1
        mujoco.mj_forward(model,data);robot.control(model,data,0.)
        forward=data.ctrl[robot.act[0]]
        data.qpos[robot.base+3:robot.base+7]=[0.,0.,0.,1.]
        mujoco.mj_forward(model,data);robot.control(model,data,0.)
        reverse=data.ctrl[robot.act[0]]
        self.assertGreater(forward,0.)
        self.assertLess(reverse,0.)
    def test_inverse_kinematics_reconstructs_foot_target(self):
        for x,z in ((0.,-.265),(.025,-.265),(-.025,-.265)):
            thigh,knee=leg_ik(x,z)
            self.assertAlmostEqual(-LINK*(math.sin(thigh)+math.sin(thigh+knee)),x)
            self.assertAlmostEqual(-LINK*(math.cos(thigh)+math.cos(thigh+knee)),z)
        with self.assertRaises(ValueError): leg_ik(.5,0.)

    def test_ik_matches_mujoco_foot_within_model_offset(self):
        model=build_scene(1,counterbalance=True);data=mujoco.MjData(model);robot=Robot(model,data,0,0.)
        thigh,knee=leg_ik(.025,-.265);data.qpos[robot.qpos[1:3]]=[thigh,knee];mujoco.mj_forward(model,data)
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'r0_FL_thigh')
        actual=data.geom_xpos[robot.feet[0]]-data.xpos[body]
        np.testing.assert_allclose(actual[[0,2]],[.025,-.265],atol=.003)

    def test_stance_initially_preserves_home(self):
        home=np.array([0.,.9,-1.8]*4);gait=FootTrajectory(home)
        np.testing.assert_array_equal(gait.targets(.5,[True]*4,0,0),home)

    def test_blend_extremes_and_midpoint(self):
        home=np.array([0.,.9,-1.8]*4);gait=BlendedTrajectory(home)
        gait.blend=0.; stance=gait.targets(2.1,[True]*4,0,0)
        gait.blend=1.; walking=gait.targets(2.1,[True]*4,0,0)
        gait.blend=.5; midpoint=gait.targets(2.1,[True]*4,0,0)
        np.testing.assert_allclose(stance,home)
        np.testing.assert_allclose(midpoint,(stance+walking)/2)


if __name__=='__main__':unittest.main()
