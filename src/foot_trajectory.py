"""Experimental planar foot trajectories and analytic two-link leg IK."""
import math
import numpy as np

LINK = .213


def leg_ik(x, z):
    distance2 = x*x + z*z
    cosine = (distance2 - 2 * LINK * LINK) / (2 * LINK * LINK)
    if not -1. <= cosine <= 1.:
        raise ValueError(f'Unreachable foot target x={x:.4f}, z={z:.4f} m')
    knee = -math.acos(cosine)
    thigh = math.atan2(-x, -z) - math.atan2(LINK * math.sin(knee), LINK + LINK * math.cos(knee))
    return thigh, knee


class FootTrajectory:
    def __init__(self, home, stride=.025, height=.025, frequency=1.):
        self.home = np.asarray(home).copy()
        self.stride, self.height, self.frequency = stride, height, frequency
        thigh, knee = self.home[1:3]
        self.x0 = -LINK * (math.sin(thigh) + math.sin(thigh + knee))
        self.z0 = -LINK * (math.cos(thigh) + math.cos(thigh + knee))

    def targets(self, time_s, contacts, roll, pitch):
        target = self.home.copy()
        if time_s < 1.:
            return target
        ramp = min(1., (time_s - 1.) / 2.)
        for leg, offset in enumerate((0., .5, .5, 0.)):
            phase = ((time_s - 1.) * self.frequency + offset) % 1.
            if phase < .6:
                x = self.stride * (1. - 2. * phase / .6)
                z = 0.
            else:
                swing = (phase - .6) / .4
                x = self.stride * (-1. + 2. * swing)
                z = self.height * math.sin(math.pi * swing)
            thigh, knee = leg_ik(self.x0 + ramp*x, self.z0 + ramp*z)
            target[leg*3 + 1] = thigh - .003*pitch
            target[leg*3 + 2] = knee
            target[leg*3] -= .003*roll
        return target


class BlendedTrajectory(FootTrajectory):
    """Blend cyclic joint targets with active stance over a finite transition."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blend = 1.

    def targets(self, time_s, contacts, roll, pitch):
        walking = super().targets(time_s, contacts, roll, pitch)
        stance = super().targets(0., contacts, roll, pitch)
        return stance + self.blend * (walking - stance)
