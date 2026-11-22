"""One robot's joint controller, driven only by its own joints and contacts."""
import numpy as np


class Gait:
    def __init__(self, home):
        self.home = np.asarray(home).copy()

    def targets(self, time_s, contacts, roll, pitch):
        target = self.home.copy()
        if time_s < 1:
            return target
        for leg, phase in enumerate((0, np.pi, np.pi, 0)):
            swing = np.sin(2 * np.pi * 1.5 * (time_s - 1) + phase)
            target[3 * leg + 1] += .3 * swing - .003 * pitch
            target[3 * leg + 2] += .5 * max(0, swing)
            # Search down gently if an expected stance foot loses contact.
            if swing < -.5 and not contacts[leg]:
                target[3 * leg + 2] -= .001
            target[3 * leg] -= .003 * roll
        return target
