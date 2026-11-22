import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from coordination import AdaptiveNegotiator, Bus, Message, Negotiator
from gait import Gait


class CoordinationTests(unittest.TestCase):
    def test_adaptive_uses_contact_and_peer_measurements(self):
        policy = AdaptiveNegotiator(0)
        peer = Message(1, 0., 'WAIT')
        policy.update(-1.3, False, [peer], 0., 0., .1, .2, False)
        self.assertEqual((policy.phase, policy.pace), ('CROSS', 1.))
        policy.update(-.5, False, [peer], .2, 0., .02, .1, True)
        self.assertEqual((policy.phase, policy.pace), ('ADAPT', .85))
        policy.update(-.5, False, [peer], .4, 0., .001, .01, True)
        self.assertEqual((policy.phase, policy.pace), ('CROSS', 1.))
        waiting = AdaptiveNegotiator(1)
        disturbed = Message(0, 0., 'EXIT', deck_angle_rad=.02, deck_rate_rad_s=.1)
        waiting.update(-1., False, [disturbed], 0.)
        self.assertEqual(waiting.phase, 'WAIT')
        waiting.update(-1., False, [Message(0, .2, 'EXIT', deck_angle_rad=0., deck_rate_rad_s=0.)], .2)
        self.assertEqual(waiting.phase, 'CROSS')

    def test_range_and_message_driven_entry(self):
        bus = Bus(max_range_m=2.)
        messages = [Message(0, 0., 'REQUEST'), Message(1, 0., 'REQUEST')]
        inbox = bus.deliver(0., messages, {0: (0., 0.), 1: (1., 0.)})
        first, second = Negotiator(0), Negotiator(1)
        first.update(-1.3, False, inbox[0])
        second.update(-1.3, False, inbox[1])
        self.assertEqual((first.phase, second.phase), ('CROSS', 'WAIT'))
        second.update(-1.3, False, [Message(0, 10., 'EXIT')], 10.)
        self.assertEqual(second.phase, 'CROSS')
        self.assertTrue(second.peer_influenced)
        self.assertEqual(bus.deliver(1., messages, {0: (0., 0.), 1: (3., 0.)}), {0: [], 1: []})
        self.assertEqual([e[:5] for e in bus.events if e[0] == 'dropped'],
                         [('dropped', 1., 0, 1, 'REQUEST'), ('dropped', 1., 1, 0, 'REQUEST')])

    def test_stopped_peer_does_not_clear_bridge(self):
        policy = Negotiator(1)
        policy.update(-1., False, [Message(0, 1., 'SAFE_STOP')], 1.)
        self.assertEqual(policy.phase, 'WAIT')

    def test_connection_loss_stops_crossing(self):
        policy = Negotiator(0)
        policy.update(-1., False, [Message(1, 0., 'REQUEST')], 0.)
        policy.update(-.5, False, [], .4)
        self.assertEqual(policy.phase, 'CROSS')
        policy.update(-.4, False, [], .8)
        self.assertEqual(policy.phase, 'SAFE_STOP')

    def test_safe_stop_is_latched(self):
        policy = Negotiator(0)
        policy.update(-1., False, [], 120.)
        policy.update(-1., False, [Message(1, 121., 'EXIT')], 121.)
        self.assertEqual(policy.phase, 'SAFE_STOP')

    def test_gait_uses_only_given_contacts_and_attitude(self):
        gait = Gait([0., .9, -1.8] * 4)
        a = gait.targets(1.5, [True]*4, 0., 0.)
        b = gait.targets(1.5, [False]*4, .1, .1)
        self.assertFalse((a == b).all())


if __name__ == '__main__':
    unittest.main()
