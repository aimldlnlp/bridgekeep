"""Time-stamped peer messages and decentralized entry negotiation."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Message:
    sender: int
    time_s: float
    phase: str
    vibration_rad_s: float = 0.
    gait_frequency_hz: float = 0.
    entry_request: bool = False
    fallen: bool = False
    deck_angle_rad: float | None = None
    deck_rate_rad_s: float | None = None


class Bus:
    def __init__(self, max_range_m=4.):
        self.max_range_m = max_range_m
        self.events = []

    def deliver(self, now, messages, positions):
        inbox = {i: [] for i in positions}
        for message in messages:
            self.events.append(('sent', now, message.sender, message.phase, message.__dict__.copy()))
            for receiver, position in positions.items():
                if receiver == message.sender:
                    continue
                if math.dist(positions[message.sender], position) <= self.max_range_m:
                    inbox[receiver].append(message)
                    self.events.append(('delivered', now, message.sender, receiver, message.phase, message.__dict__.copy()))
                else:
                    self.events.append(('dropped', now, message.sender, receiver, message.phase, message.__dict__.copy()))
        return inbox


class Negotiator:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.phase = 'REQUEST'
        self.peer_influenced = False
        self.entered = False
        self.last_peer_time = 0.

    def update(self, local_x, fallen, inbox, time_s=0., angular_rate=0.):
        old = self.phase
        if inbox:
            self.last_peer_time = time_s
        disconnected = time_s - self.last_peer_time > .6
        if fallen or angular_rate > 1.5 or (not self.entered and time_s >= 120.) or (disconnected and self.phase != 'EXIT') or self.phase == 'SAFE_STOP':
            self.phase = 'SAFE_STOP'
        elif self.phase == 'EXIT' or local_x > 1.1:
            self.phase = 'EXIT'
        elif self.entered:
            self.phase = 'CROSS'
        else:
            priority_peer = any(m.sender < self.robot_id and m.phase != 'EXIT' for m in inbox)
            if priority_peer:
                self.phase = 'WAIT'
                self.peer_influenced = True
            elif inbox:
                self.phase = 'CROSS'
                self.entered = True
                if any(m.phase == 'EXIT' for m in inbox):
                    self.peer_influenced = True
        return old != self.phase


class AdaptiveNegotiator(Negotiator):
    """Use a deck-mounted local measurement only while a foot touches the deck."""

    def __init__(self, robot_id):
        super().__init__(robot_id)
        self.pace = 1.

    def update(self, local_x, fallen, inbox, time_s=0., angular_rate=0.,
               deck_angle=None, deck_rate=None, deck_contact=False):
        old = (self.phase, self.pace)
        had_entered = self.entered
        super().update(local_x, fallen, inbox, time_s, angular_rate)
        observations = [(m.deck_angle_rad, m.deck_rate_rad_s) for m in inbox
                        if m.deck_angle_rad is not None and m.deck_rate_rad_s is not None]
        if deck_contact and deck_angle is not None and deck_rate is not None:
            observations.append((deck_angle, deck_rate))
        unsettled = any(abs(a) > .012 or abs(v) > .08 for a, v in observations)
        if not had_entered and self.phase == 'CROSS' and unsettled:
            self.phase = 'WAIT'
            self.entered = False
        if self.phase == 'CROSS' and not self.entered:
            raise RuntimeError('Crossing without entry')
        self.pace = .85 if self.phase == 'CROSS' and unsettled else 1.
        if self.phase == 'CROSS' and self.pace < 1.:
            self.phase = 'ADAPT'
        elif self.phase == 'ADAPT':
            self.phase = 'CROSS'
        return old != (self.phase, self.pace)
