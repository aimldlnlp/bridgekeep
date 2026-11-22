"""Shared MuJoCo scene with separately named, free-base Go2 robots."""
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET
import warnings
import mujoco

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'vendor/unitree_go2'


def build_scene(count, visual=False, counterbalance=False, half_width=1., bridge_stiffness=10000., transition_lip=False, hinged_lip=False):
    if count not in (1, 2):
        raise ValueError('Only one or two robots supported')
    if not 0.5 <= half_width <= 3.:
        raise ValueError('Bridge half-width must be between 0.5 and 3 m')
    if not 500. <= bridge_stiffness <= 50000.:
        raise ValueError('Bridge stiffness must be between 500 and 50000 Nm/rad')
    if hinged_lip and (not counterbalance or transition_lip):
        raise ValueError('Hinged lips require counterbalance and cannot be combined with fixed lips')
    root = ET.parse(ASSET / 'go2.xml').getroot()
    root.find('compiler').set('meshdir', str(ASSET / 'assets').replace('\\', '/'))
    world = root.find('worldbody')
    original = world.find('body')
    world.remove(original)
    actuators = root.find('actuator')
    original_actuators = list(actuators)
    actuators.clear()
    for i in range(count):
        prefix = f'r{i}_'
        robot = deepcopy(original)
        for element in robot.iter():
            if 'name' in element.attrib:
                element.set('name', prefix + element.get('name'))
        world.append(robot)
        for actuator in original_actuators:
            new = deepcopy(actuator)
            new.set('name', prefix + new.get('name'))
            new.set('joint', prefix + new.get('joint'))
            actuators.append(new)
    root.remove(root.find('keyframe'))
    ET.SubElement(world, 'light', {'pos': '0 0 4', 'dir': '0 0 -1', 'directional': 'true',
                                   **({'castshadow': 'false'} if counterbalance else {})})
    platforms = [('approach', '-3.65', '-.05'), ('exit', '3.65', '-.05')] if counterbalance else [('approach', '-2.35', '-.05'), ('exit', '2.35', '-.065')]
    for name, x, z in platforms:
        ET.SubElement(world, 'geom', {'name': name, 'type': 'box', 'pos': f'{x} 0 {z}',
                      'size': f'1.65 {half_width if counterbalance else 1.} .05', 'contype': '2', 'conaffinity': '1'})
    if transition_lip:
        if not counterbalance:
            raise ValueError('Transition lips require the counterbalance scene')
        warnings.warn('Fixed lips overlap the deck with collisions filtered; use only to reproduce the rejected geometry.', RuntimeWarning)
        for name, x, yrot in [('left_lip', -2., .2), ('right_lip', 2., -.2)]:
            ET.SubElement(world, 'geom', {'name': name, 'type': 'box', 'pos': f'{x} 0 -.04',
                'size': f'.15 {half_width} .01', 'euler': f'0 {yrot} 0',
                'contype': '2', 'conaffinity': '1', 'rgba': '.45 .39 .30 1'})
    if hinged_lip:
        for name, pivot, offset in [('left_lip', -2.15, .15), ('right_lip', 2.15, -.15)]:
            plate = ET.SubElement(world, 'body', {'name': name+'_body', 'pos': f'{pivot} 0 .02'})
            ET.SubElement(plate, 'joint', {'name': name+'_hinge', 'type': 'hinge', 'axis': '0 1 0',
                                          'damping': '1', 'range': '-.3 .3'})
            ET.SubElement(plate, 'geom', {'name': name, 'type': 'box', 'pos': f'{offset} 0 -.01',
                'size': f'.15 {half_width} .01', 'mass': '.5', 'contype': '1', 'conaffinity': '3',
                'rgba': '.45 .39 .30 1'})
    bridge = ET.SubElement(world, 'body', {'name': 'bridge', 'pos': '0 0 -.05'})
    ET.SubElement(bridge, 'joint', {'name': 'bridge_hinge', 'type': 'hinge', 'axis': '0 1 0',
                                   'stiffness': str(bridge_stiffness if counterbalance else 10000), 'damping': '80', 'range': '-0.15 0.15'})
    ET.SubElement(bridge, 'geom', {'name': 'deck', 'type': 'box', 'size': f'2 {half_width} .05' if counterbalance else '.7 1 .05',
                                  'mass': '20', 'contype': '2', 'conaffinity': '1'})
    if visual:
        asset = root.find('asset')
        ET.SubElement(asset, 'texture', {'name': 'lab_grid', 'type': '2d', 'builtin': 'checker',
                                        'rgb1': '.93 .94 .93', 'rgb2': '.86 .88 .87', 'width': '256', 'height': '256'})
        ET.SubElement(asset, 'material', {'name': 'lab_floor', 'texture': 'lab_grid', 'texrepeat': '18 18',
                                         'reflectance': '.06'})
        ET.SubElement(world, 'geom', {'name': 'visual_floor', 'type': 'plane', 'pos': '0 0 -2' if counterbalance else '0 0 -.25',
                                    'size': '12 12 .1', 'material': 'lab_floor', 'contype': '0', 'conaffinity': '0'})
        for y in (-half_width + .02, half_width - .02) if counterbalance else ('-.98', '.98'):
            ET.SubElement(bridge, 'geom', {'type': 'capsule', 'fromto': f'{-1.98 if counterbalance else -.68} {y} .055 {1.98 if counterbalance else .68} {y} .055',
                                          'size': '.012', 'rgba': '.25 .32 .33 1', 'mass': '0',
                                          'contype': '0', 'conaffinity': '0'})
        ET.SubElement(world, 'geom', {'name': 'visual_hinge', 'type': 'cylinder', 'pos': '0 0 -.05',
                                    'quat': '.70710678 .70710678 0 0', 'size': f'.035 {half_width + .05 if counterbalance else 1.05}',
                                    'rgba': '.28 .32 .33 1', 'contype': '0', 'conaffinity': '0'})
        if counterbalance:
            ET.SubElement(world, 'geom', {'name': 'visual_pier', 'type': 'box', 'pos': '0 0 -1.05',
                                          'size': '.08 .12 .95', 'rgba': '.32 .37 .39 1',
                                          'contype': '0', 'conaffinity': '0'})
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
