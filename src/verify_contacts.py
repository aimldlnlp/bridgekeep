"""Inspect replayed contact pairs, including non-foot contacts and robot collisions."""
import collections
import json
import mujoco
import numpy as np
from scene import ROOT, build_scene


def main():
    result = {}
    for name in ('s1', 's2', 's3', 's4'):
        rec = np.load(ROOT / 'outputs' / name / 'recording.npz')
        qpos, qvel, recorded = rec['qpos'], rec['qvel'], rec['foot_deck']
        model = build_scene(recorded.shape[1])
        data = mujoco.MjData(model)
        deck = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'deck')
        feet = [[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'r{i}_{leg}')
                 for leg in ('FL','FR','RL','RR')] for i in range(recorded.shape[1])]
        all_feet = set(sum(feet, []))
        owner = []
        for geom in range(model.ngeom):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom]) or ''
            owner.append(body[:2] if body.startswith(('r0_', 'r1_')) else '')
        nonfoot = collections.Counter()
        penetration, between, mismatch, checked = 0., 0, 0, 0
        for k in range(0, len(qpos), 5):
            data.qpos[:] = qpos[k]
            data.qvel[:] = qvel[k]
            mujoco.mj_forward(model, data)
            pairs = {(int(c.geom1), int(c.geom2)) for c in data.contact[:data.ncon]}
            for i, group in enumerate(feet):
                for j, foot in enumerate(group):
                    observed = (deck, foot) in pairs or (foot, deck) in pairs
                    mismatch += bool(observed != recorded[k, i, j])
                    checked += 1
            for c in data.contact[:data.ncon]:
                first, second = int(c.geom1), int(c.geom2)
                other = second if first == deck else first if second == deck else -1
                if other >= 0 and owner[other] and other not in all_feet:
                    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[other])
                    nonfoot[body] += 1
                    penetration = max(penetration, -float(c.dist))
                if owner[first] and owner[second] and owner[first] != owner[second]:
                    between += 1
        if mismatch:
            raise RuntimeError(f'{name}: {mismatch} replayed foot contact mismatches')
        result[name] = {'sample_interval_s': .1, 'foot_contact_bools_checked': checked,
                        'foot_contact_mismatches': mismatch, 'nonfoot_deck_contacts_by_body': dict(nonfoot),
                        'max_nonfoot_penetration_m': penetration, 'robot_robot_contact_count': between}
    out = ROOT / 'outputs' / 'contact_audit.json'
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
