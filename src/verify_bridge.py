"""Verify the exact scene bridge at equilibrium, after torque, and under foot load."""
import json
import mujoco
import numpy as np
from scene import ROOT, build_scene


def main():
    model = build_scene(1)
    data = mujoco.MjData(model)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'r0_base')
    base_qpos = model.jnt_qposadr[model.body_jntadr[base]]
    data.qpos[base_qpos:base_qpos+3] = [10., 0., .27]
    hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'bridge_hinge')
    qpos, dof = model.jnt_qposadr[hinge], model.jnt_dofadr[hinge]
    equilibrium, times, angle = [], [], []
    for step in range(1500):
        data.qfrc_applied[dof] = 30. if 500 <= step < 525 else 0.
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError(f'Non-finite physics at {data.time:.3f} s')
        if step < 500:
            equilibrium.append(float(data.qpos[qpos]))
        times.append(float(data.time))
        angle.append(float(data.qpos[qpos]))
    t, a = np.asarray(times), np.asarray(angle)
    maxima = np.flatnonzero((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:])) + 1
    peaks = maxima[(t[maxima] > 1.05) & (a[maxima] > 1e-9)]
    if len(peaks) < 3 or not np.all(np.diff(a[peaks]) < 0) or np.max(np.abs(equilibrium)) > 1e-10:
        raise RuntimeError('Scenario bridge equilibrium or free decay failed')
    rec = np.load(ROOT / 'outputs/s1/recording.npz')
    contact = rec['foot_deck'].any(axis=(1, 2))
    deck = rec['qpos'][:, qpos]
    first_contact = int(np.flatnonzero(contact)[0])
    first_motion = int(np.flatnonzero(np.abs(deck) > 1e-5)[0])
    if first_motion < first_contact or np.max(np.abs(deck)) < .001:
        raise RuntimeError('S1 bridge response is not consistent with physical foot contact')
    logarithmic_decrement = float(np.log(a[peaks[0]] / a[peaks[1]]))
    result = {'scene_bridge_stiffness_nm_rad': float(model.jnt_stiffness[hinge]),
              'scene_bridge_damping_nms_rad': float(model.dof_damping[dof]),
              'equilibrium_max_rad': float(np.max(np.abs(equilibrium))),
              'torque_nm': 30., 'torque_duration_s': .05,
              'free_decay_frequency_hz': float(1 / np.mean(np.diff(t[peaks]))),
              'logarithmic_decrement': logarithmic_decrement,
              'damping_ratio': float(logarithmic_decrement / np.sqrt((2*np.pi)**2 + logarithmic_decrement**2)),
              'first_peak_rad': float(a[peaks[0]]), 'last_peak_rad': float(a[peaks[-1]]),
              's1_first_foot_contact_s': float(rec['time'][first_contact]),
              's1_first_bridge_motion_s': float(rec['time'][first_motion]),
              's1_peak_angle_rad': float(np.max(np.abs(deck)))}
    (ROOT / 'outputs/b1/scenario_bridge_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
