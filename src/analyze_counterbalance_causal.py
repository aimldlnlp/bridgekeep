"""Compare a delayed-entry run with a B-waits counterfactual on matched timestamps."""
import argparse,hashlib,json
from pathlib import Path
import mujoco
import numpy as np
from scene import build_scene


def main(pair_path,reference_path,output):
    p=json.loads((pair_path/'episode.json').read_text());r=json.loads((reference_path/'episode.json').read_text())
    for key in ('bridge_length_m','bridge_mass_kg','stiffness_nm_rad','damping_nms_rad','initial_x_m','lateral_offset_m','controller','timestep_s'):
        if p[key]!=r[key]:raise ValueError(f'Mismatched {key}')
    for key,default in (('bridge_half_width_m',1.),('exit_margin_m',0.),('lane_gain',0.)):
        if p.get(key,default)!=r.get(key,default):raise ValueError(f'Mismatched {key}')
    if p.get('transition_lip',False)!=r.get('transition_lip',False):raise ValueError('Mismatched transition_lip')
    if p.get('hinged_lip',False)!=r.get('hinged_lip',False):raise ValueError('Mismatched hinged_lip')
    a=np.load(pair_path/'recording.npz');b=np.load(reference_path/'recording.npz')
    np.testing.assert_array_equal(a['qpos'][0],b['qpos'][0]);np.testing.assert_array_equal(a['qvel'][0],b['qvel'][0])
    common=np.intersect1d(np.round(a['time'],6),np.round(b['time'],6));ia=np.searchsorted(np.round(a['time'],6),common);ib=np.searchsorted(np.round(b['time'],6),common)
    model=build_scene(2,counterbalance=True,half_width=p.get('bridge_half_width_m',1.),bridge_stiffness=p['stiffness_nm_rad'],transition_lip=p.get('transition_lip',False),hinged_lip=p.get('hinged_lip',False));hinge=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'bridge_hinge');q=model.jnt_qposadr[hinge];v=model.jnt_dofadr[hinge]
    entry=p['first_foot_deck_contact_s'][1]
    if entry is None:raise ValueError('B never touches deck')
    actions=[d['time_s'] for d in p.get('decisions',[]) if d.get('robot')==1 and d.get('decision')=='ENTER']
    before_action=actions[0] if actions else p['delay_s']
    before=common<before_action
    if not np.any(before) or not np.array_equal(a['qpos'][ia[before],q],b['qpos'][ib[before],q]) or not np.array_equal(a['qvel'][ia[before],v],b['qvel'][ib[before],v]):raise ValueError('Bridge histories differ before B acts')
    mask=(common>=entry+.1)&(common<=20.);indices=ia[mask];control=ib[mask]
    if not len(indices):raise ValueError('Missing counterfactual analysis window')
    torque=a['contact_torque_nm'][indices];angles=a['qpos'][indices,q];other=b['qpos'][control,q]
    result={'pair':str(pair_path),'reference':str(reference_path),'initial_state_equal':True,'bridge_before_B_action_equal':True,'B_action_s':before_action,
            'B_first_contact_s':entry,'window_s':[float(common[mask][0]),float(common[mask][-1])],
            'opposing_torque_samples':int(np.sum(torque[:,0]*torque[:,1]<0)), 'samples':len(indices),
            'mean_foot_torque_nm':np.mean(torque,axis=0).tolist(),
            'bridge_angle_difference_rms_rad':float(np.sqrt(np.mean((angles-other)**2))),
            'pair_peak_absolute_angle_rad':float(np.max(np.abs(angles))), 'reference_peak_absolute_angle_rad':float(np.max(np.abs(other))),
            'limitation':'Foot-deck moments only; B holds its initial stance in the reference and can contact a hinged plate. Plate-deck forces also act on the bridge. Closed-loop trajectories diverge after B acts; this does not isolate all later causes.',
            'recording_sha256':{str(path):hashlib.sha256((path/'recording.npz').read_bytes()).hexdigest() for path in (pair_path,reference_path)}}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('pair',type=Path);p.add_argument('reference',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.pair,a.reference,a.output)
