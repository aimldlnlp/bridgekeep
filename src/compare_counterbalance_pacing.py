"""Compare paired episodes over exactly equal recorded timestamps."""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
from scene import build_scene


def main(control,coordinated,output):
    paths=(control,coordinated);episodes=[json.loads((p/'episode.json').read_text()) for p in paths]
    for key in ('bridge_length_m','bridge_half_width_m','bridge_mass_kg','stiffness_nm_rad','damping_nms_rad','timestep_s','initial_x_m','lateral_offset_m','exit_margin_m','lane_gain','delay_s','controller','staggered_entry','disturbance'):
        if episodes[0][key]!=episodes[1][key]:raise ValueError(f'Mismatched {key}')
    if episodes[0].get('transition_lip',False)!=episodes[1].get('transition_lip',False):raise ValueError('Mismatched transition_lip')
    if episodes[0].get('hinged_lip',False)!=episodes[1].get('hinged_lip',False):raise ValueError('Mismatched hinged_lip')
    if episodes[0]['balance_progress'] or not episodes[1]['balance_progress']:
        raise ValueError('Expected no-pacing control and progress-pacing treatment')
    recs=[np.load(p/'recording.npz') for p in paths]
    np.testing.assert_array_equal(recs[0]['qpos'][0],recs[1]['qpos'][0]);np.testing.assert_array_equal(recs[0]['qvel'][0],recs[1]['qvel'][0])
    t=np.intersect1d(np.round(recs[0]['time'],6),np.round(recs[1]['time'],6))
    if len(t)<2:raise ValueError('No shared window')
    m=build_scene(2,counterbalance=True,half_width=episodes[0]['bridge_half_width_m'],bridge_stiffness=episodes[0]['stiffness_nm_rad'],transition_lip=episodes[0].get('transition_lip',False),hinged_lip=episodes[0].get('hinged_lip',False));j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'bridge_hinge');q=m.jnt_qposadr[j];v=m.jnt_dofadr[j]
    result={'common_window_s':[float(t[0]),float(t[-1])], 'initial_state_equal':True,'scenarios':{},
            'limitation':'One deterministic paired trial. Window ends at earlier episode end; outcomes and team times cover full respective episodes.'}
    for name,rec,e in zip(('no_progress_pacing','progress_pacing'),recs,episodes):
        idx=np.searchsorted(np.round(rec['time'],6),t)
        result['scenarios'][name]={'bridge_rate_rms_rad_s':float(np.sqrt(np.mean(rec['qvel'][idx,v]**2))),
             'peak_angle_rad':float(np.max(np.abs(rec['qpos'][idx,q]))),'team_time_s':e['simulation_seconds'],
             'status':e['stop_reason'],'peak_tilt_rad':e['peak_tilt_rad'],'peak_foot_deck_tangential_speed_m_s':e['peak_foot_deck_tangential_speed_m_s'],
             'robot_contact_steps':e['robot_contact_steps'],'sampled_actuator_saturations':e['actuator_saturation_samples']}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('control',type=Path);p.add_argument('coordinated',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.control,a.coordinated,a.output)
