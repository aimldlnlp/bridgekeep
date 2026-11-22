"""Check every recorded foot/deck contact against forward kinematics replay."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scene import build_scene


def verify(source, output):
    rec=np.load(source/'recording.npz');episode=json.loads((source/'episode.json').read_text())
    model=build_scene(2,counterbalance=True,half_width=episode.get('bridge_half_width_m',1.),bridge_stiffness=episode['stiffness_nm_rad'],transition_lip=episode.get('transition_lip',False),hinged_lip=episode.get('hinged_lip',False));data=mujoco.MjData(model)
    deck=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'deck')
    feet=[[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f'r{i}_{leg}') for leg in ('FL','FR','RL','RR')] for i in range(2)]
    mismatches=[]
    for k in range(len(rec['time'])):
        data.qpos[:]=rec['qpos'][k];data.qvel[:]=rec['qvel'][k];mujoco.mj_forward(model,data)
        pairs={frozenset((int(c.geom1),int(c.geom2))) for c in data.contact[:data.ncon]}
        for i in range(2):
            for leg,foot in enumerate(feet[i]):
                present=frozenset((deck,foot)) in pairs
                if present != rec['foot_deck'][k,i,leg]:
                    mismatches.append({'time_s':float(rec['time'][k]),'robot':i,'leg':leg})
    result={'source':str(source),'frames':len(rec['time']),'contact_booleans_checked':len(rec['time'])*8,
            'mismatch_count':len(mismatches),'mismatches':mismatches,
            'note':'Replays all recorded contact snapshots; does not check contacts between samples.'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='mismatches'},indent=2))
    if mismatches: raise SystemExit(f'{len(mismatches)} contact mismatches; inspect {output}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();verify(a.source,a.output)
