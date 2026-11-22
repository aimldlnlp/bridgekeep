"""Replay a saved counterbalance trace for sampled contact and landing diagnostics."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scene import build_scene


def main(source, output):
    rec=np.load(source/'recording.npz'); model=build_scene(2,counterbalance=True); data=mujoco.MjData(model)
    deck=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'deck')
    feet=[[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f'r{i}_{leg}') for leg in ('FL','FR','RL','RR')] for i in range(2)]
    samples=[]
    for k in range(0,len(rec['time']),5):
        data.qpos[:]=rec['qpos'][k];data.qvel[:]=rec['qvel'][k];data.ctrl[:]=rec['ctrl'][k]
        mujoco.mj_forward(model,data)
        rows=[]
        for i in range(2):
            for leg,geom in enumerate(feet[i]):
                pairs=[]
                jacobian=np.zeros((3,model.nv))
                mujoco.mj_jacGeom(model,data,jacobian,None,geom)
                velocity=jacobian@data.qvel
                for cidx,c in enumerate(data.contact[:data.ncon]):
                    if geom not in (c.geom1,c.geom2): continue
                    other=c.geom2 if c.geom1==geom else c.geom1
                    force=np.zeros(6);mujoco.mj_contactForce(model,data,cidx,force)
                    pairs.append({'surface':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,other) or 'unnamed',
                                  'normal_force_n':float(force[0])})
                rows.append({'robot':i,'leg':leg,'position_m':data.geom_xpos[geom].tolist(),
                             'velocity_m_s':velocity.tolist(),'contacts':pairs})
        rotation=data.xmat[model.geom_bodyid[deck]].reshape(3,3)
        endpoint=(data.geom_xpos[deck]+rotation@np.array([-2.,0.,.05])).tolist()
        samples.append({'time_s':float(rec['time'][k]),'left_deck_endpoint_m':endpoint,'feet':rows})
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'source':str(source),'sample_interval_s':.1,'limitation':'Replay from 50 Hz saved states; forces recomputed by mj_forward, not original per-step forces.','samples':samples},indent=2),encoding='utf-8')
    print(f'{len(samples)} samples -> {output}')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.source,a.output)
