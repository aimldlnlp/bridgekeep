"""Replay and diagnose the saved solo pause experiment at 10 Hz."""
import json
from pathlib import Path
import mujoco
import numpy as np
from scene import ROOT,build_scene


def main():
    source=ROOT/'outputs/blend_pause50/bridge';rec=np.load(source/'recording.npz');ep=json.loads((source/'episode.json').read_text())
    model=build_scene(1,counterbalance=True,half_width=ep.get('half_width_m',1.));data=mujoco.MjData(model)
    body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'r0_base')
    feet=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'r0_'+leg) for leg in ('FL','FR','RL','RR')]
    rows=[]
    for k in range(0,len(rec['time']),5):
        t=float(rec['time'][k]);data.qpos[:]=rec['qpos'][k];data.qvel[:]=rec['qvel'][k];data.ctrl[:]=rec['ctrl'][k];mujoco.mj_forward(model,data)
        contacts=[]
        for foot in feet:
            foot_contacts=[]
            for i,c in enumerate(data.contact[:data.ncon]):
                if foot not in (c.geom1,c.geom2):continue
                other=c.geom2 if c.geom1==foot else c.geom1;force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
                foot_contacts.append({'surface':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,other) or 'unnamed','normal_force_n':float(force[0])})
            contacts.append(foot_contacts)
        rotation=data.xmat[body].reshape(3,3);roll=float(np.arctan2(rotation[2,1],rotation[2,2]));pitch=float(np.arctan2(-rotation[2,0],np.hypot(rotation[2,1],rotation[2,2])))
        elapsed=t-ep['pause_at_s'];gait_time=t if elapsed<1 else ep['pause_at_s']+1 if elapsed<3 else t-2
        rows.append({'time_s':t,'gait_clock_s':gait_time,'gait_phase':((gait_time-1)*ep['frequency_hz'])%1 if gait_time>=1 else None,
                     'body_position_m':data.xpos[body].tolist(),'roll_rad':roll,'pitch_rad':pitch,'foot_position_m':data.geom_xpos[feet].tolist(),
                     'foot_contacts':contacts,'actuator_saturations':int(np.count_nonzero(np.isclose(data.ctrl,model.actuator_ctrlrange[:,0],atol=1e-8,rtol=0)|np.isclose(data.ctrl,model.actuator_ctrlrange[:,1],atol=1e-8,rtol=0)))})
    out=source/'diagnostics.json';out.write_text(json.dumps({'source':str(source),'sample_interval_s':.1,'limitation':'Contacts and forces recomputed from recorded state; values are not original integration-step forces.','samples':rows},indent=2),encoding='utf-8');print(f'{len(rows)} samples -> {out}')


if __name__=='__main__':main()
