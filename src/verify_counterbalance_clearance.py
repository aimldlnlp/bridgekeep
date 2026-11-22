"""Sample inter-robot clearance and inspect the recorded terminal landing state."""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
from scene import build_scene


def main(source,output):
    ep=json.loads((source/'episode.json').read_text());rec=np.load(source/'recording.npz')
    model=build_scene(2,counterbalance=True,half_width=ep.get('bridge_half_width_m',1.),bridge_stiffness=ep['stiffness_nm_rad'],transition_lip=ep.get('transition_lip',False),hinged_lip=ep.get('hinged_lip',False));data=mujoco.MjData(model)
    bases=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f'r{i}_base') for i in range(2)];owner=[]
    for g in range(model.ngeom):
        body=int(model.geom_bodyid[g])
        while body and body not in bases:body=int(model.body_parentid[body])
        owner.append(bases.index(body) if body in bases else None)
    geoms=[[g for g in range(model.ngeom) if owner[g]==i and (model.geom_contype[g] or model.geom_conaffinity[g])] for i in range(2)]
    pairs=[(a,b) for a in geoms[0] for b in geoms[1] if (model.geom_contype[a]&model.geom_conaffinity[b]) or (model.geom_contype[b]&model.geom_conaffinity[a])]
    minimum=2.;at=None;who=None
    for k in range(0,len(rec['time']),5):
        data.qpos[:]=rec['qpos'][k];data.qvel[:]=rec['qvel'][k];mujoco.mj_forward(model,data)
        for a,b in pairs:
            distance=float(mujoco.mj_geomDistance(model,data,a,b,minimum,None))
            if distance<minimum:minimum,at,who=distance,float(rec['time'][k]),[int(a),int(b)]
    data.qpos[:]=rec['terminal_qpos'];data.qvel[:]=rec['terminal_qvel'];mujoco.mj_forward(model,data)
    landed=[]
    for i in range(2):
        landing=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,('exit','approach')[i])
        feet=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f'r{i}_{leg}') for leg in ('FL','FR','RL','RR')]
        contacts=[leg for leg,g in zip(('FL','FR','RL','RR'),feet) if any({int(c.geom1),int(c.geom2)}=={g,landing} for c in data.contact[:data.ncon])]
        x=data.geom_xpos[feet,0]; direction=(1,-1)[i]
        landed.append({'robot':i,'foot_centers_x_m':x.tolist(),'all_foot_spheres_clear_edge':bool(all(direction*x[k]-model.geom_size[g,0]>2. for k,g in enumerate(feet))), 'landing_contact_feet':contacts})
    result={'sample_interval_s':.1,'sampled_minimum_robot_geom_distance_m':minimum,'time_s':at,'geom_ids':who,'geometry_pairs_checked':len(pairs),'terminal_landing':landed,'robot_contact_steps':ep.get('robot_contact_steps'),'limitation':'Clearance is sampled at 10 Hz and capped at 2 m; final contact is instantaneous, not a sustained stance.'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.source,a.output)
