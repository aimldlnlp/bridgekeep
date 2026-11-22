"""Reproducible environment and reduced-order bridge physics checks."""
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "vendor/unitree_go2/scene.xml"


def environment():
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert model.nu == 12 and model.nq == 19 and data.time > 0
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        renderer.update_scene(data)
        frame = renderer.render()
    assert frame.shape == (240, 320, 3) and frame.std() > 1
    path = ROOT / "outputs/b0/smoke.mp4"
    with imageio.get_writer(path, fps=10, codec="libx264", macro_block_size=1) as writer:
        for _ in range(10):
            writer.append_data(frame)
    with imageio.get_reader(path) as reader:
        assert sum(1 for _ in reader) == 10
    return {"mujoco": mujoco.__version__, "numpy": np.__version__, "nq": model.nq,
            "actuators": model.nu, "render_shape": list(frame.shape), "decoded_frames": 10}


def bridge(load=False, impulse=False):
    # The hinge is a reduced-order bridge surrogate, not a deformable beam.
    ball = '<body pos="0.5 0 0.56"><freejoint/><geom name="load" type="sphere" size=".1" mass="2"/></body>' if load else ''
    xml = f'''<mujoco><option timestep="0.002"/><worldbody>
    <body name="deck" pos="0 0 .4"><joint name="hinge" type="hinge" axis="0 1 0" stiffness="100" damping="3"/>
    <geom name="deck_geom" type="box" size="1 .3 .05" mass="10"/></body>{ball}</worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    angles, times, contacts = [], [], 0
    start = time.perf_counter()
    for step in range(5000):
        if impulse and step < 50:
            data.qfrc_applied[0] = 5.0
        else:
            data.qfrc_applied[0] = 0
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            raise RuntimeError(f"Non-finite state at {data.time}s")
        angles.append(float(data.qpos[0])); times.append(float(data.time))
        contacts += data.ncon > 0
    return np.array(times), np.array(angles), int(contacts), time.perf_counter() - start


def main():
    env = environment()
    (ROOT / 'outputs/b0/environment.json').write_text(json.dumps(env, indent=2))
    _, equilibrium, _, _ = bridge()
    t, decay, _, elapsed = bridge(impulse=True)
    _, loaded, contacts, _ = bridge(load=True)
    peaks = np.flatnonzero((decay[1:-1] > decay[:-2]) & (decay[1:-1] > decay[2:])) + 1
    positive = peaks[(decay[peaks] > 0) & (t[peaks] > .1)]
    assert np.max(np.abs(equilibrium)) < 1e-10
    assert len(positive) >= 2 and abs(decay[positive[-1]]) < abs(decay[positive[0]])
    assert contacts > 0 and np.max(np.abs(loaded)) > .001
    result = {"equilibrium_max_rad": float(np.max(np.abs(equilibrium))),
              "free_decay_first_peak_rad": float(decay[positive[0]]),
              "free_decay_last_peak_rad": float(decay[positive[-1]]),
              "frequency_hz": float(1 / np.mean(np.diff(t[positive]))),
              "load_peak_rad": float(np.max(np.abs(loaded))), "load_contact_steps": contacts,
              "simulation_seconds": float(t[-1]), "wall_seconds": elapsed}
    (ROOT / 'outputs/b1/bridge.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({"environment": env, "bridge": result}, indent=2))


if __name__ == '__main__':
    main()
