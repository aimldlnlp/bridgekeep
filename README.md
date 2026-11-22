# BRIDGEKEEP

Two simulated Unitree Go2 quadrupeds cross a passive seesaw bridge from opposite sides. Their foot contacts load the same deck, so each robot changes the surface beneath the other. This project explores how entry timing and gait pacing affect bridge motion, stability, and crossing time.

https://github.com/user-attachments/assets/f6428f46-8aed-4efb-977b-f1ef866d0239

The video shows a successful crossing, followed by a trial where a disturbance causes one robot to fall. The graphs show measured bridge motion and contact forces from the simulation.

## How it works

The scene uses two free-base Go2 models and a rigid deck on a passive spring-damper hinge. Passive hinged transition plates connect the deck to the landings and can transfer forces to it. The bridge is a reduced-order mechanical model; it is not a flexible beam simulation.

Each robot follows a prescribed foot trajectory. Analytical inverse kinematics maps foot targets to joint angles, and torque-limited PD controllers track those targets. Robot-local lateral position feedback helps keep each robot in its lane. A range-limited message bus carries readiness, progress, and contact status. In the progress-pacing experiment, the robot ahead slows its gait clock when the other robot is on the deck.

The local lateral position comes from ideal simulator state. The controller is experimental and is not validated on hardware.

## Results

Each result comes from a repeatable simulation using the hinged transition plates. Times are in simulated seconds.

| Trial | Result | Time |
| --- | --- | ---: |
| Team crossing | Both robots reached the opposite landing | 98.79 s |
| Team, shifted 5 cm left | Both reached the landing | 88.06 s |
| Team, shifted 5 cm right | Both reached the landing | 97.12 s |
| 100 Nm disturbance for 0.1 s | Robot B fell | 83.09 s |
| Solo stop and restart, ±0.05 m offsets | Robot fell in both trials | 96.08 s and 96.06 s |

The nominal trial recorded zero robot-to-robot contact steps. A robot is considered to have landed when all four feet have cleared the deck and at least one foot touches the opposite landing. Sustained balance after termination has not been tested.

Between 10.98 and 20.00 s, the turning forces from the robots’ feet opposed each other in 429 of 452 measurements. The transition plates also load the deck, and the closed-loop trajectories diverge after robot B enters. The comparison therefore does not isolate a single cause for the later bridge motion.

With the same initial conditions and bridge geometry, progress pacing reduced team time from 107.05 s to 98.79 s in one paired trial. Over the same 0–98.78 s window, the root-mean-square angular speed of the deck increased from 0.06284 to 0.10930 rad/s. These trials are not a statistical robustness study.

## Run locally

Python 3.12 and a working MuJoCo rendering backend are required. Install the pinned dependencies from the project root:

```powershell
python -m pip install -r requirements.txt
```

Run the team crossing and a comparison where robot B stays on its starting platform:

```powershell
python -B src/counterbalance.py coordinated --delay 4 --foot-gait --staggered --balance-progress --exit-margin 0.03 --lane-gain 0.3 --hinged-lip --output outputs/hinged_instrumented
python -B src/counterbalance.py reference --foot-gait --exit-margin 0.03 --lane-gain 0.3 --hinged-lip --output outputs/hinged_instrumented_reference
```

To reproduce the disturbance trial:

```powershell
python -B src/counterbalance.py coordinated --delay 4 --foot-gait --staggered --balance-progress --exit-margin 0.03 --lane-gain 0.3 --hinged-lip --disturbance --output outputs/hinged_disturbance
```

Failed episodes return a nonzero exit code after saving their results. Commands overwrite files in their specified output directories. Generated recordings are excluded from Git because they can be large.

Run the focused tests with:

```powershell
python -B -m unittest discover -s tests -p test_counterbalance.py -v
python -B -m unittest discover -s tests -p test_foot_trajectory.py -v
python -B -m unittest discover -s tests -p test_physics.py -v
python -B -m unittest discover -s tests -p test_coordination.py -v
```

Tests that check past recordings are skipped until their fixtures exist under `outputs/`. The full suite also requires generated recordings.

## Repository contents

- `src/scene.py` builds the MuJoCo scene.
- `src/counterbalance.py` runs the team trials and records contacts, messages, and forces.
- `src/foot_trajectory.py` and `src/foot_control.py` implement the experimental gait and joint control.
- `src/analyze_counterbalance_causal.py` compares the team trial with the B-waits reference.

Use `--hinged-lip` for the physical transition plates. The older `--transition-lip` option allows overlapping geometry and is unsuitable for physical evaluation.

## Robot asset

The Go2 model is from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/unitree_go2), revision `370c43965323c0944b433b202aac5058b5f907a9`, under the [BSD 3-Clause license](vendor/unitree_go2/LICENSE). The model does not include this project's walking controller. See [asset provenance](ASSET_PROVENANCE.txt).
