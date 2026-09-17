# Lab 3: A* maze navigation

This implementation is adapted from teammate repository `sphero-h1-latest`,
commit `93aa75e` (Lab 3, the required Lab 2 stack, logging helpers and compatible
environment/robot interfaces). OpenAI Codex assisted with migration, dependency
inspection, student-ID configuration and local verification on 2026-09-17.
The teammate's original algorithm comments and calibration provenance are
retained. Calibration values and historical hardware observations in those
comments belong to the teammate's robot; they are not new measurements made
for this repository.

## Run

From the repository root with the project dependencies installed:

```powershell
$env:PYTHONPATH = "$PWD/src"
python labs/lab3/lab3.py --sim --no-render --no-archive
python labs/lab3/check_lab3.py
python labs/lab3/verify_lab3.py
```

Default output ID: `35301856`, taken from the existing M1-2 filename. Change
`STUDENT_ID` in `labs/lab2/lab2.py` and `labs/lab3/lab3.py` if necessary.
`Planner.py` preserves the skeleton's `Planner(map, dt).plan(state, goal)` API.
It uses four-neighbour A* with Manhattan distance and 0.125 m grid cells.
The mission uses the Lab 2 EKF and waypoint follower, with collision/deviation
replanning. The shared simulator, visualiser, maze and robot modules are
synchronised because the teammate's runner needs their newer interfaces.

## Results and provenance

`exports/35301856_lab3.csv` is freshly generated simulation output. Its
`sim_x,sim_y` columns are simulator truth and `real_x,real_y` are the EKF
estimate in that simulation, as described by the lab's CSV schema. This is
not a physical robot recording. The JSON verdict explicitly records `mode: sim`.
No teammate CSV, logs, plots or Git history were imported.

Seed 0 verification: 179 rows, 0 collisions; final estimate error 0.0094 m,
final simulator error 0.0136 m; maximum path deviations 0.0219 m and 0.0136 m.
All four local threshold checks pass. A separate BFS check verifies A* path
optimality, and tests cover walls, disconnected goals and identical endpoints.
These are local checks, not results from the university automarker.

Physical execution is supported by running without `--sim`; it requires a
connected, aimed robot and validation of the inherited calibration. Hardware
execution has not been performed during this migration.

`commit_history.txt` at the repository root is exported from the real Git log
after the implementation commit. It is intentionally not included in that
commit, so it can contain the commit that implements this work.
