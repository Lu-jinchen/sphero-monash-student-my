# Lab 3: Path planning

In Lab 3, you need to implement an A* path planner (or something else, eg. RRT, RL) to escape a known maze.

## Key learning outcomes

- Understand and implement a global planner into a robot navigation stack.
- Interface this with low level controllers and estimators from your previous labs to escape a maze.

## What you need to do

- **Implement a global waypoint planner** to escape a grid-like mazeworld.
- **Add a lower level controller** to move between these waypoints.
- **Add replanning as required** to ensure the robot escapes a known maze.

The lab3.py skeleton is pre-configured with code to generate a simulated maze matching our labs, with the end goal as a target.

If you would like to see more detail about this map, run

```python3 src/sphero_env/envs/custom_maze_full.py```

for a visualisation and resolution and scale settings.

If you want to test your planner on other maps, you can generate new mazes with the python code below.

```
from sphero_env.envs.maze_generator import MazeGenerator
mg = MazeGenerator(width,height,seed)
map,start,goal = mg.generate()
```

## Submission requirements

- Submit a CSV result output for assessment. It should comprise all steps from start to end of a run, or to within a specified goal tolerance.
- During assessment, you will be evaluated on how well your robot navigates in a maze.

## Automark assessment

The automarker reads a CSV log from `studentid_lab3.csv`.

### Required CSV columns

| Column | Description |
| --- | --- |
| `sim_x` | Simulator x position |
| `sim_y` | Simulator y position |
| `real_x` | Estimated x position |
| `real_y` | Estimated y position |

### Evaluated metrics and pass thresholds

| Metric | Default threshold |
| --- | --- |
| Final distance to goal (real) | ≤ 0.10 m | 
| Distance to optimal path (real) | ≤ 0.20 m | 
| Final distance to goal (sim) | ≤ 0.10 m |
| Distance to optimal path (sim) | ≤ 0.20 m | 

## Setting yourself up for Lab 4

- Lab 4 requires you to reproduce this lab using a learning based approach
- You could use Lab 3 to generate training data, or go back to lab 2 and use teleoperation.

---
