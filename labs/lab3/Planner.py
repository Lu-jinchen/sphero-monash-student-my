import numpy as np

from sphero_env.envs.custom_maze_full import build_occupancy_grid

map = build_occupancy_grid()

### Implement a planner and controller for the Sphero robot to navigate to a goal position in the environment.
class Planner:
    def __init__(self, map, dt=0.1):
        """
        The map (input) is the occupancy grid produced by the maze generator:
            map : a 2D numpy array of shape (h, w) with values in {0, 1}
                  1 = wall / obstacle cell
                  0 = free cell
        World origin (0, 0) is at the centre of the grid, so cell (i, j) maps
        to world coordinates via the environment's grid resolution.
        """
        self.map = map
        self.dt = dt

    def plan(self, state, goal):
        """
        Fill in this function to implement a simple planner that computes the action based on the current state,
        the map and the goal position. The planner should return a sequence of 2D waypoints to reach the goal.

        The inputs are:
            state : the current state, same format as the previous labs:
                    [x, y, heading, speed]
            goal  : the goal position, a position-only array: [x, y]
        Returns:
            waypoints : a list of 2D waypoints to reach the goal, where each
                        waypoint is a position-only array: [x, y]
        """

        waypoints = [np.array([0.5, 0.5])]  # Replace this with your planner's output

        return waypoints  # Replace this with your planner's output

