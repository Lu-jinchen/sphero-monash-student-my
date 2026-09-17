"""Self-contained A* waypoint planner for the Lab 3 occupancy grid.

The implementation follows the Week 5 graph-search formulation directly:
free grid cells are nodes, legal four-neighbour moves are edges with unit cost,
and A* expands the node with the smallest ``f(n) = g(n) + h(n)``.  Manhattan
distance is an admissible and consistent heuristic for these moves, so the
returned path has the same cost as Dijkstra/BFS while normally expanding fewer
nodes.

Only Python's standard library and NumPy are imported because this file is the
self-contained M3-2 submission surface.
"""

from __future__ import annotations

import heapq
import math

import numpy as np


class Planner:
    """Plan an optimal path through a binary occupancy grid.

    Parameters match the supplied skeleton.  ``map`` uses 1 for an obstacle
    and 0 for free space.  The staff Lab 3 map has 0.125 m occupancy cells and
    the world origin at its centre.
    """

    GRID_RESOLUTION = 0.125
    MOVES = ((-1, 0), (0, -1), (0, 1), (1, 0))

    def __init__(self, map, dt=0.1):
        occupancy = np.asarray(map)
        if occupancy.ndim != 2 or occupancy.size == 0:
            raise ValueError("map must be a non-empty 2D occupancy grid")
        if not np.all(np.isin(occupancy, (0, 1))):
            raise ValueError("map values must be 0 (free) or 1 (obstacle)")

        self.map = occupancy.astype(np.uint8, copy=True)
        self.dt = float(dt)
        self.grid_resolution = self.GRID_RESOLUTION

        # Evidence retained for the dashboard, checker and viva.
        self.last_cell_path = []
        self.last_full_path = []
        self.last_expanded = 0
        self.last_cost = math.inf

    def _in_bounds(self, cell):
        row, col = cell
        height, width = self.map.shape
        return 0 <= row < height and 0 <= col < width

    def _is_free(self, cell):
        return self._in_bounds(cell) and self.map[cell] == 0

    @staticmethod
    def _heuristic(cell, goal_cell):
        """Manhattan distance: exact in an obstacle-free four-neighbour grid."""

        return abs(cell[0] - goal_cell[0]) + abs(cell[1] - goal_cell[1])

    def world_to_cell(self, point):
        x, y = float(point[0]), float(point[1])
        height, width = self.map.shape
        col = int(np.floor(x / self.grid_resolution + width / 2.0))
        row = int(np.floor(-y / self.grid_resolution + height / 2.0))
        return row, col

    def cell_to_world(self, cell):
        row, col = cell
        height, width = self.map.shape
        x = (col + 0.5 - width / 2.0) * self.grid_resolution
        y = -(row + 0.5 - height / 2.0) * self.grid_resolution
        return np.array([x, y], dtype=float)

    def _nearest_free(self, cell):
        """Snap a noisy/off-grid pose to the closest free planning node."""

        if self._is_free(cell):
            return cell
        free = np.argwhere(self.map == 0)
        if len(free) == 0:
            return None
        target = np.asarray(cell, dtype=float)
        distances = np.sum((free - target) ** 2, axis=1)
        row, col = free[int(np.argmin(distances))]
        return int(row), int(col)

    def _astar(self, start_cell, goal_cell):
        """Return the optimal cell path, or an empty list if disconnected."""

        if not self._is_free(start_cell) or not self._is_free(goal_cell):
            return []

        frontier = [(self._heuristic(start_cell, goal_cell), 0, start_cell)]
        came_from = {}
        g_score = {start_cell: 0}
        expanded = 0

        while frontier:
            _f_score, cost_so_far, current = heapq.heappop(frontier)
            if cost_so_far != g_score.get(current):
                continue

            expanded += 1
            if current == goal_cell:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                self.last_expanded = expanded
                self.last_cost = float(cost_so_far)
                return path[::-1]

            for d_row, d_col in self.MOVES:
                neighbour = (current[0] + d_row, current[1] + d_col)
                if not self._is_free(neighbour):
                    continue
                tentative = cost_so_far + 1
                if tentative >= g_score.get(neighbour, math.inf):
                    continue
                came_from[neighbour] = current
                g_score[neighbour] = tentative
                estimate = tentative + self._heuristic(neighbour, goal_cell)
                heapq.heappush(frontier, (estimate, tentative, neighbour))

        self.last_expanded = expanded
        self.last_cost = math.inf
        return []

    @staticmethod
    def simplify_waypoints(waypoints):
        """Keep the start, goal and direction changes of a grid path."""

        points = [np.asarray(point, dtype=float).reshape(-1)[:2] for point in waypoints]
        if len(points) < 3:
            return [point.copy() for point in points]

        simplified = [points[0].copy()]
        for previous, current, following in zip(points, points[1:], points[2:]):
            before = current - previous
            after = following - current
            if abs(before[0] * after[1] - before[1] * after[0]) > 1e-12:
                simplified.append(current.copy())
        simplified.append(points[-1].copy())
        return simplified

    def plan(self, state, goal):
        """Return an optimal sequence of ``[x, y]`` world-frame waypoints."""

        state = np.asarray(state, dtype=float).reshape(-1)
        goal = np.asarray(goal, dtype=float).reshape(-1)
        if state.size < 2 or goal.size < 2:
            raise ValueError("state and goal must each provide x and y")
        if not np.all(np.isfinite(state[:2])) or not np.all(np.isfinite(goal[:2])):
            raise ValueError("state and goal positions must be finite")

        start_cell = self._nearest_free(self.world_to_cell(state[:2]))
        goal_cell = self._nearest_free(self.world_to_cell(goal[:2]))
        if start_cell is None or goal_cell is None:
            self.last_cell_path = []
            self.last_full_path = []
            self.last_cost = math.inf
            return []

        cells = self._astar(start_cell, goal_cell)
        self.last_cell_path = list(cells)
        self.last_full_path = [self.cell_to_world(cell) for cell in cells]
        return [point.copy() for point in self.last_full_path]
