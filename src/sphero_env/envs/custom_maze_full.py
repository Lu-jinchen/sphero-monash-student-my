"""Official 5 x 5 plate maze used for Labs 3 and 4.

The staff map represents each 25 cm road plate by a free cell at an odd
occupancy-grid index.  The cell between two plate centres is opened when there
is no wall between those plates.  This produces the 11 x 11 grid consumed by
``SpheroEnv`` while retaining the plate graph shown in the lab diagram.

Coordinates follow the published convention: logical ``x`` increases right,
logical ``y`` increases down, while world ``+y`` points up.  The world origin
is at the centre of the maze.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


Cell = Tuple[int, int]
Edge = Tuple[Cell, Cell]

PLATE_SIZE = 0.25
GRID_RESOLUTION = PLATE_SIZE / 2.0
MAZE_WIDTH = 5
MAZE_HEIGHT = 5

ACTIVE_CELLS = {
    (x, y)
    for y in range(MAZE_HEIGHT)
    for x in range(MAZE_WIDTH)
}

START_CELL = (0, 4)
GOAL_CELL = (4, 0)

# Walls between adjacent plates, transcribed from the staff maze update.  A
# set removes the duplicate (3, 1) <-> (4, 1) entry in the published file.
BLOCKED_EDGES = {
    ((0, 0), (0, 1)),
    ((3, 0), (3, 1)),
    ((0, 1), (1, 1)),
    ((1, 1), (2, 1)),
    ((3, 1), (4, 1)),
    ((0, 1), (0, 2)),
    ((2, 1), (2, 2)),
    ((3, 2), (4, 2)),
    ((1, 2), (1, 3)),
    ((3, 2), (3, 3)),
    ((0, 3), (1, 3)),
    ((1, 3), (2, 3)),
    ((3, 3), (4, 3)),
    ((4, 3), (4, 4)),
    ((2, 4), (3, 4)),
}


def maze_width() -> int:
    return MAZE_WIDTH


def maze_height() -> int:
    return MAZE_HEIGHT


def canonical_edge(cell_a: Cell, cell_b: Cell) -> Edge:
    """Return an undirected edge in a deterministic order."""

    a = tuple(int(v) for v in cell_a)
    b = tuple(int(v) for v in cell_b)
    return tuple(sorted((a, b)))


def blocked_edge_set() -> set[Edge]:
    return {canonical_edge(a, b) for a, b in BLOCKED_EDGES}


def are_adjacent(cell_a: Cell, cell_b: Cell) -> bool:
    return abs(cell_a[0] - cell_b[0]) + abs(cell_a[1] - cell_b[1]) == 1


def validate_custom_maze() -> None:
    """Reject malformed cells or walls before constructing the grid."""

    for cell in ACTIVE_CELLS:
        x, y = cell
        if not (0 <= x < MAZE_WIDTH and 0 <= y < MAZE_HEIGHT):
            raise ValueError(f"active cell {cell} is outside the maze")
    if START_CELL not in ACTIVE_CELLS or GOAL_CELL not in ACTIVE_CELLS:
        raise ValueError("start and goal must be active road plates")
    for cell_a, cell_b in BLOCKED_EDGES:
        if cell_a not in ACTIVE_CELLS or cell_b not in ACTIVE_CELLS:
            raise ValueError(f"wall {cell_a} <-> {cell_b} uses an inactive plate")
        if not are_adjacent(cell_a, cell_b):
            raise ValueError(f"wall {cell_a} <-> {cell_b} is not between neighbours")


def logical_to_occ(cell: Cell) -> Tuple[int, int]:
    """Convert logical ``(x, y)`` plate coordinates to ``(row, col)``."""

    x, y = cell
    return 2 * y + 1, 2 * x + 1


def is_transition_blocked(cell_a: Cell, cell_b: Cell) -> bool:
    if cell_a not in ACTIVE_CELLS or cell_b not in ACTIVE_CELLS:
        return True
    if not are_adjacent(cell_a, cell_b):
        return True
    return canonical_edge(cell_a, cell_b) in blocked_edge_set()


def build_occupancy_grid() -> np.ndarray:
    """Return the staff 11 x 11 grid, where 1 is a wall and 0 is free."""

    validate_custom_maze()
    occupancy = np.ones((2 * MAZE_HEIGHT + 1, 2 * MAZE_WIDTH + 1), dtype=np.uint8)

    for cell in ACTIVE_CELLS:
        occupancy[logical_to_occ(cell)] = 0

    # Right and down are enough because every undirected pair is visited once.
    for x, y in ACTIVE_CELLS:
        for dx, dy in ((1, 0), (0, 1)):
            neighbour = (x + dx, y + dy)
            if neighbour not in ACTIVE_CELLS or is_transition_blocked((x, y), neighbour):
                continue
            row0, col0 = logical_to_occ((x, y))
            row1, col1 = logical_to_occ(neighbour)
            occupancy[(row0 + row1) // 2, (col0 + col1) // 2] = 0

    return occupancy


def cell_to_world(cell: Cell) -> np.ndarray:
    """Convert a logical plate centre to world metres."""

    x, y = cell
    world_x = (x - (MAZE_WIDTH - 1) / 2.0) * PLATE_SIZE
    world_y = ((MAZE_HEIGHT - 1) / 2.0 - y) * PLATE_SIZE
    return np.array([world_x, world_y], dtype=np.float32)


def path_to_world(path: Iterable[Cell]) -> list[np.ndarray]:
    return [cell_to_world(cell) for cell in path]


def start_world() -> np.ndarray:
    return cell_to_world(START_CELL)


def goal_world() -> np.ndarray:
    return cell_to_world(GOAL_CELL)


def plot_custom_maze(
    ax=None,
    *,
    optimal_path: Sequence[Sequence[float]] | None = None,
    sim_path: Sequence[Sequence[float]] | None = None,
    real_path: Sequence[Sequence[float]] | None = None,
    start: Sequence[float] | None = None,
):
    """Plot the physical plate maze and optional world-frame trajectories."""

    import matplotlib.pyplot as plt

    if ax is None:
        _figure, ax = plt.subplots(figsize=(7, 7))

    half = PLATE_SIZE / 2.0
    for x, y in sorted(ACTIVE_CELLS, key=lambda c: (c[1], c[0])):
        wx, wy = cell_to_world((x, y))
        ax.add_patch(plt.Rectangle(
            (wx - half, wy - half), PLATE_SIZE, PLATE_SIZE,
            facecolor="#e7eaee", edgecolor="#bdc5ce", linewidth=0.8,
        ))

    wall_kwargs = dict(color="#171a1f", linewidth=4.0, solid_capstyle="butt")
    left = -(MAZE_WIDTH * PLATE_SIZE) / 2.0
    right = -left
    bottom = -(MAZE_HEIGHT * PLATE_SIZE) / 2.0
    top = -bottom
    ax.plot([left, right], [top, top], **wall_kwargs)
    ax.plot([left, right], [bottom, bottom], **wall_kwargs)
    ax.plot([left, left], [bottom, top], **wall_kwargs)
    ax.plot([right, right], [bottom, top], **wall_kwargs)

    for cell_a, cell_b in blocked_edge_set():
        a = cell_to_world(cell_a).astype(float)
        b = cell_to_world(cell_b).astype(float)
        midpoint = 0.5 * (a + b)
        if cell_a[0] != cell_b[0]:
            ax.plot([midpoint[0], midpoint[0]],
                    [midpoint[1] - half, midpoint[1] + half], **wall_kwargs)
        else:
            ax.plot([midpoint[0] - half, midpoint[0] + half],
                    [midpoint[1], midpoint[1]], **wall_kwargs)

    def draw(path, color, label, width, style="-"):
        if path is None:
            return
        points = np.asarray(path, dtype=float).reshape(-1, 2)
        if len(points):
            ax.plot(points[:, 0], points[:, 1], style, color=color,
                    linewidth=width, label=label, zorder=5)

    draw(optimal_path, "#2f6fed", "Optimal A*", 3.0, "--")
    draw(sim_path, "#1f9d68", "Simulation", 2.2)
    draw(real_path, "#e24a4a", "Real / estimate", 2.2)

    start = np.asarray(start_world() if start is None else start, dtype=float)
    goal = goal_world()
    ax.scatter(*start, s=120, color="#1f9d68", edgecolor="white", linewidth=1.5,
               label="Start", zorder=7)
    ax.scatter(*goal, s=180, marker="*", color="#e63946", edgecolor="#171a1f",
               linewidth=1.0, label="Goal", zorder=7)
    ax.set_aspect("equal")
    ax.set_xlim(left - 0.06, right + 0.06)
    ax.set_ylim(bottom - 0.06, top + 0.06)
    ax.set_xlabel("Maze x (m)")
    ax.set_ylabel("Maze y (m)")
    ax.set_title("Lab 3: official 5 x 5 plate maze")
    ax.grid(False)
    return ax


__all__ = [
    "ACTIVE_CELLS", "BLOCKED_EDGES", "GOAL_CELL", "GRID_RESOLUTION",
    "MAZE_HEIGHT", "MAZE_WIDTH", "PLATE_SIZE", "START_CELL",
    "are_adjacent", "blocked_edge_set", "build_occupancy_grid",
    "canonical_edge", "cell_to_world", "goal_world", "is_transition_blocked",
    "logical_to_occ", "maze_height", "maze_width", "path_to_world",
    "plot_custom_maze", "start_world", "validate_custom_maze",
]
