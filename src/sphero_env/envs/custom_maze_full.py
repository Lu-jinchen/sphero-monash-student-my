"""
custom_maze.py

Fixed physical maze used for Lab 3.

IMPORTANT IDEA
--------------

Each logical A* cell represents ONE physical road plate.

    one logical cell = one 25 cm x 25 cm road plate

All listed plates are traversable.

The physical barriers/walls are represented separately as
BLOCKED_EDGES between adjacent plates.

For example:

    ((0, 0), (1, 0))

means there is a wall between:

        cell (0,0)

and

        cell (1,0)

so A* cannot move directly between them.


Coordinate convention
---------------------

The coordinate system follows the maze drawing:

                NORTH / TOP

           x=0  x=1  x=2  x=3  x=4

    y=0     O    O    O
    y=1     O    O    O    O
    y=2     O    O    O    O    O

                SOUTH / BOTTOM


x increases to the RIGHT.
y increases DOWNWARD in logical maze coordinates.

In the Sphero world frame:

+x = right
+y = up
"""

from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# Physical dimensions
# ============================================================

# Each real road plate is 25 cm x 25 cm.
PLATE_SIZE = 0.25


# ============================================================
# A* / occupancy-grid representation
# ============================================================
#
# AStarPlanner represents logical cells at odd occupancy-grid
# indices:
#
#       logical cell      passage      logical cell
#            |               |              |
#            v               v              v
#
#            0       0       0
#
# Therefore there are TWO occupancy cells between adjacent
# logical-cell centres.
#
# To make neighbouring logical cells physically 0.25 m apart:
#
#       2 * GRID_RESOLUTION = 0.25
#
# therefore:
#
#       GRID_RESOLUTION = 0.125 m
#
# ============================================================

GRID_RESOLUTION = PLATE_SIZE / 2.0


# ============================================================
# Maze dimensions
# ============================================================
#
# Looking at the custom maze as drawn:
#
# row 0: 3 plates
# row 1: 4 plates
# row 2: 5 plates
#
# The bounding logical grid is therefore:
#
#       width  = 5
#       height = 3
#
# ============================================================

MAZE_WIDTH = 5

MAZE_HEIGHT = 5


# ============================================================
# Physical road plates
# ============================================================
#
# Every entry here corresponds to a real 25 cm road plate.
#
# Missing coordinates simply mean there is no plate there.
#
# Maze footprint:
#
#               x
#           0  1  2  3  4
#
#       0   O  O  O
#   y   1   O  O  O  O
#       2   O  O  O  O  O
#
# ============================================================

ACTIVE_CELLS = {
    (x, y)
    for y in range(MAZE_HEIGHT)
    for x in range(MAZE_WIDTH)
}


# ============================================================
# Start and goal
# ============================================================
#
# Change these to whichever physical plates you want.
#
# For now:
#
# START = top-left plate
# GOAL  = bottom-right plate
#
# ============================================================

START_CELL = (0, 4)
GOAL_CELL = (4, 0)


# ============================================================
# Physical walls
# ============================================================
#
# THIS IS THE IMPORTANT PART OF THE CUSTOM MAZE.
#
# Add one entry for every physical wall that prevents movement
# between two neighbouring 25 cm plates.
#
#
# Example:
#
#     ((0, 0), (1, 0))
#
# means:
#
#        (0,0)  |  (1,0)
#               ^
#               wall
#
#
# Another example:
#
#     ((2, 1), (2, 2))
#
# means:
#
#        (2,1)
#          ---
#          wall
#          ---
#        (2,2)
#
#
# Order does NOT matter:
#
#     ((0,0), (1,0))
#
# and
#
#     ((1,0), (0,0))
#
# represent the same wall.
#
# ============================================================


BLOCKED_EDGES = {

    
    ((0, 0), (0, 1)),
    ((3, 0), (3, 1)),
    ((0, 1), (1, 1)),
    ((1, 1), (2, 1)),
    ((3, 1), (4, 1)),
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


# ============================================================
# Helper functions
# ============================================================

def maze_width() -> int:
    """Return logical maze width."""

    return MAZE_WIDTH


def maze_height() -> int:
    """Return logical maze height."""

    return MAZE_HEIGHT


# ============================================================
# Canonical edge representation
# ============================================================

def canonical_edge(
    cell_a: Tuple[int, int],
    cell_b: Tuple[int, int],
):
    """
    Return an edge in a consistent order.

    This means:

        ((0,0), (1,0))

    and

        ((1,0), (0,0))

    are treated as exactly the same wall.
    """

    return tuple(
        sorted(
            (
                tuple(cell_a),
                tuple(cell_b),
            )
        )
    )


def blocked_edge_set():
    """
    Return all manually specified walls in canonical form.
    """

    return {
        canonical_edge(a, b)
        for a, b in BLOCKED_EDGES
    }


# ============================================================
# Adjacency
# ============================================================

def are_adjacent(
    cell_a: Tuple[int, int],
    cell_b: Tuple[int, int],
) -> bool:
    """
    Return True when two logical maze cells share one edge.
    """

    ax, ay = cell_a
    bx, by = cell_b

    return (
        abs(ax - bx)
        + abs(ay - by)
        == 1
    )


# ============================================================
# Maze validation
# ============================================================

def validate_custom_maze() -> None:
    """
    Check that the custom physical maze definition is valid.
    """

    # --------------------------------------------------------
    # Check active cells
    # --------------------------------------------------------

    for cell in ACTIVE_CELLS:

        x, y = cell

        if not (
            0 <= x < MAZE_WIDTH
            and
            0 <= y < MAZE_HEIGHT
        ):

            raise ValueError(
                f"Active cell {cell} is outside "
                f"the {MAZE_WIDTH}x{MAZE_HEIGHT} maze."
            )

    # --------------------------------------------------------
    # Start cell
    # --------------------------------------------------------

    if START_CELL not in ACTIVE_CELLS:

        raise ValueError(
            f"START_CELL {START_CELL} is not "
            f"a physical road plate."
        )

    # --------------------------------------------------------
    # Goal cell
    # --------------------------------------------------------

    if GOAL_CELL not in ACTIVE_CELLS:

        raise ValueError(
            f"GOAL_CELL {GOAL_CELL} is not "
            f"a physical road plate."
        )

    # --------------------------------------------------------
    # Validate physical walls
    # --------------------------------------------------------

    for cell_a, cell_b in BLOCKED_EDGES:

        cell_a = tuple(cell_a)
        cell_b = tuple(cell_b)

        if cell_a not in ACTIVE_CELLS:

            raise ValueError(
                f"Wall uses cell {cell_a}, "
                f"but that cell is not a road plate."
            )

        if cell_b not in ACTIVE_CELLS:

            raise ValueError(
                f"Wall uses cell {cell_b}, "
                f"but that cell is not a road plate."
            )

        if not are_adjacent(
            cell_a,
            cell_b,
        ):

            raise ValueError(
                f"Wall {cell_a} <-> {cell_b} is invalid. "
                f"Walls must separate directly adjacent plates."
            )


# ============================================================
# Logical cell -> occupancy-grid cell
# ============================================================

def logical_to_occ(
    cell: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Convert logical plate coordinate (x,y) into the
    occupancy-grid centre corresponding to that plate.

    Logical cell:

        (x, y)

    becomes:

        row = 2*y + 1
        col = 2*x + 1
    """

    x, y = cell

    row = 2 * y + 1
    col = 2 * x + 1

    return row, col


# ============================================================
# Check whether movement is blocked
# ============================================================

def is_transition_blocked(
    cell_a: Tuple[int, int],
    cell_b: Tuple[int, int],
) -> bool:
    """
    Return True if movement between two adjacent road plates
    should not be allowed.
    """

    # Missing physical plate
    if (
        cell_a not in ACTIVE_CELLS
        or
        cell_b not in ACTIVE_CELLS
    ):
        return True

    # Not neighbours
    if not are_adjacent(
        cell_a,
        cell_b,
    ):
        return True

    # Physical wall
    edge = canonical_edge(
        cell_a,
        cell_b,
    )

    return (
        edge
        in blocked_edge_set()
    )


# ============================================================
# Build occupancy grid
# ============================================================

def build_occupancy_grid() -> np.ndarray:
    """
    Build the occupancy-grid representation required by
    AStarPlanner and SpheroEnv.

    Convention:

        0 = free
        1 = wall / obstacle

    Physical road plates become free logical cells.

    Adjacent road plates are connected unless their shared
    boundary appears in BLOCKED_EDGES.
    """

    validate_custom_maze()

    occ_width = (
        2 * MAZE_WIDTH + 1
    )

    occ_height = (
        2 * MAZE_HEIGHT + 1
    )

    # Start with everything blocked.
    occupancy_grid = np.ones(
        (
            occ_height,
            occ_width,
        ),
        dtype=np.uint8,
    )

    # ========================================================
    # 1. Open every physical road plate
    # ========================================================

    for cell in ACTIVE_CELLS:

        row, col = logical_to_occ(
            cell
        )

        occupancy_grid[
            row,
            col,
        ] = 0

    # ========================================================
    # 2. Connect neighbouring plates unless a physical
    #    barrier blocks them.
    # ========================================================

    directions = [
        (1, 0),     # right
        (0, 1),     # down
    ]

    # We only need RIGHT and DOWN because each pair only
    # needs to be processed once.

    for cell in ACTIVE_CELLS:

        x, y = cell

        for dx, dy in directions:

            neighbour = (
                x + dx,
                y + dy,
            )

            if neighbour not in ACTIVE_CELLS:
                continue

            if is_transition_blocked(
                cell,
                neighbour,
            ):
                continue

            # -----------------------------------------------
            # Logical occupancy centres
            # -----------------------------------------------

            row0, col0 = (
                logical_to_occ(
                    cell
                )
            )

            row1, col1 = (
                logical_to_occ(
                    neighbour
                )
            )

            # -----------------------------------------------
            # The occupancy cell halfway between the two
            # logical cells represents the shared boundary.
            # -----------------------------------------------

            wall_row = (
                row0 + row1
            ) // 2

            wall_col = (
                col0 + col1
            ) // 2

            # No physical wall -> free passage.
            occupancy_grid[
                wall_row,
                wall_col,
            ] = 0

    return occupancy_grid


# ============================================================
# Logical cell -> metric world position
# ============================================================

def cell_to_world(
    cell: Tuple[int, int],
) -> np.ndarray:
    """
    Convert logical physical-plate coordinate into world
    coordinates in metres.

    Each neighbouring cell centre is exactly 0.25 m apart.

    World convention:

        +x = right
        +y = up
    """

    x, y = cell

    # Centre the entire maze around world origin.
    world_x = (
        x
        - (MAZE_WIDTH - 1) / 2.0
    ) * PLATE_SIZE

    # Logical y increases downward.
    # World y increases upward.
    world_y = (
        (MAZE_HEIGHT - 1) / 2.0
        - y
    ) * PLATE_SIZE

    return np.array(
        [
            world_x,
            world_y,
        ],
        dtype=np.float32,
    )


# ============================================================
# Convert complete A* path -> metric waypoints
# ============================================================

def path_to_world(
    path: List[Tuple[int, int]],
) -> List[np.ndarray]:
    """
    Convert A* logical path into physical waypoint positions.
    """

    return [
        cell_to_world(cell)
        for cell in path
    ]


# ============================================================
# Print physical maze
# ============================================================

def print_tile_map() -> None:
    """
    Print the physical road-plate footprint.
    """

    print()
    print(
        "Physical road-plate map"
    )

    print(
        "-----------------------"
    )

    for y in range(
        MAZE_HEIGHT
    ):

        row_text = []

        for x in range(
            MAZE_WIDTH
        ):

            cell = (x, y)

            if cell == START_CELL:

                symbol = " S "

            elif cell == GOAL_CELL:

                symbol = " G "

            elif cell in ACTIVE_CELLS:

                symbol = " O "

            else:

                symbol = "   "

            row_text.append(
                symbol
            )

        print(
            f"y={y}: "
            + "|".join(
                row_text
            )
        )

    print()


# ============================================================
# Print all open/blocked transitions
# ============================================================

def print_transitions() -> None:

    print(
        "Physical walls / transitions"
    )

    print(
        "----------------------------"
    )

    for y in range(
        MAZE_HEIGHT
    ):

        for x in range(
            MAZE_WIDTH
        ):

            cell = (
                x,
                y,
            )

            if cell not in ACTIVE_CELLS:
                continue

            for dx, dy in [
                (1, 0),
                (0, 1),
            ]:

                neighbour = (
                    x + dx,
                    y + dy,
                )

                if neighbour not in ACTIVE_CELLS:
                    continue

                if is_transition_blocked(
                    cell,
                    neighbour,
                ):

                    state = "BLOCKED"

                else:

                    state = "OPEN"

                print(
                    f"{cell} <-> "
                    f"{neighbour}: "
                    f"{state}"
                )

    print()



# ============================================================
# Visualise custom physical maze
# ============================================================

def plot_custom_maze(
    path=None,
    show=True,
):
    """
    Draw the physical road-plate maze.

    Parameters
    ----------
    path:
        Optional A* logical path such as:

            [(0,0), (0,1), (1,1), ...]

        If supplied, the path is drawn over the maze.

    show:
        If True, call plt.show().
    """

    validate_custom_maze()

    blocked = blocked_edge_set()

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    # ========================================================
    # 1. Draw every physical 25 cm road plate
    # ========================================================

    for y in range(MAZE_HEIGHT):
        for x in range(MAZE_WIDTH):

            cell = (x, y)

            if cell not in ACTIVE_CELLS:
                continue

            # Each logical cell is drawn as a 1x1 square.
            rect = Rectangle(
                (x, y),
                1.0,
                1.0,
                facecolor="lightgray",
                edgecolor="gray",
                linewidth=1.0,
                zorder=1,
            )

            ax.add_patch(rect)

            # ------------------------------------------------
            # Cell coordinate label
            # ------------------------------------------------

            ax.text(
                x + 0.5,
                y + 0.5,
                f"({x},{y})",
                ha="center",
                va="center",
                fontsize=9,
                zorder=5,
            )

    # ========================================================
    # 2. Draw walls
    # ========================================================

    for y in range(MAZE_HEIGHT):
        for x in range(MAZE_WIDTH):

            cell = (x, y)

            if cell not in ACTIVE_CELLS:
                continue

            # ------------------------------------------------
            # NORTH boundary
            # ------------------------------------------------

            north = (x, y - 1)

            if (
                north not in ACTIVE_CELLS
                or canonical_edge(cell, north) in blocked
            ):

                ax.plot(
                    [x, x + 1],
                    [y, y],
                    linewidth=4,
                    color="black",
                    zorder=4,
                )

            # ------------------------------------------------
            # SOUTH boundary
            # ------------------------------------------------

            south = (x, y + 1)

            if (
                south not in ACTIVE_CELLS
                or canonical_edge(cell, south) in blocked
            ):

                ax.plot(
                    [x, x + 1],
                    [y + 1, y + 1],
                    linewidth=4,
                    color="black",
                    zorder=4,
                )

            # ------------------------------------------------
            # WEST boundary
            # ------------------------------------------------

            west = (x - 1, y)

            if (
                west not in ACTIVE_CELLS
                or canonical_edge(cell, west) in blocked
            ):

                ax.plot(
                    [x, x],
                    [y, y + 1],
                    linewidth=4,
                    color="black",
                    zorder=4,
                )

            # ------------------------------------------------
            # EAST boundary
            # ------------------------------------------------

            east = (x + 1, y)

            if (
                east not in ACTIVE_CELLS
                or canonical_edge(cell, east) in blocked
            ):

                ax.plot(
                    [x + 1, x + 1],
                    [y, y + 1],
                    linewidth=4,
                    color="black",
                    zorder=4,
                )

    # ========================================================
    # 3. Draw START
    # ========================================================

    sx, sy = START_CELL

    ax.scatter(
        sx + 0.5,
        sy + 0.5,
        s=350,
        marker="o",
        color="green",
        edgecolor="black",
        linewidth=2,
        zorder=8,
        label="Start",
    )

    ax.text(
        sx + 0.5,
        sy + 0.5,
        "S",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="white",
        zorder=9,
    )

    # ========================================================
    # 4. Draw GOAL
    # ========================================================

    gx, gy = GOAL_CELL

    ax.scatter(
        gx + 0.5,
        gy + 0.5,
        s=350,
        marker="*",
        color="red",
        edgecolor="black",
        linewidth=1.5,
        zorder=8,
        label="Goal",
    )

    # ========================================================
    # 5. Draw A* path if supplied
    # ========================================================

    if path is not None and len(path) > 0:

        path_x = [
            cell[0] + 0.5
            for cell in path
        ]

        path_y = [
            cell[1] + 0.5
            for cell in path
        ]

        ax.plot(
            path_x,
            path_y,
            marker="o",
            markersize=7,
            linewidth=3,
            color="blue",
            zorder=7,
            label="A* path",
        )

    # ========================================================
    # 6. Plot formatting
    # ========================================================

    ax.set_xlim(
        -0.25,
        MAZE_WIDTH + 0.25,
    )

    ax.set_ylim(
        MAZE_HEIGHT + 0.25,
        -0.25,
    )

    # y=0 should appear at the top,
    # matching your physical maze drawing.
    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xticks(
        np.arange(MAZE_WIDTH)
        + 0.5
    )

    ax.set_xticklabels(
        np.arange(MAZE_WIDTH)
    )

    ax.set_yticks(
        np.arange(MAZE_HEIGHT)
        + 0.5
    )

    ax.set_yticklabels(
        np.arange(MAZE_HEIGHT)
    )

    ax.set_xlabel(
        "Maze x cell"
    )

    ax.set_ylabel(
        "Maze y cell"
    )

    ax.set_title(
        "Custom Sphero Maze"
    )

    ax.legend(
        loc="best"
    )

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


# ============================================================
# Debug/test
# ============================================================

if __name__ == "__main__":

    validate_custom_maze()

    print_tile_map()

    print_transitions()

    occupancy_grid = (
        build_occupancy_grid()
    )

    print(
        "Occupancy grid"
    )

    print(
        "--------------"
    )

    print(
        occupancy_grid.astype(
            int
        )
    )

    print()

    print(
        f"Plate size = "
        f"{PLATE_SIZE:.3f} m"
    )

    print(
        f"A* occupancy resolution = "
        f"{GRID_RESOLUTION:.3f} m"
    )

    print(
        f"START_CELL = "
        f"{START_CELL}"
    )

    print(
        f"START world = "
        f"{cell_to_world(START_CELL)}"
    )

    print(
        f"GOAL_CELL = "
        f"{GOAL_CELL}"
    )

    print(
        f"GOAL world = "
        f"{cell_to_world(GOAL_CELL)}"
    )

    # --------------------------------------------------------
    # SHOW MAZE
    # --------------------------------------------------------

    plot_custom_maze()