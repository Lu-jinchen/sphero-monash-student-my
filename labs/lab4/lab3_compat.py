"""Fill in the Lab 3 names that Lab 4 relies on but this repo's lab3.py lacks.

Lab 4 treats Lab 3 as its expert and its run harness: the A* route, the PID
follower, the run log, the CSV columns. The teammate stack (37482025) grew a
few extra names in lab3.py for the M3 demo rules, and lab4.py reads them:

    BALL_DIAMETER_M       collision clearance and the robot radius in the MPC
    DEADLINE_S            the --deadline default (TA stopwatch, 30 s)
    fully_in_goal_plate   the "whole ball on the plate" stop test

This repo's lab3.py (35301856) does not define them. Rather than edit Lab 3,
which was already submitted and assessed, this module adds them to the
imported lab3 module when they are missing. Anything lab3.py does define is
left alone, so a later Lab 3 that grows these names wins automatically.
"""

from __future__ import annotations

try:
    from sphero_env.envs.custom_maze_full import PLATE_SIZE
except ImportError:  # keep Lab 4 importable without the simulator package
    PLATE_SIZE = 0.25

#: Demo time limit, seconds. The TA starts timing at first movement (#122).
DEADLINE_S = 30.0
#: Sphero BOLT shell diameter.
BALL_DIAMETER_M = 0.073
#: "Fully within the plate" (#122) means the whole ball, so the centre must be
#: within (PLATE_SIZE - BALL) / 2 = 8.85 cm of plate centre on EACH axis.
GOAL_PLATE_HALF_M = (PLATE_SIZE - BALL_DIAMETER_M) / 2.0


def _make_fully_in_goal_plate(lab3_module):
    def fully_in_goal_plate(xy, goal=None):
        """True when the whole ball sits inside the goal plate."""
        goal = lab3_module.GOAL if goal is None else goal
        half = getattr(lab3_module, "GOAL_PLATE_HALF_M", GOAL_PLATE_HALF_M)
        return (abs(float(xy[0]) - float(goal[0])) <= half
                and abs(float(xy[1]) - float(goal[1])) <= half)
    return fully_in_goal_plate


def ensure(lab3_module):
    """Add any missing Lab 4 names to ``lab3_module``; return what was added."""
    defaults = {
        "DEADLINE_S": DEADLINE_S,
        "BALL_DIAMETER_M": BALL_DIAMETER_M,
        "GOAL_PLATE_HALF_M": GOAL_PLATE_HALF_M,
        "fully_in_goal_plate": _make_fully_in_goal_plate(lab3_module),
    }
    added = []
    for name, value in defaults.items():
        if not hasattr(lab3_module, name):
            setattr(lab3_module, name, value)
            added.append(name)
    return added
