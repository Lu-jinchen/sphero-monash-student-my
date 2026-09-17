"""Integration verification; takes the repository to test as its argument."""
import os
import sys
from pathlib import Path
from collections import deque

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['MPLBACKEND'] = 'Agg'
sys.path[:0] = [str(root / 'src'), str(root / 'labs/lab3')]
import numpy as np
import lab3
import check_lab3
from Planner import Planner

assert Path(lab3.__file__).resolve().is_relative_to(root)
import sphero_env.envs.sphero_env as environment
assert Path(environment.__file__).resolve().is_relative_to(root)
planner = Planner(lab3.MAP)
path = planner.plan([*lab3.START, 0, 0], lab3.GOAL)
start, goal = planner.world_to_cell(lab3.START), planner.world_to_cell(lab3.GOAL)
queue = deque([(start, 0)])
seen = {start}
while queue:
    cell, cost = queue.popleft()
    if cell == goal:
        break
    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
        nxt = cell[0]+dr, cell[1]+dc
        if 0 <= nxt[0] < lab3.MAP.shape[0] and 0 <= nxt[1] < lab3.MAP.shape[1]:
            if lab3.MAP[nxt] == 0 and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, cost+1))
assert cell == goal and len(path)-1 == cost
assert all(lab3.MAP[planner.world_to_cell(p)] == 0 for p in path)
assert all(np.isclose(np.linalg.norm(a-b), .125) for a,b in zip(path,path[1:]))
blocked = Planner(np.array([[0,1,0],[0,1,0],[0,1,0]]))
assert blocked.plan(blocked.cell_to_world((0,0)), blocked.cell_to_world((0,2))) == []
same = planner.plan(lab3.START, lab3.START)
assert len(same) == 1
print(f'Planner: BFS-optimal {cost} edges; walls, disconnected goal and same-cell tests PASS')

result = lab3.main(['--sim', '--no-render', '--no-archive'])
assert result['mode'] == 'sim'
assert result['report']['pass'], result['report']
assert result['report']['collisions'] == 0, result['report']
assert check_lab3.main([]) == 0
print('Simulation and exported CSV checks PASS')
