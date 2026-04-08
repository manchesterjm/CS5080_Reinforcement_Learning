"""
Parking Lot Environment for HW2 Q-Learning

Configurable grid with parking spots, barrier rows, and entrance.
Default: 7x6 grid with 8 parking spots (matches assignment Figure 1).
Larger grids generated procedurally for scaling experiments.

Default Layout (0-indexed, 7 rows x 6 cols):
    Col: 0    1    2    3    4    5
Row 0: [  ] [  ] [  ] [  ] [  ] [  ]  <- driving lane
Row 1: [  ] [  ] [  ] [  ] [  ] [  ]  <- driving lane
Row 2: [  ] [P1] [P2] [P3] [P4] [  ]  <- top parking
Row 3: [  ] [##] [##] [##] [##] [  ]  <- barrier (cols 1-4)
Row 4: [  ] [P5] [P6] [P7] [P8] [  ]  <- bottom parking
Row 5: [  ] [  ] [  ] [  ] [  ] [  ]  <- driving lane
Row 6: [  ] [  ] [  ] [  ] [  ] [E ]  <- entrance
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = {UP: "UP", DOWN: "DOWN", LEFT: "LEFT", RIGHT: "RIGHT"}
ACTION_ARROWS = {UP: "^", DOWN: "v", LEFT: "<", RIGHT: ">"}
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
NUM_ACTIONS = 4

# --- Default 7x6 layout constants (matches assignment Figure 1) ---
ROWS = 7
COLS = 6
ENTRANCE = (6, 5)
PARKING_SPOTS = {
    0: (2, 1), 1: (2, 2), 2: (2, 3), 3: (2, 4),
    4: (4, 1), 5: (4, 2), 6: (4, 3), 7: (4, 4),
}
NUM_GOALS = len(PARKING_SPOTS)
SPOT_NAMES = {i: f"P{i+1}" for i in range(NUM_GOALS)}
BARRIER_CELLS = {(3, c) for c in range(1, 5)}
ALL_PARKING_POSITIONS = set(PARKING_SPOTS.values())


def generate_layout(rows, cols, spots_per_row=4):
    """Generate a parking lot layout for arbitrary grid dimensions.

    Layout pattern (repeating unit = parking row, barrier, parking row):
        - Row 0-1: driving lanes
        - Repeating blocks of: parking row, barrier row, parking row
        - Final row: driving lane with entrance at bottom-right

    Args:
        rows: Total grid rows (minimum 6).
        cols: Total grid columns (minimum 7).
        spots_per_row: Parking spots per parking row.

    Returns:
        Tuple of (entrance, parking_spots dict, barrier_cells set).
    """
    entrance = (rows - 1, cols - 1)
    parking_spots = {}
    barrier_cells = set()
    spot_id = 0

    # Start parking blocks at row 2
    # Each block: parking_row, barrier_row, parking_row = 3 rows
    # Leave row 0-1 as driving lanes, last row as entrance lane
    available_rows = rows - 3  # rows 2 through rows-2
    row = 2

    while row + 2 <= rows - 1 and available_rows >= 3:
        # Top parking row
        _add_parking_row(row, cols, spots_per_row, parking_spots, spot_id)
        spot_id += spots_per_row

        # Barrier row (leave rightmost column open for passage)
        barrier_row = row + 1
        for c in range(cols - 1):
            barrier_cells.add((barrier_row, c))

        # Bottom parking row
        _add_parking_row(row + 2, cols, spots_per_row, parking_spots, spot_id)
        spot_id += spots_per_row

        row += 4  # skip to next block (parking, barrier, parking, lane)
        available_rows -= 4

    return entrance, parking_spots, barrier_cells


def _add_parking_row(row, cols, spots_per_row, parking_spots, start_id):
    """Add parking spots to a row, centered with margins."""
    # Place spots starting at col 1, spaced evenly
    max_spots = min(spots_per_row, cols - 2)  # leave margins
    for i in range(max_spots):
        col = 1 + i
        if col < cols - 1:
            parking_spots[start_id + i] = (row, col)


class ParkingLot:
    """Goal-conditioned parking lot environment.

    Args:
        rows: Grid height (default 6 for original layout).
        cols: Grid width (default 7 for original layout).
        spots_per_row: Spots per parking row (default 4).
        max_steps: Episode step limit (default 100, scales with grid).
    """

    def __init__(self, rows=None, cols=None, spots_per_row=4,
                 max_steps=None):
        if rows is None and cols is None:
            # Default: original 6x7 hardcoded layout
            self.rows = ROWS
            self.cols = COLS
            self.entrance = ENTRANCE
            self.parking_spots = dict(PARKING_SPOTS)
            self.barrier_cells = set(BARRIER_CELLS)
        else:
            # Configurable layout
            self.rows = rows or 6
            self.cols = cols or 7
            self.entrance, self.parking_spots, self.barrier_cells = \
                generate_layout(self.rows, self.cols, spots_per_row)

        self.num_goals = len(self.parking_spots)
        self.spot_names = {i: f"P{i+1}" for i in range(self.num_goals)}
        self.all_parking_positions = set(self.parking_spots.values())

        # Scale max_steps with grid size if not specified
        if max_steps is not None:
            self.max_steps = max_steps
        else:
            self.max_steps = max(100, self.rows * self.cols)

        # Verify all goals reachable via BFS
        self._verify_reachability()

        # Precompute BFS paths
        self.bfs_paths = {}
        for goal_id in range(self.num_goals):
            self.bfs_paths[goal_id] = self._bfs_shortest_path(goal_id)

        # Current episode state
        self.state = None
        self.goal_id = None
        self.goal_pos = None
        self.steps = 0

    def _get_walkable_cells(self, goal_id):
        """Return set of walkable cells for a given goal.

        All grid cells are walkable EXCEPT:
        - Barrier cells
        - Parking spots that are NOT the current goal
        """
        blocked = set(self.barrier_cells)
        for pid, pos in self.parking_spots.items():
            if pid != goal_id:
                blocked.add(pos)

        walkable = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in blocked:
                    walkable.add((r, c))
        return walkable

    def _bfs_shortest_path(self, goal_id):
        """BFS from entrance to goal. Returns path or None."""
        goal_pos = self.parking_spots[goal_id]
        walkable = self._get_walkable_cells(goal_id)

        if self.entrance not in walkable or goal_pos not in walkable:
            return None

        queue = deque([(self.entrance, [self.entrance])])
        visited = {self.entrance}

        while queue:
            pos, path = queue.popleft()
            if pos == goal_pos:
                return path

            for action in range(NUM_ACTIONS):
                dr, dc = ACTION_DELTAS[action]
                nr, nc = pos[0] + dr, pos[1] + dc
                npos = (nr, nc)
                if (0 <= nr < self.rows and 0 <= nc < self.cols
                        and npos in walkable and npos not in visited):
                    visited.add(npos)
                    queue.append((npos, path + [npos]))
        return None

    def _verify_reachability(self):
        """Verify all parking spots are reachable from entrance."""
        for goal_id in range(self.num_goals):
            path = self._bfs_shortest_path(goal_id)
            if path is None:
                raise RuntimeError(
                    f"{self.spot_names[goal_id]} at "
                    f"{self.parking_spots[goal_id]} "
                    f"is unreachable from entrance {self.entrance}"
                )

    def reset(self, goal_id=None):
        """Reset environment. Returns initial state (row, col).

        Args:
            goal_id: Target parking spot. Random if None.
        """
        if goal_id is None:
            goal_id = np.random.randint(self.num_goals)

        self.goal_id = goal_id
        self.goal_pos = self.parking_spots[goal_id]
        self.state = self.entrance
        self.steps = 0
        self._walkable = self._get_walkable_cells(goal_id)
        return self.state

    def step(self, action):
        """Take action. Returns (next_state, reward, done).

        Rewards:
            +10.0  reaching goal
            -0.1   each step (encourages efficiency)
            -1.0   hitting wall/barrier/edge (stays in place)
        """
        assert self.state is not None, "Call reset() first"
        assert 0 <= action < NUM_ACTIONS, f"Invalid action: {action}"

        self.steps += 1
        dr, dc = ACTION_DELTAS[action]
        nr, nc = self.state[0] + dr, self.state[1] + dc
        new_pos = (nr, nc)

        if (0 <= nr < self.rows and 0 <= nc < self.cols
                and new_pos in self._walkable):
            self.state = new_pos
            if self.state == self.goal_pos:
                return self.state, 10.0, True
            return self.state, -0.1, False
        else:
            return self.state, -1.0, False

    def get_valid_actions(self, state, goal_id):
        """Return list of valid actions from state for a given goal."""
        walkable = self._get_walkable_cells(goal_id)
        valid = []
        for action in range(NUM_ACTIONS):
            dr, dc = ACTION_DELTAS[action]
            nr, nc = state[0] + dr, state[1] + dc
            if (0 <= nr < self.rows and 0 <= nc < self.cols
                    and (nr, nc) in walkable):
                valid.append(action)
        return valid

    def get_all_states(self, goal_id):
        """Return all walkable states for a given goal."""
        return sorted(self._get_walkable_cells(goal_id))

    def render(self, policy=None, goal_id=None, path=None):
        """ASCII render of the parking lot."""
        gid = goal_id if goal_id is not None else self.goal_id
        walkable = (self._get_walkable_cells(gid)
                    if gid is not None else set())
        goal_pos = (self.parking_spots[gid]
                    if gid is not None else None)
        path_set = set(path) if path else set()

        print(f"\n  Parking Lot (Goal: {self.spot_names.get(gid, '?')} "
              f"at {goal_pos})")
        print("  " + "----" * self.cols + "-")

        for r in range(self.rows):
            row_str = "  |"
            for c in range(self.cols):
                pos = (r, c)
                if pos == goal_pos:
                    cell = " G "
                elif pos == self.entrance:
                    cell = " E "
                elif pos in self.barrier_cells:
                    cell = "###"
                elif pos in self.all_parking_positions:
                    sid = [k for k, v in self.parking_spots.items()
                           if v == pos][0]
                    if gid is not None and pos not in walkable:
                        cell = f"[{self.spot_names[sid][1:]}]"
                    else:
                        cell = f" {self.spot_names[sid]} "
                elif policy and pos in policy and pos in walkable:
                    cell = f" {ACTION_ARROWS[policy[pos]]} "
                elif pos in path_set:
                    cell = " * "
                else:
                    cell = "   "
                row_str += cell + "|"
            print(row_str)
            print("  " + "----" * self.cols + "-")

    def render_with_values(self, values, goal_id):
        """ASCII render showing state values."""
        walkable = self._get_walkable_cells(goal_id)
        goal_pos = self.parking_spots[goal_id]

        print(f"\n  Value Map (Goal: {self.spot_names[goal_id]})")
        print("  " + "------" * self.cols + "-")

        for r in range(self.rows):
            row_str = "  |"
            for c in range(self.cols):
                pos = (r, c)
                if pos == goal_pos:
                    cell = " GOAL "
                elif pos in self.barrier_cells:
                    cell = " #### "
                elif pos not in walkable:
                    cell = " [BLK]"
                elif pos in values:
                    cell = f"{values[pos]:6.1f}"
                else:
                    cell = "      "
                row_str += cell + "|"
            print(row_str)
            print("  " + "------" * self.cols + "-")


def print_bfs_summary(env):
    """Print BFS shortest paths for all goals."""
    print("\n" + "=" * 60)
    print(f"BFS REACHABILITY VERIFICATION ({env.rows}x{env.cols}, "
          f"{env.num_goals} spots)")
    print("=" * 60)
    for goal_id in range(env.num_goals):
        path = env.bfs_paths[goal_id]
        goal_pos = env.parking_spots[goal_id]
        if path:
            print(f"  {env.spot_names[goal_id]} at {goal_pos}: "
                  f"{len(path)-1} steps")
        else:
            print(f"  {env.spot_names[goal_id]} at {goal_pos}: UNREACHABLE!")


if __name__ == "__main__":
    # Default layout
    env = ParkingLot()
    print_bfs_summary(env)
    for gid in range(env.num_goals):
        env.render(goal_id=gid, path=env.bfs_paths[gid])

    # Larger layout demo
    print("\n\n=== LARGER LAYOUT (10x12) ===")
    big_env = ParkingLot(rows=10, cols=12, spots_per_row=8)
    print_bfs_summary(big_env)
    big_env.render(goal_id=0, path=big_env.bfs_paths[0])
