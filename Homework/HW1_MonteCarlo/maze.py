"""
Maze Environment for Monte Carlo RL

5x5 Grid Maze with barriers as specified in HW1.
Uses 1-indexed coordinates to match homework specification.

Maze Layout:
     Col 1   Col 2   Col 3   Col 4   Col 5
    +-------+-------+-------+-------+-------+
Row 1| START |       |       |       |       |
    +-------+-------+-------+-------+-------+
Row 2|       |       |       |       |       |
    +-------+-------+-------+-------+-------+
Row 3|       |XXXXXXX|XXXXXXX|XXXXXXX|XXXXXXX|  <- Barriers
    +-------+-------+-------+-------+-------+
Row 4|       |       |       |       |       |
    +-------+-------+-------+-------+-------+
Row 5|       |       |XXXXXXX|       | GOAL  |  <- Barrier at (5,3)
    +-------+-------+-------+-------+-------+

Actions: UP=0, DOWN=1, LEFT=2, RIGHT=3
"""

import numpy as np
from typing import Tuple, List, Set, Dict, Optional


class Maze:
    """
    Maze environment for reinforcement learning.

    Attributes:
        rows: Number of rows in the maze
        cols: Number of columns in the maze
        start: Starting position (row, col) - 1-indexed
        goal: Goal position (row, col) - 1-indexed
        barriers: Set of barrier positions
    """

    # Action definitions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    ACTION_NAMES = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT'}
    ACTION_SYMBOLS = {UP: '^', DOWN: 'v', LEFT: '<', RIGHT: '>'}

    # Movement deltas (row_delta, col_delta)
    ACTION_DELTAS = {
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1)
    }

    def __init__(self, rows: int = 5, cols: int = 5,
                 start: Tuple[int, int] = (1, 1),
                 goal: Tuple[int, int] = (5, 5),
                 barriers: Optional[Set[Tuple[int, int]]] = None):
        """
        Initialize the maze.

        Args:
            rows: Number of rows (default 5)
            cols: Number of columns (default 5)
            start: Starting position (row, col), 1-indexed
            goal: Goal position (row, col), 1-indexed
            barriers: Set of barrier positions, or None for default HW1 barriers
        """
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal

        # Default barriers for HW1 maze
        if barriers is None:
            self.barriers = {
                (3, 2), (3, 3), (3, 4), (3, 5),  # Row 3: cols 2-5
                (5, 3)                            # Row 5: col 3
            }
        else:
            self.barriers = barriers

        # Build state space: all grid cells that are not barriers
        self.states = self._build_state_space()
        self.n_states = len(self.states)  # 20 for default HW1 maze (25 - 5 barriers)
        self.n_actions = 4  # UP, DOWN, LEFT, RIGHT

        # Bidirectional mappings for tabular indexing
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.idx_to_state = {i: s for i, s in enumerate(self.states)}

    def _build_state_space(self) -> List[Tuple[int, int]]:
        """Build list of valid states (non-barrier cells)."""
        states = []
        for r in range(1, self.rows + 1):      # 1-indexed rows
            for c in range(1, self.cols + 1):  # 1-indexed cols
                if (r, c) not in self.barriers:
                    states.append((r, c))
        return states  # Ordered row-major for consistent iteration

    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """Check if a state is valid (within bounds and not a barrier)."""
        r, c = state
        if r < 1 or r > self.rows or c < 1 or c > self.cols:
            return False  # Off the grid edge
        if state in self.barriers:
            return False  # Inside a barrier cell
        return True

    def get_valid_actions(self, state: Tuple[int, int]) -> List[int]:
        """
        Get list of valid actions from a state.

        An action is valid if it doesn't lead into a barrier or off the grid.
        """
        valid = []
        for action in range(self.n_actions):
            next_state = self._get_next_state(state, action)
            if self.is_valid_state(next_state):
                valid.append(action)
        return valid

    def _get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Get the next state given current state and action (before validation)."""
        dr, dc = self.ACTION_DELTAS[action]
        return (state[0] + dr, state[1] + dc)

    def step(self, state: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take an action in the maze.

        Args:
            state: Current state (row, col)
            action: Action to take (0-3)

        Returns:
            next_state: The resulting state
            reward: The reward received
            done: Whether the episode is finished (reached goal)
        """
        next_state = self._get_next_state(state, action)

        # If next state is invalid (wall or barrier), stay in place
        # Agent still receives step penalty — "bumping" a wall costs a step
        if not self.is_valid_state(next_state):
            next_state = state

        # Check if reached goal
        done = (next_state == self.goal)

        # Reward structure:
        # +1.0 for reaching goal, -0.01 per step (encourages shortest path)
        # Step penalty is small enough not to overwhelm goal reward
        if done:
            reward = 1.0
        else:
            reward = -0.01

        return next_state, reward, done

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """Check if state is terminal (goal)."""
        return state == self.goal

    def reset(self) -> Tuple[int, int]:
        """Reset to start state."""
        return self.start

    def render(self, current_state: Optional[Tuple[int, int]] = None,
               policy: Optional[Dict[Tuple[int, int], int]] = None) -> str:
        """
        Render the maze as a string.

        Args:
            current_state: Highlight current position with 'A' (agent)
            policy: If provided, show policy arrows in each cell

        Returns:
            String representation of the maze
        """
        lines = []
        cell_w = 7  # width of each cell's content
        row_label_w = max(len(f"Row {r}") for r in range(1, self.rows + 1))

        # Header
        header = " " * row_label_w + " " + "".join(
            f" Col {c}".ljust(cell_w) + " " for c in range(1, self.cols + 1))
        lines.append(header)
        lines.append(" " * row_label_w + "+" + ("-" * cell_w + "+") * self.cols)

        for r in range(1, self.rows + 1):
            label = f"Row {r}".rjust(row_label_w)
            row_str = label + "|"
            for c in range(1, self.cols + 1):
                pos = (r, c)

                if pos in self.barriers:
                    cell = "X" * cell_w
                elif pos == current_state:
                    cell = "A".center(cell_w)
                elif pos == self.start and pos == self.goal:
                    cell = "S/G".center(cell_w)
                elif pos == self.start:
                    cell = "START".center(cell_w)
                elif pos == self.goal:
                    cell = "GOAL".center(cell_w)
                elif policy is not None and pos in policy:
                    action = policy[pos]
                    symbol = self.ACTION_SYMBOLS[action]
                    cell = symbol.center(cell_w)
                else:
                    cell = " " * cell_w

                row_str += cell + "|"

            lines.append(row_str)
            lines.append(" " * row_label_w + "+" + ("-" * cell_w + "+") * self.cols)

        return "\n".join(lines)

    def render_values(self, V: Dict[Tuple[int, int], float]) -> str:
        """
        Render the maze with state values.

        Args:
            V: Dictionary mapping states to values

        Returns:
            String representation with values
        """
        lines = []
        cell_w = 7
        row_label_w = max(len(f"Row {r}") for r in range(1, self.rows + 1))

        # Header
        header = " " * row_label_w + " " + "".join(
            f" Col {c}".ljust(cell_w) + " " for c in range(1, self.cols + 1))
        lines.append(header)
        lines.append(" " * row_label_w + "+" + ("-" * cell_w + "+") * self.cols)

        for r in range(1, self.rows + 1):
            label = f"Row {r}".rjust(row_label_w)
            row_str = label + "|"
            for c in range(1, self.cols + 1):
                pos = (r, c)

                if pos in self.barriers:
                    cell = "X" * cell_w
                elif pos in V:
                    val = V[pos]
                    cell = f"{val:7.2f}"
                else:
                    cell = "?".center(cell_w)

                row_str += cell + "|"

            lines.append(row_str)
            lines.append(" " * row_label_w + "+" + ("-" * cell_w + "+") * self.cols)

        return "\n".join(lines)

    def render_q_values(self, Q: Dict[Tuple[Tuple[int, int], int], float],
                        state: Tuple[int, int]) -> str:
        """Render Q-values for a specific state."""
        lines = [f"Q-values for state {state}:"]
        for action in range(self.n_actions):
            key = (state, action)
            val = Q.get(key, 0.0)
            lines.append(f"  {self.ACTION_NAMES[action]:5s}: {val:.4f}")
        return "\n".join(lines)


def create_hw1_maze() -> Maze:
    """Create the default maze specified in Homework 1."""
    return Maze(
        rows=5,
        cols=5,
        start=(1, 1),
        goal=(5, 5),
        barriers={(3, 2), (3, 3), (3, 4), (3, 5), (5, 3)}
    )


if __name__ == "__main__":
    # Test the maze
    maze = create_hw1_maze()

    print("=== HW1 Maze ===")
    print(maze.render())
    print()

    print(f"States: {maze.n_states}")
    print(f"Actions: {maze.n_actions}")
    print(f"Start: {maze.start}")
    print(f"Goal: {maze.goal}")
    print(f"Barriers: {maze.barriers}")
    print()

    # Test valid actions from start
    print(f"Valid actions from {maze.start}: {[maze.ACTION_NAMES[a] for a in maze.get_valid_actions(maze.start)]}")

    # Test a few steps
    state = maze.start
    print(f"\nStarting at {state}")

    for action in [maze.DOWN, maze.DOWN, maze.RIGHT]:
        next_state, reward, done = maze.step(state, action)
        print(f"Action {maze.ACTION_NAMES[action]}: {state} -> {next_state}, reward={reward}, done={done}")
        state = next_state
