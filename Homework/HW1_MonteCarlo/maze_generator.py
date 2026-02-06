"""
Random Maze Generator for Monte Carlo RL Experiments

Generates random mazes with similar characteristics to the HW1 maze:
- Start in top-left region
- Goal in bottom-right region
- Barriers that create non-trivial paths without blocking all routes
"""

import random
from typing import Set, Tuple, Optional
from collections import deque

from maze import Maze


def is_path_exists(rows: int, cols: int,
                   start: Tuple[int, int],
                   goal: Tuple[int, int],
                   barriers: Set[Tuple[int, int]]) -> bool:
    """
    Check if a path exists from start to goal using BFS.

    Args:
        rows: Number of rows in the maze
        cols: Number of columns in the maze
        start: Starting position (row, col)
        goal: Goal position (row, col)
        barriers: Set of barrier positions

    Returns:
        True if a path exists, False otherwise
    """
    if start in barriers or goal in barriers:
        return False

    visited = set()
    queue = deque([start])
    visited.add(start)

    # Movement deltas
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()

        if (r, c) == goal:
            return True

        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            next_pos = (nr, nc)

            # Check bounds
            if nr < 1 or nr > rows or nc < 1 or nc > cols:
                continue
            # Check barriers and visited
            if next_pos in barriers or next_pos in visited:
                continue

            visited.add(next_pos)
            queue.append(next_pos)

    return False


def generate_random_maze(rows: int, cols: int,
                         barrier_density: float = 0.2,
                         seed: Optional[int] = None) -> Maze:
    """
    Generate a random maze with guaranteed path from start to goal.

    Args:
        rows: Number of rows (minimum 3)
        cols: Number of columns (minimum 3)
        barrier_density: Fraction of cells to attempt as barriers (0.0-0.5)
        seed: Random seed for reproducibility

    Returns:
        A valid Maze object with random barriers

    Raises:
        ValueError: If dimensions too small or density out of range
    """
    if rows < 3 or cols < 3:
        raise ValueError("Maze must be at least 3x3")
    if barrier_density < 0 or barrier_density > 0.5:
        raise ValueError("Barrier density must be between 0.0 and 0.5")

    if seed is not None:
        random.seed(seed)

    # Start in top-left region, goal in bottom-right
    start = (1, 1)
    goal = (rows, cols)

    # Calculate target number of barriers
    total_cells = rows * cols
    target_barriers = int(total_cells * barrier_density)

    # Generate candidate barrier positions (exclude start, goal, and their neighbors)
    protected = {start, goal}
    # Protect cells adjacent to start and goal
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        protected.add((start[0] + dr, start[1] + dc))
        protected.add((goal[0] + dr, goal[1] + dc))

    candidates = []
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if (r, c) not in protected:
                candidates.append((r, c))

    random.shuffle(candidates)

    # Add barriers one at a time, verifying path still exists
    barriers: Set[Tuple[int, int]] = set()

    for pos in candidates:
        if len(barriers) >= target_barriers:
            break

        # Try adding this barrier
        test_barriers = barriers | {pos}

        if is_path_exists(rows, cols, start, goal, test_barriers):
            barriers.add(pos)

    return Maze(rows=rows, cols=cols, start=start, goal=goal, barriers=barriers)


def generate_maze_sizes(sizes: list = None) -> list:
    """
    Generate mazes of different sizes for experimentation.

    Args:
        sizes: List of (rows, cols) tuples. Defaults to small/medium/large.

    Returns:
        List of Maze objects
    """
    if sizes is None:
        sizes = [(5, 5), (7, 7), (10, 10)]

    mazes = []
    for i, (rows, cols) in enumerate(sizes):
        # Use different seeds for reproducibility
        maze = generate_random_maze(rows, cols, barrier_density=0.2, seed=42 + i)
        mazes.append(maze)

    return mazes


if __name__ == "__main__":
    # Demo: Generate random mazes of different sizes
    print("=== Random Maze Generator Demo ===\n")

    sizes = [(5, 5), (7, 7), (10, 10)]

    for rows, cols in sizes:
        print(f"--- {rows}x{cols} Random Maze ---")
        maze = generate_random_maze(rows, cols, barrier_density=0.2, seed=42)
        print(maze.render())
        print(f"States: {maze.n_states}, Barriers: {len(maze.barriers)}")
        print()
