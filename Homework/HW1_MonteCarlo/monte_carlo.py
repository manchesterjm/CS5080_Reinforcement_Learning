"""
Monte Carlo Reinforcement Learning Algorithms

Implements the Monte Carlo ES (Exploring Starts) algorithm for finding π*,
following the pseudocode from 301-MonteCarloAlgorithms.pdf (course handout)
and Sutton & Barto Chapter 5, p.99.

The core algorithm is first-visit MC-ES:
    1. Initialize π(s), Q(s,a), Returns(s,a) for all s, a
    2. For each episode:
       a. Choose S₀, A₀ randomly (exploring starts)
       b. Generate episode following π
       c. Loop backwards t = T-1, ..., 0:
          - G ← γG + R_{t+1}
          - If (S_t, A_t) is a first visit:
            * Append G to Returns(S_t, A_t)
            * Q(S_t, A_t) ← average(Returns(S_t, A_t))
            * π(S_t) ← argmax_a Q(S_t, a)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random

from maze import Maze


class MonteCarloAgent:
    """
    Monte Carlo agent using Exploring Starts (ES) for control.

    This implements the Monte Carlo ES algorithm from Sutton & Barto p.99
    to find the optimal policy for a given maze environment.

    Attributes:
        maze: The maze environment
        gamma: Discount factor
        Q: Action-value function Q(s, a)
        returns: Lists of returns for each (s, a) pair
        policy: Current policy mapping states to actions
    """

    def __init__(self, maze: Maze, gamma: float = 0.99):
        """
        Initialize the Monte Carlo agent.

        Args:
            maze: The maze environment
            gamma: Discount factor (default 0.99)
        """
        self.maze = maze
        self.gamma = gamma

        # --- Pseudocode: Initialize ---
        # Q(s,a) ∈ R (arbitrarily), for all s ∈ S, a ∈ A(s)
        # Key: ((row, col), action_int) → Value: estimated return
        self.Q: Dict[Tuple[Tuple[int, int], int], float] = {}
        for state in maze.states:
            for action in range(maze.n_actions):
                self.Q[(state, action)] = 0.0  # Zero is a valid arbitrary init

        # Returns(s,a) ← empty list, for all s ∈ S, a ∈ A(s)
        # Stores all observed returns for averaging; defaultdict auto-creates empty lists
        self.returns: Dict[Tuple[Tuple[int, int], int], List[float]] = defaultdict(list)

        # π(s) ∈ A(s) (arbitrarily), for all s ∈ S
        # Maps each state to a single action (deterministic greedy policy)
        self.policy: Dict[Tuple[int, int], int] = {}
        self._initialize_random_policy()

        # Track training progress for learning curve plots
        self.episode_lengths: List[int] = []
        self.episode_returns: List[float] = []

    def _initialize_random_policy(self) -> None:
        """Initialize with a random policy (equiprobable over valid actions)."""
        for state in self.maze.states:
            if not self.maze.is_terminal(state):
                valid_actions = self.maze.get_valid_actions(state)
                if valid_actions:
                    self.policy[state] = random.choice(valid_actions)

    def get_random_policy(self) -> Dict[Tuple[int, int], List[float]]:
        """
        Get the equiprobable random policy.

        Returns:
            Dictionary mapping states to action probability distributions
        """
        random_policy = {}
        for state in self.maze.states:
            if not self.maze.is_terminal(state):
                valid_actions = self.maze.get_valid_actions(state)
                n_valid = len(valid_actions)
                # Build probability vector over ALL 4 actions (invalid actions get 0.0)
                probs = [0.0] * self.maze.n_actions
                for a in valid_actions:
                    probs[a] = 1.0 / n_valid  # Equiprobable over valid actions only
                random_policy[state] = probs
        return random_policy

    def generate_episode(self, exploring_starts: bool = True,
                        max_steps: int = 1000) -> List[Tuple[Tuple[int, int], int, float]]:
        """
        Generate an episode following the current policy.

        Args:
            exploring_starts: If True, start from random state-action pair
            max_steps: Maximum steps before truncating episode, set to 1000 to prevent inf loops

        Returns:
            List of (state, action, reward) tuples for each step
        """
        episode = []

        # --- Pseudocode: Choose S₀ ∈ S, A₀ ∈ A(S₀) randomly ---
        # "such that all pairs have probability > 0"
        if exploring_starts:
            # Pick any non-terminal state uniformly at random
            non_terminal_states = [s for s in self.maze.states
                                   if not self.maze.is_terminal(s)]
            state = random.choice(non_terminal_states)
            # Pick any valid action from that state uniformly at random
            valid_actions = self.maze.get_valid_actions(state)
            action = random.choice(valid_actions)
        else:
            # Non-ES mode: always start from maze start, follow current policy
            state = self.maze.reset()
            action = self.policy.get(state, 0)

        # --- Pseudocode: Generate episode from S₀, A₀ following π ---
        # Produces: S₀, A₀, R₁, S₁, A₁, R₂, ..., S_{T-1}, A_{T-1}, R_T
        # Stored as list of (S_t, A_t, R_{t+1}) tuples
        for _ in range(max_steps):  # Truncate at max_steps to prevent infinite wandering
            next_state, reward, done = self.maze.step(state, action)
            episode.append((state, action, reward))  # Store (S_t, A_t, R_{t+1})

            if done:  # Reached goal (terminal state)
                break

            state = next_state
            # Follow π for subsequent actions: A_t = π(S_t)
            if state in self.policy:
                action = self.policy[state]
            else:
                # Fallback: random action if state somehow has no policy entry
                valid_actions = self.maze.get_valid_actions(state)
                action = random.choice(valid_actions) if valid_actions else 0

        return episode

    def update_from_episode(self, episode: List[Tuple[Tuple[int, int], int, float]],
                           first_visit: bool = True) -> float:
        """
        Update Q-values and policy from an episode.

        Implements first-visit MC update with policy improvement.

        Args:
            episode: List of (state, action, reward) tuples
            first_visit: If True, only update on first visit to (s, a)

        Returns:
            Total return for the episode
        """
        # --- Pseudocode: G ← 0 ---
        G = 0.0  # Running discounted return, built up backwards from episode end

        # --- Pseudocode: Loop for each step of episode, t = T-1, T-2, ..., 0 ---
        # Walk backwards so G accumulates future discounted rewards correctly
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]

            # --- Pseudocode: G ← γG + R_{t+1} ---
            # Discount previously accumulated future return, then add this step's reward
            G = self.gamma * G + reward

            sa_pair = (state, action)

            # --- Pseudocode: Unless the pair S_t, A_t appears in ---
            # ---   S₀, A₀, S₁, A₁, ..., S_{t-1}, A_{t-1}       ---
            # First-visit: skip if this (state, action) was already seen earlier in episode
            if first_visit:
                earlier_pairs = [(episode[i][0], episode[i][1]) for i in range(t)]
                if sa_pair in earlier_pairs:
                    continue  # Not the first visit — don't update

            # --- Pseudocode: Append G to Returns(S_t, A_t) ---
            # Store this return sample; will be averaged across all episodes
            self.returns[sa_pair].append(G)

            # --- Pseudocode: Q(S_t, A_t) ← average(Returns(S_t, A_t)) ---
            # Simple mean of all observed returns for this (state, action) pair
            self.Q[sa_pair] = np.mean(self.returns[sa_pair])

            # --- Pseudocode: π(S_t) ← argmax_a Q(S_t, a) ---
            # Greedy policy improvement: pick the action with highest Q-value
            self.policy[state] = self._get_greedy_action(state)

        # Calculate total episode return (from start, forward direction)
        # This is separate from G — used for tracking learning progress
        total_return = 0.0
        discount = 1.0
        for _, _, reward in episode:
            total_return += discount * reward
            discount *= self.gamma  # Each subsequent reward discounted by γ^t

        return total_return

    def _get_greedy_action(self, state: Tuple[int, int]) -> int:
        """Get the greedy action for a state based on current Q-values."""
        valid_actions = self.maze.get_valid_actions(state)
        if not valid_actions:
            return 0  # Safety fallback — should never happen in valid maze

        # Linear scan over valid actions to find argmax Q(s, a)
        best_action = valid_actions[0]
        best_value = self.Q.get((state, best_action), 0.0)

        for action in valid_actions[1:]:
            value = self.Q.get((state, action), 0.0)
            if value > best_value:  # Strict > means ties go to first action found
                best_value = value
                best_action = action

        return best_action

    def train(self, n_episodes: int = 10000,
              exploring_starts: bool = True,
              first_visit: bool = True,
              verbose: bool = True,
              print_interval: int = 1000) -> None:
        """
        Train the agent using Monte Carlo ES.

        Args:
            n_episodes: Number of episodes to run
            exploring_starts: Use exploring starts
            first_visit: Use first-visit MC (vs every-visit)
            verbose: Print progress
            print_interval: How often to print progress
        """
        # --- Pseudocode: Loop forever (for each episode) ---
        # Bounded to n_episodes for practical convergence
        for ep in range(n_episodes):
            episode = self.generate_episode(exploring_starts=exploring_starts)
            total_return = self.update_from_episode(episode, first_visit=first_visit)

            # Track for learning curve visualization
            self.episode_lengths.append(len(episode))
            self.episode_returns.append(total_return)

            if verbose and (ep + 1) % print_interval == 0:
                # Sliding window average over last print_interval episodes
                avg_len = np.mean(self.episode_lengths[-print_interval:])
                avg_ret = np.mean(self.episode_returns[-print_interval:])
                print(f"Episode {ep + 1}/{n_episodes}: "
                      f"Avg Length = {avg_len:.1f}, Avg Return = {avg_ret:.4f}")

    def get_state_values(self) -> Dict[Tuple[int, int], float]:
        """
        Compute V(s) from Q(s, a) using the current policy.

        Returns:
            Dictionary mapping states to values
        """
        V = {}
        for state in self.maze.states:
            if self.maze.is_terminal(state):
                V[state] = 0.0  # Terminal state has no future return
            elif state in self.policy:
                action = self.policy[state]
                V[state] = self.Q.get((state, action), 0.0)  # V(s) = Q(s, π(s))
            else:
                V[state] = 0.0
        return V

    def evaluate_policy(self, n_episodes: int = 100,
                       max_steps: int = 1000) -> Tuple[float, float, float]:
        """
        Evaluate the current policy by running episodes from the start.

        Args:
            n_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode

        Returns:
            (success_rate, avg_steps_to_goal, avg_return)
        """
        successes = 0
        total_steps = []
        total_returns = []

        for _ in range(n_episodes):
            state = self.maze.reset()  # Always start from (1,1) for evaluation
            episode_return = 0.0
            discount = 1.0

            for step in range(max_steps):
                action = self.policy.get(state, 0)  # Follow learned greedy policy
                next_state, reward, done = self.maze.step(state, action)

                episode_return += discount * reward  # Accumulate γ^t * R_{t+1}
                discount *= self.gamma

                if done:  # Reached goal
                    successes += 1
                    total_steps.append(step + 1)  # +1 because step is 0-indexed
                    break

                state = next_state

            total_returns.append(episode_return)

        success_rate = successes / n_episodes  # Fraction of episodes that reached goal
        avg_steps = np.mean(total_steps) if total_steps else float('inf')
        avg_return = np.mean(total_returns)

        return success_rate, avg_steps, avg_return

    def print_policy(self) -> None:
        """Print the learned policy."""
        print("\n=== Learned Policy ===")
        print(self.maze.render(policy=self.policy))

    def print_values(self) -> None:
        """Print the state values."""
        V = self.get_state_values()
        print("\n=== State Values V(s) ===")
        print(self.maze.render_values(V))


if __name__ == "__main__":
    from maze import create_hw1_maze

    # Create maze and agent
    maze = create_hw1_maze()
    agent = MonteCarloAgent(maze, gamma=0.99)

    print("=== Initial Random Policy ===")
    print(maze.render(policy=agent.policy))

    print("\n=== Training with Monte Carlo ES ===")
    agent.train(n_episodes=5000, verbose=True, print_interval=1000)

    # Show results
    agent.print_policy()
    agent.print_values()

    # Evaluate
    success_rate, avg_steps, avg_return = agent.evaluate_policy(n_episodes=100)
    print(f"\n=== Policy Evaluation (100 episodes) ===")
    print(f"Success Rate: {success_rate * 100:.1f}%")
    print(f"Avg Steps to Goal: {avg_steps:.1f}")
    print(f"Avg Return: {avg_return:.4f}")
