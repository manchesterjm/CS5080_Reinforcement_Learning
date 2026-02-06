"""
Monte Carlo Reinforcement Learning Algorithms

Implements Monte Carlo methods for policy evaluation and control,
following Sutton & Barto Chapter 5.

Algorithms:
- First-Visit MC Prediction (estimates V_π)
- Monte Carlo ES (Exploring Starts) for finding π*
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

        # Initialize Q(s, a) arbitrarily (zeros)
        # Key: (state, action), Value: estimated return
        self.Q: Dict[Tuple[Tuple[int, int], int], float] = {}
        for state in maze.states:
            for action in range(maze.n_actions):
                self.Q[(state, action)] = 0.0

        # Returns(s, a) - list of returns for averaging
        self.returns: Dict[Tuple[Tuple[int, int], int], List[float]] = defaultdict(list)

        # Initialize policy arbitrarily (random valid action for each state)
        self.policy: Dict[Tuple[int, int], int] = {}
        self._initialize_random_policy()

        # Episode history for analysis
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
                # Probability distribution over all actions
                probs = [0.0] * self.maze.n_actions
                for a in valid_actions:
                    probs[a] = 1.0 / n_valid
                random_policy[state] = probs
        return random_policy

    def generate_episode(self, exploring_starts: bool = True,
                        max_steps: int = 1000) -> List[Tuple[Tuple[int, int], int, float]]:
        """
        Generate an episode following the current policy.

        Args:
            exploring_starts: If True, start from random state-action pair
            max_steps: Maximum steps before truncating episode

        Returns:
            List of (state, action, reward) tuples for each step
        """
        episode = []

        if exploring_starts:
            # Choose random starting state (excluding goal)
            non_terminal_states = [s for s in self.maze.states
                                   if not self.maze.is_terminal(s)]
            state = random.choice(non_terminal_states)

            # Choose random first action from valid actions
            valid_actions = self.maze.get_valid_actions(state)
            action = random.choice(valid_actions)
        else:
            state = self.maze.reset()
            action = self.policy.get(state, 0)

        # Generate episode
        for _ in range(max_steps):
            next_state, reward, done = self.maze.step(state, action)
            episode.append((state, action, reward))

            if done:
                break

            state = next_state
            # Follow policy for subsequent actions
            if state in self.policy:
                action = self.policy[state]
            else:
                # Fallback to random if state not in policy
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
        # Calculate returns backwards through episode
        G = 0.0
        visited = set()  # Track visited (state, action) pairs

        # Process episode backwards
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = self.gamma * G + reward

            sa_pair = (state, action)

            # First-visit check
            if first_visit:
                # Check if (s, a) appears earlier in episode
                earlier_pairs = [(episode[i][0], episode[i][1]) for i in range(t)]
                if sa_pair in earlier_pairs:
                    continue

            # Update returns list and Q-value
            self.returns[sa_pair].append(G)
            self.Q[sa_pair] = np.mean(self.returns[sa_pair])

            # Policy improvement: greedy w.r.t. Q
            best_action = self._get_greedy_action(state)
            self.policy[state] = best_action

        # Calculate total episode return (from start)
        total_return = 0.0
        discount = 1.0
        for _, _, reward in episode:
            total_return += discount * reward
            discount *= self.gamma

        return total_return

    def _get_greedy_action(self, state: Tuple[int, int]) -> int:
        """Get the greedy action for a state based on current Q-values."""
        valid_actions = self.maze.get_valid_actions(state)
        if not valid_actions:
            return 0

        best_action = valid_actions[0]
        best_value = self.Q.get((state, best_action), 0.0)

        for action in valid_actions[1:]:
            value = self.Q.get((state, action), 0.0)
            if value > best_value:
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
        for ep in range(n_episodes):
            episode = self.generate_episode(exploring_starts=exploring_starts)
            total_return = self.update_from_episode(episode, first_visit=first_visit)

            self.episode_lengths.append(len(episode))
            self.episode_returns.append(total_return)

            if verbose and (ep + 1) % print_interval == 0:
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
                V[state] = 0.0
            elif state in self.policy:
                action = self.policy[state]
                V[state] = self.Q.get((state, action), 0.0)
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
            state = self.maze.reset()
            episode_return = 0.0
            discount = 1.0

            for step in range(max_steps):
                action = self.policy.get(state, 0)
                next_state, reward, done = self.maze.step(state, action)

                episode_return += discount * reward
                discount *= self.gamma

                if done:
                    successes += 1
                    total_steps.append(step + 1)
                    break

                state = next_state

            total_returns.append(episode_return)

        success_rate = successes / n_episodes
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
