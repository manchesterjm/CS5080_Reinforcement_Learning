"""
Deep Q-Network (DQN) Agent

Implements DQN with experience replay and target network for the
parking lot navigation problem.

No RL libraries used — only PyTorch for the neural network.

Reference: Mnih et al. (2015), "Human-level control through deep
reinforcement learning", Nature.
"""

import copy
import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from parking_lot import ParkingLot, NUM_ACTIONS
from replay_memory import ReplayBuffer, Transition


class QNetwork(nn.Module):
    """Simple MLP for Q-value approximation.

    Input: [row/max_row, col/max_col, goal_row/max_row, goal_col/max_col]
    Output: Q-values for each of 4 actions
    """

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, NUM_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Deep Q-Network agent with experience replay and target network."""

    def __init__(self, env: ParkingLot,
                 hidden_size: int = 64,
                 lr: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 batch_size: int = 64,
                 memory_size: int = 10000,
                 target_update: int = 100,
                 seed: Optional[int] = None):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Networks
        self.q_net = QNetwork(hidden_size)
        self.target_net = QNetwork(hidden_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.memory = ReplayBuffer(capacity=memory_size, seed=seed)

        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []

        # Normalization constants
        self.row_norm = float(env.rows - 1)
        self.col_norm = float(env.cols - 1)

    def _encode_state(self, state: Tuple[int, int],
                      goal_id: int) -> np.ndarray:
        """Encode state as normalized float vector."""
        goal_pos = self.env.parking_spots[goal_id]
        return np.array([
            state[0] / self.row_norm,
            state[1] / self.col_norm,
            goal_pos[0] / self.row_norm,
            goal_pos[1] / self.col_norm,
        ], dtype=np.float32)

    def _state_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Convert numpy state to torch tensor."""
        return torch.FloatTensor(state).unsqueeze(0)

    def select_action(self, state: Tuple[int, int],
                      goal_id: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)

        encoded = self._encode_state(state, goal_id)
        with torch.no_grad():
            q_values = self.q_net(self._state_tensor(encoded))
        return int(q_values.argmax(dim=1).item())

    def greedy_action(self, state: Tuple[int, int],
                      goal_id: int) -> int:
        """Pure greedy action selection."""
        encoded = self._encode_state(state, goal_id)
        with torch.no_grad():
            q_values = self.q_net(self._state_tensor(encoded))
        return int(q_values.argmax(dim=1).item())

    def _optimize(self):
        """Sample batch from replay memory and perform one gradient step."""
        if not self.memory.is_ready(self.batch_size):
            return

        batch = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(np.array([t.state for t in batch]))
        actions = torch.LongTensor([t.action for t in batch]).unsqueeze(1)
        rewards = torch.FloatTensor([t.reward for t in batch]).unsqueeze(1)
        next_states = torch.FloatTensor(
            np.array([t.next_state for t in batch]))
        dones = torch.FloatTensor(
            [float(t.done) for t in batch]).unsqueeze(1)

        # Current Q-values for chosen actions
        current_q = self.q_net(states).gather(1, actions)

        # Target Q-values (from target network)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())

    def train(self, n_episodes: int = 2000,
              verbose: bool = False,
              print_interval: int = 500) -> Dict:
        """Train the DQN agent."""
        for ep in range(n_episodes):
            goal_id = np.random.randint(self.env.num_goals)
            state = self.env.reset(goal_id)
            encoded = self._encode_state(state, goal_id)
            total_reward = 0.0
            steps = 0

            for step in range(self.env.max_steps):
                action = self.select_action(state, goal_id)
                next_state, reward, done = self.env.step(action)
                next_encoded = self._encode_state(next_state, goal_id)

                # Store in replay memory
                self.memory.push(encoded, action, reward,
                                 next_encoded, done)

                # Optimize
                self._optimize()

                total_reward += reward
                steps += 1
                state = next_state
                encoded = next_encoded

                if done:
                    break

            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min,
                               self.epsilon * self.epsilon_decay)

            # Update target network
            if (ep + 1) % self.target_update == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            if verbose and (ep + 1) % print_interval == 0:
                avg_r = np.mean(self.episode_rewards[-print_interval:])
                avg_l = np.mean(self.episode_lengths[-print_interval:])
                avg_loss = (np.mean(self.losses[-1000:])
                            if self.losses else 0.0)
                print(f"  Episode {ep+1:5d} | "
                      f"Avg Reward: {avg_r:7.2f} | "
                      f"Avg Length: {avg_l:5.1f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Epsilon: {self.epsilon:.4f}")

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "losses": self.losses,
        }

    def evaluate(self, n_episodes: int = 100) -> Dict:
        """Evaluate learned policy (greedy, no exploration)."""
        results = {gid: {"successes": 0, "total_steps": [],
                         "total_rewards": []}
                   for gid in range(self.env.num_goals)}

        for _ in range(n_episodes):
            for goal_id in range(self.env.num_goals):
                state = self.env.reset(goal_id)
                total_reward = 0.0

                for step in range(self.env.max_steps):
                    action = self.greedy_action(state, goal_id)
                    next_state, reward, done = self.env.step(action)
                    total_reward += reward
                    state = next_state
                    if done:
                        results[goal_id]["successes"] += 1
                        break

                results[goal_id]["total_steps"].append(step + 1)
                results[goal_id]["total_rewards"].append(total_reward)

        summary = {}
        for gid in range(self.env.num_goals):
            r = results[gid]
            summary[gid] = {
                "success_rate": r["successes"] / n_episodes,
                "avg_steps": np.mean(r["total_steps"]),
                "avg_reward": np.mean(r["total_rewards"]),
            }
        return summary

    def get_policy(self, goal_id: int) -> Dict[Tuple[int, int], int]:
        """Extract greedy policy for a specific goal."""
        policy = {}
        for state in self.env.get_all_states(goal_id):
            if state != self.env.parking_spots[goal_id]:
                policy[state] = self.greedy_action(state, goal_id)
        return policy

    def get_values(self, goal_id: int) -> Dict[Tuple[int, int], float]:
        """Extract state values V(s) = max_a Q(s,a) for a goal."""
        values = {}
        for state in self.env.get_all_states(goal_id):
            encoded = self._encode_state(state, goal_id)
            with torch.no_grad():
                q_vals = self.q_net(self._state_tensor(encoded))
            values[state] = float(q_vals.max().item())
        return values


if __name__ == "__main__":
    from q_learning import print_evaluation

    env = ParkingLot()

    print("=" * 60)
    print("DEEP Q-NETWORK (DQN)")
    print("=" * 60)
    agent = DQNAgent(env, seed=42)
    agent.train(n_episodes=3000, verbose=True, print_interval=500)
    summary = agent.evaluate()
    print_evaluation(summary, "DQN")

    # Show policy for P1
    policy = agent.get_policy(goal_id=0)
    env.render(policy=policy, goal_id=0)
