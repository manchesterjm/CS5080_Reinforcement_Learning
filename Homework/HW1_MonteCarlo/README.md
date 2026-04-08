# HW1: Monte Carlo Maze Solver

**CS 5080 — Reinforcement Learning**
**Author:** Josh Manchester
**Date:** February 2026

## Overview

Implementation of Monte Carlo Exploring Starts (MC-ES) from scratch, following Sutton & Barto (2018) p.99. The agent learns an optimal policy to navigate a 5×5 grid maze from start (1,1) to goal (5,5), avoiding barriers.

## Requirements

- Python 3.10+
- numpy
- matplotlib

## Setup

```bash
pip install -r requirements.txt
```

## Running

**Run all experiments with plots:**
```bash
python main.py
```

**Run text-only (no matplotlib windows):**
```bash
python main.py --no-plots
```

## Experiments

`main.py` runs five experiments in sequence:

| # | Experiment                  | HW Question | What It Does                                                  |
|---|------------------------------|-------------|---------------------------------------------------------------|
| 1 | HW1 Maze                    | Q1–Q3       | Train MC-ES on the 5×5 HW1 maze, show learned policy/values  |
| 2 | Policy Consistency           | Q4          | Run 5 independent training runs, compare learned policies     |
| 3 | Parameter Sensitivity        | Q4          | Vary gamma (0.9–0.999) and episode count (1K–10K)             |
| 4 | Random Mazes                 | Q5          | Generate and solve random 5×5, 7×7, and 10×10 mazes           |
| 5 | First-Visit vs Every-Visit   | Q7 (extra)  | Compare first-visit MC vs every-visit MC                      |

## File Descriptions

| File                 | Purpose                                                    |
|----------------------|------------------------------------------------------------|
| `main.py`            | Experiment runner — runs all HW questions                  |
| `maze.py`            | 5×5 grid environment (states, actions, barriers, rewards)  |
| `monte_carlo.py`     | First-visit MC-ES with Q-table and greedy policy           |
| `maze_generator.py`  | Random maze generator with BFS path verification           |
| `visualize.py`       | Matplotlib policy arrows, value heatmaps, learning curves  |
| `generate_figures.py`| Publication-quality figure generation for the paper        |
| `requirements.txt`   | Python dependencies (numpy, matplotlib)                    |

## Results

- 100% success rate across all maze configurations
- Optimal path length of 8 steps on the HW1 maze
- Converges within ~1,500 episodes
- Scales to 7×7 (12 steps) and 10×10 (18 steps) random mazes

## Paper

`HW1_MonteCarlo_Paper.pdf` — 4-page writeup in AAAI format covering implementation, results, and analysis.
