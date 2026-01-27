# Proposal Analysis: Comparison with 2024 Grad Student Examples

**Date:** January 27, 2026
**Purpose:** Document comparison of our proposal against CS 5080 grad student examples from 2024

---

## Papers Compared

### Our Proposal
- **Title:** Can Modern Deep RL Overcome Sparse Rewards? A Case Study in Minesweeper
- **Author:** Josh Manchester
- **References:** 12

### 2024 Grad Student Examples (identified by 10+ references)

| Student | Topic | References | Level |
|---------|-------|------------|-------|
| Sakib | Autonomous Driving Collision Avoidance | 11 | Grad |
| Kelso | Traffic Signal Control | 15+ | Grad |
| Wilson | Delivery Optimization with Decision Transformers | 11 | Grad |
| Walker | 2048 Game | 7 | Undergrad (excluded) |

---

## Abstract Comparison

### Our Abstract
> "Sparse reward environments remain a fundamental challenge in reinforcement learning, where agents receive feedback only upon task completion rather than incrementally. This proposal investigates whether modern deep RL algorithms can achieve competent performance in Minesweeper---a domain characterized by sparse rewards, large state spaces, and partial observability. Prior work combining Deep Q-Networks with constraint propagation logic achieved an 88% win rate, while pure DQN achieved only 1%. This dramatic gap raises a central question: can advanced RL techniques such as Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC), reward shaping, and curriculum learning close this performance gap, or is domain knowledge fundamentally required? This project will systematically evaluate these approaches to provide empirical guidance on the limitations of pure RL in sparse reward domains."

### Comparison Table

| Aspect | Our Abstract | Sakib | Kelso | Wilson |
|--------|--------------|-------|-------|--------|
| **Research question** | Explicit | Implicit | Implicit | None |
| **Quantitative baseline** | Yes (1% vs 88%) | No | No | Yes (2.2M items) |
| **Specific algorithms** | PPO, SAC, β-DQN, curriculum | Q-Learning, SARSA, MLP-SARSA | "various methods" | Decision Transformer |
| **Problem significance** | Sparse rewards challenge | Safety in autonomy | Societal (pollution, delays) | Industry scale |
| **Expected contribution** | "Empirical guidance" | "Evaluate performance" | "Improve performance" | "Efficiently handling" |
| **Prior work cited** | Yes (own 1%/88%) | No | No | Yes (2017 study) |

### Abstract Verdict
Our abstract is stronger than the grad examples in:
- **Research question clarity** - Explicitly asks falsifiable question
- **Quantitative motivation** - 1% vs 88% gap is compelling
- **Personal prior work** - References own baseline results
- **Specific methodology** - Names 5+ specific techniques

---

## Structure Comparison

| Section | Our Proposal | Sakib | Kelso | Wilson |
|---------|--------------|-------|-------|--------|
| Abstract | ✓ | ✓ | ✓ | ✓ |
| Introduction | ✓ | ✓ | ✓ | ✓ |
| Background | ✓ (4 subsections) | ✓ (3 subsections) | ✓ (Problem Formulation) | ✓ |
| Related Work | Integrated in Background | ✓ | ✓ (Existing Methods) | ✓ |
| Proposed Methods | ✓ (5 subsections) | ✓ (4 subsections) | ✓ | ✓ (Environment section) |
| Evaluation | ✓ | ✓ (brief) | ✓ (Performance Metrics) | ✓ |
| Timeline | ✓ (table) | ✓ (table) | ✓ (table) | ✓ (table) |
| Expected Contributions | ✓ | ✗ | ✗ | ✗ |
| Conclusion | ✓ | ✓ | ✓ | ✓ |
| References | 12 | 11 | 15+ | 11 |

---

## Section-by-Section Analysis

### 1. Introduction

| Aspect | Our Proposal | Sakib | Kelso | Wilson |
|--------|--------------|-------|-------|--------|
| Opens with broad context | ✓ (RL successes) | ✓ (collision avoidance) | ✓ (transportation) | ✓ (e-commerce) |
| Narrows to specific problem | ✓ (sparse rewards → Minesweeper) | ✓ (dynamic obstacles) | ✓ (TSC with RL) | ✓ (logistics) |
| States research question | ✓ (explicit, italicized) | Weak | ✗ | ✗ |
| Cites prior work | ✓ (4 citations) | ✓ (5 citations) | ✓ (3 citations) | ✓ (1 citation) |
| Personal prior work | ✓ (1% vs 88%) | ✗ | ✗ | ✗ |

**Verdict:** Our introduction is the strongest due to explicit research question and personal baseline data.

---

### 2. Background / Literature Review

| Aspect | Our Proposal | Sakib | Kelso | Wilson |
|--------|--------------|-------|-------|--------|
| Organized subsections | ✓ (4 clear topics) | ✓ (3 algorithms) | ✓ (state/action/reward) | ✓ (transformer explained) |
| Defines key concepts | ✓ | ✓ | ✓ | ✓ |
| Mathematical notation | Minimal | Minimal | ✓ (equations 1-3) | Minimal |
| Cites recent work (2023+) | ✓ (Zhang 2025, Chen 2024, Han 2024) | ✓ | ✓ | ✓ |
| Connects to problem | ✓ | Partial | ✓ | Partial |

**Our Background Subsections:**
1. Sparse Rewards in RL
2. Deep Q-Network Improvements
3. Policy Gradient Methods
4. Curriculum Learning

**Note:** Kelso's background is more mathematically rigorous with formal state space equations. Consider whether adding formal MDP definition for Minesweeper would strengthen ours.

---

### 3. Proposed Methods

| Aspect | Our Proposal | Sakib | Kelso | Wilson |
|--------|--------------|-------|-------|--------|
| Clear baseline | ✓ (DQN at 1%) | ✗ | ✓ (Fixed Time, etc.) | ✗ |
| Methods enumerated | ✓ (5 methods) | ✓ (3 methods) | ✓ (IDQN justified) | ✓ (Decision Transformer) |
| Implementation details | Partial | ✓ (states, actions, rewards) | ✓ (ε-greedy, binary DQN) | ✓ (detailed) |
| Justifies method choices | ✓ (cites papers) | Partial | ✓ (cites performance) | ✓ |
| Specific hyperparameters | Partial | ✗ | ✓ (ε decay 1.0→0.1) | ✗ |

**Our Reward Shaping Specifics:**
- +1 for each safe cell revealed
- -0.5 for incorrect flag placement
- +10 for game completion (scaled by remaining cells)

**Our Curriculum Specifics:**
- 5×5 grid with 3 mines (easy)
- 8×8 grid with 10 mines (medium)
- 16×16 grid with 40 mines (expert)

**Verdict:** Our methods section is comprehensive. Could add formal state/action space definitions like Sakib and Wilson.

---

### 4. Evaluation

| Aspect | Our Proposal | Sakib | Kelso | Wilson |
|--------|--------------|-------|-------|--------|
| Primary metric | Win rate (10K episodes) | Not specified | Cumulative delay (s) | Time/cost efficiency |
| Secondary metrics | Sample efficiency, learning curves | ✗ | Queue length, trip time | Actions, miles |
| Baselines for comparison | ✓ (4 baselines) | ✗ | ✓ (Table with 10 methods) | ✗ |
| Statistical rigor | 10K test episodes | Not specified | RESCO benchmark | Not specified |
| Comparison to prior work | ✓ (Phan 2025) | ✗ | ✓ (extensive table) | ✗ |

**Our Baselines:**
1. Pure DQN baseline (1% win rate)
2. Hybrid DQN + AC-3 system (88% win rate)
3. Random baseline
4. Results from Phan 2025 on comparable board sizes

**Note:** Kelso includes a comparison table with specific numbers from literature. Consider adding similar table showing expected/target performance ranges.

---

### 5. Timeline

| Aspect | Our Proposal | Sakib | Kelso | Wilson |
|--------|--------------|-------|-------|--------|
| Format | Table | Table | Table | Table |
| Milestones | 8 | 6 | 5 | 4 |
| Granularity | Bi-weekly | ~Weekly | Weekly | Weekly |
| Extends to final | ✓ (May 12) | ✓ (May 1) | ✓ (Apr 1) | ✓ (Mar 18) |

**Our Timeline:**
| Date | Milestone |
|------|-----------|
| Feb 17-19 | Proposal presentation |
| Feb 19 | Proposal paper due |
| Mar 1 | PPO, SAC, β-DQN implementations |
| Mar 15 | Reward shaping experiments complete |
| Mar 31 | Curriculum learning experiments complete |
| Apr 1 | Midterm presentation and paper |
| Apr 15 | Full experimental results |
| May 7 | Final presentation |
| May 12 | Final paper due |

**Verdict:** Our timeline is most complete, covering all semester milestones.

---

### 6. Expected Contributions

| Aspect | Our Proposal | Sakib | Kelso | Wilson |
|--------|--------------|-------|-------|--------|
| Has this section | ✓ | ✗ | ✗ | ✗ |
| Contributions listed | 3 items | N/A | N/A | N/A |

**Our Contributions:**
1. Empirical comparison of modern RL algorithms (including β-DQN) on challenging sparse reward task
2. Analysis of when pure RL succeeds versus when domain knowledge is required
3. Practical guidance for practitioners facing sparse reward problems

**Verdict:** We are the only proposal with an explicit contributions section.

---

### 7. Conclusion

| Aspect | Our Proposal | Sakib | Kelso | Wilson |
|--------|--------------|-------|-------|--------|
| Summarizes approach | ✓ | ✓ | ✓ | ✓ |
| Restates research question | ✓ | ✗ | ✗ | ✗ |
| Forward-looking | Partial | ✓ | ✓ | ✓ |

**Note:** Kelso includes "Expected Challenges" section before conclusion discussing potential issues (non-stationarity, divergence). This is a nice addition to consider.

---

### 8. References

| Aspect | Our Proposal | Sakib | Kelso | Wilson |
|--------|--------------|-------|-------|--------|
| Total count | 12 | 11 | 15+ | 11 |
| Meets grad requirement (10+) | ✓ | ✓ | ✓ | ✓ |
| Foundational papers | ✓ (Mnih 2015, Sutton 2018) | ✓ | ✓ | ✓ |
| Recent papers (2023+) | ✓ (4 papers) | ✓ (3) | ✓ (4) | ✓ (3) |
| Domain-specific | ✓ (Phan 2025, Jiang 2025) | ✓ | ✓ | ✓ |

---

## Overall Quality Ratings

| Criterion | Our Proposal | Sakib | Kelso | Wilson |
|-----------|--------------|-------|-------|--------|
| Clarity of research question | ★★★ | ★ | ★ | ★ |
| Technical depth | ★★★ | ★★ | ★★★ | ★★★ |
| Prior work integration | ★★★ | ★★ | ★★★ | ★★ |
| Methodology specificity | ★★★ | ★★ | ★★★ | ★★★ |
| Evaluation plan | ★★★ | ★ | ★★★ | ★★ |
| Writing quality | ★★★ | ★★ | ★★★ | ★★★ |
| Personal baseline data | ★★★ | ✗ | ✗ | ✗ |

---

## Summary

### Our Strengths vs. Grad Examples

1. **Explicit research question** - None of the examples have this
2. **Personal prior work** (1% vs 88%) - Unique among examples
3. **Expected Contributions section** - Only ours has this
4. **Complete timeline** - Covers full semester
5. **Specific experimental parameters** - Reward values, curriculum stages

### Potential Improvements (from examples)

1. **From Kelso:** Formal MDP definition with state space equation
2. **From Kelso:** Performance comparison table showing baseline numbers from literature
3. **From Kelso:** "Expected Challenges" section discussing potential issues
4. **From Wilson/Sakib:** More explicit state/action space definitions

### Optional Additions to Consider

- Add formal state representation (how is the board encoded for the neural network?)
- Add table comparing expected performance of methods from literature
- Consider "Expected Challenges" paragraph (e.g., sample efficiency, partial observability issues)

---

## Final Verdict

**Our proposal is at or above the quality level of the best grad examples (Kelso).** The explicit research question, quantitative baseline from prior work, and contributions section set it apart from all three comparison papers.

---

*Analysis completed: January 27, 2026*
