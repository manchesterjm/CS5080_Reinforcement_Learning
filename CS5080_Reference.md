# CS 5080 - Reinforcement Learning

## Course Info

| | |
|---|---|
| **Instructor** | Dr. Jugal Kalita (jkalita@uccs.edu, 255-3432) |
| **Class** | Tu/Th 4:45-6:00pm, ENG 109 |
| **Office Hours** | MW 1:00-2:30pm, Osborne A-340 |

## Textbooks

**Required:**
1. *Reinforcement Learning: An Introduction* (2nd ed.) - Sutton & Barto, MIT Press 2018
   - The "bible" of RL, covers fundamentals + math
2. *Grokking Deep Reinforcement Learning* - Morales, Manning 2020
   - Gentler intro with worked examples and Python code

**Supplemental:**
- *Spinning Up in Deep RL!* - OpenAI (online tutorial)
- *Deep Learning* - Goodfellow, Bengio, Courville (for DL background)

**Videos (optional):**
- David Silver - Introduction to RL (10 videos, ~100 min each)
- Emma Brunskill - Stanford CS 234 (16 videos, ~75 min each)

### Textbook Access (All Free)

| Book | Access | Location/Link |
|------|--------|---------------|
| **Sutton & Barto** | PDF downloaded | `References/Sutton_Barto_RL_Introduction_2nd_Ed.pdf` |
| **Grokking Deep RL** | UCCS O'Reilly | See instructions below + `References/Grokking_Deep_RL_Chapter_*.docx` |
| **Spinning Up** | PDF downloaded | `References/Spinning_Up_Deep_RL_OpenAI.pdf` |
| **Goodfellow Deep Learning** | PDF downloaded | `References/Goodfellow_Deep_Learning.pdf` |

**To access Grokking Deep Reinforcement Learning (O'Reilly):**
1. Go to https://learning.oreilly.com/
2. Click **"Sign In"** → **"Select your institution"**
3. Search for **"University of Colorado Colorado Springs"**
4. Login with UCCS credentials (jmanches@uccs.edu)
5. Search for "Grokking Deep Reinforcement Learning"
6. Direct link (after login): https://www.oreilly.com/library/view/grokking-deep-reinforcement/9781617295454/

## Grading

| Component | Weight |
|-----------|--------|
| Semester Project | 50% |
| Homework Assignments (3-4) | 35% |
| Short Questions (Canvas) | 15% |
| Class Participation | 5% |

## Project Breakdown

| Deliverable | Weight | Details |
|-------------|--------|---------|
| Proposal presentation | 2.5% | ~10 min |
| Proposal paper | 5% | 2-3 pages, 10+ refs |
| Midterm presentation | 2.5% | ~10 min |
| Midterm paper | 7.5% | 4-5 pages, 10+ refs |
| Midterm demo | 2.5% | During office hours |
| Final presentation | 5% | Longer |
| Final paper | 17.5% | 6-8 pages |
| Final demo | 7.5% | Week of final |

**Format:** All papers in LaTeX using AAAI author style. No Word.

## Key Deadlines

| Deliverable | Date |
|-------------|------|
| Proposal Presentations | 2/17-19/2026 |
| Proposal Paper | 2/19/2026 |
| Midterm Presentations | 3/31-4/1/2026 |
| Midterm Paper | 4/1/2026 |
| Midterm Demo | Week of 4/1/2026 |
| Final Presentations | 5/7, 5/12/2026 (12:40-2:40) |
| Final Report | On or before 5/12/2026 |
| Final Demo | Week of 5/12/2026 |

## Topics Sequence

| Classes | Topic | Reading |
|---------|-------|---------|
| 2 | Intro to RL | SB Ch. 1 |
| 2 | Finite Markov Decision Processes | SB Ch. 3, M Ch. 2 |
| 2 | Monte Carlo methods | SB Ch. 5 |
| 2 | Proposal presentations | |
| 3 | TD Learning, Q-learning, SARSA | SB Ch. 6, M Ch. 6-7 |
| 2 | Function approximation, Value function approx | SB Ch. 9, M Ch. 5, 8-9 |
| 2 | Midterm presentations | |
| 4 | Deep Learning: CNN, DQN, Double DQN, Dueling DQN | Papers, SUDR |
| 4 | Policy search methods | SB Ch. 13, papers |
| 4 | Actor-Critic Methods | Papers, handouts |
| 2 | Monte Carlo Tree Search | Papers, handouts |
| 2 | Final presentations | |

## AI Policy

- **AI allowed for:** Grammar and spell-checking only
- **NOT allowed:** Content generation
- Quoted content (including from AI) max 5% of paper, must be attributed
- Plagiarism: First instance = F on deliverable, repeated = F in class

## Homework Format

- 2-4 page paper per assignment, conference paper style
- Must have: title, abstract, intro, conclusions, references
- Each assignment must be demoed

---

## Prior Experience & Prerequisites

### Minesweeper ML Project (Direct Experience)

**Location:** `D:\Simulations\AI_ML\minesweeper_ml_project\`
**GitHub:** https://github.com/manchesterjm/minesweeper_ml_project

Already implemented key course topics:

| Course Topic | Your Experience |
|--------------|-----------------|
| DQN (Deep Q-Network) | ✓ Implemented `dqn_agent.py` |
| Double DQN | ✓ Used in DQN agent |
| Dueling DQN | ✓ Architecture implemented |
| Experience Replay | ✓ Standard replay buffer |
| Gym-style Environments | ✓ Custom `environment.py` |
| Function Approximation | ✓ Neural network for Q-values |
| Policy vs Value methods | ✓ Hybrid agent comparison |

**Key Insight from Project:**
- Pure DQN achieved only 1% win rate (sparse rewards, huge state space)
- Hybrid (AC-3 logic + NN for guessing) achieved 88%
- Demonstrates when RL struggles vs. when problem structure helps

**Conda Environment:** `minesweeper-gpu` (PyTorch 2.7.0, CUDA 12.8)

### CS 4820 - Artificial Intelligence (Fall 2025)

**Instructor:** Dr. Atyabi
**Grade:** A

Directly relevant course covering:
- Search algorithms (state space exploration)
- Neural Networks (function approximation)
- Reinforcement Learning fundamentals
- NLP and sequence modeling

**BiLSTM Exoplanet Detection Project:**
- Built 3-layer BiLSTM (256 hidden units, bidirectional) for TESS lightcurve classification
- Combined with K-means clustering (k=5) on BLS features
- 2.1M parameters, trained on 655 windows
- Achieved AUC 0.6947 on real TESS data
- Experience with sequence modeling directly applicable to temporal RL problems

### Mathematical Background

From CS 2300 and general coursework:
- Probability/expected values (MDPs)
- Linear algebra (neural network foundations)
- Python programming (all implementations)

---

## Grad vs Undergrad Requirements

| Aspect | Undergrad (4885) | Grad (5080) |
|--------|------------------|-------------|
| Project work | Alone or pairs | Must work alone |
| Proposal refs | 5+ | 10+ |
| Paper quality | Standard | Higher quality, deeper understanding |
| Homework | Standard | May have additional problems |

## Project Ideas

**Requirement:** "Projects are expected to use modern or deep reinforcement learning"

**"Modern" RL (2024-2025):**
- Foundational: DQN, Double DQN, Dueling DQN, PPO, A3C, SAC
- Cutting-edge: Model-based (Dreamer), Transformer-based, Multi-Agent RL, RLHF

**Strong candidate: Extend Minesweeper ML project**
- Already have working codebase with DQN/Double DQN/Dueling DQN
- Add PPO or SAC to compare value-based vs policy-gradient methods
- Compare policy gradient methods vs. value-based methods
- Write-up already familiar with problem domain

**Past student examples:** See `Course_Materials/101-StudentPapers2021Handout.pdf` and `102-RLClass2024StudentWork.zip`

---

## Assignments

| Assignment        | Due      | Status    | File                                        |
| ----------------- | -------- | --------- | ------------------------------------------- |
| Short Questions 1 | 1/27     | Submitted | `Homework/Submissions/ShortQuestions1_Answers.pdf` |
| Short Questions 2 | 2/3      | Submitted | `Homework/Submissions/ShortQuestions2_Answers.pdf` |
| HW1 Monte Carlo   | 2/13     | Complete  | `Homework/HW1_MonteCarlo/` + `Homework/Source/HW1_MonteCarlo_Paper.pdf` |

### HW1: Monte Carlo Maze Solver

**Implementation:** Monte Carlo ES (Exploring Starts) from scratch, following Sutton & Barto p.99.

| File                 | Purpose                                          |
| -------------------- | ------------------------------------------------ |
| `maze.py`            | 5×5 grid environment (start (1,1), goal (5,5))   |
| `monte_carlo.py`     | First-visit MC-ES with Q-table and greedy policy |
| `maze_generator.py`  | Random maze generator with BFS path verification |
| `visualize.py`       | Matplotlib policy/value visualization            |
| `main.py`            | Experiment runner for all HW questions           |
| `generate_figures.py`| Publication-quality figure generation            |

**Results:**
- 100% success rate across all configurations
- 8 steps optimal path on HW1 maze
- Converges within ~1,500 episodes
- Scales to 7×7 (12 steps) and 10×10 (18 steps) mazes

**Paper:** `Homework/Source/HW1_MonteCarlo_Paper.pdf` (4 pages, AAAI format)

---

## Semester Project

**Title:** How Accurate Are Learned World Models? An Empirical Analysis of DreamerV3's Imagination

**Research Question:** Does imagination accuracy correlate with task performance in model-based RL?

**Approach:**
1. Train DreamerV3 on 3 Atari games (Breakout, Pong, Seaquest)
2. Extract imagined trajectories at multiple training checkpoints
3. Measure prediction accuracy (SSIM, MSE, perceptual metrics)
4. Correlate imagination quality with game performance
5. Analyze which game elements are well-modeled vs poorly-modeled

**Key Insight:** DreamerV3 trains policies entirely in "imagination" - if the world model is inaccurate, how does the agent still succeed?

**Deliverables:**

| Deliverable           | Due          | Weight | Status     |
| --------------------- | ------------ | ------ | ---------- |
| Proposal presentation | Feb 17-19    | 2.5%   | Ready      |
| Proposal paper        | Feb 19       | 5%     | Complete   |
| Midterm presentation  | Mar 31-Apr 1 | 2.5%   |            |
| Midterm paper         | Apr 1        | 7.5%   |            |
| Midterm demo          | Week of Apr 1| 2.5%   |            |
| Final presentation    | May 7/12     | 5%     |            |
| Final paper           | May 12       | 17.5%  |            |
| Final demo            | Week of May 12| 7.5%  |            |

**References:** 14 papers in `Project/Proposal/references.bib` including:
- Hafner et al. (2023) - DreamerV3
- Hafner et al. (2020, 2021) - DreamerV1, DreamerV2
- Ha & Schmidhuber (2018) - World Models
- Wang et al. (2004) - SSIM metric

**Proposal:** `Project/Proposal/proposal.pdf` (3 pages, AAAI format)

---

## Folder Structure

```
CS5080_Reinforcement_Learning/
├── Course_Materials/          # Syllabus, lectures, handouts
│   ├── CS5080_Lecture_*.md    # Transcribed lecture notes
│   └── *.pdf                  # Original lecture PDFs
├── Homework/
│   ├── Assignments/           # Assignment PDFs (questions)
│   ├── Submissions/           # Submitted answer PDFs
│   ├── Source/                # LaTeX source files
│   │   ├── figures/           # Generated figures for papers
│   │   ├── aaai24.sty/bst     # AAAI style files
│   │   └── *.tex              # Paper source files
│   └── HW1_MonteCarlo/        # HW1 Python implementation
├── LaTeX/                     # Shared AAAI style files
├── Project/
│   ├── Proposal/              # DreamerV3 proposal (due Feb 19)
│   ├── Midterm/               # Midterm paper (due Apr 1)
│   └── Final/                 # Final paper (due May 12)
├── References/                # Textbooks and supplemental PDFs
└── CS5080_Reference.md        # This file
```

### References/ (Textbooks)
- `Sutton_Barto_RL_Introduction_2nd_Ed.pdf` (70MB) - Required textbook
- `Spinning_Up_Deep_RL_OpenAI.pdf` (1.2MB) - OpenAI tutorial
- `Goodfellow_Deep_Learning.pdf` (16MB) - Deep learning background
- `Grokking_Deep_RL_Chapter_01.docx` - Introduction to DRL
- `Grokking_Deep_RL_Chapter_02.docx` - Mathematical foundations (MDPs)
- `Grokking_Deep_RL_Chapter_03.docx` - Balancing immediate and long-term goals
- `Grokking_Deep_RL_Chapter_04.docx` - Balancing evaluation and control
- `Grokking_Deep_RL_Chapter_05.docx` - Evaluating agent behaviors
- `Grokking_Deep_RL_Chapter_06.docx` - Improving agent behaviors
- `Grokking_Deep_RL_Chapter_07.docx` - Achieving goals more effectively
- `Grokking_Deep_RL_Chapter_08.docx` - Introduction to value-based deep RL
- `Grokking_Deep_RL_Chapter_09.docx` - More stable value-based methods
- `Grokking_Deep_RL_Chapter_10.docx` - Policy gradient and actor-critic methods
- `Grokking_Deep_RL_Chapter_11.docx` - Advanced actor-critic methods
- `Grokking_Deep_RL_Chapter_12.docx` - Toward curiosity-driven agents
- `Grokking_Deep_RL_Chapter_13.docx` - Achieving human-level performance

### LaTeX/
- `aaai24.sty` - AAAI 2024 LaTeX style file
- `aaai24.bst` - AAAI 2024 BibTeX style file

### Homework/
- `references.bib` - Shared BibTeX citations for all assignments

---

## LaTeX Compilation

MiKTeX installed at: `C:\Users\manch\AppData\Local\Programs\MiKTeX\miktex\bin\x64\`

**Compile from Homework folder (uses LaTeX/ for style files):**
```bash
cd "/mnt/d/OneDrive/Desktop/CS5080_Reinforcement_Learning/Homework"
export TEXINPUTS="../LaTeX:."
export BSTINPUTS="../LaTeX:."
pdflatex -interaction=nonstopmode file.tex
bibtex file
pdflatex -interaction=nonstopmode file.tex
pdflatex -interaction=nonstopmode file.tex
```

**One-liner:**
```bash
cd "/mnt/d/OneDrive/Desktop/CS5080_Reinforcement_Learning/Homework" && TEXINPUTS="../LaTeX:." BSTINPUTS="../LaTeX:." pdflatex -interaction=nonstopmode file.tex && bibtex file && TEXINPUTS="../LaTeX:." pdflatex -interaction=nonstopmode file.tex && TEXINPUTS="../LaTeX:." pdflatex -interaction=nonstopmode file.tex
```

**Notes:**
- AAAI style automatically sets `\bibliographystyle{aaai24}`, so don't add a separate `\bibliographystyle` command
- `hyperref` package is NOT allowed with AAAI style
- `TEXINPUTS` tells LaTeX where to find .sty files; `BSTINPUTS` tells BibTeX where to find .bst files

---

## Publication Venues (for references)

From syllabus + Google Scholar H5-index (2025):

| Venue | Type | H5-Index |
|-------|------|----------|
| ICML | Conference | 272 |
| AAAI | Conference | 232 |
| IEEE Trans. Neural Networks & Learning Systems | Journal | 165 |
| IJCAI | Conference | 136 |
| JMLR | Journal | 130 |

---

## Notes

*Add class notes and observations here*

---
*Created: 2026-01-24*
*Updated: 2026-02-05*
