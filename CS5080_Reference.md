# CS 5080 - Reinforcement Learning

## Course Info

| | |
|---|---|
| **Instructor** | Dr. Jugal Kalita (jkalita@uccs.edu, 255-3432) |
| **Class** | Tu/Th 4:45-6:00pm, ENG 109 |
| **Office Hours** | Tu/Th 6:05-7:00pm, Wed 4:00-5:00pm, or by appointment (Osborne A-340) |

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

| Assignment        | Due  | Status    | Grade     | File                                                                    |
| ----------------- | ---- | --------- | --------- | ----------------------------------------------------------------------- |
| Short Questions 1 | 1/27 | Submitted | **10/10** | `Homework/Submissions/ShortQuestions1_Answers.pdf`                       |
| Short Questions 2 | 2/3  | Graded    | **10/10** | `Homework/Submissions/ShortQuestions2_Answers.pdf`                       |
| Short Questions 5 | 2/24 | Submitted |           | `Homework/Submissions/ShortQuestions5_Answers.pdf`                       |
| Short Questions 6 | 3/3  | Graded    | **10/10** | `Homework/Submissions/ShortQuestions6_Answers.pdf`                       |
| Short Questions 7 | 3/10 | Graded    | **10/10** | `Homework/Submissions/ShortQuestions7_Answers.pdf`                       |
| Short Questions 8 | 3/17 | Ready     |           | `Homework/Submissions/ShortQuestions8_Answers.pdf`                       |
| HW1 Monte Carlo   | 2/26 | Submitted |           | `Homework/HW1_MonteCarlo/` + `Homework/Source/HW1_MonteCarlo_Paper.pdf` |
| HW1 Demo          | 3/5  | Pending   |           | Demo during office hours (Tu/Th 6:05pm or Wed 4pm)                      |
| HW2 DQN           | 4/9  | Not Started |         | `Homework/Assignments/702-hw2.pdf`; **code at `D:\CS5080_HW2_QLearning\`** (not OneDrive) |

### HW2: Parking Lot DQN Agent

**Assignment:** Implement RL agent from scratch (no RL libraries) to navigate a car from entrance to any of 8 parking spots.

**Key clarifications from Lecture 15 (March 17):**
- Grid is **7x6** (Kalita corrected in class — assignment PDF says 6x7)
- Can modify dimensions per question 1(a)
- Use SOFA principles (CS 4300)
- **Demo:** Kalita will specify which parking spot during demo — agent must handle all 8

**Environment:**
- 7x6 grid, 8 parking spots (P1-P8), entrance (E) at bottom-right
- Barrier (hatched) between two rows of parking — no movement through it
- Car moves: forward, backward, sideways (no diagonal/angular steering)
- Cannot drive over parking spots en route to goal

**Requirements:**

| Part | Task                                                                                          |
| ---- | --------------------------------------------------------------------------------------------- |
| 1a   | Environment design: state/action representation, barriers, edge handling                      |
| 1b   | Implement Q-learning AND Double Q-learning, pseudocode with explanation                       |
| 2a   | DQN architecture: NN inputs/outputs, replay memory design and sizing                         |
| 2b   | Implement DQN, run multiple times, vary parameters, show policies, graphs/tables              |
| 2c   | Enhancement: make problem harder (bigger grid, obstacles, etc.) and adapt                     |
| 3    | 4-page paper (AAAI format) + 1 page refs                                                     |
| 4    | Extra credit for substantial additional work                                                  |

**Relevant lecture notes:**
- **Lecture 10:** Double Q-Learning pseudocode (S&B Eq. 6.10) — two Q-tables, one selects action, other evaluates, reduces maximization bias
- **Lecture 11:** Parking lot as example of traditional RL limitations (discrete states, limited view)
- **Lecture 15:** DQN algorithm (Mnih et al.), replay memory, Q-table→Q-network mapping, HW2 clarifications

**Code location (desktop, 2026-04-12):**
- **Path:** `D:\CS5080_HW2_QLearning\` (moved off OneDrive because the venv is ~720 MB)
- **Originally created on:** Surface Pro
- **Transferred via:** external SSD E: (Windows Backup, 1 TB)
- **Contents:** `demo.py`, `dqn.py`, `parking_lot.py`, `q_learning.py`, `models/` (trained checkpoints: Q, DQ, DQN at grid sizes 7x6, 10x12, 14x16, 18x20), `venv/` (Windows venv from Surface — may need regeneration on this machine)
- **Total size:** 734 MB (719 MB venv, 16 MB models, ~50 KB source)

**No code submission required** — written answers in the paper are sufficient. Demo required the week of April 9.

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

**Research Question:** Does DreamerV3 need accurate imagination to learn effective policies, or can it succeed even when its world model is wrong?

**Hypotheses:**
- **H1:** Imagination accuracy improves monotonically as training progresses
- **H2:** Better imagination leads to better play (accuracy correlates with performance)
- **H3:** Semantic accuracy (game state) matters more than pixel-level accuracy (MSE/SSIM)

**Approach:**
1. Train DreamerV3 on Snake (Gymnasium-compatible, 64×64 RGB, 10×10 grid)
2. Extract imagined trajectories at 20 checkpoints (every 50K steps over 1M total)
3. Measure frame-level accuracy (MSE, SSIM) and semantic accuracy (head position, body segments, food location, collision prediction)
4. Correlate imagination quality with game performance across checkpoints
5. Analyze which semantic metrics are most predictive of performance

**Key Insight:** DreamerV3 trains policies entirely in "imagination" - if the world model is inaccurate, how does the agent still succeed? Snake's simple, deterministic dynamics make imagination errors precisely measurable.

**Deliverables:**

| Deliverable           | Due          | Weight | Status     |
| --------------------- | ------------ | ------ | ---------- |
| Proposal presentation | Feb 17-19    | 2.5%   | Submitted  |
| Proposal paper        | Feb 19       | 5%     | Resubmitted Feb 20 (hypotheses), resubmitted Feb 24 (removed first-person) |
| Midterm presentation  | Mar 31 (Tue) | 2.5%   | Complete (`Project/Midterm/DreamerV3_Snake_Midterm_Presentation_v2.pptx`, 12 slides + talking paper). Submit slides on Canvas before 12:00 on presentation day. Josh presents Tue 3/31, other students Thu 4/2. |
| Midterm paper         | Apr 2 (Canvas)| 7.5%  | Complete (`Project/Midterm/midterm.pdf`, 4 pages, interim report). Canvas date may be extended (Kalita extended proposal paper deadline previously). |
| Midterm demo          | Week of Apr 1| 2.5%   | Deferred — will ask Kalita on Tue 3/31 about expected format |
| Final presentation    | May 7/12     | 5%     |            |
| Final paper           | May 12       | 17.5%  |            |
| Final demo            | Week of May 12| 7.5%  |            |

**References:** 12 papers in `Project/Proposal/references.bib` including:
- Hafner et al. (2025) - DreamerV3 (Nature)
- Hafner et al. (2020, 2021) - DreamerV1, DreamerV2
- Hafner et al. (2019) - PlaNet (RSSM)
- Ha & Schmidhuber (2018) - World Models
- Sutton & Barto (2018) - RL textbook

**Proposal:** `Project/Proposal/proposal.pdf` (4 pages, AAAI format)
**Presentation:** `Project/Proposal/DreamerV3_Snake_Proposal_Presentation.pptx`
**Talking Paper:** `Project/Proposal/Talking_Paper.md`

### Midterm Demo Plan (Kalita's Office, Week of Apr 1)

Demo should tell the research story, not just show an agent playing:

1. **Agent plays Snake** — quick visual proof the system works (just setup)
2. **Imagination vs reality** — real frames next to imagined frames from early and late checkpoints, showing world model improving over training (H1)
3. **Performance tied to imagination** — correlation figures showing better imagination = higher scores (H2)
4. **Head position is what matters** — head error is strongest predictor, outperforms pixel metrics like MSE/SSIM. World model doesn't need sharp images, just correct head position (H3)

### Paper Writing Rules (from Proposal Feedback, A-)

**These apply to ALL project papers (midterm, final):**

1. **No first-person singular.** No "I", "my", "me" anywhere. Use passive voice ("is evaluated", "the analysis correlates"), "this work", "the approach", "the author" instead. Professor highlighted every single instance and offered resubmission specifically for this.
2. **Make contributions general.** Frame contributions as methodology/findings applicable to the broader field, not just "we did X on Y." Think: what does this tell the community?
3. **Justify gaps in related work.** When identifying that something is unexplored, add a sentence explaining *why it matters* — don't just state the gap exists.
4. **AI Usage Statement:** Use "the author's" not "my."

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

### Course_Materials/ (Handouts)

| File                                      | Content                                                          |
| ----------------------------------------- | ---------------------------------------------------------------- |
| `001-syllabus..pdf`                       | Course syllabus                                                  |
| `002-RubricForPresentations.pdf`          | Presentation grading rubric                                      |
| `101-StudentPapers2021Handout.pdf`        | Example student papers from 2021                                 |
| `102-RLClass2024StudentWork`              | Student work from 2024 class                                     |
| `103-ExamplesoReinforcementLearning.docx` | RL examples document                                             |
| `105-CommentsOnProposals.pdf`             | Professor's feedback on proposal papers (writing/format rules)   |
| `301-MonteCarloAlgorithms.pdf`            | Monte Carlo algorithm boxes                                      |
| `303-WeightedImportanceSamplingFormula.pdf`| Handwritten derivation of S&B eq 5.8 (weighted IS incremental)  |
| `304-TDLearning.pptx`                    | TD Learning lecture slides                                       |
| `305-TD0Algorithms.pdf`                  | Algorithm boxes: TD(0), SARSA, Q-Learning, Double Q-Learning     |
| `CS5080_Lecture_*.md`                     | Transcribed lecture notes                                        |
| `CS_5080_Lecture_*.pdf`                   | Original handwritten lecture PDFs                                |

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

| Venue                                          | Type       | H5-Index |
| ---------------------------------------------- | ---------- | -------- |
| Nature                                         | Journal    | 490      |
| NeurIPS                                        | Conference | 371      |
| ICLR                                           | Conference | 362      |
| ICML                                           | Conference | 272      |
| AAAI                                           | Conference | 232      |
| IEEE Trans. Neural Networks & Learning Systems | Journal    | 165      |
| IJCAI                                          | Conference | 136      |
| JMLR                                           | Journal    | 130      |

---

## DreamerV3 Training (Linux/WSL)

### Project Location

| Item                | Path                                                                            |
| ------------------- | ------------------------------------------------------------------------------- |
| **Refactored root** | `/home/josh/cs5080_project_refactored/` **(use this for all work)**             |
| **Original root**   | `/home/josh/cs5080_project/` **(DO NOT MODIFY — preserved as baseline)**        |
| **Python venv**     | `/home/josh/cs5080_project/.venv/` (symlinked from refactored)                  |
| **DreamerV3 code**  | `/home/josh/cs5080_project/dreamerv3-torch/` (symlinked from refactored)        |
| **Snake env**       | `/home/josh/cs5080_project_refactored/snake_env/`                               |
| **Training output** | `/home/josh/cs5080_project/training_output/` (symlinked from refactored)        |
| **Analysis output** | `/home/josh/cs5080_project/analysis_output*/` (symlinked from refactored)       |
| **Config**          | `/home/josh/cs5080_project/dreamerv3-torch/configs.yaml` (section: `snake`)     |
| **GitHub**          | https://github.com/manchesterjm/CS5080_Semester_Project (`original/` + `refactored/`) |

**Refactored code (2026-03-29):** All project code was refactored for submission quality. The refactored project is behavior-identical to the original (verified byte-level identical frames and rewards across multiple seeds). DreamerV3's snake adapter (`dreamerv3-torch/envs/snake.py`) now imports from the refactored project.

**Symlinks in refactored project:**
- `.venv` → original venv
- `dreamerv3-torch` → original DreamerV3 code
- `training_output` → original training data
- `analysis_output*` → original analysis results

### Snake Config (key settings)

| Setting         | Value      | Notes                                  |
| --------------- | ---------- | -------------------------------------- |
| `envs`          | 10         | Parallel environments                  |
| `batch_size`    | 32         | Sequences per gradient update (default 16) |
| `train_ratio`   | 512        | Gradient updates per env step (default 512) |
| `steps`         | 2e6        | Default total env steps                |
| `eval_every`    | 1e4        | Checkpoint + eval every 10K steps      |
| `time_limit`    | 2000       | Max steps per episode                  |
| `discount`      | 0.999      | High discount for long-horizon         |
| `size`          | [64, 64]   | RGB observation size                   |
| `action_repeat` | 1          | No frame skipping                      |
| `actor.dist`    | onehot     | Discrete actions (4 directions)        |
| `imag_gradient` | reinforce  | Policy gradient in imagination         |

### Snake Env Reward Structure

Built into `snake_env/snake_env.py`:
- **+1** eat food
- **-1** collision (death)
- **-0.002** per step (idle penalty to prevent safe looping)

### Training Runs

| Directory        | Agent | Steps Target | Steps Completed | Status             |
| ---------------- | ----- | ------------ | --------------- | ------------------ |
| `snake_1M`       | 2     | 1M           | ~1.1M           | Complete           |
| `snake_2M_long`  | 2     | 2M           | 2,015,965       | Complete           |

`snake_2M_long` extends Agent 2 training to 2M total steps. `snake_1M` was the initial training run.

### How to Run Training

**Start new training:**
```bash
cd /home/josh/cs5080_project_refactored
tmux new -s dreamer
source .venv/bin/activate
python3 dreamerv3-torch/dreamer.py --configs snake --logdir training_output/<run_name> --steps <N>
```

**Resume existing training** (auto-resumes from `latest.pt`):
```bash
cd /home/josh/cs5080_project_refactored
tmux new -s dreamer
source .venv/bin/activate
python3 dreamerv3-torch/dreamer.py --configs snake --logdir training_output/snake_2M_long
```

**One-liner (background in tmux):**
```bash
tmux new-session -d -s dreamer "cd /home/josh/cs5080_project_refactored && source .venv/bin/activate && python3 dreamerv3-torch/dreamer.py --configs snake --logdir training_output/snake_2M_long 2>&1 | tee -a training_output/snake_2M_long_resume.log"
```

**Override steps (e.g., Agent 3 at 500K):**
```bash
python3 dreamerv3-torch/dreamer.py --configs snake --logdir training_output/snake_500K_short --steps 500000
```

**Check training progress:**
```bash
tail -3 /home/josh/cs5080_project/training_output/snake_2M_long/metrics.jsonl
```

**Check GPU load:**
```bash
nvidia-smi
```

**Stop training:** `tmux kill-session -t dreamer`

**Attach to running training:** `tmux attach -t dreamer`

### Agent Plan

| Agent | Description                    | Steps  | Run Directory       | Status                                      |
| ----- | ------------------------------ | ------ | ------------------- | ------------------------------------------- |
| 2     | Step penalty (primary, 64x64)  | 2M     | `snake_2M_long`     | Complete (2,015,965 steps)                   |
| 2a    | Resolution test (32x32)        | 500K   | `snake_32x32`       | Complete (~530K steps, 54 ckpts, analyzed)   |
| 2b    | Resolution test (16x16)        | 500K   | `snake_16x16`       | Not started (wipe and restart)               |

### Resolution Scaling Experiment (H3 Direct Test)

**Motivation:** H3 (semantic > pixel accuracy) showed the right trend but lacked statistical power. Instead of testing indirectly via correlation, directly degrade visual fidelity and measure impact.

**Design:**
- Same game logic (10x10 grid, same rewards, same hyperparams)
- Only change: observation resolution (64x64 baseline, 32x32, 16x16)
- At 16x16, each grid cell is ~1.5 pixels — pixel-level metrics (MSE, SSIM) are inherently degraded
- If agent still performs well AND semantic metrics (body_accuracy, food_correct) hold up while pixel metrics degrade, that's direct evidence for H3

**Implementation (3 changes needed):**

| File                              | Change                                                              |
| --------------------------------- | ------------------------------------------------------------------- |
| `dreamerv3-torch/envs/snake.py`  | Use existing `size` parameter to resize observations via PIL/numpy  |
| `dreamerv3-torch/configs.yaml`   | Add `snake_32x32` and `snake_16x16` config sections                 |
| `snake_env/state_extractor.py`   | Handle 32x32 and 16x16 grids for semantic metric extraction        |

**Architecture auto-adjustment:** ConvEncoder calculates stages as `log2(h) - log2(minres)`:
- 64x64 → 4 stages (64→32→16→8→4), embedding 4096
- 32x32 → 3 stages (32→16→8→4), embedding 2048
- 16x16 → 2 stages (16→8→4), embedding 1024

**Config step counts:** Both `snake_32x32` and `snake_16x16` were mistakenly set to `5e5` (500K steps) in `configs.yaml` instead of `2e6`. Training completed at ~557K and ~514K respectively. Standard target is 2M. Config error discovered 2026-03-29.

**Estimated GPU time:** ~2 hours per run × 2 runs = ~4 hours total

**Analysis plan:** Run same analysis pipeline on all 3 resolutions, then cross-resolution comparison:
1. Performance vs resolution (does the agent degrade gracefully?)
2. Semantic metrics vs resolution (do they hold up better than pixel metrics?)
3. Within-resolution H2 test (does body_accuracy still predict performance at each resolution?)

### Code Quality (Refactored Project)

Refactored 2026-03-29 for submission. All metrics measured on `/home/josh/cs5080_project_refactored/`.

| Metric      | Score   | Details                                              |
| ----------- | ------- | ---------------------------------------------------- |
| **Pylint**  | 10.0/10 | All 12 source files                                  |
| **Tests**   | 121     | 58 original + 63 new                                 |
| **Coverage**| 88%     | Target was 80%+                                      |
| **SOFA**    | Pass    | All functions ≤20 lines, few arguments               |

**New files added in refactor:**

| File           | Purpose                                                  |
| -------------- | -------------------------------------------------------- |
| `constants.py` | All magic numbers centralized (rewards, grid, metrics, display, thresholds) |
| `shared.py`    | Deduplicated utilities (`upscale()`, `load_config()`, `NumpyEncoder`)       |
| `pyproject.toml`| pytest and pylint configuration                         |

**New test files:**

| File                       | Tests | What's Tested                                        |
| -------------------------- | ----- | ---------------------------------------------------- |
| `tests/test_constants.py`  | 8     | Constant values, relationships, completeness         |
| `tests/test_shared.py`     | 6     | upscale, NumpyEncoder                                |
| `tests/test_analyze.py`    | 20    | H1/H2/H3 functions, data loading, matching, pipeline |
| `tests/test_compute_metrics.py` | 12 | Frame/semantic metrics, aggregation, load/save      |
| `tests/test_imagine.py`    | 6     | Episode loading, tensor building                     |
| `tests/test_run_analysis.py`| 6    | Path discovery, checkpoint detection                 |

**Behavioral identity verified:** Byte-identical frames and rewards between original and refactored code across multiple seeds and action sequences (3 seeds × 10 actions each, all 33 frames and 30 rewards matched exactly).

### Project File Inventory

**Custom code (refactored project):**

| File                                     | Purpose                                                    |
| ---------------------------------------- | ---------------------------------------------------------- |
| `constants.py`                           | Centralized constants (no magic numbers)                   |
| `shared.py`                              | Deduplicated utilities (upscale, config, JSON encoder)     |
| `snake_env/snake_game.py`                | Core Snake game logic (grid, movement, collision, food)    |
| `snake_env/snake_env.py`                 | Gymnasium wrapper, 64x64 RGB rendering, reward structure   |
| `snake_env/state_extractor.py`           | Parse RGB frames → head/body/food positions + `compare()`  |
| `dreamerv3-torch/envs/snake.py`          | DreamerV3 adapter (wraps SnakeEnv into DreamerV3 format)   |
| `imagine.py`                             | Extract imagined trajectories from checkpoints             |
| `compute_metrics.py`                     | Compute MSE, SSIM, head_error, body_accuracy, food_correct |
| `analyze.py`                             | Cross-checkpoint hypothesis testing + figures              |
| `run_analysis.py`                        | Orchestration (runs all 3 stages, skips existing)          |
| `scripts/demo.py`                        | Run trained agent visually                                 |
| `scripts/play_trained.py`                | Play back a trained agent                                  |
| `tests/` (10 files)                      | 121 tests, 88% coverage                                   |
| `TRAINING_PLAN.md`                       | Model history (1-3), hyperparameters, way ahead (4-7)      |
| `trained_agent.mp4`                      | Video of trained agent playing                             |

**Training data (`training_output/snake_1M/`):**

| Item               | Count/Details                                               |
| ------------------ | ----------------------------------------------------------- |
| Checkpoints (`.pt`) | 21 files (55K through ~1.1M steps, every ~50K)             |
| Eval episodes       | 252 `.npz` files (observations, actions, rewards per episode) |
| Train episodes      | 4,204 `.npz` files                                         |
| Metrics log         | `metrics.jsonl` — 4,366 entries with step, return, length  |

**State extractor capabilities:**
- Handles both crisp real frames (exact color match) and blurry imagined frames (nearest-neighbor with distance threshold)
- `compare()` returns: head_error (Euclidean), body_accuracy (fraction correct), food_correct (bool)
- Colors: background=black, head=bright green, body=dark green, food=red

### Hypothesis Data Requirements

**H1 (imagination improves over training):**

| Needed                               | Status                        |
| ------------------------------------ | ----------------------------- |
| Checkpoints at regular intervals     | DONE (21 checkpoints)         |
| Imagination trajectory extraction    | DONE (`imagine.py`)           |
| Frame-level metrics (MSE, SSIM)      | DONE (`compute_metrics.py`)   |
| Semantic metrics via state_extractor | DONE (`compute_metrics.py`)   |

**H2 (accuracy correlates with performance):**

| Needed                         | Status                      |
| ------------------------------ | --------------------------- |
| Game performance per checkpoint | DONE (from `metrics.jsonl`) |
| Accuracy metrics per checkpoint | DONE (20 checkpoints)      |
| Correlation analysis script    | DONE (`analyze.py`)         |

**H3 (semantic > pixel accuracy):**

| Needed                          | Status                              |
| ------------------------------- | ----------------------------------- |
| MSE/SSIM per checkpoint         | DONE                                |
| Semantic metrics per checkpoint | DONE                                |
| Comparison analysis             | DONE (Fisher's z-test in `analyze.py`) |

### Analysis Pipeline (Built 2026-02-24)

| Script              | Purpose                                                |
| ------------------- | ------------------------------------------------------ |
| `imagine.py`        | Extract imagined trajectories from checkpoints         |
| `compute_metrics.py`| Compute MSE, SSIM, head_error, body_accuracy, food_correct |
| `analyze.py`        | Cross-checkpoint hypothesis testing + figures          |
| `run_analysis.py`   | Orchestration (runs all 3 stages, skips existing)      |

**Multi-resolution support (fixed 2026-03-02):**
- `imagine.py` accepts `config_name` parameter to load correct architecture per resolution
- `run_analysis.py` accepts `--config` flag (defaults to `--run` value)
- Example: `python3 run_analysis.py --run snake_32x32 --config snake_32x32`

**Output:** `analysis_output_<run_name>/` — checkpoint dirs (each with `imagination_data.npz` + `metrics.json`), plus `summary_table.csv`, `hypothesis_results.json`, and `figures/`.

**Key Results (Model 2, 1.1M steps — `analysis_output/`):**
- H1: 4/5 metrics significantly improving (head_error ρ = -0.85, p < 0.001)
- H2: 4/5 metrics significantly correlated with performance (head_error ρ = -0.84, p < 0.001)
- H3: Head error is best single predictor (|ρ| = 0.843), but avg semantic ≈ avg pixel

**Key Results (Model 2, 2M steps — `analysis_output_snake_2M_long/`):**
- H1: NOT significant. Imagination quality plateaus after ~500K steps; trends correct but noisy.
- H2: **body_accuracy significantly correlated** (Spearman ρ=0.511 p=0.021, Pearson r=0.569 p=0.009)
- H3: Semantic avg |ρ|=0.255 vs pixel |ρ|=0.110, but Fisher z-test p=0.196 (not significant)
- Early vs late t-test (first 5 vs last 5 checkpoints): body_accuracy improved (0.835→0.869) but p=0.189
- **Takeaway:** World model learns visuals quickly then plateaus; policy learning is the slow part. body_accuracy predicts performance but more power needed for H1/H3.

**Key Results (Agent 2a, 32x32, ~530K steps — `analysis_output_snake_32x32/`):**
- H1: **5/5 metrics improving, 5/5 significant** (SSIM ρ=+0.831, food_correct ρ=+0.821, MSE ρ=-0.783)
- H2: **5/5 metrics significantly correlated** (SSIM ρ=+0.660, food_correct ρ=+0.654, MSE ρ=-0.603)
- H3: Pixel avg |ρ|=0.631 vs semantic avg |ρ|=0.486, Fisher z p=0.96 (not significant, pixel slightly better)
- Early vs late t-test: MSE improved (0.0040→0.0012, p=0.021*), food_correct improved (0.114→0.797, p<0.0001*)
- Final eval return: ~7-14 (agent early in learning on 1024-cell board)
- Final metrics: SSIM 0.985, body_accuracy 0.98, food_correct 0.84

**Cross-Resolution Comparison (10x10 vs 32x32):**

| Metric                      | 10x10 (2M steps) | 32x32 (530K steps) |
| --------------------------- | ----------------- | ------------------- |
| H1 significant              | 0/5               | 5/5                 |
| H2 significant              | 1/5               | 5/5                 |
| H3 direction                | Semantic slightly  | Pixel slightly      |
| Final eval return           | ~35-54            | ~7-14               |
| Final SSIM                  | ~0.974            | ~0.985              |
| Final food_correct          | ~0.27             | ~0.84               |
| Final body_accuracy         | ~0.89             | ~0.98               |

**Interpretation:**
- All hypotheses dramatically stronger on 32x32 — the agent is still actively improving, so imagination-performance coupling is strong
- 32x32 model imagines more accurately because board states are simpler (short snake, lots of empty space)
- H3 direction flipped: on easy boards pixel metrics suffice; semantic advantage may only emerge with complex/crowded states
- 10x10 agent already plateaued on imagination quality, making trends noisy and non-significant

---

## Notes

*Add class notes and observations here*

---
*Created: 2026-01-24*
*Updated: 2026-03-29*
