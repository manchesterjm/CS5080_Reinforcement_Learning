# Claude Code Instructions - CS 5080 Reinforcement Learning

## Important: AI Policy Restrictions

**This course has strict AI usage policies:**
- AI is ONLY allowed for grammar and spell-checking
- AI is NOT allowed for content generation
- Quoted content (including from AI) must be under 5% of paper and attributed
- Violating this policy results in F on deliverable or F in class

**When working on this repo:**
- DO NOT generate paper content, answers, or solutions
- DO help with: LaTeX formatting, bibliography management, code debugging, file organization
- Always remind the user about AI policy when they ask for content help

---

## Project Overview

**Student:** Josh Manchester (jmanches@uccs.edu)
**Course:** CS 5080 - Reinforcement Learning, Spring 2026, UCCS
**Instructor:** Dr. Jugal Kalita

**Semester Project:** "Can Modern Deep RL Overcome Sparse Rewards? A Case Study in Minesweeper"
- Comparing DQN variants (baseline, Beta-DQN, Elastic Step DQN) with policy gradient methods (PPO, SAC)
- Related codebase: https://github.com/manchesterjm/minesweeper_ml_project

---

## Folder Structure

```
CS5080_Reinforcement_Learning/
├── CLAUDE.md              # This file - Claude Code instructions
├── README.md              # Project overview
├── CS5080_Reference.md    # Detailed course reference (deadlines, grading, topics)
├── Course_Materials/      # Professor-provided materials (syllabus, lectures, past student work)
├── Homework/              # Assignment submissions
│   ├── *.tex              # LaTeX source files
│   ├── *.pdf              # Compiled submissions
│   └── references.bib     # Shared bibliography
├── LaTeX/                 # AAAI style files
│   ├── aaai24.sty         # AAAI 2024 LaTeX style
│   └── aaai24.bst         # AAAI 2024 BibTeX style
├── Project/               # Semester project
│   ├── Proposal/          # Due Feb 19
│   │   ├── Papers/        # Reference PDFs for citations
│   │   ├── proposal.tex   # Proposal LaTeX source
│   │   └── references.bib # Proposal bibliography
│   ├── Midterm/           # Due Apr 1
│   └── Final/             # Due May 12
└── References/            # Textbooks (PDFs)
    ├── Sutton_Barto_RL_Introduction_2nd_Ed.pdf  # Primary textbook (70MB)
    ├── Spinning_Up_Deep_RL_OpenAI.pdf           # OpenAI tutorial
    ├── Goodfellow_Deep_Learning.pdf             # DL background
    └── Grokking_Deep_RL_Chapter_*.docx          # Supplemental chapters
```

---

## Key Deadlines

| Deliverable | Due Date | Weight |
|-------------|----------|--------|
| Proposal Presentation | Feb 17-19, 2026 | 2.5% |
| Proposal Paper | Feb 19, 2026 | 5% |
| Midterm Presentation | Mar 31-Apr 1, 2026 | 2.5% |
| Midterm Paper | Apr 1, 2026 | 7.5% |
| Midterm Demo | Week of Apr 1, 2026 | 2.5% |
| Final Presentation | May 7/12, 2026 | 5% |
| Final Paper | May 12, 2026 | 17.5% |
| Final Demo | Week of May 12, 2026 | 7.5% |

---

## LaTeX Compilation

All papers must use AAAI format. Style files are in `LaTeX/`.

**To compile a .tex file:**
```bash
# Set paths to find style files
export TEXINPUTS="../LaTeX:."
export BSTINPUTS="../LaTeX:."

# Compile (run pdflatex 3x for references)
pdflatex -interaction=nonstopmode filename.tex
bibtex filename
pdflatex -interaction=nonstopmode filename.tex
pdflatex -interaction=nonstopmode filename.tex
```

**Notes:**
- AAAI style auto-sets `\bibliographystyle{aaai24}` - don't add another
- `hyperref` package is NOT compatible with AAAI style
- Papers must have: title, abstract, intro, conclusions, references

---

## Reference Materials

### Textbooks in References/
| File | Description |
|------|-------------|
| `Sutton_Barto_RL_Introduction_2nd_Ed.pdf` | Primary textbook - RL fundamentals |
| `Spinning_Up_Deep_RL_OpenAI.pdf` | Practical deep RL guide |
| `Goodfellow_Deep_Learning.pdf` | Neural network foundations |
| `Grokking_Deep_RL_Chapter_*.docx` | Accessible worked examples |

### Project Reference Papers in Project/Proposal/Papers/
Key papers for the semester project on Minesweeper RL:
- Mnih2015_DQN_Nature.pdf - Original DQN paper
- VanHasselt2016_DoubleDQN.pdf - Double DQN
- Schulman2017_PPO.pdf - PPO algorithm
- Haarnoja2018_SAC.pdf - Soft Actor-Critic
- Zhang2025_BetaDQN_AAMAS.pdf - Beta-DQN exploration
- Han2024_ElasticDQN.pdf - Elastic Step DQN
- Narvekar2020_Curriculum_Survey.pdf - Curriculum learning survey

---

## What Claude CAN Help With

1. **LaTeX formatting** - Fix compilation errors, format tables, equations
2. **Bibliography management** - Format BibTeX entries, check citations
3. **Code debugging** - Help with Python/PyTorch for the related Minesweeper project
4. **File organization** - Suggest structure, find files
5. **Explaining concepts** - Clarify RL algorithms, math notation from textbooks
6. **Proofreading** - Grammar and spell-check (per AI policy)

## What Claude CANNOT Help With

1. **Writing paper content** - Violates AI policy
2. **Generating homework answers** - Academic integrity violation
3. **Creating presentation slides content** - Must be original work
4. **Summarizing papers for submission** - Content generation not allowed

---

## Related Repositories

- **Minesweeper ML Project:** https://github.com/manchesterjm/minesweeper_ml_project
  - Contains DQN implementation used as baseline for semester project
  - Conda environment: `minesweeper-gpu` (PyTorch 2.7.0, CUDA 12.8)
