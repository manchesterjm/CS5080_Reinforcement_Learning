---
name: Training workflow rules
description: When user says "start training", resume the current incomplete model — don't skip ahead. A model is not done until it reaches its target step count (e.g., 2M). No early kills.
type: feedback
---

When user says "start training," start/resume whatever model is currently in progress — don't skip to the next one.

**Why:** User wants all models run to completion (2M steps) regardless of interim performance. Killing a run early or skipping ahead is not the user's call to make — only theirs.

**How to apply:** Check `TRAINING_PLAN.md` for the current model. If its last checkpoint is below the target step count, resume it. Only move to the next model after the current one finishes its full run.
