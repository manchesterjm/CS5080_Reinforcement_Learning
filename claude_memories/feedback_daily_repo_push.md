---
name: Daily GitHub Repo Push
description: Push CS5080 and CS5610 repos to GitHub at start of every session
type: feedback
---

Push both course repos to GitHub daily as part of session startup.

**Repos:**
- `D:\OneDrive\Desktop\CS5080_Reinforcement_Learning\` → github.com/manchesterjm/CS5080_Reinforcement_Learning
- `D:\OneDrive\Desktop\CS5610_Applied_Convex_Optimization\` → github.com/manchesterjm/CS5610_Applied_Convex_Optimization

**Why:** Repos went 2+ months without a push (last push Feb 5 / Jan 28). User wants all local work backed up to GitHub daily.

**How to apply:** At session start, after reading core files, check both repos for uncommitted changes. If any exist, `git add -A`, commit with a summary of what changed, and push. The user wants everything on the computer in the repo — don't skip files. Only things in the repo but not on the computer should stay that way (don't pull down repo-only content).
