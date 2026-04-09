---
name: Session logs go to Session_Logs directory only
description: Never write session logs to the desktop — always use D:\Documents\Claude_References\Session_Logs\
type: feedback
---

Session log files must ONLY be written to `D:\Documents\Claude_References\Session_Logs\session_YYYY_MM_DD.md`. A previous session incorrectly wrote one to `C:\Users\manch\Desktop\`, creating a misplaced duplicate.

**Why:** User expects a clean desktop. Session logs on the desktop are clutter and indicate Claude didn't follow the CLAUDE.md instructions.

**How to apply:** When creating or updating session logs, always use the full path `D:\Documents\Claude_References\Session_Logs\`. Never use the desktop, home directory, or any other location.
