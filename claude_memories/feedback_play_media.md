---
name: Play media files via Windows path
description: When opening files in Windows from WSL, use powershell.exe with Windows D:\ paths, not /mnt/d/ paths
type: feedback
---

When playing media or opening files in Windows from WSL, use `powershell.exe -c "Start-Process 'D:\path\to\file'"` with the **Windows path** (e.g., `D:\Music\...`).

**Why:** WSL `/mnt/d/` paths don't resolve when Windows tries to open them — the error dialog shows it literally looking for `/mnt/d/...` which doesn't exist on the Windows side.

**How to apply:** Any time you need to open a file in a Windows application from WSL (media, PDFs, etc.), translate the path to Windows format and use `powershell.exe Start-Process`. Escape single quotes in paths by doubling them (`''`).
