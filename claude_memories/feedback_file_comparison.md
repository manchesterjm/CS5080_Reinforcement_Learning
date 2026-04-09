---
name: File comparison — hash then read
description: When comparing files for duplicates, use md5 hash first to detect differences, then read content to determine which version is which (e.g., blank assignment vs solutions)
type: feedback
---

When comparing files to determine if they are duplicates or different versions:
1. **First:** Use md5sum to quickly identify whether files are identical or different
2. **Then:** If different, read the actual content to determine what each version is (e.g., professor's blank template vs student's submitted solutions)
3. **Name accordingly:** Rename files based on their content role (e.g., `HW2_assignment.pdf` for the original, `HW2.pdf` for solutions)

**Why:** Identical filenames can contain completely different content (professor's template vs student's solutions). Hash comparison alone tells you they differ but not which is which. Content inspection is required to make the right decision about what to keep, rename, or delete.

**How to apply:** Any time files need to be moved, deduplicated, or organized — especially course materials downloaded from Canvas alongside solution files.
