## CRITICAL: Current Year is 2026

When calculating Unix timestamps, dates, or any time-related values, the current year is **2026**, NOT 2025. This has caused repeated errors (e.g., Discord timestamps off by a year). Double-check all date math.

## CRITICAL: Never Overwrite Files

**Never compile over, save over, or overwrite an existing file.** Always preserve the original first:
1. Rename the original (e.g., `proposal.pdf` → `proposal_original.pdf`)
2. THEN compile/create the new version (e.g., `proposal_revised.pdf`)

This applies to PDFs, source files, configs, backups — everything. Losing data is unacceptable. If a revised version is needed, both versions must coexist.

## CS 5080 Project Location
- **All project code is on Linux:** `/home/josh/cs5080_project/`
- **DO NOT look at Windows paths** for this project. The OneDrive folder only has the proposal/paper LaTeX.
- Code, training data, checkpoints, venv — all on Linux side

## CS 5080 Lecture Notes
- Handwritten PDFs are the **user's own class notes**, not professor-provided materials
- When user writes "Claude" in their notes, it's a request to insert the referenced algorithm/content (e.g., from Sutton & Barto) because they couldn't copy it fast enough during lecture
- Transcribe to markdown in `Course_Materials/CS5080_Lecture_N.md`, copy original PDF there too

## "Can you" = Capability Question, Not a Request

When the user asks "can you do X" or similar, they're asking about **capabilities**, not requesting action. Answer whether it's possible, then stop. Only take action when explicitly told to do something (e.g., "grab the wallpapers" vs "can you grab the wallpapers?").

## Wallpaper Preferences
- **Location:** `D:\Pictures\Wallpapers\` — ultrawide 2560x1080
- **Strongly prefers:** Real photography over drawn/digital art
- **YES:** Landscapes (mountains, cliffs, coasts, lakes, forests), moody/atmospheric nature, dark minimalist abstracts (kept blue & gray glass renders, periodic table on black bg), sunset cityscapes with real architecture (Cordoba bridge)
- **NO:** Anime, video game art, people (unless Dune), drawn/digital paintings (rejected ALL Gracile art, all pixel art), bright/pastoral scenes, animal-focused photos
- **Color palette:** Dark, cool-toned — deep blues, teal, black, forest green, amber/orange accents
- **Mood:** Contemplative, vast scale, night/twilight/golden hour, long exposures, mist/fog
- **When cropping:** Center-crop to 21:9, resize with Lanczos to 2560x1080, save only cropped version (no originals)

## Session Logging: Log BEFORE and DURING, Not After

**Always log tasks to the session log (`Session_Logs\session_YYYY_MM_DD.md`) BEFORE starting work and UPDATE as progress is made.** Do not wait until a task is complete to log it. If a crash or disconnect happens mid-task, the log should already reflect what was in progress so context can be recovered.

## CS 5610 Exam 1 Results (March 4, 2026)

**Score:** 45/60 (75%) + 2 bonus points

**What went well:** T/F (18/20), quadratic convexity check (10/10), PSD shortcut ac≥b²

**What to review for Exam 2:**
- **Optimality diagrams:** If -∇f₀(x) points into feasible set interior → NOT optimal (missed j)
- **f(x)=x₁/x₂ is NOT linear** — linear means f(x)=Ax. This is a ratio; check Hessian (indefinite → neither convex nor concave)
- **Feasible set drawing:** Lost 6 pts on precision. Practice plotting constraint intersections
- **Minimizer identification:** Use ∇f₀(x*)ᵀ(y−x*)≥0 optimality condition; don't erase work
- **Problem ordering:** Keith tied 2b to 2a after Josh had already done 2b independently — read full problem before starting subparts

## CS 5080 DreamerV3 Resolution Experiment

**Status:** 32x32 done, 16x16 done, 64x64 not started. Full analysis complete for all 4 models.
- Results documented in `TRAINING_PLAN.md` under "Resolution Experiment"
- Key finding: 64x64 dramatically outperforms lower resolutions (peak 54.26 vs 13.60 vs 9.13)
- Lower resolutions show cleaner H1/H2 signals; H3 inconclusive across all
- Analysis outputs: `analysis_output_snake_{16x16,32x32}/`, `analysis_output/`, `analysis_output_snake_2M_long/`
- All code at `/home/josh/cs5080_project/`

## CS 5080 Homework PDF Workflow

**No duplicate PDFs.** The workflow is:
1. Edit `.tex` file in `Homework/Source/`
2. LaTeX build outputs draft PDF into `Source/` — this is the working copy
3. User reviews and approves the draft
4. Move+rename the PDF from `Source/` into `Homework/Submissions/` — no copy, just move
5. Only the `.tex` source remains in `Source/`, only the final PDF in `Submissions/`

## Document Game Testing Immediately
- [feedback_document_game_testing.md](feedback_document_game_testing.md) — Log game experiments/testing to session log AND reference files as they happen, not after

## Forecast Preferences
- **"Hourly breakdown"** = just the hourly table (Time, Temp, Sky, Wind, Precip). No summary, no commentary, no analysis on top. Just the data.

## Speedtest: Use Ookla CLI
- [feedback_speedtest.md](feedback_speedtest.md) — Use `speedtest` (Ookla multi-connection), not old Python speedtest-cli

## Use Cloudflare for Pings
- [feedback_ping_cloudflare.md](feedback_ping_cloudflare.md) — Use 1.1.1.1 not 8.8.8.8 for ping/DNS tests

## NSAID Sensitivity
- [user_health_nsaid.md](user_health_nsaid.md) — Motrin/ibuprofen causes stomach pain; hot drinks are the only remedy

## Military Background
- [user_military_background.md](user_military_background.md) — Air Force brat + 24-yr enlisted career (1994-2018); comm/space ops; retired from Peterson

## Open Files in Windows from WSL
- [feedback_play_media.md](feedback_play_media.md) — Use powershell.exe Start-Process with D:\ paths, not /mnt/d/

## Verify Day of Week
- [Verify Day of Week](feedback_verify_day_of_week.md) — Compute day from date once, don't guess or flip-flop

## No Guessing — Use Real Data
- [feedback_no_guessing.md](feedback_no_guessing.md) — Never estimate or guess when real data is available; compute from logs/timestamps/metrics

## Session Log Path
- [feedback_session_log_path.md](feedback_session_log_path.md) — ONLY write to `D:\Documents\Claude_References\Session_Logs\`, never to desktop or other locations

## Dune Books Discussion Rules
- [feedback_dune_books_discussion.md](feedback_dune_books_discussion.md) — Answer ONLY from book text files, no guessing, internet only for expert/author commentary when requested

## Training Workflow Rules
- [feedback_training_workflow.md](feedback_training_workflow.md) — "Start training" means resume current incomplete model. A model isn't done until it hits target steps. Don't skip ahead or kill early.

## Never Delete Data from Project Files
- [feedback_never_delete_data.md](feedback_never_delete_data.md) — TRAINING_PLAN.md and similar files are data records. Only append, never remove or summarize over existing data.

## Cross-Reference Session Context with Script Output
- [feedback_cross_reference_session_context.md](feedback_cross_reference_session_context.md) — Before presenting advisor/script results, check session logs for prior decisions on the same topic and lead with that context

## Cell Phone
- [user_phone.md](user_phone.md) — Motorola Edge Plus 2022

## CS 5080 HW3 Reference Papers
- [project_cs5080_hw3_papers.md](project_cs5080_hw3_papers.md) — Dueling DQN, Rainbow, Decision Transformer papers from 2026-04-07 lecture; will be key for upcoming HW3

## File Comparison: Hash Then Read
- [feedback_file_comparison.md](feedback_file_comparison.md) — Use md5 hash to detect differences, then read content to determine which version is which (template vs solutions, etc.)

## Daily GitHub Repo Push
- [feedback_daily_repo_push.md](feedback_daily_repo_push.md) — Push CS5080 and CS5610 repos to GitHub at session start; don't let them go stale

## Past Coursework Location
- Non-MSCS courses moved to `D:\Documents\Past_Coursework\` (Apr 8, 2026) to free OneDrive space
- MSCS-relevant courses remain in `D:\OneDrive\Desktop\`

## Writing Samples
- Location: `D:\Documents\Josh Writting Samples\` (note spelling)
- Contains essays from ENG 122, CS 3050 (Dr. Sullivan), and other courses
- Use these to match Josh's voice when drafting coursework
- **Style traits:** Direct/matter-of-fact, uses "Right now," "So," "But," "This means" transitions, rhetorical questions, explains what things DO before what they ARE, short punchy sentences mixed with longer ones, "we"/"us" inclusive language, no stuffy academic words