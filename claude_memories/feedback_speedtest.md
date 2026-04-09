---
name: Speedtest default tool
description: Use Ookla official CLI (multi-connection) for speed tests, not the old Python speedtest-cli
type: feedback
---

Use the Ookla official CLI (`speedtest` at `/usr/local/bin/speedtest`) for speed tests.

**Why:** Multi-connection testing saturates the link properly — gets accurate results (~1200 Mbps) vs the old single-connection Python script (~700 Mbps).

**How to apply:** When user asks for a speed test, run `speedtest` (already installed, license accepted). Combine with `ping -c 10 1.1.1.1` for ping tests.
