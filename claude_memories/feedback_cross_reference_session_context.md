---
name: Cross-reference session logs when presenting advisor/script output
description: Always connect script output to relevant prior session log context before presenting recommendations
type: feedback
---

When running any advisor script (AC pre-cool, forecasts, etc.), cross-reference the session logs for prior decisions or context about the same topic BEFORE presenting the output. Don't just dump the script results — frame them with what was already discussed.

**Why:** User ran AC Pre-Cool Advisor on Mar 26 startup. Session log from Mar 25 already noted "may not need AC tomorrow; cold pattern for 7+ days." Claude had read this but presented the advisor's 2 PM recommendation without mentioning the prior context, making it a useless recommendation.

**How to apply:** After running any recurring advisor/script, check if the last 1-2 session logs mention the same topic. If they do, lead with that context (e.g., "Yesterday you noted you probably won't need AC today — the advisor still says X, but here's the forecast context for your call").
