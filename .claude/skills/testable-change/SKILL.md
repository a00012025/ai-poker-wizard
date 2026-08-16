---
name: testable-change
description: Use for every bug fix, feature, refactor, prompt, hook, skill, or provider integration that changes production behavior. Requires a regression test first and keeps third-party dependencies outside deterministic core logic.
---

# Testable Change

1. Trace the changed behavior through every caller. Fix the shared root cause.
2. Add or update a deterministic behavior test that fails before the production edit. Bug reports must preserve the exact reported values.
3. Keep domain decisions in pure functions or injected callables. Telegram, LLM, database, and GTO Wizard calls stay at adapter boundaries and are replaced with fakes in the default test suite.
4. Reuse an existing boundary before adding an interface. Do not test source shape, private implementation details, live credentials, timing, or model wording when a behavior assertion works.
5. Run the smallest relevant test first, then:

```bash
python scripts/agent_change_guard.py --base origin/main
python scripts/run_chat_contracts.py  # when chat behavior changes
python scripts/regression_test.py     # when a core analysis file changes
```

Do not finish with failing checks or a production diff that has no test diff. Keep live-provider tests explicitly marked and separate from the offline suite.
