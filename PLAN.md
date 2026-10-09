# PashuPlan: solo MVP plan (Reva hackathon)

Problem: Multi-Agent AI for Livestock Farm Decision Support.
Deadline: judging starts Thu 2026-10-09 18:00. Build window about 10:00 to 18:00 (8h), solo.

## Scope (decided)
- One synthetic 40-cow dairy herd, 60 days of data, one hero scenario with four planted causes:
  mastitis in 3 cows, a lower-protein feed batch, a heat wave, plus a red herring (normal late-lactation decline).
- 5 specialist agents (Health, Nutrition, Breeding, Inventory, Finance) + Farm Manager.
- **Agent tool-use loop** (decided 2026-10-08): each specialist is an agent that calls read-only tools over the farm data
  until it can answer. Tools are parametrised (`get_cow_detail(cow_id)`), so agents choose what to look into.
  Guardrails: per-agent tool allowlist, max 6 tool turns per agent, tool errors are fed back instead of raising.
- **Hindi** (decided 2026-10-08): the farmer can ask in Hindi or English; language is detected from the query
  (Devanagari). Agent output and the UI follow that language. Numbers, rupee amounts and cow tags stay in Latin digits.
- Streamlit UI, replay mode for an offline demo.
- Cut: React, voice, cross-examination round (folded into Manager prompt), evaluation harness, extra scenarios.

## Layout
```
pashuplan/
  domain.py       # constants + formulas: THI, Wood's curve, SCC/fever thresholds, withdrawal days
  data_gen.py     # seeded farm generator -> FarmData (DataFrames + ground_truth)
  analytics.py    # deterministic fact functions (pure, JSON-safe)
  tools.py        # tool registry: schemas, per-agent allowlists, run_tool() dispatch
  i18n.py         # en/hi strings, language detection, prompt language instruction
  agents.py       # (13:30 slot) tool-use loop + prompts
  orchestrator.py # (13:30 slot) route -> parallel specialists -> manager
app.py            # (afternoon) Streamlit
tests/            # TDD: written before each module
```

## Timeline and status (Thu 9 Oct)
| Slot | Block | Status |
|---|---|---|
| 10:00-11:00 | domain + data generator (tests first) | done (2026-10-08), 99% coverage |
| 11:00-12:00 | analytics + tool registry + i18n (tests first) | done (2026-10-08), 69 tests green |
| **STOP HERE** | user asked to stop after this slot | reached; resume at agents.py |
| 12:00-13:30 | agents.py tool-use loop, orchestrator, CLI run | todo |
| 13:30-15:30 | Streamlit UI with Hindi toggle | todo |
| 15:30 | feature freeze | |
| 15:30-16:15 | replay mode + backup video | todo |
| 16:15-17:15 | slides + 3 rehearsals | todo |
| 17:15-18:00 | buffer, bug fixes only | |

## Fallbacks
- Behind at 13:30: skip the router, always run all 5 agents.
- Tool loop flaky: pre-compute facts and give them to the agent in one prompt (analytics functions already return them).
- Behind at 15:00: `st.metric` numbers instead of charts.

## Run
```
"../../../.venv/Scripts/python.exe" -m pytest
```
(the venv lives in the `reva hackathon` folder, two levels above this worktree's `.claude/worktrees/`)
