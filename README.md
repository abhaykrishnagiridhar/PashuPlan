# PashuPlan

Multi-agent AI decision support for livestock farms, built solo for the Reva hackathon
(problem statement: *Multi-Agent AI for Livestock Farm Decision Support*).

A farmer asks a plain question, in English or Hindi:

> "Milk production has decreased this week"

and PashuPlan works out why, using the farm's own data.

## How it works

- **Specialist agents:** Health, Nutrition, Breeding, Inventory and Finance. Each is an
  LLM agent with a tool-use loop. It chooses which read-only data tools to call
  (for example `get_cow_detail("C16")`), reads the results, and reports back.
- **Farm Manager agent:** combines the specialist reports into one prioritised action plan.
- **Deterministic analytics:** every number comes from tested Python functions, not
  from the model. The LLM decides what to look at and explains it. It never does the maths.
- **Guardrails:** per-agent tool allowlists, a cap of 6 tool turns per agent, and tool
  errors fed back to the model instead of crashing the run.
- **Hindi support:** the language is detected from the question. Answers and the UI follow
  it, and numbers, rupee amounts and cow tags stay in Latin digits.

## Demo scenario

A synthetic 40-cow dairy herd with 60 days of data. Milk falls about 14% in a week,
with four planted causes the agents have to untangle:

1. Mastitis in 3 cows (high somatic cell count and fever)
2. A cheaper, lower-protein feed batch
3. A heat wave
4. A red herring: normal late-lactation decline in older cows

All data is simulated. No real farm data is used.

## Status

- [x] Domain model, seeded data generator, analytics, tool registry, Hindi/English i18n (69 tests, 99% coverage)
- [x] Streamlit dashboard preview
- [ ] Agent tool-use loop and Farm Manager orchestration
- [ ] Hindi UI end to end
- [ ] Offline replay mode for demos

## Run

    pip install -r requirements.txt
    python -m pytest
    streamlit run app.py

The agents need an `ANTHROPIC_API_KEY` environment variable. The dashboard and tests do not.

## Tech

Python, pandas, Streamlit, Plotly, the Anthropic API.
