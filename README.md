# Session 9: Browser Agent with DAG Orchestration

A production-grade browser automation agent built on a DAG-based multi-agent orchestrator. The agent navigates web pages through a cost-optimized cascade, interacting with forms, filters, dropdowns, calendars, and search interfaces — extracting structured data from live websites.

## Architecture

```
User Query
    |
Planner LLM
    |
    v
+----------------------------------+
|        DAG Orchestrator          |
|  (NetworkX graph, parallel exec) |
+----------------------------------+
    |           |           |
    v           v           v
Researcher  Browser Skill  Formatter
            |
            v
    4-Layer Cascade:
    1. Extract (httpx + trafilatura)
    2. Deterministic (CSS selectors)
    3. A11y (DOM element list + LLM)
    4. Vision (screenshot on-demand)
```

### Browser Skill Cascade

| Layer | Method | LLM Cost | When Used |
|-------|--------|----------|-----------|
| Extract | httpx + trafilatura | Validation only | Static pages (HN, blogs) |
| Deterministic | Playwright + CSS selectors | None | Known site structures |
| A11y | DOM element list + text LLM | Per-turn | Interactive pages (forms, filters) |
| Vision | Screenshot sent to VLM | Per-screenshot | Sliders, canvas, visual-only controls |

The cascade tries the cheapest layer first. Vision is invoked only when the LLM explicitly requests a screenshot (cannot find needed control in DOM).

### Element Detection

Two-pass detection inspired by browser-use:

1. **Pass 1** (targeted selectors): standard HTML tags, ARIA roles, tabindex, onclick, labels
2. **Pass 2** (cursor:pointer scan): catches React/Vue/Angular framework components

Element name resolution uses a 10-step fallback chain: aria-label, aria-labelledby, innerText, value, placeholder, title, alt, data-tooltip, data-testid, name.

Outermost-wins dedup removes nested decorative elements while preserving calendar cells (role=gridcell), form controls (input/select/textarea/button), and elements with distinct text.

### Unified Interaction Loop

Each turn:
1. Dismiss overlays (cookie banners, login modals)
2. Extract DOM elements with bounding boxes
3. Send element list (+ optional screenshot) to LLM
4. LLM responds with structured JSON: `{"thought": "...", "actions": [...]}`
5. Execute actions via Playwright DOM locators
6. Auto-click first autocomplete suggestion after typing (for non-search fields)

Supported actions: click (by element #), type, press, scroll, drag, go_back, screenshot, done.

## Setup

### Prerequisites

- Python 3.11+
- AWS credentials (Bedrock profile via `aws login --profile bedrock`)
- Chromium browser (installed via Playwright)

### Installation

```bash
pip install -e .
playwright install chromium
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your API keys and AWS region
```

### Running

```bash
# Start the dashboard
python dashboard_server.py

# Open http://localhost:8080 in browser
```

## Dashboard

The web dashboard provides:

- **Live Trace**: Real-time node-by-node execution log with per-turn browser actions
- **Execution Graph**: Horizontal DAG visualization with clickable nodes
- **Answer**: Final formatted output (markdown rendered)
- **Browser Replay**: 8-section assignment report (goal, DAG, path chosen, actions, screenshots, extracted data, comparison table, cost summary)
- **Browser I/O**: Full per-turn debugging (elements sent, LLM response, tokens used)
- **Node I/O**: Input/output data for each orchestrator node

Sessions are persisted to `state/sessions/` and can be replayed by clicking in the session list.

## Cost Profile

| Query Type | Layer | Tokens | Estimated Cost |
|-----------|-------|--------|---------------|
| Static page (HN, Product Hunt) | Extract | ~200 (validation) | < $0.01 |
| Simple interaction (HuggingFace sort) | A11y | ~8-10K | ~$0.03 |
| Complex form (Cleartrip flights) | A11y | ~30-40K | ~$0.12 |
| Visual control (price slider) | A11y + Vision | ~40-50K | ~$0.16 |

## Supported Sites (Tested)

| Site | Interaction | Status |
|------|------------|--------|
| HuggingFace Models | Filter + sort | Working |
| Amazon India | Search + extract | Working |
| Cleartrip Flights | Form fill + autocomplete + date + search | Working |
| Skyscanner | Form fill + date + search | Working (Cloudflare intermittent) |
| GitHub Trending | Extract | Working |
| npm Search | Search + extract | Working (sort triggers Cloudflare) |
| Google Scholar | Search + date filter + extract | Working |
| YouTube | Search + filter | Working |
| NoBroker | Search + BHK filter + extract | Working |
| 99acres | Search + filter | Partial (price slider needs vision) |

## Key Design Decisions

1. **Unified loop over cascade**: DOM elements always available, vision on-demand. Eliminates fragile escalation detection.
2. **Element-based execution**: All clicks go through DOM element indices (stable across viewports). No coordinate guessing.
3. **Auto-click autocomplete**: After typing in non-search fields, first suggestion is auto-selected. Saves a turn.
4. **Playwright-stealth**: Reduces Cloudflare bot detection triggers.
5. **Structured LLM output**: `{"thought": "...", "actions": [...]}` format enables clean parsing and debugging.
6. **LLM validates Layer 1**: Static extraction is checked by LLM before accepting ("does this answer the goal?").

## Project Structure

```
.
├── browser/                 # Browser skill package
│   ├── skill.py             # Cascade orchestrator + unified loop
│   ├── driver.py            # Playwright lifecycle + stealth + overlay dismissal
│   ├── dom.py               # Element detection (CDP a11y tree + clickable elements)
│   ├── extract.py           # Layer 1: static extraction
│   ├── highlight.py         # Set-of-marks annotation (dashed colored boxes)
│   ├── precondition.py      # CAPTCHA/block detection
│   └── selectors.py         # Layer 2a: site-specific CSS selectors
├── agent/                   # LLM gateway + memory + perception
│   ├── config.py            # Settings (AWS profile, region)
│   └── llm_gateway/         # Bedrock client with auto-credential refresh
├── prompts/                 # Skill prompt files (14 skills)
├── flow.py                  # DAG orchestrator (NetworkX graph, parallel execution)
├── skills.py                # Skill catalogue loader
├── schemas_v2.py            # AgentResult, NodeSpec, RunBudget
├── dashboard_server.py      # FastAPI + WebSocket server
├── dashboard_s8.html        # Single-page dashboard UI
├── report.py                # Replay report generator (text + HTML)
├── persistence.py           # Atomic session persistence
├── tracing.py               # Per-node span logging
├── mcp_server.py            # MCP tools (web_search, fetch_url, etc.)
├── mcp_runner.py            # Tool-use loop with Bedrock protocol
├── agent_config.yaml        # Skill definitions
└── pyproject.toml           # Dependencies
```

## Differences from Session 9 Lesson

The lesson teaches a strict 4-layer cascade with separate a11y and vision loops. This implementation evolves that into a production architecture:

| Aspect | Session 9 Lesson | This Implementation |
|--------|-----------------|---------------------|
| A11y/Vision | Separate loops, cascade between them | Unified loop, vision on-demand |
| Element detection | CDP accessibility tree only | cursor:pointer + ARIA + tabindex + labels |
| Dedup | Bounding-box containment | Outermost-wins with role/tag exemptions |
| Autocomplete | LLM clicks suggestion manually | Auto-click first suggestion after typing |
| Action format | Free text JSON array | Structured `{"thought", "actions"}` |
| Anti-detection | Basic UA + webdriver flag removal | playwright-stealth (full fingerprint) |
| Cost control | Separate cheap/expensive models | Single model, vision only when requested |

## Demo Queries

```
Compare top 3 Hugging Face text-generation models sorted by likes on https://huggingface.co/models
```

```
Find cheapest flights from Bangalore to Delhi next weekend on https://www.cleartrip.com
```

```
Compare 3 laptops under Rs 80,000 on https://www.amazon.in
```

```
Find recent papers about browser agents published in 2026 on https://scholar.google.com
```

```
Find 2BHK flats for rent under 25000 in Koramangala on https://www.nobroker.in
```

## Execution Logs

<!-- Add session traces / screenshots / Browser I/O outputs here -->

## Screenshots

<!-- Add dashboard screenshots here -->

## Known Limitations

- **Cloudflare Turnstile**: Sites with aggressive bot detection (Product Hunt) cannot be accessed. The agent correctly reports `gateway_blocked`.
- **Price sliders**: Drag-based controls require vision mode. Works when LLM requests screenshot.
- **Calendar date selection**: Works on most sites (Cleartrip, Skyscanner) but requires the date element to be in the viewport.
- **Non-determinism**: Same query may produce slightly different results across runs due to LLM temperature=0 not guaranteeing identical outputs across different prompts.
- **npm sort**: Sorting on npm triggers a page reload that drops the search query (npm bug, not agent bug). Agent works around by extracting unsorted results.

## References

- [Session 9 Lesson Material](./Session9Materials.html)
- [browser-use](https://github.com/browser-use/browser-use) — DOM extraction patterns
- [Playwright Stealth](https://pypi.org/project/playwright-stealth/) — Anti-detection
- [Set-of-Marks (Yang et al. 2023)](https://arxiv.org/abs/2310.11441) — Visual element annotation
