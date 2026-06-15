# AXON — Browser Agent

### Autonomous Web Navigation with DAG Orchestration

> *A cost-optimized browser agent that navigates the live web through a 4-layer cascade. DOM-first interaction for speed; vision on-demand for precision. The orchestrator plans a graph of skills; the browser skill handles everything from static extraction to complex form workflows.*

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![Playwright](https://img.shields.io/badge/Browser-Playwright-green)](https://playwright.dev/python/)
[![NetworkX](https://img.shields.io/badge/Graph-NetworkX-orange)](https://networkx.org)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                               USER QUERY                                      │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PLANNER                                                                      │
│  Decomposes query into skill nodes with typed inputs and dependencies          │
└─────────────────────────────────────┬────────────────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│    Researcher    │    │   Browser Skill  │    │    Formatter     │
│  web_search      │    │   4-Layer Cascade│    │  final answer    │
│  fetch_url       │    │                  │    │                  │
└──────────────────┘    └────────┬─────────┘    └──────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Layer 1: Extract       │──→ httpx + trafilatura
                    │  Layer 2: Deterministic │──→ CSS selectors
                    │  Layer 3: A11y          │──→ DOM elements + LLM
                    │  Layer 4: Vision        │──→ Screenshot + VLM
                    └─────────────────────────┘
```

---

## Browser Skill: 4-Layer Cascade

The browser skill picks the cheapest correct path for each page:

| Layer | Method | When Used |
|-------|--------|-----------|
| **Extract** | Raw HTTP + trafilatura text extraction | Static pages with content accessible without JavaScript |
| **Deterministic** | Playwright + hand-written CSS selectors | Sites with known, stable DOM structures |
| **A11y** | DOM element list + LLM decides actions | Interactive pages requiring clicks, typing, filtering |
| **Vision** | Screenshot sent to VLM on-demand | Visual-only controls (sliders, canvas, icon-only buttons) |

The A11y layer handles the majority of interactions. Vision is invoked only when the LLM explicitly requests a screenshot — typically for drag-based controls or pages where DOM elements are insufficient.

---

## Element Detection

Two-pass detection inspired by [browser-use](https://github.com/browser-use/browser-use):

**Pass 1** — Targeted selectors: standard HTML tags (`a`, `button`, `input`, `select`, `textarea`, `label`), ARIA roles (`gridcell`, `combobox`, `option`, `menuitem`), `tabindex`, `onclick`.

**Pass 2** — Cursor:pointer scan: catches framework components (React, Vue, Angular) that use CSS pointer without semantic markup.

**Name resolution** uses a 10-step fallback: `aria-label` → `aria-labelledby` → `innerText` → `value` → `placeholder` → `title` → `alt` → `data-tooltip` → `data-testid` → `name`.

**Dedup** uses outermost-wins: drops nested decorative wrappers while preserving calendar cells (`role=gridcell`), form controls, and elements with distinct text.

---

## Interaction Loop

Each turn of the unified interaction loop:

1. Dismiss overlays (cookie banners, login modals, popups)
2. Detect Cloudflare challenges — wait for auto-resolution
3. Extract all interactive DOM elements with bounding boxes
4. Build element list and include page content (if available)
5. Send to LLM — structured response: `{"thought": "...", "actions": [...]}`
6. Execute actions via Playwright DOM locators
7. Auto-select first autocomplete suggestion after typing (non-search fields)

**Supported actions**: `click`, `type`, `press`, `scroll`, `drag`, `go_back`, `screenshot`, `done`

---

## Setup

### Prerequisites

- Python 3.11+
- AWS credentials configured via `aws login`
- Chromium (installed via Playwright)

### Installation

```bash
git clone https://github.com/Shwethaamrutha/EAGv3-Session9.git
cd EAGv3-Session9
pip install -e .
playwright install chromium
```

### Configuration

```bash
cp .env.example .env
# Configure API keys and region
```

### Running

```bash
python dashboard_server.py
# Open http://localhost:8080
```

---

## Dashboard

The live dashboard provides full observability into agent execution:

| Tab | Description |
|-----|-------------|
| **Live Trace** | Real-time execution log — per-turn browser actions, thoughts, element clicks |
| **Execution Graph** | Horizontal DAG visualization with clickable nodes |
| **Answer** | Final rendered output (markdown tables, structured data) |
| **Browser Replay** | Complete session report — goal, path chosen, actions, screenshots, extracted data, metrics |
| **Browser I/O** | Full per-turn debugging — element list sent, LLM response, token usage |
| **Node I/O** | Orchestrator-level input/output per skill node |

Sessions persist to disk and can be replayed by selecting from the session history.

---

## Tested Sites

| Site | Interaction Type | Path |
|------|-----------------|------|
| Hacker News | Static content | Extract |
| GitHub Trending | JS-rendered listing | A11y |
| HuggingFace Models | Filter + sort dropdown | A11y |
| Amazon India | Search + extract results | A11y |
| Cleartrip Flights | Form fill + autocomplete + date picker + search | A11y |
| Skyscanner | Form fill + date + search | A11y |
| Google Scholar | Search + year filter + extract | A11y |
| YouTube | Search + filter | A11y |
| NoBroker | Search + BHK filter + extract | A11y |
| npm | Search + extract | A11y |
| 99acres | Search + multi-filter | A11y + Vision |
| tldraw / Excalidraw | Canvas drawing (drag) | A11y |

---

## Project Structure

```
.
├── browser/                 # Browser skill package
│   ├── skill.py             # Cascade orchestrator + unified interaction loop
│   ├── driver.py            # Playwright lifecycle, stealth, overlay dismissal
│   ├── dom.py               # Element detection (a11y tree + clickable elements)
│   ├── extract.py           # Layer 1: static extraction
│   ├── highlight.py         # Set-of-marks annotation (dashed colored boxes)
│   ├── precondition.py      # Gateway block detection
│   └── selectors.py         # Layer 2: site-specific CSS selectors
├── agent/                   # LLM gateway, memory, perception
│   ├── config.py            # Settings (profile, region)
│   └── llm_gateway/         # Provider client with credential refresh
├── prompts/                 # Skill prompt templates (14 skills)
├── flow.py                  # DAG orchestrator (parallel execution, recovery)
├── skills.py                # Skill catalogue loader
├── schemas_v2.py            # Typed contracts (AgentResult, NodeSpec, RunBudget)
├── dashboard_server.py      # FastAPI + WebSocket server
├── dashboard_s8.html        # Single-page dashboard UI
├── report.py                # Session replay report generator
├── persistence.py           # Atomic session state persistence
├── tracing.py               # Per-node span logging
├── mcp_server.py            # MCP tools (web_search, fetch_url, run_command)
├── mcp_runner.py            # Tool-use loop
├── agent_config.yaml        # Skill definitions (14 skills)
└── pyproject.toml           # Project metadata and dependencies
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Unified loop over strict cascade | Eliminates fragile escalation detection between a11y and vision |
| Element-based execution | Clicks by DOM index (viewport-independent, no coordinate drift) |
| Vision on-demand | LLM requests screenshots only when element list is insufficient |
| Auto-click first suggestion | Saves a turn on autocomplete fields (airports, cities) |
| Playwright-stealth | Reduces bot detection on Cloudflare-protected sites |
| Structured LLM response | `{"thought", "actions"}` enables clean parsing and turn-by-turn debugging |
| LLM validates Layer 1 | Prevents accepting garbage static extraction from JS-heavy pages |
| Outermost-wins dedup | Removes nested decorative elements without breaking calendar cells |
| Per-turn persistence | Browser I/O saved to JSONL for post-run debugging |

---

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

```
What are the trending Python repositories on GitHub this week?
```

---

## Execution Logs

<!-- Session traces and Browser I/O outputs will be added here -->

---

## Screenshots

<!-- Dashboard screenshots will be added here -->

---

## Known Limitations

- **Cloudflare Turnstile**: Sites with "press and hold" verification (Product Hunt) block all automated browsers. Agent correctly reports `gateway_blocked`.
- **Drag-based controls**: Price sliders require the LLM to request a screenshot first (vision escalation). Works but adds latency.
- **Non-determinism**: Temperature 0 does not guarantee identical outputs across different prompt contexts. Complex forms may require 1-2 retries.
- **npm sort**: Sorting on npm triggers a URL change that drops the search query (site behavior, not agent bug). Agent extracts from unsorted results.

---

## References

- [browser-use](https://github.com/browser-use/browser-use) — Element detection and interaction patterns
- [Playwright](https://playwright.dev/python/) — Browser automation framework
- [Set-of-Marks (Yang et al. 2023)](https://arxiv.org/abs/2310.11441) — Visual element annotation for VLMs
- [Playwright Stealth](https://pypi.org/project/playwright-stealth/) — Anti-detection patches
