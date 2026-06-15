"""Layer 2a — Deterministic extraction via hand-written CSS selectors for known sites."""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from playwright.async_api import Page


@dataclass
class SelectorAction:
    selector: str
    action: str      # "click", "fill", "extract", "extract_all", "wait", "press"
    value: str = ""
    field: str = ""
    wait_ms: int = 1000


# Site-specific selector maps
SITE_SELECTORS: dict[str, list[SelectorAction]] = {
    "www.amazon.in": [
        SelectorAction(selector="#twotabsearchtextbox", action="fill", value="{query}"),
        SelectorAction(selector="#nav-search-submit-button", action="click"),
        SelectorAction(selector=".s-result-item", action="wait", wait_ms=2000),
        SelectorAction(selector=".s-result-item[data-component-type='s-search-result']", action="extract_all", field="products"),
    ],
    "www.flipkart.com": [
        SelectorAction(selector="input[name='q']", action="fill", value="{query}"),
        SelectorAction(selector="input[name='q']", action="press", value="Enter"),
        SelectorAction(selector="._1AtVbE", action="wait", wait_ms=2000),
        SelectorAction(selector="._1AtVbE", action="extract_all", field="products"),
    ],
}


async def try_deterministic(page: Page, url: str, goal: str) -> str | None:
    """If the domain has selectors, run them. Returns extracted text or None."""
    domain = urlparse(url).netloc
    actions = SITE_SELECTORS.get(domain)
    if not actions:
        return None

    query = _extract_query_from_goal(goal)
    results = []

    try:
        for action in actions:
            selector = action.selector
            value = action.value.replace("{query}", query) if action.value else ""

            if action.action == "fill":
                await page.fill(selector, value, timeout=5000)
            elif action.action == "click":
                await page.click(selector, timeout=5000)
            elif action.action == "press":
                await page.press(selector, value)
            elif action.action == "wait":
                await page.wait_for_selector(selector, timeout=action.wait_ms)
            elif action.action == "extract":
                el = await page.query_selector(selector)
                if el:
                    text = await el.text_content()
                    results.append(text.strip() if text else "")
            elif action.action == "extract_all":
                elements = await page.query_selector_all(selector)
                for el in elements[:10]:
                    text = await el.text_content()
                    if text and text.strip():
                        results.append(text.strip()[:500])

            if action.action in ("click", "fill", "press"):
                import asyncio
                await asyncio.sleep(action.wait_ms / 1000)

    except Exception:
        return None

    if not results:
        return None

    return "\n---\n".join(results)


def _extract_query_from_goal(goal: str) -> str:
    """Pull a search query from the goal text."""
    for prefix in ["search for ", "find ", "look up ", "compare "]:
        if prefix in goal.lower():
            idx = goal.lower().index(prefix) + len(prefix)
            return goal[idx:].split(".")[0].split(",")[0].strip()
    return goal[:60]
