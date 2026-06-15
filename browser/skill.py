"""Browser skill — 4-layer cascade orchestrator.

Layer 1: extract (httpx + readability)    — 0 LLM cost
Layer 2a: deterministic (CSS selectors)   — 0 LLM cost
Layer 2b: a11y (accessibility tree + LLM) — cheap text LLM
Layer 3: vision (set-of-marks + VLM)      — vision model cost
Precondition: CAPTCHA/block detection     — returns error_code
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

# Load .env early so BROWSER_HEADLESS is available regardless of entry point.
from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))

from browser.precondition import check_blocked_http, check_page_content
from browser.extract import extract_static
from browser.driver import BrowserDriver
from browser.dom import get_a11y_snapshot, get_clickable_elements
from browser.selectors import try_deterministic
from browser.highlight import draw_set_of_marks


@dataclass
class BrowserResult:
    success: bool
    content: str = ""
    layer_used: str = ""   # "extract" | "deterministic" | "a11y" | "vision"
    actions: list[dict] = field(default_factory=list)
    turns: int = 0
    error_code: str | None = None
    screenshots: list[str] = field(default_factory=list)
    final_url: str = ""
    tokens_in: int = 0
    tokens_out: int = 0


DROPDOWN_SIGNALS = ["▾", ":", "sort", "filter", "select", "dropdown", "menu"]


async def run_browser(
    url: str,
    goal: str,
    session_id: str,
    node_id: str,
    *,
    on_event=None,
    force_layer: str | None = None,
    max_turns: int = 20,
) -> BrowserResult:
    """Execute the 4-layer cascade against a URL with a goal."""

    screenshots_dir = Path(f"state/sessions/{session_id}/screenshots")
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    async def emit(browser_action: str, **kwargs):
        if on_event:
            try:
                await on_event("browser_action", node_id=node_id, action=browser_action, **kwargs)
            except Exception:
                pass

    # --- Precondition check ---
    # Note: 403 from httpx doesn't mean the page won't work in Playwright.
    # Many SPAs return 403 to raw HTTP but render fine in a real browser.
    # Only hard-block on connection failures, not on HTTP status codes.
    http_blocked = False
    if not force_layer:
        blocked = await check_blocked_http(url)
        if blocked:
            http_blocked = True  # remember, but don't fail yet — try Playwright

    # --- Layer 1: Static extraction ---
    # Skip if goal requires interaction or HTTP was blocked
    needs_interaction = _goal_needs_interaction(goal)

    if (not force_layer or force_layer == "extract") and not needs_interaction and not http_blocked:
        goal_keywords = _extract_keywords(goal)
        text, sufficient = await extract_static(url, goal_keywords)
        if sufficient and text:
            # Validate: does the extracted content actually answer the goal?
            is_valid = await _validate_extraction(text, goal)
            if is_valid:
                await emit("extract", url=url, chars=len(text))
                return BrowserResult(
                    success=True, content=text, layer_used="extract",
                    turns=0, final_url=url,
                )
        if force_layer == "extract":
            return BrowserResult(
                success=False, content=text or "", layer_used="extract",
                error_code="extraction_failed", final_url=url,
            )

    # --- Layers 2-3 require Playwright ---
    headless = False
    # DPR=1: screenshot pixels match CSS pixels for coordinate accuracy in vision layer
    driver = BrowserDriver(headless=headless, dpr=1.0)
    try:
        page = await driver.launch()
        await driver.goto(url)

        # Check if page is blocked — but wait for JS challenges to resolve first
        page_blocked = False
        for wait_attempt in range(3):
            page_html = await driver.get_content()
            page_title = await driver.get_title()
            page_blocked = check_page_content(page_html, page_title)
            if not page_blocked:
                break
            await asyncio.sleep(3)  # Wait for Cloudflare/JS challenge to resolve
        if page_blocked:
            await emit("blocked", url=url, reason="page_blocked_after_render")
            return BrowserResult(success=False, error_code="gateway_blocked", final_url=url)

        # --- Layer 2a: Deterministic selectors ---
        # Skip deterministic when interaction is required — use a11y instead
        # for visible browser actions that satisfy the assignment requirement
        if (not force_layer or force_layer == "deterministic") and not needs_interaction:
            det_result = await try_deterministic(page, url, goal)
            if det_result:
                screenshot_path = str(screenshots_dir / f"{node_id}_deterministic.png")
                await driver.screenshot(screenshot_path)
                await emit("deterministic", url=url, chars=len(det_result))
                return BrowserResult(
                    success=True, content=det_result, layer_used="deterministic",
                    turns=0, final_url=await driver.get_url(),
                    screenshots=[screenshot_path],
                )
            if force_layer == "deterministic":
                return BrowserResult(
                    success=False, layer_used="deterministic",
                    error_code="interaction_failed", final_url=url,
                )

        # --- Unified loop: a11y by default, one-shot vision when LLM needs it ---
        result = await _run_unified_loop(
            driver, goal, session_id, node_id, screenshots_dir,
            emit, max_turns=max_turns,
        )
        if result.success and result.content and len(result.content) > 100:
            return result

        return result

    finally:
        await driver.close()


async def _run_unified_loop(
    driver: BrowserDriver,
    goal: str,
    session_id: str,
    node_id: str,
    screenshots_dir: Path,
    emit,
    max_turns: int = 20,
) -> BrowserResult:
    """Unified browser loop: DOM elements (always) + screenshot (on-demand).

    Like browser-use: LLM always sees the element list, can request screenshot when needed.
    Screenshots auto-included after typing (to see autocomplete) and after failures.
    """
    from llm_gateway import gateway

    actions_log = []
    total_tokens_in = 0
    total_tokens_out = 0
    screenshots = []
    prior_actions_desc = []
    include_screenshot_next = False
    last_action_key = ""
    repeat_count = 0
    vision_was_used = False  # Track if screenshot was actually sent to LLM

    for turn in range(max_turns):
        await driver._dismiss_overlays()

        # Get DOM elements — if page is transitioning (0 elements), wait and retry
        elements = await get_clickable_elements(driver.page)
        if not elements:
            await asyncio.sleep(2)
            elements = await get_clickable_elements(driver.page)
        if not elements:
            await asyncio.sleep(3)
            elements = await get_clickable_elements(driver.page)
        if not elements:
            prior_actions_desc.append("Page transitioning (0 elements) — waited")
            continue

        # Detect Cloudflare/bot challenge — wait for it to resolve
        element_texts = " ".join(el.get("text", "").lower() for el in elements[:5])
        if len(elements) <= 3 and ("cloudflare" in element_texts or "privacy" in element_texts):
            await emit("a11y_thought", turn=turn, thought="Cloudflare challenge detected — waiting 5s for resolution")
            await asyncio.sleep(5)
            elements = await get_clickable_elements(driver.page)
            if len(elements) <= 3:
                await asyncio.sleep(5)
                elements = await get_clickable_elements(driver.page)
            if len(elements) <= 3:
                prior_actions_desc.append("Cloudflare challenge — could not resolve")
                continue

        a11y_text = await get_a11y_snapshot(driver.page)
        elem_index = {el["index"]: el for el in elements}

        # Build element list — filter noise only (no text dedup — dom.py handles structural dedup)
        SHOW_ROLES = {'slider', 'combobox', 'switch', 'spinbutton', 'gridcell', 'option', 'tab', 'menuitem'}
        filtered = [el for el in elements if el['text'].strip() or el['tag'] in ('input', 'select', 'textarea') or el.get('role') in SHOW_ROLES]
        def _el_label(el):
            role = el.get('role', '')
            display_type = role if role in SHOW_ROLES else el['tag']
            text = el['text'][:30]
            if role == 'slider' or (el.get('tag') == 'input' and el.get('valuemax')):
                vnow = el.get('valuenow', '?')
                return f"  #{el['index']} slider \"{text}\" (current={vnow}) [use set_range with real target amount e.g. 35000]"
            return f"  #{el['index']} {display_type} \"{text}\""
        element_list = "\n".join(_el_label(el) for el in filtered)

        # Save turn legend for debugging
        legend_path = str(screenshots_dir / f"{node_id}_turn{turn}_legend.txt")
        Path(legend_path).write_text(element_list)

        # Page content — only include on later turns and keep short
        page_text = ""
        if turn >= 2:
            page_content_raw = await _extract_page_content(driver)
            if page_content_raw and len(page_content_raw) > 200:
                page_text = f"\n\nPage content:\n{page_content_raw[:4000]}"

        # If we've been scrolling with page content available, force extraction
        scroll_count = sum(1 for a in prior_actions_desc if "Scrolled" in a)
        if page_text and scroll_count >= 4:
            page_text += "\n\nYou have scrolled 4+ times and page content is available. You MUST use 'done' NOW and extract the data."

        # Screenshot: one-shot only when LLM explicitly requests, then back to a11y
        screenshot_bytes = None
        screenshot_note = ""
        if include_screenshot_next:
            try:
                screenshot_bytes = await driver.screenshot()
                ss_path = str(screenshots_dir / f"{node_id}_turn{turn}.png")
                Path(ss_path).parent.mkdir(parents=True, exist_ok=True)
                Path(ss_path).write_bytes(screenshot_bytes)
                screenshots.append(ss_path)
                screenshot_note = "\n[Screenshot attached — use it for visual context]"
            except Exception:
                screenshot_note = "\n[Screenshot failed — decide based on element list only]"
            include_screenshot_next = False

        # Prior actions
        prior_str = ""
        if prior_actions_desc:
            prior_str = "\nPrior actions (last 10):\n" + "\n".join(f"  {a}" for a in prior_actions_desc[-10:])

        # Build prompt
        # Two-phase prompt: Phase 1 (turns 0-5) = pure a11y, Phase 2 (turns 6+) = vision available
        screenshot_action = ''

        prompt = f"""You are a browser automation agent.

Goal: {goal}

Interactive elements (click by #number):
{element_list}
{prior_str}{screenshot_note}{page_text}

Respond ONLY as a JSON object with "thought" (1 short sentence) and "actions" (array):

{{"thought": "brief reason for next action", "actions": [{{"action": "click", "element": 7}}]}}

Available actions:
- click: {{"action": "click", "element": <#>}}
- type: {{"action": "type", "element": <#>, "text": "value"}}
- press: {{"action": "press", "key": "Enter"}}
- scroll: {{"action": "scroll", "direction": "down"}}
- go_back: {{"action": "go_back"}}
- drag: {{"action": "drag", "startX": <x>, "startY": <y>, "endX": <x>, "endY": <y>}}
- set_range: {{"action": "set_range", "element": <#>, "value": <target_amount>}} (for sliders — pass the REAL target amount, e.g., 35000 for ₹35,000. The system will auto-calibrate.)
- done: {{"action": "done", "content": "extracted data here"}}

Rules:
- Max 2 actions per turn. Use element # for all clicks.
- After typing, first autocomplete suggestion is auto-selected.
- For airports/cities, type codes (BLR, DEL, BOM).
- Complete ALL interactions mentioned in the goal (filtering, sorting, setting ranges) BEFORE extracting.
- Only use "done" after all required actions are completed. Don't shortcut.
- For sliders: use set_range with the REAL target amount (e.g., 35000 for ₹35,000). System auto-calibrates. Do NOT convert to slider scale yourself.
- Do NOT scroll more than 3 times. Never repeat failed actions.

Example responses:
{{"thought": "Need to search for the query", "actions": [{{"action": "click", "element": 5}}, {{"action": "type", "element": 5, "text": "browser agents"}}]}}
{{"thought": "Set max rent to 35000", "actions": [{{"action": "set_range", "element": 12, "value": 35000}}]}}
{{"thought": "All filters applied, extracting results", "actions": [{{"action": "done", "content": "1. Result A\\n2. Result B"}}]}}"""

        await emit("a11y_thought", turn=turn,
                  thought=f"a11y: {len(elements)} elements | content: {'yes' if page_text else 'no'} | screenshot: {'yes' if screenshot_bytes else 'no'}")

        # Call LLM (with or without screenshot)
        if screenshot_bytes:
            vision_was_used = True
            resp = await asyncio.to_thread(
                gateway.vision,
                messages=[{"role": "user", "content": prompt}],
                image_bytes=screenshot_bytes,
                temperature=0.0,
                max_tokens=4096,
            )
        else:
            resp = await asyncio.to_thread(
                gateway.chat,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=4096,
            )

        total_tokens_in += resp.input_tokens
        total_tokens_out += resp.output_tokens

        # Emit detailed turn I/O for debugging + persist to disk
        turn_io_data = {
            "turn": turn,
            "elements_count": len(filtered),
            "elements_sent": element_list,
            "page_content_chars": len(page_text),
            "screenshot_included": bool(screenshot_bytes),
            "prior_actions": prior_actions_desc[-10:] if prior_actions_desc else [],
            "llm_response": resp.text or "",
            "tokens_in": resp.input_tokens,
            "tokens_out": resp.output_tokens,
        }
        await emit("browser_turn_io", **turn_io_data)
        # Persist to JSONL for replay (node_id scoped to avoid mixing)
        bio_path = screenshots_dir.parent / "browser_io.jsonl"
        turn_io_data["node_id"] = node_id
        with open(bio_path, "a") as f:
            f.write(json.dumps(turn_io_data) + "\n")

        if resp.is_error:
            await emit("a11y_thought", turn=turn, thought=f"LLM ERROR: {resp.text[:80]}")
            continue

        parsed_response = _parse_structured_response(resp.text or "")
        if not parsed_response:
            await emit("a11y_thought", turn=turn, thought=f"Unparseable: {(resp.text or '')[:80]}")
            continue

        thought = parsed_response.get("thought", "")
        parsed = parsed_response.get("actions", [])
        if not parsed:
            await emit("a11y_thought", turn=turn, thought=f"No actions in response")
            continue

        # Log what LLM decided
        actions_summary = " | ".join(f"{a.get('action','?')}({a.get('element', a.get('text', a.get('key', '')))})" for a in parsed[:3])
        await emit("a11y_thought", turn=turn, thought=f"{thought} → {actions_summary}")

        # Execute actions — all DOM-based, no coordinate guessing
        done_result = None
        for action_data in parsed[:2]:
            action_type = action_data.get("action", "")

            if action_type == "done":
                content = action_data.get("content", "")
                if not content:
                    content = await _extract_page_content(driver)
                done_result = content
                break

            elif action_type == "screenshot":
                if not vision_was_used and not include_screenshot_next:
                    include_screenshot_next = True
                    prior_actions_desc.append("Screenshot granted (one-shot only — will not be available again)")
                else:
                    prior_actions_desc.append("Screenshot NOT available. Use element list and set_range for sliders. Do NOT request again.")

            elif action_type == "click":
                el_idx = action_data.get("element")
                success = False
                if el_idx is not None and int(el_idx) in elem_index:
                    el = elem_index[int(el_idx)]
                    try:
                        # Use Playwright locator for the DOM element (not coordinates)
                        bbox = el["bbox"]
                        x = int(bbox["x"] + bbox["width"] / 2)
                        y = int(bbox["y"] + bbox["height"] / 2)
                        await driver.page.mouse.click(x, y)
                        success = True
                        await emit("a11y_thought", turn=turn, thought=f"Click #{el_idx} \"{el['text'][:30]}\"")
                        actions_log.append({"type": "click", "target": el["text"][:60], "element": el_idx, "turn": turn})
                        prior_actions_desc.append(f"Clicked #{el_idx}: \"{el['text'][:40]}\"")
                    except Exception as e:
                        prior_actions_desc.append(f"FAILED click #{el_idx}: {str(e)[:40]}")
                        pass  # LLM can request screenshot if needed
                if not success:
                    prior_actions_desc.append(f"FAILED: element #{el_idx} not found")
                    pass  # LLM can request screenshot if needed
                await asyncio.sleep(0.8)

            elif action_type == "type":
                el_idx = action_data.get("element")
                text = action_data.get("text", "")
                target_locator = None

                if el_idx is not None and int(el_idx) in elem_index:
                    el = elem_index[int(el_idx)]
                    try:
                        bbox = el["bbox"]
                        x = int(bbox["x"] + bbox["width"] / 2)
                        y = int(bbox["y"] + bbox["height"] / 2)
                        await driver.page.mouse.click(x, y)
                        await asyncio.sleep(0.3)
                        # Clear existing value
                        await driver.page.keyboard.press("Meta+a")
                        await driver.page.keyboard.press("Backspace")
                        await asyncio.sleep(0.2)
                    except Exception:
                        pass

                # Clear any existing text first
                await driver.page.keyboard.press("Meta+a")
                await driver.page.keyboard.press("Backspace")
                await asyncio.sleep(0.2)
                # Use keyboard.type with delay (triggers per-keystroke events for autocomplete)
                await driver.page.keyboard.type(text, delay=80)
                await emit("a11y_thought", turn=turn, thought=f"Type \"{text}\" (keystroke-by-keystroke)")
                actions_log.append({"type": "type", "target": text, "turn": turn})
                prior_actions_desc.append(f"Typed: \"{text}\"")
                pass  # LLM can request screenshot if needed

                # Check if this is a search field (don't auto-select for search — user wants to search, not pick a suggestion)
                is_search_field = False
                if el_idx is not None and int(el_idx) in elem_index:
                    el_text = elem_index[int(el_idx)].get("text", "").lower()
                    if "search" in el_text:
                        is_search_field = True
                # Also check if the input has type=search
                if not is_search_field:
                    try:
                        input_type = await driver.page.evaluate("() => document.activeElement?.type || ''")
                        if input_type == "search":
                            is_search_field = True
                    except: pass

                if not is_search_field:
                    # Auto-click first autocomplete suggestion (for form fields like airport/city selectors)
                    suggestion_clicked = False
                    try:
                        await driver.page.get_by_role("option").first.wait_for(state="visible", timeout=2500)
                        await driver.page.get_by_role("option").first.click()
                        suggestion_clicked = True
                        prior_actions_desc[-1] += f" — auto-selected first suggestion"
                    except Exception:
                        try:
                            suggestion = driver.page.locator('[role="listbox"] li, [class*="dropdown"] li, [class*="suggestion"] li, ul.airportList li').first
                            await suggestion.wait_for(state="visible", timeout=1500)
                            await suggestion.click()
                            suggestion_clicked = True
                            prior_actions_desc[-1] += f" — auto-selected first dropdown item"
                        except Exception:
                            await asyncio.sleep(1.0)
                    if suggestion_clicked:
                        await emit("a11y_thought", turn=turn, thought=f"Auto-selected first suggestion after typing \"{text}\"")
                        actions_log.append({"type": "click_text", "target": f"first suggestion for '{text}'", "turn": turn})
                else:
                    # Search field — don't auto-click, let LLM press Enter or click Search button
                    await asyncio.sleep(1.5)
                    prior_actions_desc[-1] += f" — search field, awaiting next action"

            elif action_type == "click_text":
                raw_text = action_data.get("text", "")
                if raw_text:
                    # Clean up: remove ellipsis, trailing dots, extra whitespace
                    text = raw_text.rstrip(".").rstrip("…").strip()
                    # Build search variants — most specific to least specific
                    search_variants = [text]
                    if len(text) > 20:
                        search_variants.append(text[:20])
                    # Extract parts: "Bangalore (BLR)" → ["Bangalore", "BLR"]
                    first_part = text.split(",")[0].split(" - ")[0].split("(")[0].strip()
                    if first_part and first_part != text:
                        search_variants.insert(0, first_part)
                    # Extract code in parentheses: "Bangalore (BLR)" → "BLR"
                    import re as _re
                    paren_match = _re.search(r'\(([A-Z]{2,5})\)', text)
                    if paren_match:
                        search_variants.append(paren_match.group(1))

                    clicked_text = False
                    for search in search_variants:
                        if clicked_text:
                            break
                        # Strategy 1: ARIA listbox/dropdown containers
                        for container_sel in ['[role="listbox"]', '[role="menu"]', '[class*="dropdown"]', '[class*="suggestion"]', '[class*="autocomplete"]', '[class*="airport"]']:
                            try:
                                await driver.page.locator(f'{container_sel} >> text={search}').first.click(timeout=1500)
                                clicked_text = True
                                await emit("a11y_thought", turn=turn, thought=f"Click in dropdown: \"{search}\"")
                                break
                            except Exception:
                                continue
                        # Strategy 2: role=option
                        if not clicked_text:
                            try:
                                await driver.page.get_by_role("option", name=re.compile(re.escape(search[:15]), re.IGNORECASE)).first.click(timeout=1500)
                                clicked_text = True
                                await emit("a11y_thought", turn=turn, thought=f"Click role=option: \"{search}\"")
                            except Exception:
                                pass
                        # Strategy 3: li elements containing the text
                        if not clicked_text:
                            try:
                                await driver.page.locator(f'li:has-text("{search}")').first.click(timeout=1500)
                                clicked_text = True
                                await emit("a11y_thought", turn=turn, thought=f"Click li: \"{search}\"")
                            except Exception:
                                pass
                        # Strategy 4: calendar/date cells (for numeric dates like "21")
                        if not clicked_text and search.isdigit():
                            for cal_sel in ['[role="gridcell"]', 'button', 'td', '[class*="calendar"] button', '[class*="date"]']:
                                try:
                                    await driver.page.locator(f'{cal_sel}:text-is("{search}")').first.click(timeout=1500)
                                    clicked_text = True
                                    await emit("a11y_thought", turn=turn, thought=f"Click calendar cell: \"{search}\"")
                                    break
                                except Exception:
                                    continue

                    # No broad text search — only dropdown/autocomplete containers above
                    # If none matched, the click_text fails and LLM should use element # instead

                    if clicked_text:
                        actions_log.append({"type": "click_text", "target": raw_text, "turn": turn})
                        prior_actions_desc.append(f"Clicked text: \"{first_part}\"")
                        # Don't auto-screenshot after click_text — element list refresh is enough
                        # LLM can request screenshot if it needs visual context
                    else:
                        prior_actions_desc.append(f"FAILED click_text: \"{raw_text[:40]}\" not found")
                        pass  # LLM can request screenshot if needed
                await asyncio.sleep(1.0)

            elif action_type == "press":
                key = action_data.get("key", "Enter")
                await emit("a11y_thought", turn=turn, thought=f"Press: {key}")
                await driver.page.keyboard.press(key)
                actions_log.append({"type": "press", "target": key, "turn": turn})
                prior_actions_desc.append(f"Pressed: {key}")
                await asyncio.sleep(0.8)

            elif action_type == "go_back":
                await emit("a11y_thought", turn=turn, thought="Navigating back")
                await driver.page.go_back(timeout=5000)
                actions_log.append({"type": "go_back", "target": "back", "turn": turn})
                prior_actions_desc.append("Navigated back")
                await asyncio.sleep(1)

            elif action_type == "drag":
                sx = action_data.get("startX", 0)
                sy = action_data.get("startY", 0)
                ex = action_data.get("endX", 0)
                ey = action_data.get("endY", 0)
                await emit("a11y_thought", turn=turn, thought=f"Drag ({sx},{sy}) to ({ex},{ey})")
                # Prevent page scroll during drag by temporarily locking overflow
                await driver.page.evaluate("document.body.style.overflow = 'hidden'")
                await driver.page.mouse.move(sx, sy)
                await asyncio.sleep(0.1)
                await driver.page.mouse.down()
                await asyncio.sleep(0.05)
                steps = max(10, int(((ex - sx)**2 + (ey - sy)**2)**0.5 / 3))
                await driver.page.mouse.move(ex, ey, steps=steps)
                await asyncio.sleep(0.05)
                await driver.page.mouse.up()
                await driver.page.evaluate("document.body.style.overflow = ''")
                actions_log.append({"type": "drag", "target": f"({sx},{sy})->({ex},{ey})", "turn": turn})
                prior_actions_desc.append(f"Dragged ({sx},{sy}) to ({ex},{ey})")
                await asyncio.sleep(0.5)

            elif action_type == "set_range":
                el_idx = action_data.get("element")
                target_value = action_data.get("value", 0)
                if el_idx is not None and int(el_idx) in elem_index:
                    el = elem_index[int(el_idx)]
                    bbox = el["bbox"]
                    cx = int(bbox["x"] + bbox["width"] / 2)
                    cy = int(bbox["y"] + bbox["height"] / 2)

                    await emit("a11y_thought", turn=turn, thought=f"set_range #{el_idx}: clicking ({cx},{cy}) to focus, then keyboard")

                    # Step 1: Click the slider handle to focus it
                    await driver.page.mouse.click(cx, cy)
                    await asyncio.sleep(0.3)

                    # Step 2: Press Home to go to minimum
                    await driver.page.keyboard.press("Home")
                    await asyncio.sleep(0.3)

                    # Step 3: Use ArrowRight to reach target value
                    # Read aria-valuemax to know the scale
                    slider_meta = await driver.page.evaluate(f"""() => {{
                        const el = document.elementFromPoint({cx}, {cy});
                        if (!el) return null;
                        const slider = el.closest('[role=slider]') || el;
                        const vmax = parseInt(slider.getAttribute('aria-valuemax') || '100');
                        const vmin = parseInt(slider.getAttribute('aria-valuemin') || '0');
                        const vnow = parseInt(slider.getAttribute('aria-valuenow') || '0');
                        // Find displayed text near slider
                        const container = slider.closest('[class*=filter], [class*=slider], [class*=Slider], [class*=rent], [class*=Rent], [class*=range]') || slider.parentElement?.parentElement?.parentElement;
                        let display = '';
                        if (container) {{
                            const spans = container.querySelectorAll('span, div, label');
                            for (const s of spans) {{
                                const t = (s.innerText || '').trim();
                                if (t && (t.includes('₹') || t.includes('Lac') || /\\d{{3,}}/.test(t.replace(/,/g,'')))) {{
                                    display += t + ' | ';
                                }}
                            }}
                        }}
                        return {{vmax, vmin, vnow, display: display.slice(0, 150)}};
                    }}""")

                    if slider_meta:
                        vmax = slider_meta['vmax']
                        vmin = slider_meta['vmin']
                        vnow = slider_meta['vnow']
                        total_steps = vmax - vmin
                        await emit("a11y_thought", turn=turn,
                                  thought=f"  Slider: {vmin}-{vmax}, now={vnow}")

                        # Press Home to reset to minimum
                        await driver.page.keyboard.press("Home")
                        await asyncio.sleep(0.4)

                        # Helper: read the slider's displayed rupee value from its container
                        async def _read_slider_rupees():
                            """Read ₹ value displayed near the slider element."""
                            val = await driver.page.evaluate(f"""() => {{
                                const el = document.elementFromPoint({cx}, {cy});
                                if (!el) return '';
                                const slider = el.closest('[role=slider]') || el;
                                // Walk up to find the container with the ₹ display
                                let container = slider.parentElement;
                                for (let i = 0; i < 5 && container; i++) {{
                                    const text = container.innerText || '';
                                    // Look for the range display like "₹ 0 to ₹ 2.5 k" or "₹ 35,000"
                                    // But SKIP if it's the main page heading with "₹ 0 to ₹ 5 Lacs" (static)
                                    const matches = text.match(/₹\\s*[\\d,.]+\\s*(?:Lacs?|lac|K|k)?/g);
                                    if (matches && matches.length >= 2) {{
                                        // Return the last/highest ₹ amount in this container
                                        return matches[matches.length - 1];
                                    }}
                                    container = container.parentElement;
                                }}
                                return '';
                            }}""")
                            if not val:
                                return None
                            val = val.replace('₹', '').replace(',', '').strip()
                            lac_m = re.search(r'([\d.]+)\s*[Ll]ac', val)
                            if lac_m:
                                return float(lac_m.group(1)) * 100000
                            k_m = re.search(r'([\d.]+)\s*[Kk]', val)
                            if k_m:
                                return float(k_m.group(1)) * 1000
                            num_m = re.search(r'[\d.]+', val)
                            if num_m:
                                return float(num_m.group())
                            return None

                        # Incremental approach: press Right in batches, check value after each
                        # This handles non-linear sliders correctly
                        steps_done = 0
                        batch_size = 5
                        final_amount = 0

                        for _ in range(20):  # max 20 batches = 100 steps
                            for _ in range(batch_size):
                                await driver.page.keyboard.press("ArrowRight")
                                steps_done += 1
                                if steps_done >= total_steps:
                                    break
                            await asyncio.sleep(0.3)

                            current_amount = await _read_slider_rupees()
                            if current_amount is not None:
                                final_amount = current_amount
                                await emit("a11y_thought", turn=turn,
                                          thought=f"  step {steps_done}: ₹{current_amount:,.0f}")

                                if current_amount >= target_value:
                                    # At or past target — back up if overshot
                                    if current_amount > target_value * 1.15:
                                        back_steps = batch_size
                                        for _ in range(back_steps):
                                            await driver.page.keyboard.press("ArrowLeft")
                                            steps_done -= 1
                                        await asyncio.sleep(0.2)
                                    break
                            else:
                                # Can't read — just keep going
                                pass

                            if steps_done >= total_steps:
                                break

                            # Speed up if we're still far from target
                            if current_amount and current_amount < target_value * 0.3:
                                batch_size = 10
                            elif current_amount and current_amount < target_value * 0.7:
                                batch_size = 5
                            else:
                                batch_size = 2

                        prior_actions_desc.append(f"Set slider #{el_idx} to ₹{target_value:,} → reached ₹{final_amount:,.0f} after {steps_done} steps. DONE — do not call set_range again.")
                    else:
                        # No meta — try direct keyboard approach anyway
                        await driver.page.keyboard.press("Home")
                        await asyncio.sleep(0.2)
                        # Conservative: press 70 times (works for ₹500/step × 70 = ₹35,000)
                        for _ in range(70):
                            await driver.page.keyboard.press("ArrowRight")
                        await asyncio.sleep(0.3)
                        prior_actions_desc.append(f"Set slider #{el_idx}: Home + Right 70x (fallback). DONE — do not call set_range again.")

                    actions_log.append({"type": "set_range", "target": f"target={target_value}", "turn": turn})
                else:
                    prior_actions_desc.append(f"FAILED set_range: element #{el_idx} not found")
                await asyncio.sleep(1)

            elif action_type == "scroll":
                direction = action_data.get("direction", "down")
                delta = -300 if direction == "up" else 300
                # Move mouse to center-right of viewport (main content area) to avoid scrolling sidebars
                vp = driver.page.viewport_size or {"width": 1280, "height": 720}
                await driver.page.mouse.move(int(vp["width"] * 0.65), int(vp["height"] * 0.5))
                await driver.page.mouse.wheel(0, delta)
                actions_log.append({"type": "scroll", "target": direction, "turn": turn})
                prior_actions_desc.append(f"Scrolled {direction}")
                await asyncio.sleep(0.8)

            elif action_type == "wait":
                await asyncio.sleep(2)
                prior_actions_desc.append("Waited")

        # Stuck detection
        first = parsed[0] if parsed else {}
        current_key = f"{first.get('action','')}:{first.get('element', first.get('text', first.get('description','')))}" if parsed else ""
        if current_key == last_action_key:
            repeat_count += 1
        else:
            repeat_count = 0
            last_action_key = current_key

        # Handle done
        if done_result is not None:
            await emit("a11y_thought", turn=turn, thought=f"DONE — extracted {len(done_result)} chars")
            # Final screenshot (full page for complete evidence)
            final_ss = str(screenshots_dir / f"{node_id}_final.png")
            await driver.screenshot(final_ss, full_page=True)
            screenshots.append(final_ss)
            # Determine path label based on whether VLM was actually called
            path_label = "a11y+vision" if vision_was_used else "a11y"
            return BrowserResult(
                success=True, content=done_result, layer_used=path_label,
                actions=actions_log, turns=turn + 1,
                final_url=await driver.get_url(), screenshots=screenshots,
                tokens_in=total_tokens_in, tokens_out=total_tokens_out,
            )

    # Exhausted turns
    final_content = await _extract_page_content(driver)
    return BrowserResult(
        success=bool(final_content and len(final_content) > 100),
        content=final_content or "", layer_used="a11y+vision" if vision_was_used else "a11y",
        actions=actions_log, turns=max_turns,
        final_url=await driver.get_url(), screenshots=screenshots,
        tokens_in=total_tokens_in, tokens_out=total_tokens_out,
    )


async def _run_a11y_loop(
    driver: BrowserDriver,
    goal: str,
    session_id: str,
    node_id: str,
    screenshots_dir: Path,
    emit,
    max_turns: int,
) -> BrowserResult:
    """Layer 2b — multi-turn loop using accessibility tree + text LLM."""
    from llm_gateway import gateway

    actions_log = []
    total_tokens_in = 0
    total_tokens_out = 0
    screenshots = []

    last_action_error = ""
    consecutive_failures = 0

    for turn in range(max_turns):
        # Escalate to vision if a11y actions keep failing
        if consecutive_failures >= 2:
            await emit("a11y_thought", turn=turn, thought=f"ESCALATING TO VISION: {consecutive_failures} consecutive failures")
            break

        # Dismiss any popups/modals that appeared after page load
        await driver._dismiss_overlays()

        a11y_text = await get_a11y_snapshot(driver.page)
        if not a11y_text or len(a11y_text.strip()) < 20:
            await emit("a11y_thought", turn=turn, thought="A11y tree empty — escalating to vision")
            break

        current_url = await driver.get_url()
        prompt = _build_a11y_prompt(goal, a11y_text, current_url, turn, actions_log)

        # Include page content once actions have been taken (so LLM can see results and say done)
        page_context = ""
        content_available = False
        if turn >= 1:
            page_content_raw = await _extract_page_content(driver)
            if page_content_raw and len(page_content_raw) > 200:
                content_available = True
                page_context = f"\n\n--- PAGE CONTENT (this is what you should extract from) ---\n{page_content_raw[:4000]}\n--- END PAGE CONTENT ---"

        # Feed back errors from prior actions
        error_feedback = ""
        if last_action_error:
            error_feedback = f"\n\nLast action FAILED: {last_action_error}. Try a different approach."
            last_action_error = ""

        # When content is available, shrink the a11y tree
        a11y_limit = 2000 if content_available else 6000
        extract_nudge = ""
        if content_available and turn >= max_turns - 2:
            extract_nudge = "\n\nRunning out of turns. If PAGE CONTENT has what you need, respond with done=true and extract it."

        user_msg = f"Goal: {goal}\n\nAccessibility tree:\n{a11y_text[:a11y_limit]}{page_context}{extract_nudge}{error_feedback}"
        if turn == max_turns - 1:
            user_msg += "\n\nThis is your LAST turn. You MUST respond with done=true and extract all available content NOW."

        # Emit what we're sending to LLM
        a11y_count = len(a11y_text.splitlines())
        await emit("a11y_thought", turn=turn,
                  thought=f"a11y: {a11y_count} elements | content: {'yes ('+str(len(page_content_raw))+' chars)' if content_available else 'no'} | failures: {consecutive_failures}")

        # Run synchronous gateway.chat in thread pool to avoid blocking event loop
        resp = await asyncio.to_thread(
            gateway.chat,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=1024,
        )

        total_tokens_in += resp.input_tokens
        total_tokens_out += resp.output_tokens

        if resp.is_error:
            await emit("a11y_thought", turn=turn, thought=f"LLM ERROR: {resp.text[:80]}")
            break

        action_data = _parse_action_response(resp.text or "")
        if not action_data:
            await emit("a11y_thought", turn=turn, thought=f"LLM returned unparseable response: {(resp.text or '')[:80]}")
            break

        # Emit raw LLM decision
        llm_decision = action_data.get("thought", "")
        if action_data.get("done"):
            llm_decision = f"DONE — extracting {len(action_data.get('content',''))} chars"
        elif action_data.get("actions"):
            acts = action_data["actions"]
            llm_decision = " + ".join(f"{a.get('type')}(\"{a.get('target','')}\")" for a in acts)
        await emit("a11y_thought", turn=turn, thought=f"LLM decided: {llm_decision}")

        if action_data.get("done"):
            await emit("a11y_action", turn=turn, action_type="extract",
                      target=f"Extracting content ({len(action_data.get('content',''))} chars)",
                      url=current_url, session_id=session_id, phase="executing")

            screenshot_path = str(screenshots_dir / f"{node_id}_turn{turn}.png")
            await driver.screenshot(screenshot_path)
            screenshots.append(screenshot_path)

            # Final screenshot (full page for complete evidence)
            final_ss_path = str(screenshots_dir / f"{node_id}_final.png")
            await driver.screenshot(final_ss_path, full_page=True)
            screenshots.append(final_ss_path)

            content = action_data.get("content", "")
            if not content:
                content = await _extract_page_content(driver)

            await emit("a11y_done", url=current_url, turns=turn + 1)
            return BrowserResult(
                success=True, content=content, layer_used="a11y",
                actions=actions_log, turns=turn + 1,
                final_url=current_url, screenshots=screenshots,
                tokens_in=total_tokens_in, tokens_out=total_tokens_out,
            )

        # Emit LLM thought and turn context
        thought = action_data.get("thought", "")
        a11y_count = len(a11y_text.splitlines())
        turn_summary = f"a11y: {a11y_count} elements | content: {'yes' if content_available else 'no'}"
        if thought:
            turn_summary += f" | thought: {thought}"
        await emit("a11y_thought", turn=turn, thought=turn_summary)

        # Emit planned actions BEFORE execution (so dashboard shows what's about to happen)
        planned_actions = action_data.get("actions", [])
        for a in planned_actions:
            await emit("a11y_action", turn=turn, action_type=a.get("type", ""),
                      target=a.get("target", ""), url=current_url,
                      session_id=session_id, phase="executing")

        executed_actions = await _execute_actions(driver, planned_actions, turn)
        actions_log.extend(executed_actions)

        # Capture errors from this turn for feedback to LLM
        turn_errors = [a.get("error", "") for a in executed_actions if a.get("error")]
        only_scrolled = all(a.get("type") == "scroll" for a in executed_actions) if executed_actions else False
        if turn_errors:
            last_action_error = turn_errors[-1]
            consecutive_failures += 1
        elif only_scrolled:
            new_a11y = await get_a11y_snapshot(driver.page)
            if new_a11y == a11y_text:
                consecutive_failures += 1
                last_action_error = "Scrolling did not reveal new elements. Try a different approach or extract available data."
        elif any(a.get("type") == "fill" for a in executed_actions):
            # Fill actions that didn't error but also didn't change the a11y tree — likely failed silently
            new_a11y = await get_a11y_snapshot(driver.page)
            if len(new_a11y) == len(a11y_text):
                consecutive_failures += 1
                last_action_error = "Fill action did not change page state. The input field may not have been found."
            else:
                consecutive_failures = 0
        else:
            consecutive_failures = 0

        screenshot_path = str(screenshots_dir / f"{node_id}_turn{turn}.png")
        await driver.screenshot(screenshot_path)
        screenshots.append(screenshot_path)

        # Emit results with screenshot AFTER execution
        for a in executed_actions:
            if a.get("error"):
                await emit("a11y_action", turn=turn, action_type=a.get("type", ""),
                          target=a.get("target", ""), url=current_url,
                          screenshot_path=screenshot_path, session_id=session_id,
                          phase="failed", error=a.get("error", ""))

        await asyncio.sleep(0.5)

    final_content = await _extract_page_content(driver)
    final_url = await driver.get_url()

    # If we broke out due to consecutive failures, escalate to vision
    if consecutive_failures >= 2:
        return BrowserResult(
            success=False, layer_used="a11y", error_code="interaction_failed",
            actions=actions_log, turns=turn + 1 if 'turn' in dir() else 0,
            final_url=final_url, screenshots=screenshots,
            tokens_in=total_tokens_in, tokens_out=total_tokens_out,
        )

    # Otherwise we exhausted turns — return whatever content we have
    if final_content and len(final_content) > 100:
        return BrowserResult(
            success=True, content=final_content, layer_used="a11y",
            actions=actions_log, turns=max_turns,
            final_url=final_url, screenshots=screenshots,
            tokens_in=total_tokens_in, tokens_out=total_tokens_out,
        )

    return BrowserResult(
        success=False, layer_used="a11y", error_code="interaction_failed",
        actions=actions_log, turns=max_turns,
        final_url=final_url, screenshots=screenshots,
        tokens_in=total_tokens_in, tokens_out=total_tokens_out,
    )


async def _run_vision_loop(
    driver: BrowserDriver,
    goal: str,
    session_id: str,
    node_id: str,
    screenshots_dir: Path,
    emit,
    max_turns: int = 12,
) -> BrowserResult:
    """Layer 3 — Computer-use style: screenshot → VLM decides action → execute → repeat.

    Supports full action space: click (coordinates), type, press, scroll, done.
    Works like ChatGPT Operator / Claude Computer Use.
    """
    from llm_gateway import gateway

    actions_log = []
    total_tokens_in = 0
    total_tokens_out = 0
    screenshots = []

    last_action_key = ""
    repeat_count = 0

    await emit("a11y_thought", turn=0, thought="VISION LAYER ACTIVATED — computer-use mode")

    # Dismiss overlays and scroll to top before starting
    await driver._dismiss_overlays()
    await driver.page.evaluate("window.scrollTo(0, 0)")
    await asyncio.sleep(0.5)

    prior_actions_desc = []

    for turn in range(max_turns):
        # Take screenshot + get element bounding boxes for hybrid approach
        screenshot_bytes = await driver.screenshot()
        elements = await get_clickable_elements(driver.page)
        screenshot_path = str(screenshots_dir / f"{node_id}_vision_turn{turn}.png")
        Path(screenshot_path).parent.mkdir(parents=True, exist_ok=True)
        Path(screenshot_path).write_bytes(screenshot_bytes)
        screenshots.append(screenshot_path)

        # Build element index for reference
        elem_index = {el["index"]: el for el in elements}

        await emit("a11y_thought", turn=turn, thought=f"Vision turn {turn}: {len(elements)} elements, sending screenshot to VLM")

        # Build element list for the VLM
        element_list = "\n".join(
            f"  #{el['index']} {el['role']} \"{el['text'][:40]}\" at ({int(el['bbox']['x'] + el['bbox']['width']/2)},{int(el['bbox']['y'] + el['bbox']['height']/2)})"
            for el in elements[:40]
        )

        # Build prompt
        prior_str = ""
        if prior_actions_desc:
            prior_str = "\nActions taken so far:\n" + "\n".join(f"  {a}" for a in prior_actions_desc)

        page_text = ""
        if turn >= 2:
            page_text_raw = await _extract_page_content(driver)
            if page_text_raw and len(page_text_raw) > 200:
                page_text = f"\n\nPage text:\n{page_text_raw[:2000]}"

        # Build action feedback from last turn
        last_action_feedback = ""
        if prior_actions_desc:
            last = prior_actions_desc[-1]
            if "Clicked" in last:
                last_action_feedback = f"\nLAST ACTION RESULT: {last}. If you clicked an input field, it is now focused — use 'type' to enter text. Do NOT click the same field again."
            elif "Typed" in last:
                last_action_feedback = f"\nLAST ACTION RESULT: {last}. Autocomplete suggestions may now be visible — look for a dropdown and CLICK the correct suggestion."
            else:
                last_action_feedback = f"\nLAST ACTION RESULT: {last}"

        prompt = f"""You are a browser automation agent. Look at this screenshot and perform ONE action toward the goal.

Goal: {goal}
{prior_str}{last_action_feedback}
{page_text}

Interactive elements on page:
{element_list}

ACTIONS (respond as JSON array — you can batch up to 3 actions per turn):
- Click by element: {{"action": "click", "element": <#number>, "description": "what"}}
- Click by coords (if element not listed): {{"action": "click", "x": <x>, "y": <y>, "description": "what"}}
- Type (into focused field): {{"action": "type", "text": "text to type"}}
- Press key: {{"action": "press", "key": "Enter"}}  (Tab, Escape, ArrowDown, Backspace)
- Scroll: {{"action": "scroll", "direction": "down"}}  (or "up")
- Wait: {{"action": "wait"}}
- Done: {{"action": "done", "content": "extracted data"}}

RULES:
- PREFER clicking by element # (stable across viewports). Use x,y only for elements not in the list.
- You can batch: [{{"action":"click","element":5,"description":"From field"}}, {{"action":"type","text":"Bangalore"}}]
- After typing in an autocomplete field, wait for next screenshot to see suggestions, then click the suggestion.
- Do NOT repeat the same action if it already worked.
- When results are visible, "done" with ALL extracted data.

Respond with ONLY a JSON array."""

        resp = await asyncio.to_thread(
            gateway.vision,
            messages=[{"role": "user", "content": prompt}],
            image_bytes=screenshot_bytes,
            temperature=0.0,
            max_tokens=1024,
        )

        total_tokens_in += resp.input_tokens
        total_tokens_out += resp.output_tokens

        if resp.is_error:
            await emit("a11y_thought", turn=turn, thought=f"Vision VLM ERROR: {resp.text[:100]}")
            break

        parsed = _parse_vision_actions(resp.text or "")
        if not parsed:
            await emit("a11y_thought", turn=turn, thought=f"Vision unparseable: {(resp.text or '')[:100]}")
            break

        # Execute all actions in the batch
        done_result = None
        for action_data in parsed:
            action_type = action_data.get("action", "")
            description = action_data.get("description", "")

            if action_type == "done":
                content = action_data.get("content", "")
                if not content:
                    content = await _extract_page_content(driver)
                done_result = content
                break

            elif action_type == "click":
                el_idx = action_data.get("element")
                x = action_data.get("x", 0)
                y = action_data.get("y", 0)
                clicked = False

                # Strategy 1: element index (most stable)
                if el_idx and int(el_idx) in elem_index:
                    el = elem_index[int(el_idx)]
                    bbox = el["bbox"]
                    x = int(bbox["x"] + bbox["width"] / 2)
                    y = int(bbox["y"] + bbox["height"] / 2)
                    await emit("a11y_thought", turn=turn, thought=f"Vision click #{el_idx} \"{el['text'][:30]}\" at ({x},{y})")
                    await driver.page.mouse.click(x, y)
                    clicked = True

                # Strategy 2: try finding element by description text (DOM-based, robust)
                # Extract meaningful keywords from VLM description for text matching
                if not clicked and description:
                    # Try the full description first, then key parts
                    search_texts = [description[:40]]
                    # Extract quoted text or text after common prefixes
                    for prefix in ["Click on ", "Select ", "Click "]:
                        if description.startswith(prefix):
                            search_texts.insert(0, description[len(prefix):].rstrip(" suggestion").rstrip(" option")[:40])
                    for search_text in search_texts:
                        try:
                            await driver.page.get_by_text(search_text, exact=False).first.click(timeout=1500)
                            clicked = True
                            await emit("a11y_thought", turn=turn, thought=f"Vision click by text: \"{search_text}\"")
                            break
                        except Exception:
                            continue

                # Strategy 3: coordinate fallback
                if not clicked and (x or y):
                    await emit("a11y_thought", turn=turn, thought=f"Vision click coords ({x},{y}): {description}")
                    await driver.page.mouse.click(x, y)
                    clicked = True

                if clicked:
                    actions_log.append({"type": "click", "target": description, "x": x, "y": y, "turn": turn})
                    await emit("vision_click", turn=turn, index=el_idx or 0, target=description[:40])
                    prior_actions_desc.append(f"Clicked: {description}")
                    await asyncio.sleep(0.8)

            elif action_type == "type":
                text = action_data.get("text", "")
                await emit("a11y_thought", turn=turn, thought=f"Vision type: \"{text}\"")
                await driver.page.keyboard.type(text, delay=80)
                actions_log.append({"type": "type", "target": text, "turn": turn})
                prior_actions_desc.append(f"Typed: \"{text}\" — autocomplete suggestions should appear")
                await asyncio.sleep(2.5)  # Wait for autocomplete to render

            elif action_type == "press":
                key = action_data.get("key", "Enter")
                await emit("a11y_thought", turn=turn, thought=f"Vision press: {key}")
                await driver.page.keyboard.press(key)
                actions_log.append({"type": "press", "target": key, "turn": turn})
                prior_actions_desc.append(f"Pressed: {key}")
                await asyncio.sleep(0.8)

            elif action_type == "scroll":
                direction = action_data.get("direction", "down")
                delta = -300 if direction == "up" else 300
                vp = driver.page.viewport_size or {"width": 1280, "height": 720}
                await driver.page.mouse.move(int(vp["width"] * 0.65), int(vp["height"] * 0.5))
                await driver.page.mouse.wheel(0, delta)
                actions_log.append({"type": "scroll", "target": direction, "turn": turn})
                prior_actions_desc.append(f"Scrolled {direction}")
                await asyncio.sleep(0.8)

            elif action_type == "wait":
                await asyncio.sleep(2)
                prior_actions_desc.append("Waited")

        # Detect stuck: same action repeated 3+ times
        current_action_key = f"{action_type}:{description[:30]}" if 'action_type' in dir() else ""
        if current_action_key and current_action_key == last_action_key:
            repeat_count += 1
            if repeat_count >= 3:
                await emit("a11y_thought", turn=turn, thought=f"STUCK: repeated '{description[:30]}' {repeat_count} times — stopping")
                break
        else:
            repeat_count = 0
            last_action_key = current_action_key

        if done_result is not None:
            await emit("a11y_thought", turn=turn, thought=f"Vision DONE — extracted {len(done_result)} chars")
            await emit("vision_done", turns=turn + 1)
            final_ss = str(screenshots_dir / f"{node_id}_final.png")
            await driver.screenshot(final_ss, full_page=True)
            screenshots.append(final_ss)
            return BrowserResult(
                success=True, content=done_result, layer_used="vision",
                actions=actions_log, turns=turn + 1,
                final_url=await driver.get_url(), screenshots=screenshots,
                tokens_in=total_tokens_in, tokens_out=total_tokens_out,
            )

    final_content = await _extract_page_content(driver)
    return BrowserResult(
        success=bool(final_content), content=final_content or "", layer_used="vision",
        actions=actions_log, turns=max_turns,
        final_url=await driver.get_url(), screenshots=screenshots,
        tokens_in=total_tokens_in, tokens_out=total_tokens_out,
    )


async def _execute_actions(driver: BrowserDriver, actions: list[dict], turn: int) -> list[dict]:
    """Execute parsed actions from LLM response. Enforce dropdown-as-fence rule."""
    executed = []
    page = driver.page

    for i, action in enumerate(actions[:2]):
        action_type = action.get("type", "")
        target = action.get("target", "")
        value = action.get("value", "")

        is_dropdown = any(sig in target.lower() for sig in DROPDOWN_SIGNALS)
        if is_dropdown and i > 0:
            break

        try:
            if action_type == "click":
                if not target:
                    continue
                clicked = False
                for click_strategy in [
                    lambda: page.get_by_role("button", name=target).or_(
                        page.get_by_role("link", name=target)
                    ).or_(
                        page.get_by_role("tab", name=target)
                    ).or_(
                        page.get_by_role("menuitem", name=target)
                    ).first.click(timeout=3000),
                    lambda: page.get_by_text(target, exact=False).first.click(timeout=3000),
                    lambda: page.locator(f"[aria-label*='{target[:30]}']").first.click(timeout=3000),
                ]:
                    try:
                        await click_strategy()
                        clicked = True
                        break
                    except Exception:
                        continue
                if not clicked:
                    raise Exception(f"Element not found: '{target[:40]}'")

            elif action_type == "fill":
                if target and value:
                    filled = False
                    for strategy in [
                        lambda: page.get_by_role("searchbox", name=target).first.fill(value, timeout=3000),
                        lambda: page.get_by_role("textbox", name=target).first.fill(value, timeout=3000),
                        lambda: page.get_by_placeholder(target).first.fill(value, timeout=3000),
                        lambda: page.locator(f"input[aria-label*='{target[:20]}']").first.fill(value, timeout=3000),
                        lambda: page.locator("input[type='search'], input[type='text']").first.fill(value, timeout=3000),
                    ]:
                        try:
                            await strategy()
                            filled = True
                            break
                        except Exception:
                            continue
                    if not filled:
                        raise Exception(f"Could not find input matching '{target}'")

                    # Handle autocomplete: wait for suggestion dropdown and click first match
                    await asyncio.sleep(1.5)
                    try:
                        suggestion = page.get_by_role("option", name=re.compile(value[:10], re.IGNORECASE)).first
                        if await suggestion.is_visible(timeout=1500):
                            await suggestion.click()
                            await asyncio.sleep(0.5)
                    except Exception:
                        try:
                            suggestion = page.locator(f"li:has-text('{value[:15]}'), [class*='suggestion']:has-text('{value[:15]}'), [class*='option']:has-text('{value[:15]}')").first
                            if await suggestion.is_visible(timeout=1000):
                                await suggestion.click()
                                await asyncio.sleep(0.5)
                        except Exception:
                            # Last resort: try get_by_text for any visible suggestion
                            try:
                                await page.get_by_text(value[:15], exact=False).first.click(timeout=1000)
                            except Exception:
                                pass

            elif action_type == "press":
                await page.keyboard.press(value or "Enter")

            elif action_type == "scroll":
                direction = value.lower() if value else "down"
                delta = -300 if direction == "up" else 300
                await page.mouse.wheel(0, delta)

            elif action_type == "hover":
                if target:
                    await page.get_by_text(target, exact=False).first.hover(timeout=5000)

            elif action_type == "select_option":
                if target and value:
                    await page.get_by_role("combobox", name=target).or_(
                        page.locator(f"select[name*='{target}']")
                    ).first.select_option(value, timeout=5000)

            elif action_type == "go_back":
                await page.go_back(timeout=5000)

            elif action_type == "wait":
                await asyncio.sleep(1)

            executed.append({"type": action_type, "target": target, "value": value, "turn": turn})

        except Exception as e:
            executed.append({"type": action_type, "target": target, "error": str(e)[:100], "turn": turn})

        if is_dropdown:
            break

        await asyncio.sleep(0.3)

    return executed


async def _extract_page_content(driver: BrowserDriver) -> str:
    """Extract readable text from current page state, including hyperlink URLs."""
    try:
        text = await driver.page.evaluate("""() => {
            const contentSelectors = [
                '[data-component-type="s-search-results"]',
                '.s-main-slot',
                '#search',
                'ytd-section-list-renderer',
                '#contents',
                '[role="main"]',
                'main',
                'article',
            ];
            let el = null;
            for (const sel of contentSelectors) {
                const candidate = document.querySelector(sel);
                if (candidate && candidate.innerText && candidate.innerText.trim().length > 200) {
                    el = candidate;
                    break;
                }
            }
            if (!el) el = document.body;
            if (!el || !el.innerText || el.innerText.length < 100) return '';

            // Inject link URLs inline by replacing link text with "text [url]"
            const anchors = el.querySelectorAll('a[href^="http"]');
            for (const a of anchors) {
                const t = (a.innerText || '').trim();
                const h = a.href;
                if (t && h && h.length < 200 && !h.includes('accounts.') && !h.includes('/search?') && !h.includes('/signin') && !h.includes('/login')) {
                    a.setAttribute('data-original', t);
                    a.innerText = t + ' [' + h + ']';
                }
            }
            let text = el.innerText.slice(0, 10000);
            // Restore original text (in case page is used further)
            for (const a of anchors) {
                const orig = a.getAttribute('data-original');
                if (orig) a.innerText = orig;
            }
            return text;
        }""")
        return text.strip() if text else ""
    except Exception:
        return ""


async def _validate_extraction(content: str, goal: str) -> bool:
    """Ask LLM if the extracted content actually answers the goal."""
    from llm_gateway import gateway

    resp = await asyncio.to_thread(
        gateway.chat,
        messages=[
            {"role": "system", "content": "You validate whether extracted web content answers a user's goal. Respond with ONLY 'yes' or 'no'."},
            {"role": "user", "content": f"Goal: {goal}\n\nExtracted content:\n{content[:2000]}\n\nDoes this content contain specific data that directly answers the goal? (not generic page content, not a homepage, not an error)"},
        ],
        temperature=0.0,
        max_tokens=10,
    )
    answer = (resp.text or "").strip().lower()
    return answer.startswith("yes")


def _build_a11y_prompt(goal: str, a11y_text: str, current_url: str, turn: int, prior_actions: list) -> str:
    """Build the system prompt for the a11y LLM turn."""
    prior_str = ""
    if prior_actions:
        prior_str = "\nPrior actions:\n" + "\n".join(
            f"  Turn {a.get('turn', '?')}: {a.get('type', '?')} → \"{a.get('target', '')[:40]}\""
            for a in prior_actions[-5:]
        )

    return f"""You are a browser agent navigating a webpage to achieve a goal.

Current URL: {current_url}
Turn: {turn + 1}
{prior_str}

RULES:
- Max 2 actions per response.
- If an action targets a dropdown, sort menu, or filter trigger: emit ONLY that one action.
- After opening a dropdown, the next turn MUST click the correct option from the dropdown. Do NOT say done before selecting.
- Only say done=true AFTER you have completed ALL required interactions (filtering, sorting, searching). Do not extract early.
- If the goal says "sort by X" and you just opened the sort menu, you MUST click the sort option before extracting.
- If you scrolled and still cannot find the control you need, extract the best available results.
- Prefer completing the full task over extracting partial/wrong data.

Respond as JSON (no markdown):
{{
  "done": false,
  "actions": [
    {{"type": "click"|"fill"|"press"|"scroll"|"wait", "target": "element name or text", "value": "optional"}}
  ]
}}

OR when goal is achieved:
{{
  "done": true,
  "content": "extracted information as structured text"
}}"""


def _parse_structured_response(text: str) -> dict | None:
    """Parse LLM response as {"thought": "...", "actions": [...]} or fallback to array."""
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n") if not l.startswith("```"))
    text = text.strip()

    # Fix common LLM mistakes: #31 → 31
    text = re.sub(r'"element":\s*#(\d+)', r'"element": \1', text)

    # Try parsing as the structured format {"thought": ..., "actions": [...]}
    try:
        result = json.loads(text)
        if isinstance(result, dict) and "actions" in result:
            actions = result["actions"]
            if isinstance(actions, list):
                return {"thought": result.get("thought", ""), "actions": actions}
            elif isinstance(actions, dict):
                return {"thought": result.get("thought", ""), "actions": [actions]}
    except json.JSONDecodeError:
        pass

    # Try finding a JSON object with actions field
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[start:i+1])
                        if isinstance(result, dict) and "actions" in result:
                            actions = result["actions"]
                            if isinstance(actions, list):
                                return {"thought": result.get("thought", ""), "actions": actions}
                    except json.JSONDecodeError:
                        pass
                    break

    # Fallback: try parsing as plain array (old format)
    actions = _parse_vision_actions(text)
    if actions:
        return {"thought": "", "actions": actions}

    return None


def _parse_vision_actions(text: str) -> list[dict] | None:
    """Parse VLM response into a list of actions. Handles arrays and single objects."""
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n") if not l.startswith("```"))
    text = text.strip()

    # Fix common LLM mistakes: #31 → 31 in element references
    text = re.sub(r'"element":\s*#(\d+)', r'"element": \1', text)

    # Try parsing as array first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    # Find first [ or { and parse
    arr_start = text.find("[")
    obj_start = text.find("{")

    if arr_start >= 0 and (arr_start < obj_start or obj_start < 0):
        depth = 0
        for i in range(arr_start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[arr_start:i+1])
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        break

    if obj_start >= 0:
        depth = 0
        for i in range(obj_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        result = json.loads(text[obj_start:i+1])
                        if isinstance(result, dict):
                            return [result]
                    except json.JSONDecodeError:
                        break

    return None


def _parse_action_response(text: str) -> dict | None:
    """Parse LLM JSON response for actions or done signal."""
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n") if not l.startswith("```"))

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first valid JSON object by matching braces
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    break
    return None


async def _set_slider_binary_search(driver, bbox: dict, target_amount: float, el_idx, emit, turn: int) -> str:
    """Set a slider to match a target display value using binary search with drag.

    The LLM passes the real-world target (e.g., 35000 for ₹35,000). This function:
    1. Gets the track bounds and current handle position
    2. Binary searches by DRAGGING the handle to different track positions
    3. Reads the displayed value after each drag to converge

    Returns a status message for prior_actions_desc.
    """
    page = driver.page

    async def _get_slider_geometry():
        """Get fresh slider/track geometry (handle moves between attempts)."""
        return await page.evaluate(f"""() => {{
            const els = document.querySelectorAll('[role=slider], input[type=range]');
            for (const el of els) {{
                const r = el.getBoundingClientRect();
                if (Math.abs(r.x - {bbox['x']}) < 30 && Math.abs(r.y - {bbox['y']}) < 30) {{
                    const track = el.closest('[class*=slider], [class*=range], [class*=Slider], [class*=filter], [class*=Rent], [class*=rent]') || el.parentElement;
                    const tr = track ? track.getBoundingClientRect() : r;
                    return {{
                        trackX: tr.x, trackY: tr.y + tr.height/2, trackW: tr.width,
                        handleX: r.x + r.width/2, handleY: r.y + r.height/2,
                        handleW: r.width
                    }};
                }}
            }}
            return null;
        }}""")

    track_info = await _get_slider_geometry()
    if not track_info:
        return f"FAILED set_range: slider track not found for #{el_idx}"

    track_left = track_info['trackX']
    track_w = track_info['trackW']
    track_y = track_info['handleY']

    def _parse_display_amount(text: str) -> float | None:
        """Parse ₹ display text into numeric value."""
        if not text:
            return None
        text = text.replace('₹', '').replace('Rs', '').replace('Rs.', '').replace(',', '').strip()
        lac_match = re.search(r'([\d.]+)\s*[Ll]ac', text)
        if lac_match:
            return float(lac_match.group(1)) * 100000
        k_match = re.search(r'([\d.]+)\s*[Kk]', text)
        if k_match:
            return float(k_match.group(1)) * 1000
        num_match = re.search(r'[\d.]+', text)
        if num_match:
            val = float(num_match.group())
            if val > 0:
                return val
        return None

    async def _read_display_value() -> tuple[float | None, str]:
        """Read the current displayed value near the slider."""
        result = await page.evaluate(f"""() => {{
            const els = document.querySelectorAll('[role=slider], input[type=range]');
            for (const el of els) {{
                const r = el.getBoundingClientRect();
                if (Math.abs(r.x - {bbox['x']}) < 30 && Math.abs(r.y - {bbox['y']}) < 30) {{
                    const container = el.closest('[class*=slider], [class*=range], [class*=Slider], [class*=filter], [class*=Rent], [class*=rent], [class*=price], [class*=Price]') || el.parentElement?.parentElement?.parentElement;
                    if (!container) return '';
                    const texts = container.querySelectorAll('span, div, p, label, [class*=value], [class*=Value], [class*=amount], [class*=label]');
                    let allText = '';
                    for (const t of texts) {{
                        const txt = (t.innerText || '').trim();
                        if (txt && (txt.includes('₹') || txt.includes('Rs') || txt.includes('Lac') || txt.includes(',') || /\\d{{3,}}/.test(txt))) {{
                            allText += txt + ' | ';
                        }}
                    }}
                    return allText.slice(0, 200);
                }}
            }}
            return '';
        }}""")
        if result:
            amounts = re.findall(r'₹\s*[\d,.]+ ?(?:[Ll]ac[s]?|[Kk])?|[\d,.]+ ?[Ll]ac[s]?', result)
            if len(amounts) >= 2:
                parsed = _parse_display_amount(amounts[-1])
                return parsed, result
            elif amounts:
                parsed = _parse_display_amount(amounts[0])
                return parsed, result
        return None, result or ""

    # Lock scroll during slider interaction
    await page.evaluate("document.body.style.overflow = 'hidden'")

    lo_ratio = 0.0
    hi_ratio = 1.0
    best_ratio = 0.5
    best_display = ""
    tolerance = target_amount * 0.20  # 20% tolerance

    await emit("a11y_thought", turn=turn, thought=f"Binary search slider #{el_idx} for target ₹{target_amount:,.0f}")

    # Strategy: Focus slider + ArrowLeft/ArrowRight keys (universally supported by ARIA sliders)
    # First, figure out current value and step direction
    slider_state = await page.evaluate(f"""() => {{
        const els = document.querySelectorAll('[role=slider], input[type=range]');
        for (const el of els) {{
            const r = el.getBoundingClientRect();
            if (Math.abs(r.x - {bbox['x']}) < 30 && Math.abs(r.y - {bbox['y']}) < 30) {{
                return {{
                    valuenow: parseInt(el.getAttribute('aria-valuenow') || el.value || '0'),
                    valuemax: parseInt(el.getAttribute('aria-valuemax') || el.max || '100'),
                    valuemin: parseInt(el.getAttribute('aria-valuemin') || el.min || '0'),
                }};
            }}
        }}
        return null;
    }}""")

    if slider_state:
        vmin = slider_state['valuemin']
        vmax = slider_state['valuemax']
        vnow = slider_state['valuenow']
        total_steps = vmax - vmin

        # Focus the slider element by clicking it
        await page.mouse.click(int(bbox['x'] + 5), int(bbox['y'] + 5))
        await asyncio.sleep(0.3)

        # First: try direct aria-valuenow set via JS + React internals
        # This works on many React sliders by triggering the onChange handler
        direct_set = await page.evaluate(f"""() => {{
            const els = document.querySelectorAll('[role=slider], input[type=range]');
            for (const el of els) {{
                const r = el.getBoundingClientRect();
                if (Math.abs(r.x - {bbox['x']}) < 30 && Math.abs(r.y - {bbox['y']}) < 30) {{
                    // Native input range
                    if (el.tagName === 'INPUT') {{
                        const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                        nativeSetter.call(el, {target_amount});
                        el.dispatchEvent(new Event('input', {{bubbles: true}}));
                        el.dispatchEvent(new Event('change', {{bubbles: true}}));
                        return 'native';
                    }}
                    return 'aria';
                }}
            }}
            return null;
        }}""")

        await asyncio.sleep(0.3)

        if direct_set == 'native':
            # Native range input — verify
            current_val, display_text = await _read_display_value()
            if current_val and abs(current_val - target_amount) <= tolerance:
                await page.evaluate("document.body.style.overflow = ''")
                return f"Set slider #{el_idx} to ₹{target_amount:,.0f} → {display_text[:60]}"

        # Keyboard approach: ArrowLeft to decrease, ArrowRight to increase
        # Binary search: try a target_ratio, press Home first to reset, then ArrowRight N times
        # OR just calculate steps from current position

        # First read current display
        current_val, _ = await _read_display_value()
        await emit("a11y_thought", turn=turn,
                  thought=f"  Slider state: now={vnow}, range={vmin}-{vmax}, display=₹{current_val:,.0f}" if current_val else f"  Slider state: now={vnow}, range={vmin}-{vmax}")

        # Use keyboard: press Home to go to min, then ArrowRight to target
        # Home key sets slider to minimum in most implementations
        await page.keyboard.press("Home")
        await asyncio.sleep(0.3)

        # Now binary search with ArrowRight presses
        # Each ArrowRight moves by 1 step. We need to find how many steps = target_amount
        # Start with a linear estimate, then adjust
        target_ratio = target_amount / (current_val if current_val and current_val > 0 else 500000) * vnow if current_val else 0.07
        estimated_steps = max(1, int(total_steps * (target_amount / 500000)))  # rough estimate assuming 5 Lacs max

        # Press Right in chunks, checking display after each chunk
        steps_taken = 0
        chunk_size = max(1, estimated_steps // 4)

        for chunk in range(20):  # max 20 chunks
            # Press ArrowRight chunk_size times
            for _ in range(chunk_size):
                await page.keyboard.press("ArrowRight")
                steps_taken += 1
            await asyncio.sleep(0.3)

            current_val, display_text = await _read_display_value()
            if current_val is not None:
                await emit("a11y_thought", turn=turn,
                          thought=f"  step {steps_taken}: ₹{current_val:,.0f} ({display_text[:30]})")

                if abs(current_val - target_amount) <= tolerance:
                    best_display = display_text
                    break
                elif current_val > target_amount:
                    # Overshot — go back
                    for _ in range(chunk_size // 2):
                        await page.keyboard.press("ArrowLeft")
                        steps_taken -= 1
                    await asyncio.sleep(0.2)
                    # Reduce chunk size for finer control
                    chunk_size = max(1, chunk_size // 2)
                    current_val, display_text = await _read_display_value()
                    if current_val and abs(current_val - target_amount) <= tolerance:
                        best_display = display_text
                        break
                else:
                    # Not there yet — keep going, maybe increase chunk
                    if current_val < target_amount * 0.5:
                        chunk_size = max(1, chunk_size * 2)  # speed up
            else:
                # Can't read — keep pressing
                chunk_size = max(1, chunk_size // 2)

            if steps_taken > total_steps:
                break

    await page.evaluate("document.body.style.overflow = ''")

    if best_display:
        return f"Set slider #{el_idx} to ₹{target_amount:,.0f} → landed at {best_display[:80]}"
    else:
        return f"Set slider #{el_idx} to ratio={best_ratio:.2f} (target ₹{target_amount:,.0f}, could not verify display)"


def _goal_needs_interaction(goal: str) -> bool:
    """Detect goals that require browser interaction (filter, sort, click, etc.)."""
    interaction_signals = [
        "filter", "sort", "click", "select", "open", "navigate",
        "dropdown", "search", "type", "fill", "submit",
        "switch tab", "expand", "interact", "browse",
    ]
    goal_lower = goal.lower()
    return any(sig in goal_lower for sig in interaction_signals)


def _extract_keywords(goal: str) -> list[str]:
    """Extract meaningful keywords from a goal string."""
    stop_words = {"the", "a", "an", "is", "are", "was", "to", "from", "for", "of", "in", "on", "and", "or", "with", "by", "at", "it", "this", "that"}
    words = re.findall(r'\b[a-zA-Z]{3,}\b', goal.lower())
    return [w for w in words if w not in stop_words][:8]
