"""Accessibility tree + clickable element enumeration + deduplication.

Uses CDP (Chrome DevTools Protocol) for the a11y tree since
page.accessibility.snapshot() was removed in Playwright 1.50+.
"""
from __future__ import annotations

from playwright.async_api import Page


INTERACTIVE_ROLES = {
    "button", "link", "textbox", "combobox", "checkbox",
    "radio", "menuitem", "menuitemcheckbox", "menuitemradio",
    "tab", "switch", "searchbox", "spinbutton", "slider",
    "option", "treeitem", "listbox",
}

DROPDOWN_SIGNALS = {"▾", ":", "sort", "filter", "select", "more", "menu"}


async def get_a11y_snapshot(page: Page) -> str:
    """Get compact text representation of the accessibility tree via CDP.

    Returns a numbered list like:
    [1] button "Submit"
    [2] link "Home"
    [3] textbox "Search..."

    Includes headings as section markers for context.
    """
    client = None
    try:
        client = await page.context.new_cdp_session(page)
        tree = await client.send("Accessibility.getFullAXTree")
    except Exception:
        return ""
    finally:
        if client:
            try:
                await client.detach()
            except Exception:
                pass

    nodes = tree.get("nodes", [])
    if not nodes:
        return ""

    lines = []
    index = 1
    seen_names = set()

    for node in nodes:
        role = node.get("role", {}).get("value", "")
        name = node.get("name", {}).get("value", "")

        if not name:
            continue

        # Include headings as non-interactive context markers
        if role == "heading":
            heading_text = name.strip()[:60]
            if heading_text and heading_text.lower() not in seen_names:
                seen_names.add(heading_text.lower())
                lines.append(f"[--] heading \"{heading_text}\"")
            continue

        if role not in INTERACTIVE_ROLES:
            continue

        name_clean = name.strip()[:80]
        dedup_key = f"{role}:{name_clean.lower()}"
        if dedup_key in seen_names:
            continue
        seen_names.add(dedup_key)

        suffix = ""
        name_lower = name_clean.lower()
        if role == "combobox" or any(sig in name_lower for sig in DROPDOWN_SIGNALS):
            suffix = " [dropdown]"

        lines.append(f"[{index}] {role} \"{name_clean}\"{suffix}")
        index += 1

    return "\n".join(lines)


async def get_clickable_elements(page: Page) -> list[dict]:
    """Find all interactive elements on the page.

    Two-pass approach:
    1. Targeted selectors: standard HTML, ARIA roles, tabindex, onclick (fast, reliable)
    2. Cursor:pointer scan: catches framework components (React/Vue/Angular with click handlers)

    Returns list of {index, tag, text, role, bbox: {x, y, width, height}}.
    """
    elements = await page.evaluate("""() => {
        const candidates = [];
        const candSet = new Set();

        // Pass 1: Standard interactive selectors + ARIA roles (fast, reliable)
        const sels = [
            'a', 'button', 'input', 'select', 'textarea', 'label',
            '[role=button]', '[role=link]', '[role=menuitem]', '[role=tab]',
            '[role=option]', '[role=gridcell]', '[role=combobox]', '[role=searchbox]',
            '[role=switch]', '[role=slider]', '[role=treeitem]', '[tabindex]', '[onclick]',
        ];
        const selectors = sels.join(', ');
        for (const el of document.querySelectorAll(selectors)) {
            if (!candSet.has(el)) { candidates.push(el); candSet.add(el); }
        }

        // Pass 2: cursor:pointer elements (catches React/Vue/Angular components)
        // Skip SVG primitives — the wrapping button/link is what we want.
        const SVG_SKIP = new Set([
            'path','rect','circle','ellipse','line','polyline','polygon',
            'g','use','symbol','defs','mask','pattern','svg','tspan','text',
        ]);
        for (const el of document.querySelectorAll('*')) {
            if (candSet.has(el)) continue;
            if (SVG_SKIP.has(el.tagName.toLowerCase())) continue;
            const rect = el.getBoundingClientRect();
            if (rect.width < 3 || rect.height < 3) continue;
            if (rect.top > window.innerHeight * 2 || rect.bottom < 0) continue;
            try {
                if (window.getComputedStyle(el).cursor === 'pointer') {
                    candidates.push(el); candSet.add(el);
                }
            } catch(e) {}
        }

        // --- Outermost-wins dedup (from browser-use) ---
        // Drop any candidate whose ancestor is also a candidate,
        // UNLESS the candidate has role=gridcell/option or is inside a
        // grid/listbox (calendar cells, date pickers, select options).
        const KEEP_NESTED_ROLES = new Set(['gridcell', 'option', 'row', 'cell', 'slider']);
        const KEEP_NESTED_TAGS = new Set(['input', 'select', 'textarea', 'button']);
        const outermost = [];
        for (const el of candidates) {
            const elRole = el.getAttribute('role') || '';
            const elTag = el.tagName.toLowerCase();
            // Always keep: grid cells, options, form controls
            if (KEEP_NESTED_ROLES.has(elRole) || KEEP_NESTED_TAGS.has(elTag)) {
                outermost.push(el);
                continue;
            }
            let p = el.parentElement;
            let dominated = false;
            while (p) {
                if (candSet.has(p)) {
                    // This element is nested inside another candidate.
                    // Keep it only if it has substantially different text
                    // (e.g. a link inside a nav item with extra text).
                    const parentText = (p.innerText || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
                    const childText = (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
                    if (childText && parentText && childText === parentText) {
                        // Same text — child is decorative, drop it
                        dominated = true;
                    } else if (!childText || childText.length < 2) {
                        // No meaningful text on child — drop it
                        dominated = true;
                    }
                    // If child has different/additional text, keep both
                    break;
                }
                p = p.parentElement;
            }
            if (!dominated) outermost.push(el);
        }

        // --- Build results from outermost list ---
        const results = [];
        const seen = new Set();
        for (const el of outermost) {
            const rect = el.getBoundingClientRect();
            if (rect.width < 3 || rect.height < 3) continue;
            if (rect.top > window.innerHeight * 2 || rect.bottom < 0) continue;
            if (rect.left > window.innerWidth || rect.right < 0) continue;
            try {
                const style = window.getComputedStyle(el);
                if (style.visibility === 'hidden' || style.display === 'none') continue;
                if (parseFloat(style.opacity) < 0.1) continue;
            } catch(e) { continue; }
            const posKey = Math.round(rect.x/5) + ',' + Math.round(rect.y/5) + ',' + Math.round(rect.width/5);
            if (seen.has(posKey)) continue;
            seen.add(posKey);
            let text = '';
            // 10-step name resolution
            const ariaLabel = el.getAttribute('aria-label');
            const ariaLabelledBy = el.getAttribute('aria-labelledby');
            if (ariaLabel) { text = ariaLabel; }
            else if (ariaLabelledBy) { const ref = document.getElementById(ariaLabelledBy); if (ref) text = ref.textContent || ''; }
            else { text = (el.innerText || el.textContent || '').trim(); }
            if (!text || text.length < 2) text = el.value || '';
            if (!text || text.length < 2) text = el.getAttribute('placeholder') || '';
            if (!text || text.length < 2) text = el.getAttribute('title') || '';
            if (!text || text.length < 2) text = el.getAttribute('alt') || '';
            if (!text || text.length < 2) text = el.getAttribute('data-tooltip') || el.getAttribute('data-testid') || '';
            if (!text || text.length < 2) text = el.getAttribute('name') || '';
            text = (text || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
            if (results.length >= 150) break;
            const role = el.getAttribute('role') || el.tagName.toLowerCase();
            const entry = {
                tag: el.tagName.toLowerCase(),
                text: text,
                role: role,
                bbox: {x: rect.x, y: rect.y, width: rect.width, height: rect.height},
            };
            if (role === 'slider' || el.type === 'range') {
                entry.valuemax = el.getAttribute('aria-valuemax') || el.max || '';
                entry.valuemin = el.getAttribute('aria-valuemin') || el.min || '';
                entry.valuenow = el.getAttribute('aria-valuenow') || el.value || '';
            }
            results.push(entry);
        }

        return results;
    }""")

    # Outermost-wins dedup now runs in JS (preserves calendar cells via role check).
    # No Python-side dedupe needed.
    for i, el in enumerate(elements):
        el["index"] = i + 1

    return elements


def dedupe_elements(elements: list[dict]) -> list[dict]:
    """Remove nested decorations — if element A fully contains element B, keep only A."""
    if not elements:
        return []

    keep = []
    for i, el in enumerate(elements):
        bbox_i = el["bbox"]
        is_contained = False
        for j, other in enumerate(elements):
            if i == j:
                continue
            bbox_j = other["bbox"]
            if (bbox_j["x"] <= bbox_i["x"] and
                bbox_j["y"] <= bbox_i["y"] and
                bbox_j["x"] + bbox_j["width"] >= bbox_i["x"] + bbox_i["width"] and
                bbox_j["y"] + bbox_j["height"] >= bbox_i["y"] + bbox_i["height"] and
                (bbox_j["width"] * bbox_j["height"]) > (bbox_i["width"] * bbox_i["height"])):
                is_contained = True
                break
        if not is_contained:
            keep.append(el)

    return keep
