"""Precondition layer — detect CAPTCHA, login walls, geo-blocks, rate limits."""
from __future__ import annotations

import httpx

BLOCK_PATTERNS = [
    "captcha", "are you a human", "verify you are human", "confirm you are human",
    "access denied", "please sign in", "login required", "log in to continue",
    "unusual traffic", "rate limit", "too many requests",
    "403 forbidden", "401 unauthorized",
    "geo-restricted", "not available in your region", "not available in your country",
    "enable javascript", "please enable cookies",
    "robot", "automated access", "bot detection",
]

BLOCK_TITLE_PATTERNS = [
    "access denied", "just a moment", "attention required",
    "are you a robot", "security check", "verify",
]


async def check_blocked_http(url: str, timeout: float = 10.0) -> str | None:
    """Quick HTTP GET to detect blocks before launching a browser.

    Returns "gateway_blocked" if blocked, None if OK.
    Only triggers on obvious block pages — not pages that merely mention login.
    """
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"},
        ) as client:
            resp = await client.get(url)

            if resp.status_code in (401, 403, 429, 503):
                return "gateway_blocked"

            # Only check body patterns if the page is suspiciously short (block pages)
            # or looks like a challenge page (not a normal content page)
            body = resp.text
            if len(body) > 10000:
                return None

            text_lower = body[:5000].lower()
            title_match = _extract_title(text_lower)
            if title_match:
                for pattern in BLOCK_TITLE_PATTERNS:
                    if pattern in title_match:
                        return "gateway_blocked"

            block_score = 0
            for pattern in BLOCK_PATTERNS:
                if pattern in text_lower:
                    if _is_real_block(text_lower, pattern):
                        block_score += 1
            if block_score >= 2:
                return "gateway_blocked"

    except httpx.TimeoutException:
        return "gateway_blocked"
    except httpx.ConnectError:
        return "gateway_blocked"
    except Exception:
        pass

    return None


def _extract_title(html_lower: str) -> str:
    """Extract <title> content from HTML."""
    import re
    match = re.search(r'<title[^>]*>(.*?)</title>', html_lower)
    return match.group(1).strip() if match else ""


def check_page_content(html: str, title: str = "") -> str | None:
    """Check rendered page for block signals. Used after Playwright navigation.

    Only triggers on obvious block pages — short pages with block indicators in title.
    Long pages with real content are not blocked (scripts may mention captcha internally).
    """
    title_lower = title.lower()

    for pattern in BLOCK_TITLE_PATTERNS:
        if pattern in title_lower:
            return "gateway_blocked"

    # If the page has substantial content, it's not a block page
    if len(html) > 20000:
        return None

    # Short pages might be block/challenge pages
    text_lower = html[:5000].lower()
    block_score = 0
    for pattern in BLOCK_PATTERNS:
        if pattern in text_lower:
            if _is_real_block(text_lower, pattern):
                block_score += 1
    if block_score >= 2:
        return "gateway_blocked"

    return None


def _is_real_block(text: str, pattern: str) -> bool:
    """Reduce false positives — ignore mentions in nav/footer context."""
    idx = text.find(pattern)
    if idx < 0:
        return False
    context = text[max(0, idx - 100):idx + len(pattern) + 100]
    false_positive_signals = ["privacy policy", "cookie policy", "terms of", "footer"]
    for fp in false_positive_signals:
        if fp in context:
            return False
    return True
