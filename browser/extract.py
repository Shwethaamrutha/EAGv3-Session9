"""Layer 1 — Static extraction via httpx + trafilatura (primary) / readability (fallback). Zero LLM cost."""
from __future__ import annotations

import re

import httpx
import trafilatura
from readability import Document
from markdownify import markdownify


async def extract_static(url: str, goal_keywords: list[str], timeout: float = 10.0) -> tuple[str | None, bool]:
    """Fetch URL and extract main content.

    Returns (extracted_text, is_sufficient).
    is_sufficient = len >= 200 chars AND at least one goal keyword found.
    """
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except Exception:
        return None, False

    content_type = resp.headers.get("content-type", "")
    if "json" in content_type:
        text = resp.text[:10000]
        sufficient = len(text) >= 200 and _has_keyword(text, goal_keywords)
        return text, sufficient

    if "html" not in content_type and "text" not in content_type:
        return None, False

    html = resp.text
    if not html or len(html) < 100:
        return None, False

    # Primary: trafilatura (handles listing pages, articles, etc.)
    text = trafilatura.extract(html, output_format="txt")
    if text:
        text = _clean_text(text)

    # Fallback: readability + markdownify
    if not text or len(text) < 50:
        try:
            doc = Document(html)
            summary_html = doc.summary()
            text = markdownify(summary_html, strip=["img", "svg", "script", "style"])
            text = _clean_text(text)
        except Exception:
            pass

    if not text or len(text) < 50:
        return None, False

    # Sufficient if: substantial content (500+ chars) OR keyword match with 200+ chars
    sufficient = len(text) >= 500 or (len(text) >= 200 and _has_keyword(text, goal_keywords))
    return text, sufficient


def _has_keyword(text: str, keywords: list[str]) -> bool:
    """Check if any goal keyword appears in the extracted text."""
    if not keywords:
        return True
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and empty lines."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()
