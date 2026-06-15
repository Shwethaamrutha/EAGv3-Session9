"""MCP Server with 11 tools for agent7.

Tools: web_search, fetch_url, get_time, currency_convert,
       read_file, list_dir, create_file, update_file, edit_file,
       index_document, search_knowledge

Heavy deps (crawl4ai, duckduckgo-search) are lazy-imported inside tool functions
to keep subprocess startup fast (~1s instead of ~8s).
"""
from __future__ import annotations
import sys; sys.path.insert(0, "agent")

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

if os.getenv("MCP_LOG_LEVEL") == "error":
    logging.disable(logging.CRITICAL)

load_dotenv()

mcp = FastMCP("agent7-tools")

SANDBOX_DIR = Path("state/sandbox")
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily with full page content. Returns titles, URLs,
    and complete article text for each result. This is usually sufficient for
    synthesis — only use fetch_url for URLs not found via search."""
    max_results = int(max_results)

    # Primary: Tavily search with content summaries
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            response = client.search(query, max_results=max_results, search_depth="advanced")
            results = response.get("results", [])
            if results:
                lines = []
                for r in results:
                    lines.append(f"Title: {r.get('title', '')}")
                    lines.append(f"URL: {r.get('url', '')}")
                    content = r.get("content", "")
                    lines.append(f"Content: {content}")
                    lines.append("")
                # Also include Tavily's direct answer if available
                answer = response.get("answer", "")
                if answer:
                    lines.insert(0, f"Direct answer: {answer}\n")
                return "\n".join(lines)
        except Exception:
            pass

    # Fallback: DuckDuckGo
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if results:
            lines = []
            for r in results:
                lines.append(f"Title: {r['title']}")
                lines.append(f"URL: {r['href']}")
                lines.append(f"Snippet: {r['body']}")
                lines.append("")
            return "\n".join(lines)
    except Exception:
        pass

    return "[ERROR] No results found. Search services may be rate-limited."


@mcp.tool()
async def fetch_url(url: str) -> str:
    """Fetch a URL and return its content as cleaned markdown using Crawl4AI."""
    import httpx
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; Agent6/1.0)"})
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

            if "json" in content_type:
                return resp.text[:100000]

            if "text" in content_type or "html" in content_type:
                html = resp.text

                # Use readability to extract main article content (like Firefox Reader View)
                try:
                    from readability import Document
                    from markdownify import markdownify as md
                    doc = Document(html)
                    clean_html = doc.summary()
                    title = doc.title()
                    text = md(clean_html, heading_style="ATX", strip=["img", "svg"])
                    text = f"# {title}\n\n{text}" if title else text
                except ImportError:
                    import re
                    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r"<[^>]+>", " ", text)

                import re
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = re.sub(r" {2,}", " ", text)
                return text.strip()[:80000]

            return f"Binary content ({content_type}), {len(resp.content)} bytes"
    except Exception as e:
        return f"[ERROR] Fetch error: {e}"


@mcp.tool()
async def get_time(timezone: str = "UTC") -> str:
    """Get the current date and time. Timezone can be 'UTC', 'local', or an IANA timezone name."""
    from datetime import timezone as tz, timedelta
    from zoneinfo import ZoneInfo

    if timezone.lower() == "local":
        now = datetime.now().astimezone()
        tz_name = str(now.tzinfo)
    elif timezone.upper() == "UTC":
        now = datetime.now(tz.utc)
        tz_name = "UTC"
    else:
        try:
            now = datetime.now(ZoneInfo(timezone))
            tz_name = timezone
        except Exception:
            now = datetime.now(tz.utc)
            tz_name = f"UTC ('{timezone}' not recognized)"

    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({tz_name})"


@mcp.tool()
async def currency_convert(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency using a free exchange rate API."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            rate = data["rates"].get(to_currency.upper())
            if rate is None:
                return f"[ERROR] Currency {to_currency} not found."
            result = amount * rate
            return f"{amount} {from_currency.upper()} = {result:.2f} {to_currency.upper()} (rate: {rate})"
    except Exception as e:
        return f"[ERROR] Conversion error: {e}"


@mcp.tool()
async def read_file(path: str) -> str:
    """Read a file from the sandbox directory. Path is relative to state/sandbox/."""
    target = SANDBOX_DIR / path
    if not target.exists():
        return f"[ERROR] File not found: {path}"
    try:
        return target.read_text()
    except Exception as e:
        return f"[ERROR] Read error: {e}"


@mcp.tool()
async def list_dir(path: str = ".") -> str:
    """List files in a sandbox directory. Path is relative to state/sandbox/."""
    target = SANDBOX_DIR / path
    if not target.exists():
        return f"[ERROR] Directory not found: {path}"
    if not target.is_dir():
        return f"[ERROR] Not a directory: {path}"
    entries = []
    for item in sorted(target.iterdir()):
        kind = "dir" if item.is_dir() else "file"
        entries.append(f"  [{kind}] {item.name}")
    return "\n".join(entries) if entries else "(empty directory)"


@mcp.tool()
async def create_file(path: str, content: str) -> str:
    """Create or overwrite a file in the sandbox directory. Path is relative to state/sandbox/."""
    target = SANDBOX_DIR / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Created: {path} ({len(content)} bytes)"


@mcp.tool()
async def update_file(path: str, content: str) -> str:
    """Overwrite an existing file in the sandbox. Path is relative to state/sandbox/."""
    target = SANDBOX_DIR / path
    if not target.exists():
        return f"[ERROR] File not found: {path}. Use create_file instead."
    target.write_text(content)
    return f"Updated: {path} ({len(content)} bytes)"


@mcp.tool()
async def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Replace old_text with new_text in a sandbox file. Path is relative to state/sandbox/."""
    target = SANDBOX_DIR / path
    if not target.exists():
        return f"[ERROR] File not found: {path}"
    current = target.read_text()
    if old_text not in current:
        return f"[ERROR] old_text not found in {path}"
    updated = current.replace(old_text, new_text, 1)
    target.write_text(updated)
    return f"Edited: {path}"


@mcp.tool()
async def index_document(path: str, chunk_size: int = 400, overlap: int = 80) -> str:
    """Chunk a sandbox file or artifact and write the chunks into Memory as
    fact records, where they become FAISS-searchable for later queries.
    Use this when the content must be searchable across later turns or runs.
    For one-shot inspection of a file's contents, use read_file."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from memory import memory

    # Resolve source text
    if path.startswith("art:"):
        from artifacts import artifact_store
        art_id = path[4:]
        if not artifact_store.exists(art_id):
            return f"[ERROR] Artifact not found: {art_id}"
        blob = artifact_store.get_bytes(art_id)
        text = blob.decode("utf-8", errors="replace")
        source_label = f"artifact:{art_id}"
    else:
        # Strip common prefix mistakes (e.g. "sandbox:" from descriptor labels)
        clean_path = path.removeprefix("sandbox:").removeprefix("state/sandbox/")
        target = SANDBOX_DIR / clean_path
        if not target.exists():
            return f"[ERROR] File not found: {clean_path}"
        text = target.read_text()
        source_label = f"sandbox:{clean_path}"

    # Clean noisy content before chunking
    import re as _re
    # Remove everything after References/Acknowledgements
    for pattern in [r'(?i)## References.*', r'(?i)## Acknowledgements.*']:
        text = _re.split(pattern, text)[0]
    # Remove license/attribution boilerplate at the start
    text = _re.sub(r'(?i)Provided proper attribution.*?(?=\n#|\n\n)', '', text)
    # Remove citation list noise
    text = _re.sub(r'\* \[\d+\].*?\n', '', text)
    # Remove LaTeX/math rendering artifacts aggressively
    text = _re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)  # \command{...}
    text = _re.sub(r'\\[a-zA-Z]+', '', text)  # \command
    text = _re.sub(r'\{[^}]*pgf[^}]*\}', '', text)
    text = _re.sub(r'start_POSTSUPERSCRIPT.*?end_POSTSUPERSCRIPT', '', text)
    text = _re.sub(r'start_POSTSUBSCRIPT.*?end_POSTSUBSCRIPT', '', text)
    text = _re.sub(r'blackboard_[A-Z]\s*', '', text)
    text = _re.sub(r'italic_\w+', '', text)
    text = _re.sub(r'[𝐀-𝐳𝑎-𝑧𝛼-𝜔𝚫𝚲]+', '', text)  # math unicode
    text = _re.sub(r'←|→|⋅|≤|≥|∈|⁢|⊤', ' ', text)  # math symbols
    text = _re.sub(r'\|[^|]*\|_[A-Z]', '', text)  # norms like ||X||_F
    # Clean up excessive whitespace from removals
    text = _re.sub(r'\n{3,}', '\n\n', text)
    text = _re.sub(r' {2,}', ' ', text)

    # Sliding window chunking
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append(chunk_text)
        start += chunk_size - overlap

    # Write each chunk as a fact into memory
    import uuid
    run_id = uuid.uuid4().hex[:8]
    total = len(chunks)
    for i, chunk_text in enumerate(chunks):
        descriptor = f"[{source_label} chunk {i+1}/{total}] {chunk_text[:100]}"
        keywords = list(set(
            w.lower().strip(".,!?;:'\"()-[]{}/@#$%^&*")
            for w in chunk_text.split()[:20]
            if len(w) > 3
        ))[:8]
        memory.add_fact(
            descriptor=descriptor,
            value={"chunk": chunk_text, "source": source_label, "chunk_index": i, "total_chunks": total},
            keywords=keywords,
            source=source_label,
            run_id=run_id,
        )

    return f"Indexed {total} chunks from {source_label}"


@mcp.tool()
async def run_command(command: str, timeout: int = 10) -> str:
    """Run a shell command and return stdout, stderr, and exit code.
    Commands run in the sandbox directory. Dangerous commands are blocked."""
    import subprocess
    import shlex

    # Safety: block dangerous commands
    blocked = ['rm -rf /', 'sudo', 'mkfs', 'dd if=', ':(){', 'chmod -R 777 /',
               'shutdown', 'reboot', 'kill -9 1', '> /dev/sda']
    cmd_lower = command.lower()
    for b in blocked:
        if b in cmd_lower:
            return f"[ERROR] Blocked dangerous command: {command}"

    timeout = min(timeout, 30)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).parent),
        )
        output_parts = []
        if result.stdout:
            output_parts.append(f"stdout:\n{result.stdout[:5000]}")
        if result.stderr:
            output_parts.append(f"stderr:\n{result.stderr[:2000]}")
        output_parts.append(f"exit_code: {result.returncode}")
        return "\n".join(output_parts)
    except subprocess.TimeoutExpired:
        return f"[ERROR] Command timed out after {timeout}s"
    except Exception as e:
        return f"[ERROR] {e}"


@mcp.tool()
async def count_syllables(text: str) -> str:
    """Count syllables in each line of text. Returns per-line counts and total."""
    import re
    import syllapy

    def _count_word(word: str) -> int:
        word = word.lower().strip(".,!?;:'\"()-")
        if not word:
            return 0
        return max(syllapy.count(word), 1)

    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    results = []
    for line in lines:
        words = re.findall(r"[a-zA-Z']+", line)
        syllables = sum(_count_word(w) for w in words)
        results.append(f"{line} → {syllables} syllables")
    return "\n".join(results)


@mcp.tool()
async def count_characters(text: str) -> str:
    """Count the exact number of characters in the given text."""
    return f"Character count: {len(text)} (including spaces and punctuation)"


@mcp.tool()
async def search_knowledge(query: str, k: int = 5) -> str:
    """Vector search over previously indexed fact chunks. Returns the top k
    most relevant chunks (default 5, max 8). Use this rather than re-fetching
    or re-reading source files when Memory already contains indexed chunks."""
    k = min(k, 8)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from memory import memory

    hits = memory.read(query, [], kinds=["fact"], top_k=k)
    if not hits:
        return "[No matching chunks found in indexed knowledge.]"

    results = []
    for h in hits:
        chunk_text = h.value.get("chunk", h.descriptor)
        source = h.value.get("source", h.source)
        chunk_idx = h.value.get("chunk_index", "?")
        total = h.value.get("total_chunks", "?")
        results.append(f"[{source} chunk {chunk_idx}/{total}]\n{chunk_text}")
    return "\n\n---\n\n".join(results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
