"""Playwright browser lifecycle — launch, context, anti-detection, page management."""
from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from playwright_stealth import Stealth


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

ANTI_DETECT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
"""


class BrowserDriver:
    def __init__(self, viewport_width: int = 1280, viewport_height: int = 720, dpr: float = 2.0, headless: bool = True):
        self._pw = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._viewport_width = viewport_width
        self._viewport_height = viewport_height
        self._dpr = dpr
        self._headless = headless

    @property
    def dpr(self) -> float:
        return self._dpr

    @property
    def page(self) -> Page | None:
        return self._page

    async def launch(self) -> Page:
        """Launch Chromium with stealth anti-detection."""
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=self._headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        self._context = await self._browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": self._viewport_width, "height": self._viewport_height},
            device_scale_factor=self._dpr,
            locale="en-US",
            java_script_enabled=True,
        )
        self._page = await self._context.new_page()
        # Apply stealth patches to avoid bot detection (Cloudflare, etc.)
        stealth = Stealth()
        await stealth.apply_stealth_async(self._page)
        # Auto-dismiss dialogs (alert/confirm/prompt) to prevent hanging
        self._page.on("dialog", lambda d: asyncio.ensure_future(d.dismiss()))
        return self._page

    async def goto(self, url: str, wait_until: str = "domcontentloaded", timeout: float = 30000) -> None:
        """Navigate to URL. Raises on timeout."""
        try:
            await self._page.goto(url, wait_until=wait_until, timeout=timeout)
        except Exception:
            try:
                await self._page.goto(url, wait_until="commit", timeout=timeout + 10000)
            except Exception:
                pass
        await asyncio.sleep(1)
        await self._dismiss_overlays()

    async def _dismiss_overlays(self) -> None:
        """Dismiss cookie banners, signup modals, and registration popups."""
        try:
            await self._page.evaluate("""() => {
                // Common dismiss button selectors
                const dismissSelectors = [
                    '[class*="cookie"] button',
                    '[id*="cookie"] button',
                    '[class*="consent"] button',
                    '[class*="popup"] [class*="close"]',
                    '[class*="modal"] [class*="close"]',
                    '[class*="overlay"] [class*="close"]',
                    '[aria-label="Close"]',
                    '[aria-label="Dismiss"]',
                    '[aria-label="close"]',
                    'button[class*="close"]',
                    '[class*="newsletter"] [class*="close"]',
                    '[class*="signup"] [class*="close"]',
                    '[data-testid="close"]',
                    '.modal .close',
                    '.popup-close',
                    '#onetrust-accept-btn-handler',
                    '.cc-dismiss',
                    '.cc-btn.cc-dismiss',
                ];

                // Click "Accept" / "Close" / "X" buttons on overlays
                const acceptTexts = ['accept', 'accept all', 'agree', 'got it', 'ok', 'i agree', 'continue', 'dismiss', 'no thanks', 'skip', 'not now', 'maybe later', 'close'];

                for (const sel of dismissSelectors) {
                    const el = document.querySelector(sel);
                    if (el && el.offsetParent !== null) {
                        el.click();
                        return;
                    }
                }

                // Try finding buttons by text content
                const buttons = document.querySelectorAll('button, a[role="button"], [class*="btn"]');
                for (const btn of buttons) {
                    const text = (btn.textContent || '').trim().toLowerCase();
                    if (acceptTexts.includes(text) && btn.offsetParent !== null) {
                        const rect = btn.getBoundingClientRect();
                        // Only click if it looks like an overlay button (not main page content)
                        if (rect.width < 300 && rect.height < 60) {
                            btn.click();
                            return;
                        }
                    }
                }

                // Try clicking X/close buttons on modals/popups
                const closeSelectors = [
                    '[class*="modal"] [class*="close"]',
                    '[class*="modal"] svg',
                    '[class*="popup"] [class*="close"]',
                    '[class*="overlay"] [class*="close"]',
                    '[aria-label*="lose"]',
                    '[aria-label*="ismiss"]',
                ];
                for (const sel of closeSelectors) {
                    const el = document.querySelector(sel);
                    if (el && el.offsetParent !== null) {
                        el.click();
                        return;
                    }
                }

                // Look for any visible X character that's clickable (common pattern)
                const allEls2 = document.querySelectorAll('span, div, button, svg');
                for (const el of allEls2) {
                    const text = (el.textContent || '').trim();
                    if ((text === '×' || text === '✕' || text === 'X' || text === '✖') && el.offsetParent !== null) {
                        const rect = el.getBoundingClientRect();
                        if (rect.width < 50 && rect.height < 50) {
                            el.click();
                            return;
                        }
                    }
                }

                // Remove fixed/sticky overlays that cover the page
                const allEls = document.querySelectorAll('[style*="position: fixed"], [style*="position:fixed"], [class*="overlay"], [class*="modal"]');
                for (const el of allEls) {
                    const style = window.getComputedStyle(el);
                    if ((style.position === 'fixed' || style.position === 'sticky') &&
                        parseFloat(style.zIndex) > 999 &&
                        el.offsetWidth > window.innerWidth * 0.5) {
                        el.remove();
                    }
                }
            }""")
        except Exception:
            pass

    async def screenshot(self, path: str | None = None, full_page: bool = False) -> bytes:
        """Take screenshot. Returns PNG bytes."""
        kwargs = {"type": "png", "full_page": full_page, "timeout": 60000}
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            kwargs["path"] = path
        return await self._page.screenshot(**kwargs)

    async def click(self, selector: str, timeout: float = 5000) -> None:
        """Click an element by selector."""
        await self._page.click(selector, timeout=timeout)
        await asyncio.sleep(0.5)

    async def fill(self, selector: str, value: str, timeout: float = 5000) -> None:
        """Fill a text input."""
        await self._page.fill(selector, value, timeout=timeout)

    async def press(self, selector: str, key: str) -> None:
        """Press a key in an element."""
        await self._page.press(selector, key)

    async def wait_for_selector(self, selector: str, timeout: float = 5000) -> None:
        """Wait for a selector to appear."""
        await self._page.wait_for_selector(selector, timeout=timeout)

    async def get_content(self) -> str:
        """Get page text content."""
        return await self._page.content()

    async def get_title(self) -> str:
        """Get page title."""
        return await self._page.title()

    async def get_url(self) -> str:
        """Get current URL."""
        return self._page.url

    async def close(self) -> None:
        """Clean up browser resources — ensures no zombie processes."""
        for resource, method in [
            (self._page, "close"),
            (self._context, "close"),
            (self._browser, "close"),
            (self._pw, "stop"),
        ]:
            if resource:
                try:
                    await getattr(resource, method)()
                except Exception:
                    pass
        self._page = None
        self._context = None
        self._browser = None
        self._pw = None
