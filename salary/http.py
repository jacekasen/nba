"""Conservative HTTP helpers with local HTML caching and exponential backoff."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import requests

from salary.config import DEFAULT_BACKOFF_BASE_SECONDS, DEFAULT_MAX_RETRIES, DEFAULT_REQUEST_DELAY_SECONDS, USER_AGENT

logger = logging.getLogger(__name__)


class AccessBlockedError(RuntimeError):
    """Raised when Basketball Reference rejects or challenges the client."""


class HttpClient:
    """Tiny requests wrapper that caches responses on disk."""

    def __init__(
        self,
        *,
        delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS,
        user_agent: str = USER_AGENT,
        session: requests.Session | None = None,
    ) -> None:
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self._last_request_at: float | None = None

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _looks_blocked(self, response: requests.Response) -> bool:
        if response.status_code in {401, 403, 429, 503}:
            return True
        text_head = response.text[:4000].lower()
        blockers = (
            "captcha",
            "cf-challenge",
            "just a moment",
            "access denied",
            "request blocked",
            "unusual traffic",
        )
        return any(token in text_head for token in blockers) and "basketball-reference" not in text_head

    def fetch_html(self, url: str, cache_path: Path, *, refresh: bool = False) -> str:
        """Return HTML for url, using cache_path unless refresh is requested."""
        if cache_path.exists() and not refresh:
            logger.info("Using cached HTML: %s", cache_path)
            return cache_path.read_text(encoding="utf-8")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                logger.info("GET %s (attempt %s/%s)", url, attempt, self.max_retries)
                response = self.session.get(url, timeout=45)
                self._last_request_at = time.monotonic()

                if self._looks_blocked(response):
                    raise AccessBlockedError(
                        f"Basketball Reference appears to be blocking requests "
                        f"(HTTP {response.status_code}) for {url}. "
                        "Stopping to preserve collected data."
                    )

                if response.status_code >= 500:
                    raise requests.HTTPError(f"Server error HTTP {response.status_code}", response=response)

                if response.status_code >= 400:
                    response.raise_for_status()

                html = response.text
                cache_path.write_text(html, encoding="utf-8")
                return html
            except AccessBlockedError:
                raise
            except (requests.RequestException, OSError) as exc:
                last_error = exc
                sleep_for = self.backoff_base_seconds * (2 ** (attempt - 1)) + random.uniform(0, 1.5)
                logger.warning("Request failed (%s). Backing off %.1fs", exc, sleep_for)
                time.sleep(sleep_for)

        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} attempts") from last_error
