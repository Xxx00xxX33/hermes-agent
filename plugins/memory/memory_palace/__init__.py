"""Memory Palace memory plugin using the MemoryProvider interface.

MVP scope:
- Automatic recall from a durable Memory Palace store
- Selective turn capture for preference / strategy / failure_lesson events
- Session-end summaries written to durable notes
- No provider-specific tools yet (context-only provider)

Configuration via environment variables (profile-scoped via each profile's .env):
  MEMORY_PALACE_URL      — Backend base URL (required)
  MEMORY_PALACE_API_KEY  — API key for write/search endpoints (required)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_DEFAULT_API_TIMEOUT = 5.0
_DEFAULT_MAX_RECALL_RESULTS = 3
_DEFAULT_SEARCH_MODE = "hybrid"
_DEFAULT_PATH_PREFIX = "external_agents"
_DEFAULT_EVENTS_URI = "notes://external_agents/hermes_events"
_DEFAULT_SESSIONS_URI = "notes://external_agents/hermes_sessions"
_DEFAULT_PREFERENCE_ROLLUP_URI = "core://external_agents/hermes_user_preference_rollup"
_MIN_SENTENCE_LEN = 12
_TRIVIAL_RE = re.compile(r"^(ok|okay|thanks|thank you|got it|sure|yes|no|yep|nope|k|ty|thx|np)\.?$", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_PREFERENCE_PATTERNS = (
    re.compile(r"\b(i prefer|i want|i hope|i need|please|let's|lets)\b", re.IGNORECASE),
    re.compile(r"(我希望|我更喜欢|我想要|请|不要|别)")
)
_STRATEGY_PATTERNS = (
    re.compile(r"\b(prefer .* over|best .* is|recommended|should use|should keep|the best way|the right way)\b", re.IGNORECASE),
    re.compile(r"(最好的方式|最佳做法|建议|应该用|应该保持)")
)
_FAILURE_PATTERNS = (
    re.compile(r"\b(do not|don't|avoid|never put|must not)\b", re.IGNORECASE),
    re.compile(r"(不要|避免|别把|不应)")
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_uri(uri: str) -> tuple[str, str]:
    if "://" not in uri:
        return ("core", uri)
    domain, _, path = uri.partition("://")
    return domain, path


def _split_parent_and_title(path: str) -> tuple[str, str]:
    if not path:
        return ("", "")
    if "/" not in path:
        return ("", path)
    parent, _, title = path.rpartition("/")
    return parent, title


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _trim_sentence(text: str, limit: int = 280) -> str:
    normalized = _normalize_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _split_sentences(text: str) -> List[str]:
    cleaned = _normalize_text(text)
    if not cleaned:
        return []
    parts = _SENTENCE_SPLIT_RE.split(cleaned)
    return [part.strip() for part in parts if part and len(part.strip()) >= _MIN_SENTENCE_LEN]


def _is_trivial(text: str) -> bool:
    return bool(_TRIVIAL_RE.match((text or "").strip()))


def _matches_any(sentence: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(sentence or "") for pattern in patterns)


def _make_event(event_type: str, content: str, *, session_id: str, source_hook: str) -> Dict[str, str]:
    return {
        "timestamp": _utc_now(),
        "session_id": session_id,
        "event_type": event_type,
        "source_hook": source_hook,
        "content": _trim_sentence(content),
    }


def _render_event_line(event: Dict[str, str]) -> str:
    return (
        f"- [{event['timestamp']}] [{event['event_type']}] {event['content']} "
        f"(session: {event['session_id']}, source: {event['source_hook']})\n"
    )


def _unique_in_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        normalized = _normalize_text(item).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(_normalize_text(item))
    return out


def _extract_turn_events(user_content: str, assistant_content: str, *, session_id: str) -> List[Dict[str, str]]:
    events: List[Dict[str, str]] = []

    for sentence in _split_sentences(user_content):
        if _is_trivial(sentence):
            continue
        if _matches_any(sentence, _PREFERENCE_PATTERNS):
            events.append(_make_event(
                "preference",
                f"User preference: {sentence}",
                session_id=session_id,
                source_hook="sync_turn",
            ))
            break

    for sentence in _split_sentences(assistant_content):
        if _matches_any(sentence, _STRATEGY_PATTERNS):
            events.append(_make_event(
                "strategy",
                f"Strategy: {sentence}",
                session_id=session_id,
                source_hook="sync_turn",
            ))
            break

    for sentence in _split_sentences(assistant_content):
        if _matches_any(sentence, _FAILURE_PATTERNS):
            events.append(_make_event(
                "failure_lesson",
                f"Failure lesson: {sentence}",
                session_id=session_id,
                source_hook="sync_turn",
            ))
            break

    deduped: List[Dict[str, str]] = []
    seen = set()
    for event in events:
        key = f"{event['event_type']}::{event['content'].lower()}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def _build_session_summary(messages: List[Dict[str, Any]], *, session_id: str) -> str:
    user_messages = [str(m.get("content", "")).strip() for m in messages if m.get("role") == "user"]
    assistant_messages = [str(m.get("content", "")).strip() for m in messages if m.get("role") == "assistant"]
    if not user_messages and not assistant_messages:
        return ""

    preferences: List[str] = []
    strategies: List[str] = []
    lessons: List[str] = []

    for user_text in user_messages:
        for sentence in _split_sentences(user_text):
            if _matches_any(sentence, _PREFERENCE_PATTERNS):
                preferences.append(f"User preference: {sentence}")
                break

    for assistant_text in assistant_messages:
        for sentence in _split_sentences(assistant_text):
            if _matches_any(sentence, _STRATEGY_PATTERNS):
                strategies.append(f"Strategy: {sentence}")
                break
        for sentence in _split_sentences(assistant_text):
            if _matches_any(sentence, _FAILURE_PATTERNS):
                lessons.append(f"Failure lesson: {sentence}")
                break

    lines = [
        f"## Session {session_id}",
        f"- Ended: {_utc_now()}",
    ]
    if user_messages:
        lines.append(f"- Intent: {_trim_sentence(user_messages[0], 220)}")
    if preferences:
        lines.append("- Preferences confirmed:")
        lines.extend(f"  - {item}" for item in _unique_in_order(preferences))
    if strategies:
        lines.append("- Strategies learned:")
        lines.extend(f"  - {item}" for item in _unique_in_order(strategies))
    if lessons:
        lines.append("- Failure lessons:")
        lines.extend(f"  - {item}" for item in _unique_in_order(lessons))

    if len(lines) <= 2:
        return ""
    return "\n".join(lines) + "\n"


class _MemoryPalaceClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = _DEFAULT_API_TIMEOUT):
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._session = requests.Session()

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            logger.debug("Memory Palace session close failed", exc_info=True)

    def _read_headers(self) -> dict[str, str]:
        return {"Accept": "application/json"}

    def _write_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["X-MCP-API-Key"] = self._api_key
        return headers

    def search(self, query: str, *, max_results: int, mode: str, path_prefix: str = "") -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "query": query,
            "mode": mode,
            "max_results": max_results,
            "candidate_multiplier": 4,
            "include_session": True,
            "filters": {},
        }
        if path_prefix:
            payload["filters"]["path_prefix"] = path_prefix
        try:
            resp = self._session.post(
                f"{self._base}/maintenance/observability/search",
                headers=self._write_headers(),
                data=json.dumps(payload),
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
        except Exception:
            logger.debug("Memory Palace search failed", exc_info=True)
            return []
        if not isinstance(data, dict) or not data.get("ok"):
            return []
        results = data.get("results")
        return results if isinstance(results, list) else []

    def read(self, uri: str) -> Optional[str]:
        domain, path = _parse_uri(uri)
        try:
            resp = self._session.get(
                f"{self._base}/browse/node",
                headers=self._read_headers(),
                params={"path": path, "domain": domain},
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
        except Exception:
            logger.debug("Memory Palace read failed", exc_info=True)
            return None
        node = data.get("node") if isinstance(data, dict) else None
        if isinstance(node, dict):
            content = node.get("content")
            return content if isinstance(content, str) else ""
        return None

    def _create_node(
        self,
        *,
        domain: str,
        parent_path: str,
        title: str,
        content: str,
        priority: int,
        disclosure: str = "",
    ) -> bool:
        payload: Dict[str, Any] = {
            "domain": domain,
            "parent_path": parent_path,
            "title": title,
            "content": content,
            "priority": priority,
        }
        if disclosure:
            payload["disclosure"] = disclosure
        try:
            resp = self._session.post(
                f"{self._base}/browse/node",
                headers=self._write_headers(),
                data=json.dumps(payload),
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                logger.debug(
                    "Memory Palace create failed: status=%s body=%s",
                    resp.status_code,
                    getattr(resp, "text", "")[:400],
                )
                return False
            data = resp.json()
        except Exception:
            logger.debug("Memory Palace create failed", exc_info=True)
            return False
        return bool(isinstance(data, dict) and data.get("created") is True)

    def _ensure_parent_chain(self, *, domain: str, parent_path: str, priority: int, disclosure: str = "") -> bool:
        if not parent_path:
            return True

        accumulated: List[str] = []
        parent_disclosure = disclosure or "When recalling Hermes durable memory namespaces"
        for segment in [part for part in parent_path.split("/") if part]:
            accumulated.append(segment)
            current_path = "/".join(accumulated)
            current_uri = f"{domain}://{current_path}"
            if self.read(current_uri) is not None:
                continue

            ancestor_parent, title = _split_parent_and_title(current_path)
            created = self._create_node(
                domain=domain,
                parent_path=ancestor_parent,
                title=title,
                content=(
                    f"Auto-created namespace node for Hermes durable memory at {current_uri}."
                ),
                priority=priority,
                disclosure=parent_disclosure,
            )
            if not created and self.read(current_uri) is None:
                return False
        return True

    def create(self, uri: str, content: str, *, priority: int, disclosure: str = "") -> bool:
        domain, path = _parse_uri(uri)
        parent_path, title = _split_parent_and_title(path)
        if not title:
            return False
        if not self._ensure_parent_chain(
            domain=domain,
            parent_path=parent_path,
            priority=priority,
            disclosure=disclosure,
        ):
            return False
        created = self._create_node(
            domain=domain,
            parent_path=parent_path,
            title=title,
            content=content,
            priority=priority,
            disclosure=disclosure,
        )
        if created:
            return True
        return self.read(uri) is not None

    def replace(self, uri: str, content: str) -> bool:
        domain, path = _parse_uri(uri)
        try:
            resp = self._session.put(
                f"{self._base}/browse/node",
                headers=self._write_headers(),
                params={"path": path, "domain": domain},
                data=json.dumps({"content": content}),
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                return False
            data = resp.json()
        except Exception:
            logger.debug("Memory Palace replace failed", exc_info=True)
            return False
        return bool(isinstance(data, dict) and data.get("updated") is True)

    def append_or_create(self, uri: str, text: str, *, priority: int, disclosure: str = "") -> bool:
        current = self.read(uri)
        if current is None:
            return self.create(uri, text, priority=priority, disclosure=disclosure)
        if text.strip() and text.strip() in current:
            return True
        new_content = current
        if new_content and not new_content.endswith("\n"):
            new_content += "\n"
        new_content += text.lstrip()
        return self.replace(uri, new_content)


class MemoryPalaceMemoryProvider(MemoryProvider):
    def __init__(self):
        self._base_url = ""
        self._api_key = ""
        self._client: Optional[_MemoryPalaceClient] = None
        self._session_id = ""
        self._write_enabled = True
        self._sync_thread: Optional[threading.Thread] = None
        self._summary_thread: Optional[threading.Thread] = None
        self._max_recall_results = _DEFAULT_MAX_RECALL_RESULTS
        self._search_mode = _DEFAULT_SEARCH_MODE
        self._path_prefix = _DEFAULT_PATH_PREFIX

    @property
    def name(self) -> str:
        return "memory_palace"

    def is_available(self) -> bool:
        return bool(self._get_base_url() and self._get_api_key())

    def _get_base_url(self) -> str:
        return (os.environ.get("MEMORY_PALACE_URL") or os.environ.get("HERMES_MEMORY_PALACE_URL") or "").strip()

    def _get_api_key(self) -> str:
        return (os.environ.get("MEMORY_PALACE_API_KEY") or os.environ.get("HERMES_MEMORY_PALACE_API_KEY") or "").strip()

    def get_config_schema(self):
        return [
            {
                "key": "base_url",
                "description": "Memory Palace backend base URL",
                "required": True,
                "env_var": "MEMORY_PALACE_URL",
            },
            {
                "key": "api_key",
                "description": "Memory Palace API key",
                "secret": True,
                "required": True,
                "env_var": "MEMORY_PALACE_API_KEY",
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._base_url = self._get_base_url()
        self._api_key = self._get_api_key()
        agent_context = kwargs.get("agent_context", "primary")
        self._write_enabled = agent_context not in ("cron", "flush", "subagent")
        if self.is_available():
            self._client = _MemoryPalaceClient(self._base_url, self._api_key, _DEFAULT_API_TIMEOUT)
        else:
            self._client = None

    def system_prompt_block(self) -> str:
        if not self._client:
            return ""
        return (
            "# Memory Palace\n"
            "Active. Automatic recall is enabled for durable preferences, strategies, and failure lessons. "
            "Use recalled context silently and treat it as background memory, not new user input."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._client or not query.strip():
            return ""
        results = self._client.search(
            query=query,
            max_results=self._max_recall_results,
            mode=self._search_mode,
            path_prefix=self._path_prefix,
        )
        if not results:
            return ""

        lines: List[str] = []
        for item in results[: self._max_recall_results]:
            snippet = _normalize_text(str(item.get("snippet") or item.get("content") or ""))
            uri = _normalize_text(str(item.get("uri") or ""))
            if snippet:
                lines.append(f"- {snippet}")
            elif uri:
                lines.append(f"- {uri}")

        if not lines:
            return ""
        return "## Memory Palace Recall\n" + "\n".join(lines)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._client or not self._write_enabled:
            return
        events = _extract_turn_events(user_content, assistant_content, session_id=session_id or self._session_id)
        if not events:
            return

        def _run() -> None:
            preference_lines: List[str] = []
            for event in events:
                self._client.append_or_create(
                    _DEFAULT_EVENTS_URI,
                    _render_event_line(event),
                    priority=5,
                    disclosure="When recalling Hermes durable learning events",
                )
                if event["event_type"] == "preference":
                    preference_lines.append(f"- {event['content']}\n")

            for line in preference_lines:
                self._client.append_or_create(
                    _DEFAULT_PREFERENCE_ROLLUP_URI,
                    line,
                    priority=2,
                    disclosure="When recalling Hermes user preference rollups",
                )

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=2.0)
        self._sync_thread = threading.Thread(target=_run, daemon=True, name="memory-palace-sync")
        self._sync_thread.start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._client or not self._write_enabled:
            return
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        summary = _build_session_summary(messages, session_id=self._session_id)
        if not summary:
            return

        def _run() -> None:
            self._client.append_or_create(
                _DEFAULT_SESSIONS_URI,
                summary,
                priority=4,
                disclosure="When recalling Hermes session summaries",
            )

        if self._summary_thread and self._summary_thread.is_alive():
            self._summary_thread.join(timeout=2.0)
        self._summary_thread = threading.Thread(target=_run, daemon=True, name="memory-palace-summary")
        self._summary_thread.start()
        self._summary_thread.join(timeout=5.0)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def shutdown(self) -> None:
        for thread in (self._sync_thread, self._summary_thread):
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
        if self._client:
            self._client.close()
        self._sync_thread = None
        self._summary_thread = None


def register(ctx) -> None:
    ctx.register_memory_provider(MemoryPalaceMemoryProvider())
