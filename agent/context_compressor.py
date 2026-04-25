"""Automatic context window compression for long conversations.

Self-contained class with its own OpenAI client for summarization.
Uses auxiliary model (cheap/fast) to summarize middle turns while
protecting head and tail context.

Improvements over v2:
  - Structured summary template with Resolved/Pending question tracking
  - Summarizer preamble: "Do not respond to any questions" (from OpenCode)
  - Handoff framing: "different assistant" (from Codex) to create separation
  - "Remaining Work" replaces "Next Steps" to avoid reading as active instructions
  - Clear separator when summary merges into tail message
  - Iterative summary updates (preserves info across multiple compactions)
  - Token-budget tail protection instead of fixed message count
  - Tool output pruning before LLM summarization (cheap pre-pass)
  - Scaled summary budget (proportional to compressed content)
  - Richer tool call/result detail in summarizer input
"""

import hashlib
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import call_llm
from agent.context_engine import ContextEngine
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    get_model_context_length,
    estimate_messages_tokens_rough,
)

logger = logging.getLogger(__name__)

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Your current task or objective is identified near the top of the "
    "summary — resume exactly from there. "
    "Respond ONLY to the latest user message "
    "that appears AFTER this summary. The current session state (files, "
    "config, etc.) may reflect work described here — avoid repeating it:"
)
LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"
SESSION_GUIDE_HEADING = "Session Guide"
SESSION_GUIDE_SECTIONS = (
    "1. Summary",
    "2. Decisions",
    "3. Active Files",
    "4. Active Entities",
    "5. Unresolved Tasks",
    "6. Risks",
    "7. Validations",
    "8. Retrieval Handles",
)
_SESSION_GUIDE_EMPTY_ITEM = "None recorded in compacted turns."

# Minimum tokens for the summary output
_MIN_SUMMARY_TOKENS = 2000
# Proportion of compressed content to allocate for summary
_SUMMARY_RATIO = 0.20
# Absolute ceiling for summary tokens (even on very large context windows)
_SUMMARY_TOKENS_CEILING = 12_000

# For 200k models, compact a bit early to avoid hitting the wall mid-run.
_PREEMPTIVE_COMPACTION_RATIO = 0.85

# Placeholder used when pruning old tool results
_PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"

# Chars per token rough estimate
_CHARS_PER_TOKEN = 4
_SUMMARY_FAILURE_COOLDOWN_SECONDS = 600


def _summarize_tool_result(tool_name: str, tool_args: str, tool_content: str) -> str:
    """Create an informative 1-line summary of a tool call + result.

    Used during the pre-compression pruning pass to replace large tool
    outputs with a short but useful description of what the tool did,
    rather than a generic placeholder that carries zero information.

    Returns strings like::

        [terminal] ran `npm test` -> exit 0, 47 lines output
        [read_file] read config.py from line 1 (1,200 chars)
        [search_files] content search for 'compress' in agent/ -> 12 matches
    """
    try:
        args = json.loads(tool_args) if tool_args else {}
    except (json.JSONDecodeError, TypeError):
        args = {}

    content = tool_content or ""
    content_len = len(content)
    line_count = content.count("\n") + 1 if content.strip() else 0

    if tool_name == "terminal":
        cmd = args.get("command", "")
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        exit_match = re.search(r'"exit_code"\s*:\s*(-?\d+)', content)
        exit_code = exit_match.group(1) if exit_match else "?"
        return f"[terminal] ran `{cmd}` -> exit {exit_code}, {line_count} lines output"

    if tool_name == "read_file":
        path = args.get("path", "?")
        offset = args.get("offset", 1)
        return f"[read_file] read {path} from line {offset} ({content_len:,} chars)"

    if tool_name == "write_file":
        path = args.get("path", "?")
        written_lines = args.get("content", "").count("\n") + 1 if args.get("content") else "?"
        return f"[write_file] wrote to {path} ({written_lines} lines)"

    if tool_name == "search_files":
        pattern = args.get("pattern", "?")
        path = args.get("path", ".")
        target = args.get("target", "content")
        match_count = re.search(r'"total_count"\s*:\s*(\d+)', content)
        count = match_count.group(1) if match_count else "?"
        return f"[search_files] {target} search for '{pattern}' in {path} -> {count} matches"

    if tool_name == "patch":
        path = args.get("path", "?")
        mode = args.get("mode", "replace")
        return f"[patch] {mode} in {path} ({content_len:,} chars result)"

    if tool_name in ("browser_navigate", "browser_click", "browser_snapshot",
                     "browser_type", "browser_scroll", "browser_vision"):
        url = args.get("url", "")
        ref = args.get("ref", "")
        detail = f" {url}" if url else (f" ref={ref}" if ref else "")
        return f"[{tool_name}]{detail} ({content_len:,} chars)"

    if tool_name == "web_search":
        query = args.get("query", "?")
        return f"[web_search] query='{query}' ({content_len:,} chars result)"

    if tool_name == "web_extract":
        urls = args.get("urls", [])
        url_desc = urls[0] if isinstance(urls, list) and urls else "?"
        if isinstance(urls, list) and len(urls) > 1:
            url_desc += f" (+{len(urls) - 1} more)"
        return f"[web_extract] {url_desc} ({content_len:,} chars)"

    if tool_name == "delegate_task":
        goal = args.get("goal", "")
        if len(goal) > 60:
            goal = goal[:57] + "..."
        return f"[delegate_task] '{goal}' ({content_len:,} chars result)"

    if tool_name == "execute_code":
        code_preview = (args.get("code") or "")[:60].replace("\n", " ")
        if len(args.get("code", "")) > 60:
            code_preview += "..."
        return f"[execute_code] `{code_preview}` ({line_count} lines output)"

    if tool_name in ("skill_view", "skills_list", "skill_manage"):
        name = args.get("name", "?")
        return f"[{tool_name}] name={name} ({content_len:,} chars)"

    if tool_name == "vision_analyze":
        question = args.get("question", "")[:50]
        return f"[vision_analyze] '{question}' ({content_len:,} chars)"

    if tool_name == "memory":
        action = args.get("action", "?")
        target = args.get("target", "?")
        return f"[memory] {action} on {target}"

    if tool_name == "todo":
        return "[todo] updated task list"

    if tool_name == "clarify":
        return "[clarify] asked user a question"

    if tool_name == "text_to_speech":
        return f"[text_to_speech] generated audio ({content_len:,} chars)"

    if tool_name == "cronjob":
        action = args.get("action", "?")
        return f"[cronjob] {action}"

    if tool_name == "process":
        action = args.get("action", "?")
        sid = args.get("session_id", "?")
        return f"[process] {action} session={sid}"

    # Generic fallback
    first_arg = ""
    for k, v in list(args.items())[:2]:
        sv = str(v)[:40]
        first_arg += f" {k}={sv}"
    return f"[{tool_name}]{first_arg} ({content_len:,} chars result)"


class ContextCompressor(ContextEngine):
    """Default context engine — compresses conversation context via lossy summarization.

    Algorithm:
      1. Prune old tool results (cheap, no LLM call)
      2. Protect head messages (system prompt + first exchange)
      3. Protect tail messages by token budget (most recent ~20K tokens)
      4. Summarize middle turns with structured LLM prompt
      5. On subsequent compactions, iteratively update the previous summary
    """

    @property
    def name(self) -> str:
        return "compressor"

    def on_session_reset(self) -> None:
        """Reset all per-session state for /new or /reset."""
        super().on_session_reset()
        self._context_probed = False
        self._context_probe_persistable = False
        self._previous_summary = None
        self._last_compression_savings_pct = 100.0
        self._ineffective_compression_count = 0

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        """Update model info after a model switch or fallback activation."""
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.context_length = context_length
        self.threshold_tokens = self._compute_threshold_tokens(context_length)

    def _compute_threshold_tokens(self, context_length: int) -> int:
        """Return the effective auto-compaction trigger for this model.

        Local policy: 200k-context models compact at 85% usage (170k tokens)
        so the session does not run into the hard wall mid-task. Other context
        sizes keep the configured threshold percentage, with the normal minimum
        floor preserved.
        """
        if context_length == 200_000:
            threshold = int(context_length * _PREEMPTIVE_COMPACTION_RATIO)
        else:
            threshold = int(context_length * self.threshold_percent)
        return max(threshold, MINIMUM_CONTEXT_LENGTH)

    def __init__(
        self,
        model: str,
        threshold_percent: float = 0.50,
        protect_first_n: int = 3,
        protect_last_n: int = 20,
        summary_target_ratio: float = 0.20,
        quiet_mode: bool = False,
        summary_model_override: str = None,
        base_url: str = "",
        api_key: str = "",
        config_context_length: int | None = None,
        provider: str = "",
        api_mode: str = "",
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.threshold_percent = threshold_percent
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.summary_target_ratio = max(0.10, min(summary_target_ratio, 0.80))
        self.quiet_mode = quiet_mode

        self.context_length = get_model_context_length(
            model, base_url=base_url, api_key=api_key,
            config_context_length=config_context_length,
            provider=provider,
        )
        # Floor: never compress below MINIMUM_CONTEXT_LENGTH tokens even if
        # the percentage would suggest a lower value.  For 200k models we also
        # compact slightly early so long-running sessions do not slam into the
        # hard limit before the compactor can react.
        self.threshold_tokens = self._compute_threshold_tokens(self.context_length)
        self.compression_count = 0

        # Derive token budgets: ratio is relative to the threshold, not total context
        target_tokens = int(self.threshold_tokens * self.summary_target_ratio)
        self.tail_token_budget = target_tokens
        self.max_summary_tokens = min(
            int(self.context_length * 0.05), _SUMMARY_TOKENS_CEILING,
        )

        if not quiet_mode:
            logger.info(
                "Context compressor initialized: model=%s context_length=%d "
                "threshold=%d (%.0f%%) target_ratio=%.0f%% tail_budget=%d "
                "provider=%s base_url=%s",
                model, self.context_length, self.threshold_tokens,
                threshold_percent * 100, self.summary_target_ratio * 100,
                self.tail_token_budget,
                provider or "none", base_url or "none",
            )
        self._context_probed = False  # True after a step-down from context error

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

        self.summary_model = summary_model_override or ""

        # Stores the previous compaction summary for iterative updates
        self._previous_summary: Optional[str] = None
        # Anti-thrashing: track whether last compression was effective
        self._last_compression_savings_pct: float = 100.0
        self._ineffective_compression_count: int = 0
        self._summary_failure_cooldown_until: float = 0.0

    def update_from_response(self, usage: Dict[str, Any]):
        """Update tracked token usage from API response."""
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Check if context exceeds the compression threshold.

        Includes anti-thrashing protection: if the last two compressions
        each saved less than 10%, skip compression to avoid infinite loops
        where each pass removes only 1-2 messages.
        """
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if tokens < self.threshold_tokens:
            return False
        # Anti-thrashing: back off if recent compressions were ineffective
        if self._ineffective_compression_count >= 2:
            if not self.quiet_mode:
                logger.warning(
                    "Compression skipped — last %d compressions saved <10%% each. "
                    "Consider /new to start a fresh session, or /compress <topic> "
                    "for focused compression.",
                    self._ineffective_compression_count,
                )
            return False
        return True

    # ------------------------------------------------------------------
    # Tool output pruning (cheap pre-pass, no LLM call)
    # ------------------------------------------------------------------

    def _prune_old_tool_results(
        self, messages: List[Dict[str, Any]], protect_tail_count: int,
        protect_tail_tokens: int | None = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Replace old tool result contents with informative 1-line summaries.

        Instead of a generic placeholder, generates a summary like::

            [terminal] ran `npm test` -> exit 0, 47 lines output
            [read_file] read config.py from line 1 (3,400 chars)

        Also deduplicates identical tool results (e.g. reading the same file
        5x keeps only the newest full copy) and truncates large tool_call
        arguments in assistant messages outside the protected tail.

        Walks backward from the end, protecting the most recent messages that
        fall within ``protect_tail_tokens`` (when provided) OR the last
        ``protect_tail_count`` messages (backward-compatible default).
        When both are given, the token budget takes priority and the message
        count acts as a hard minimum floor.

        Returns (pruned_messages, pruned_count).
        """
        if not messages:
            return messages, 0

        result = [m.copy() for m in messages]
        pruned = 0

        # Build index: tool_call_id -> (tool_name, arguments_json)
        call_id_to_tool: Dict[str, tuple] = {}
        for msg in result:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        cid = tc.get("id", "")
                        fn = tc.get("function", {})
                        call_id_to_tool[cid] = (fn.get("name", "unknown"), fn.get("arguments", ""))
                    else:
                        cid = getattr(tc, "id", "") or ""
                        fn = getattr(tc, "function", None)
                        name = getattr(fn, "name", "unknown") if fn else "unknown"
                        args_str = getattr(fn, "arguments", "") if fn else ""
                        call_id_to_tool[cid] = (name, args_str)

        # Determine the prune boundary
        if protect_tail_tokens is not None and protect_tail_tokens > 0:
            # Token-budget approach: walk backward accumulating tokens
            accumulated = 0
            boundary = len(result)
            min_protect = min(protect_tail_count, len(result) - 1)
            for i in range(len(result) - 1, -1, -1):
                msg = result[i]
                raw_content = msg.get("content") or ""
                content_len = sum(len(p.get("text", "")) for p in raw_content) if isinstance(raw_content, list) else len(raw_content)
                msg_tokens = content_len // _CHARS_PER_TOKEN + 10
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        args = tc.get("function", {}).get("arguments", "")
                        msg_tokens += len(args) // _CHARS_PER_TOKEN
                if accumulated + msg_tokens > protect_tail_tokens and (len(result) - i) >= min_protect:
                    boundary = i
                    break
                accumulated += msg_tokens
                boundary = i
            prune_boundary = max(boundary, len(result) - min_protect)
        else:
            prune_boundary = len(result) - protect_tail_count

        # Pass 1: Deduplicate identical tool results.
        # When the same file is read multiple times, keep only the most recent
        # full copy and replace older duplicates with a back-reference.
        content_hashes: dict = {}  # hash -> (index, tool_call_id)
        for i in range(len(result) - 1, -1, -1):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content") or ""
            # Skip multimodal content (list of content blocks)
            if isinstance(content, list):
                continue
            if len(content) < 200:
                continue
            h = hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()[:12]
            if h in content_hashes:
                # This is an older duplicate — replace with back-reference
                result[i] = {**msg, "content": "[Duplicate tool output — same content as a more recent call]"}
                pruned += 1
            else:
                content_hashes[h] = (i, msg.get("tool_call_id", "?"))

        # Pass 2: Replace old tool results with informative summaries
        for i in range(prune_boundary):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            # Skip multimodal content (list of content blocks)
            if isinstance(content, list):
                continue
            if not content or content == _PRUNED_TOOL_PLACEHOLDER:
                continue
            # Skip already-deduplicated or previously-summarized results
            if content.startswith("[Duplicate tool output"):
                continue
            # Only prune if the content is substantial (>200 chars)
            if len(content) > 200:
                call_id = msg.get("tool_call_id", "")
                tool_name, tool_args = call_id_to_tool.get(call_id, ("unknown", ""))
                summary = _summarize_tool_result(tool_name, tool_args, content)
                result[i] = {**msg, "content": summary}
                pruned += 1

        # Pass 3: Truncate large tool_call arguments in assistant messages
        # outside the protected tail. write_file with 50KB content, for
        # example, survives pruning entirely without this.
        for i in range(prune_boundary):
            msg = result[i]
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue
            new_tcs = []
            modified = False
            for tc in msg["tool_calls"]:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    if len(args) > 500:
                        tc = {**tc, "function": {**tc["function"], "arguments": args[:200] + "...[truncated]"}}
                        modified = True
                new_tcs.append(tc)
            if modified:
                result[i] = {**msg, "tool_calls": new_tcs}

        return result, pruned

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _compute_summary_budget(self, turns_to_summarize: List[Dict[str, Any]]) -> int:
        """Scale summary token budget with the amount of content being compressed.

        The maximum scales with the model's context window (5% of context,
        capped at ``_SUMMARY_TOKENS_CEILING``) so large-context models get
        richer summaries instead of being hard-capped at 8K tokens.
        """
        content_tokens = estimate_messages_tokens_rough(turns_to_summarize)
        budget = int(content_tokens * _SUMMARY_RATIO)
        return max(_MIN_SUMMARY_TOKENS, min(budget, self.max_summary_tokens))

    # Truncation limits for the summarizer input.  These bound how much of
    # each message the summary model sees — the budget is the *summary*
    # model's context window, not the main model's.
    _CONTENT_MAX = 6000       # total chars per message body
    _CONTENT_HEAD = 4000      # chars kept from the start
    _CONTENT_TAIL = 1500      # chars kept from the end
    _TOOL_ARGS_MAX = 1500     # tool call argument chars
    _TOOL_ARGS_HEAD = 1200    # kept from the start of tool args

    def _serialize_for_summary(self, turns: List[Dict[str, Any]]) -> str:
        """Serialize conversation turns into parent-safe text for the summarizer.

        Tool result contents are deliberately not copied into this serialized
        handoff input.  The compressor's job is to produce a recoverable guide,
        not to preserve raw command output/logs/source dumps in the active
        parent context.  Tool calls keep names/arguments and tool results keep
        metadata so the summary model can describe what happened without seeing
        or re-emitting bulky raw output.
        """
        parts = []
        tool_call_index: Dict[str, tuple[str, str]] = {}

        def _parse_json_maybe(value: str) -> Optional[dict]:
            if not isinstance(value, str):
                return None
            stripped = value.strip()
            if not (stripped.startswith("{") and stripped.endswith("}")):
                return None
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                return None
            return parsed if isinstance(parsed, dict) else None

        for msg in turns:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    cid = tc.get("id") or ""
                    fn = (tc.get("function") or {}).get("name") or ""
                    args = (tc.get("function") or {}).get("arguments") or ""
                else:
                    cid = getattr(tc, "id", "") or ""
                    fn_obj = getattr(tc, "function", None)
                    fn = getattr(fn_obj, "name", "") if fn_obj else ""
                    args = getattr(fn_obj, "arguments", "") if fn_obj else ""
                if cid:
                    tool_call_index[str(cid)] = (str(fn or ""), str(args or ""))

        for msg in turns:
            role = msg.get("role", "unknown")
            content = msg.get("content") or ""

            if role == "tool":
                tool_id = str(msg.get("tool_call_id") or "")
                tool_name, _tool_args = tool_call_index.get(tool_id, ("", ""))
                content_text = str(content or "")
                metadata: list[str] = []
                parsed = _parse_json_maybe(content_text)
                if isinstance(parsed, dict):
                    exit_code = parsed.get("exit_code")
                    if isinstance(exit_code, int):
                        metadata.append(f"exit_code={exit_code}")
                    total_count = parsed.get("total_count")
                    if isinstance(total_count, int):
                        metadata.append(f"total_count={total_count}")
                    truncated = parsed.get("truncated")
                    if isinstance(truncated, bool):
                        metadata.append(f"truncated={truncated}")
                if content_text:
                    metadata.append(f"raw_chars={len(content_text)}")
                metadata_text = ("; " + "; ".join(metadata)) if metadata else ""
                label = f" {tool_id}" if tool_id else ""
                name = tool_name or "tool"
                parts.append(
                    f"[TOOL RESULT{label}]: {name} completed; raw output omitted from compaction input{metadata_text}."
                )
                continue

            if role == "assistant":
                if len(content) > self._CONTENT_MAX:
                    content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc_parts = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = fn.get("arguments", "")
                            if len(args) > self._TOOL_ARGS_MAX:
                                args = args[:self._TOOL_ARGS_HEAD] + "..."
                            tc_parts.append(f"  {name}({args})")
                        else:
                            fn = getattr(tc, "function", None)
                            name = getattr(fn, "name", "?") if fn else "?"
                            tc_parts.append(f"  {name}(...)")
                    content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
                continue

            if len(content) > self._CONTENT_MAX:
                content = content[:self._CONTENT_HEAD] + "\n...[truncated]...\n" + content[-self._CONTENT_TAIL:]
            parts.append(f"[{role.upper()}]: {content}")

        return "\n\n".join(parts)

    @staticmethod
    def _strip_ansi(text: str) -> str:
        """Remove ANSI escape sequences so fallback summaries stay readable."""
        return re.sub(r"\[[0-9;?]*[ -/]*[@-~]", "", text or "")

    @staticmethod
    def _looks_like_session_guide(text: str) -> bool:
        """Return True when text already uses the recoverable guide schema."""
        stripped = (text or "").strip()
        if stripped.startswith(SESSION_GUIDE_HEADING):
            return True
        return all(section in stripped for section in SESSION_GUIDE_SECTIONS[:4])

    @staticmethod
    def _plain_summary_bullets(text: str) -> str:
        """Convert plain legacy summary text into safe bullets."""
        lines = [line.strip(" -") for line in (text or "").splitlines() if line.strip()]
        if not lines:
            return f"- {_SESSION_GUIDE_EMPTY_ITEM}"
        return "\n".join(f"- {line}" for line in lines)

    @classmethod
    def _to_session_guide_body(cls, summary: str) -> str:
        """Normalize LLM/fallback output into the Phase 2 Session Guide shape."""
        text = (summary or "").strip()
        for prefix in (LEGACY_SUMMARY_PREFIX, SUMMARY_PREFIX):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()
                break
        if not text:
            text = _SESSION_GUIDE_EMPTY_ITEM
        if cls._looks_like_session_guide(text):
            if text.startswith(SESSION_GUIDE_HEADING):
                return text
            return f"{SESSION_GUIDE_HEADING}\n\n{text}"
        return "\n\n".join([
            SESSION_GUIDE_HEADING,
            "1. Summary\n" + cls._plain_summary_bullets(text),
            f"2. Decisions\n- {_SESSION_GUIDE_EMPTY_ITEM}",
            f"3. Active Files\n- {_SESSION_GUIDE_EMPTY_ITEM}",
            f"4. Active Entities\n- {_SESSION_GUIDE_EMPTY_ITEM}",
            f"5. Unresolved Tasks\n- {_SESSION_GUIDE_EMPTY_ITEM}",
            f"6. Risks\n- {_SESSION_GUIDE_EMPTY_ITEM}",
            f"7. Validations\n- {_SESSION_GUIDE_EMPTY_ITEM}",
            "8. Retrieval Handles\n- None recorded in this compacted summary. Use session search/live state for detail recovery.",
        ])

    def _build_deterministic_fallback_summary(
        self,
        turns_to_summarize: List[Dict[str, Any]],
        focus_topic: str = None,
        failure_reason: str = "summary generation failed",
    ) -> str:
        """Build a local recoverable Session Guide when remote summarization fails."""

        def _clip(value: Any, limit: int = 220) -> str:
            text = " ".join(self._strip_ansi(str(value or "")).split())
            if len(text) <= limit:
                return text
            return text[: limit - 3] + "..."

        user_points: List[str] = []
        assistant_points: List[str] = []
        tool_points: List[str] = []
        references: List[str] = []
        active_files: List[str] = []
        active_entities: List[str] = []
        ref_pattern = re.compile(r"(?:~?/[^\s\]\[)(]+|https?://\S+|\d{2,5})")

        def _remember_ref(value: str) -> None:
            cleaned = self._strip_ansi(value).strip().rstrip('.,:;')
            if not cleaned or cleaned in references:
                return
            references.append(cleaned)
            if cleaned.startswith("/") or cleaned.startswith("~/"):
                if cleaned not in active_files:
                    active_files.append(cleaned)
            elif cleaned not in active_entities:
                active_entities.append(cleaned)

        for msg in turns_to_summarize:
            role = str(msg.get("role") or "")
            raw_content = str(msg.get("content") or "")
            content = _clip(raw_content)
            if role == "user" and content:
                user_points.append(content)
            elif role == "assistant" and content:
                assistant_points.append(content)
            elif role == "tool":
                tool_id = str(msg.get("tool_call_id") or "").strip()
                label = f"tool result {tool_id}" if tool_id else "tool result"
                tool_points.append(
                    f"{label} observed; raw output omitted from compacted parent context (raw_chars={len(raw_content)})."
                )

            if role != "tool":
                for match in ref_pattern.findall(raw_content):
                    _remember_ref(match)
                    if len(references) >= 8:
                        break

            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    args = (tc.get("function") or {}).get("arguments") or ""
                else:
                    fn = getattr(tc, "function", None)
                    args = getattr(fn, "arguments", "") if fn else ""
                for match in ref_pattern.findall(str(args)):
                    _remember_ref(match)
                    if len(references) >= 8:
                        break
                if len(references) >= 8:
                    break

        objective = user_points[-1] if user_points else "Continue the in-progress conversation after compaction."
        current_state: List[str] = []
        if assistant_points:
            current_state.append(f"Latest assistant state: {assistant_points[-1]}")
        if not current_state:
            current_state.append("Recent turns were compacted with a deterministic local fallback guide.")

        decisions = ["Continue from current repo/tool state rather than replaying dropped turns."]
        unresolved_tasks = ["Resume from the preserved recent turns and current file/tool state."]
        if focus_topic:
            unresolved_tasks.insert(0, f"Prioritize the focus topic during continuation: {focus_topic}")
        if tool_points:
            unresolved_tasks.append("Re-check important tool-derived facts through live state, session search, or retrieval handles before relying on omitted raw details.")

        risks = [
            "Older raw tool outputs were intentionally omitted from the compacted parent context.",
        ]
        if failure_reason:
            risks.insert(0, f"Remote summary generation failed: {failure_reason}")

        validations = [
            "Fallback guide generated locally to preserve continuity during compaction.",
        ]
        if tool_points:
            validations.extend(tool_points[:3])

        active_files = active_files[:8] or ["Use the preserved tail messages and current workspace state as ground truth."]
        active_entities = active_entities[:8] or references[:8] or [_SESSION_GUIDE_EMPTY_ITEM]

        sections = [
            SESSION_GUIDE_HEADING,
            "1. Summary\n- Objective: " + objective + "\n" + "\n".join(f"- Current state: {item}" for item in current_state),
            "2. Decisions\n" + "\n".join(f"- {item}" for item in decisions),
            "3. Active Files\n" + "\n".join(f"- {item}" for item in active_files),
            "4. Active Entities\n" + "\n".join(f"- {item}" for item in active_entities),
            "5. Unresolved Tasks\n" + "\n".join(f"- {item}" for item in unresolved_tasks),
            "6. Risks\n" + "\n".join(f"- {item}" for item in risks),
            "7. Validations\n" + "\n".join(f"- {item}" for item in validations),
            "8. Retrieval Handles\n- None recorded in this fallback guide. Recover details through session events, session search, or live state instead of relying on raw parent-context payloads.",
        ]
        return "\n\n".join(sections)

    def _generate_summary(self, turns_to_summarize: List[Dict[str, Any]], focus_topic: str = None) -> Optional[str]:
        """Generate a parent-safe, recoverable Session Guide."""
        now = time.monotonic()
        if now < self._summary_failure_cooldown_until:
            fallback = self._build_deterministic_fallback_summary(
                turns_to_summarize,
                focus_topic=focus_topic,
                failure_reason="summary generation is in cooldown",
            )
            guide_body = self._to_session_guide_body(fallback)
            self._previous_summary = guide_body
            return self._with_summary_prefix(guide_body)

        summary_budget = self._compute_summary_budget(turns_to_summarize)
        content_to_summarize = self._serialize_for_summary(turns_to_summarize)

        summarizer_preamble = (
            "You are a summarization agent creating a context checkpoint for a DIFFERENT assistant. "
            "Do NOT respond to any questions or requests in the conversation — only output the structured Session Guide. "
            "Do NOT include any preamble, greeting, or prefix."
        )
        template_sections = f"""Use this exact recoverable Session Guide structure:

{SESSION_GUIDE_HEADING}

1. Summary
- Compactly state the user's goal and current state.

2. Decisions
- Decisions already made that future steps must respect.

3. Active Files
- File paths currently relevant to continuation.

4. Active Entities
- Commands, services, issue IDs, URLs, schemas, handles, or other named entities needed for recovery.

5. Unresolved Tasks
- Concrete next work that remains; include validation gaps.

6. Risks
- Known blockers, uncertainties, leakage risks, compatibility risks, or recovery risks.

7. Validations
- Tests/checks already run and their outcomes, or checks still pending.

8. Retrieval Handles
- Handle IDs and minimal metadata for recoverable detail. If none are present, say none are recorded and point to session search/live state.

Rules:
- Do not include raw tool outputs, long logs, large code snippets, or values from searchable_text.
- Do not quote bulky command output; summarize outcome and recovery route instead.
- Prefer handle IDs, file paths, commands, and concise evidence over transcript narrative.
- Each decision/task/risk should be grounded in the supplied turns or previous guide.
- Target ~{summary_budget} tokens. Preserve continuity and concrete facts.
- Write only the Session Guide body. Do not include any preamble or prefix."""

        if self._previous_summary:
            prompt = f"""{summarizer_preamble}

You are updating a context compaction Session Guide. A previous compaction produced the guide below. New conversation turns have occurred since then and need to be incorporated.

PREVIOUS SESSION GUIDE:
{self._previous_summary}

NEW TURNS TO INCORPORATE:
{content_to_summarize}

Update the guide. Preserve what is still relevant, add new progress, and remove only what is clearly obsolete.

{template_sections}"""
        else:
            prompt = f"""{summarizer_preamble}

Create a structured handoff Session Guide for a different assistant that will continue this conversation after earlier turns are compacted.

TURNS TO SUMMARIZE:
{content_to_summarize}

{template_sections}"""

        if focus_topic:
            prompt += f"""

FOCUS TOPIC: "{focus_topic}"
The user has requested that this compaction PRIORITISE preserving recoverable information related to the focus topic above. Include exact file paths, commands, identifiers, decisions, validation outcomes, and retrieval handles. Do not include raw tool output or long snippets; route details through handles/search/live state."""

        try:
            call_kwargs = {
                "task": "compression",
                "main_runtime": {
                    "model": self.model,
                    "provider": self.provider,
                    "base_url": self.base_url,
                    "api_key": self.api_key,
                    "api_mode": self.api_mode,
                },
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(summary_budget * 1.3),
            }
            if self.summary_model:
                call_kwargs["model"] = self.summary_model
            response = call_llm(**call_kwargs)
            content = response.choices[0].message.content
            if not isinstance(content, str):
                content = str(content) if content else ""
            summary = content.strip()
            if not summary:
                raise ValueError("Summary LLM returned empty content")
            guide_body = self._to_session_guide_body(summary)
            self._previous_summary = guide_body
            self._summary_failure_cooldown_until = 0.0
            self._summary_model_fallen_back = False
            return self._with_summary_prefix(guide_body)
        except Exception as e:
            self._summary_failure_cooldown_until = time.monotonic() + 60
            logging.warning(
                "Failed to generate context summary: %s. Using deterministic local Session Guide fallback.",
                e,
            )
            fallback = self._build_deterministic_fallback_summary(
                turns_to_summarize,
                focus_topic=focus_topic,
                failure_reason=str(e),
            )
            guide_body = self._to_session_guide_body(fallback)
            self._previous_summary = guide_body
            return self._with_summary_prefix(guide_body)

    @staticmethod
    def _with_summary_prefix(summary: str) -> str:
        """Normalize summary text to the current compaction handoff format."""
        body = ContextCompressor._to_session_guide_body(summary)
        return f"{SUMMARY_PREFIX}\n{body}" if body else SUMMARY_PREFIX

    # ------------------------------------------------------------------
    # Tool-call / tool-result pair integrity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_tool_call_id(tc) -> str:
        """Extract the call ID from a tool_call entry (dict or SimpleNamespace)."""
        if isinstance(tc, dict):
            return tc.get("id", "")
        return getattr(tc, "id", "") or ""

    def _sanitize_tool_pairs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix orphaned tool_call / tool_result pairs after compression.

        Two failure modes:
        1. A tool *result* references a call_id whose assistant tool_call was
           removed (summarized/truncated).  The API rejects this with
           "No tool call found for function call output with call_id ...".
        2. An assistant message has tool_calls whose results were dropped.
           The API rejects this because every tool_call must be followed by
           a tool result with the matching call_id.

        This method removes orphaned results and inserts stub results for
        orphaned calls so the message list is always well-formed.
        """
        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = self._get_tool_call_id(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # 1. Remove tool results whose call_id has no matching assistant tool_call
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            if not self.quiet_mode:
                logger.info("Compression sanitizer: removed %d orphaned tool result(s)", len(orphaned_results))

        # 2. Add stub results for assistant tool_calls whose results were dropped
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = self._get_tool_call_id(tc)
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "content": "[Result from earlier conversation — see context summary above]",
                                "tool_call_id": cid,
                            })
            messages = patched
            if not self.quiet_mode:
                logger.info("Compression sanitizer: added %d stub tool result(s)", len(missing_results))

        return messages

    def _align_boundary_forward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """Push a compress-start boundary forward past any orphan tool results.

        If ``messages[idx]`` is a tool result, slide forward until we hit a
        non-tool message so we don't start the summarised region mid-group.
        """
        while idx < len(messages) and messages[idx].get("role") == "tool":
            idx += 1
        return idx

    def _align_boundary_backward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """Pull a compress-end boundary backward to avoid splitting a
        tool_call / result group.

        If the boundary falls in the middle of a tool-result group (i.e.
        there are consecutive tool messages before ``idx``), walk backward
        past all of them to find the parent assistant message.  If found,
        move the boundary before the assistant so the entire
        assistant + tool_results group is included in the summarised region
        rather than being split (which causes silent data loss when
        ``_sanitize_tool_pairs`` removes the orphaned tail results).
        """
        if idx <= 0 or idx >= len(messages):
            return idx
        # Walk backward past consecutive tool results
        check = idx - 1
        while check >= 0 and messages[check].get("role") == "tool":
            check -= 1
        # If we landed on the parent assistant with tool_calls, pull the
        # boundary before it so the whole group gets summarised together.
        if check >= 0 and messages[check].get("role") == "assistant" and messages[check].get("tool_calls"):
            idx = check
        return idx

    # ------------------------------------------------------------------
    # Tail protection by token budget
    # ------------------------------------------------------------------

    def _find_last_user_message_idx(
        self, messages: List[Dict[str, Any]], head_end: int
    ) -> int:
        """Return the index of the last user-role message at or after *head_end*, or -1."""
        for i in range(len(messages) - 1, head_end - 1, -1):
            if messages[i].get("role") == "user":
                return i
        return -1

    def _ensure_last_user_message_in_tail(
        self,
        messages: List[Dict[str, Any]],
        cut_idx: int,
        head_end: int,
    ) -> int:
        """Guarantee the most recent user message is in the protected tail.

        Context compressor bug (#10896): ``_align_boundary_backward`` can pull
        ``cut_idx`` past a user message when it tries to keep tool_call/result
        groups together.  If the last user message ends up in the *compressed*
        middle region the LLM summariser writes it into "Pending User Asks",
        but ``SUMMARY_PREFIX`` tells the next model to respond only to user
        messages *after* the summary — so the task effectively disappears from
        the active context, causing the agent to stall, repeat completed work,
        or silently drop the user's latest request.

        Fix: if the last user-role message is not already in the tail
        (``messages[cut_idx:]``), walk ``cut_idx`` back to include it.  We
        then re-align backward one more time to avoid splitting any
        tool_call/result group that immediately precedes the user message.
        """
        last_user_idx = self._find_last_user_message_idx(messages, head_end)
        if last_user_idx < 0:
            # No user message found beyond head — nothing to anchor.
            return cut_idx

        if last_user_idx >= cut_idx:
            # Already in the tail; nothing to do.
            return cut_idx

        # The last user message is in the middle (compressed) region.
        # Pull cut_idx back to it directly — a user message is already a
        # clean boundary (no tool_call/result splitting risk), so there is no
        # need to call _align_boundary_backward here; doing so would
        # unnecessarily pull the cut further back into the preceding
        # assistant + tool_calls group.
        if not self.quiet_mode:
            logger.debug(
                "Anchoring tail cut to last user message at index %d "
                "(was %d) to prevent active-task loss after compression",
                last_user_idx,
                cut_idx,
            )
        # Safety: never go back into the head region.
        return max(last_user_idx, head_end + 1)

    def _find_tail_cut_by_tokens(
        self, messages: List[Dict[str, Any]], head_end: int,
        token_budget: int | None = None,
    ) -> int:
        """Walk backward from the end of messages, accumulating tokens until
        the budget is reached. Returns the index where the tail starts.

        ``token_budget`` defaults to ``self.tail_token_budget`` which is
        derived from ``summary_target_ratio * context_length``, so it
        scales automatically with the model's context window.

        Token budget is the primary criterion.  A hard minimum of 3 messages
        is always protected, but the budget is allowed to exceed by up to
        1.5x to avoid cutting inside an oversized message (tool output, file
        read, etc.).  If even the minimum 3 messages exceed 1.5x the budget
        the cut is placed right after the head so compression still runs.

        Never cuts inside a tool_call/result group.  Always ensures the most
        recent user message is in the tail (see ``_ensure_last_user_message_in_tail``).
        """
        if token_budget is None:
            token_budget = self.tail_token_budget
        n = len(messages)
        # Hard minimum: always keep at least 3 messages in the tail
        min_tail = min(3, n - head_end - 1) if n - head_end > 1 else 0
        soft_ceiling = int(token_budget * 1.5)
        accumulated = 0
        cut_idx = n  # start from beyond the end

        for i in range(n - 1, head_end - 1, -1):
            msg = messages[i]
            content = msg.get("content") or ""
            msg_tokens = len(content) // _CHARS_PER_TOKEN + 10  # +10 for role/metadata
            # Include tool call arguments in estimate
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    msg_tokens += len(args) // _CHARS_PER_TOKEN
            # Stop once we exceed the soft ceiling (unless we haven't hit min_tail yet)
            if accumulated + msg_tokens > soft_ceiling and (n - i) >= min_tail:
                break
            accumulated += msg_tokens
            cut_idx = i

        # Ensure we protect at least min_tail messages
        fallback_cut = n - min_tail
        if cut_idx > fallback_cut:
            cut_idx = fallback_cut

        # If the token budget would protect everything (small conversations),
        # force a cut after the head so compression can still remove middle turns.
        if cut_idx <= head_end:
            cut_idx = max(fallback_cut, head_end + 1)

        # Align to avoid splitting tool groups
        cut_idx = self._align_boundary_backward(messages, cut_idx)

        # Ensure the most recent user message is always in the tail so the
        # active task is never lost to compression (fixes #10896).
        cut_idx = self._ensure_last_user_message_in_tail(messages, cut_idx, head_end)

        return max(cut_idx, head_end + 1)

    # ------------------------------------------------------------------
    # Main compression entry point
    # ------------------------------------------------------------------

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None, focus_topic: str = None) -> List[Dict[str, Any]]:
        """Compress conversation messages by summarizing middle turns.

        Algorithm:
          1. Prune old tool results (cheap pre-pass, no LLM call)
          2. Protect head messages (system prompt + first exchange)
          3. Find tail boundary by token budget (~20K tokens of recent context)
          4. Summarize middle turns with structured LLM prompt
          5. On re-compression, iteratively update the previous summary

        After compression, orphaned tool_call / tool_result pairs are cleaned
        up so the API never receives mismatched IDs.

        Args:
            focus_topic: Optional focus string for guided compression.  When
                provided, the summariser will prioritise preserving information
                related to this topic and be more aggressive about compressing
                everything else.  Inspired by Claude Code's ``/compact``.
        """
        n_messages = len(messages)
        # Only need head + 3 tail messages minimum (token budget decides the real tail size)
        _min_for_compress = self.protect_first_n + 3 + 1
        if n_messages <= _min_for_compress:
            if not self.quiet_mode:
                logger.warning(
                    "Cannot compress: only %d messages (need > %d)",
                    n_messages, _min_for_compress,
                )
            return messages

        display_tokens = current_tokens if current_tokens else self.last_prompt_tokens or estimate_messages_tokens_rough(messages)

        # Phase 1: Prune old tool results (cheap, no LLM call)
        messages, pruned_count = self._prune_old_tool_results(
            messages, protect_tail_count=self.protect_last_n,
            protect_tail_tokens=self.tail_token_budget,
        )
        if pruned_count and not self.quiet_mode:
            logger.info("Pre-compression: pruned %d old tool result(s)", pruned_count)

        # Phase 2: Determine boundaries
        compress_start = self.protect_first_n
        compress_start = self._align_boundary_forward(messages, compress_start)

        # Use token-budget tail protection instead of fixed message count
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)

        if compress_start >= compress_end:
            return messages

        turns_to_summarize = messages[compress_start:compress_end]

        if not self.quiet_mode:
            logger.info(
                "Context compression triggered (%d tokens >= %d threshold)",
                display_tokens,
                self.threshold_tokens,
            )
            logger.info(
                "Model context limit: %d tokens (%.0f%% = %d)",
                self.context_length,
                self.threshold_percent * 100,
                self.threshold_tokens,
            )
            tail_msgs = n_messages - compress_end
            logger.info(
                "Summarizing turns %d-%d (%d turns), protecting %d head + %d tail messages",
                compress_start + 1,
                compress_end,
                len(turns_to_summarize),
                compress_start,
                tail_msgs,
            )

        # Phase 3: Generate structured summary
        summary = self._generate_summary(turns_to_summarize, focus_topic=focus_topic)

        # Phase 4: Assemble compressed message list
        compressed = []
        for i in range(compress_start):
            msg = messages[i].copy()
            if i == 0 and msg.get("role") == "system":
                existing = msg.get("content") or ""
                _compression_note = "[Note: Some earlier conversation turns have been compacted into a handoff summary to preserve context space. The current session state may still reflect earlier work, so build on that summary and state rather than re-doing work.]"
                if _compression_note not in existing:
                    msg["content"] = existing + "\n\n" + _compression_note
            compressed.append(msg)

        # If remote summary generation failed, build a deterministic local
        # handoff so the model still receives continuity instead of a blank gap.
        if not summary:
            if not self.quiet_mode:
                logger.warning("Summary generation failed — inserting deterministic local fallback summary")
            summary = self._with_summary_prefix(
                self._build_deterministic_fallback_summary(
                    turns_to_summarize,
                    focus_topic=focus_topic,
                    failure_reason="summary generation was unavailable",
                )
            )

        _merge_summary_into_tail = False
        last_head_role = messages[compress_start - 1].get("role", "user") if compress_start > 0 else "user"
        first_tail_role = messages[compress_end].get("role", "user") if compress_end < n_messages else "user"
        # Pick a role that avoids consecutive same-role with both neighbors.
        # Priority: avoid colliding with head (already committed), then tail.
        if last_head_role in ("assistant", "tool"):
            summary_role = "user"
        else:
            summary_role = "assistant"
        # If the chosen role collides with the tail AND flipping wouldn't
        # collide with the head, flip it.
        if summary_role == first_tail_role:
            flipped = "assistant" if summary_role == "user" else "user"
            if flipped != last_head_role:
                summary_role = flipped
            else:
                # Both roles would create consecutive same-role messages
                # (e.g. head=assistant, tail=user — neither role works).
                # Merge the summary into the first tail message instead
                # of inserting a standalone message that breaks alternation.
                _merge_summary_into_tail = True
        if not _merge_summary_into_tail:
            compressed.append({"role": summary_role, "content": summary})

        for i in range(compress_end, n_messages):
            msg = messages[i].copy()
            if _merge_summary_into_tail and i == compress_end:
                original = msg.get("content") or ""
                msg["content"] = (
                    summary
                    + "\n\n--- END OF CONTEXT SUMMARY — "
                    "respond to the message below, not the summary above ---\n\n"
                    + original
                )
                _merge_summary_into_tail = False
            compressed.append(msg)

        self.compression_count += 1

        compressed = self._sanitize_tool_pairs(compressed)

        new_estimate = estimate_messages_tokens_rough(compressed)
        saved_estimate = display_tokens - new_estimate

        # Anti-thrashing: track compression effectiveness
        savings_pct = (saved_estimate / display_tokens * 100) if display_tokens > 0 else 0
        self._last_compression_savings_pct = savings_pct
        if savings_pct < 10:
            self._ineffective_compression_count += 1
        else:
            self._ineffective_compression_count = 0

        if not self.quiet_mode:
            logger.info(
                "Compressed: %d -> %d messages (~%d tokens saved, %.0f%%)",
                n_messages,
                len(compressed),
                saved_estimate,
                savings_pct,
            )
            logger.info("Compression #%d complete", self.compression_count)

        return compressed
