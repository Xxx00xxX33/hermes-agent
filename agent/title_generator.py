"""Auto-generate short session titles from the first user/assistant exchange.

Runs asynchronously after the first response is delivered so it never
adds latency to the user-facing reply.
"""

import logging
import threading
from typing import Optional

from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)

_TITLE_PROMPT = (
    "Generate a short, descriptive title (3-7 words) for a conversation that starts with the "
    "following exchange. The title should capture the main topic or intent. "
    "Return ONLY the title text, nothing else. No quotes, no punctuation at the end, no prefixes."
)
_TITLE_TASK = "title"
_REASONING_UNSUPPORTED_NEEDLES = (
    "reasoning",
    "unsupported_parameter",
    "unknown parameter",
    "unexpected keyword",
    "extra_body",
    "extra fields not permitted",
)


def _get_title_reasoning_extra_body() -> Optional[dict]:
    """Return optional extra_body for title generation reasoning overrides.

    Reads auxiliary.title.reasoning_effort from config.yaml. When unset,
    returns None so title generation uses the provider default. Unknown values
    are ignored with a warning.
    """
    try:
        from hermes_cli.config import load_config
        from hermes_constants import parse_reasoning_effort

        config = load_config() or {}
        auxiliary = config.get("auxiliary", {}) if isinstance(config, dict) else {}
        title_cfg = auxiliary.get(_TITLE_TASK, {}) if isinstance(auxiliary, dict) else {}
        if not isinstance(title_cfg, dict):
            return None

        effort = str(title_cfg.get("reasoning_effort", "") or "").strip()
        if not effort:
            return None

        parsed = parse_reasoning_effort(effort)
        if parsed is None:
            logger.warning(
                "Title generation: unknown auxiliary.%s.reasoning_effort '%s'; ignoring",
                _TITLE_TASK,
                effort,
            )
            return None
        return {"reasoning": parsed}
    except Exception as exc:
        logger.debug("Title generation: could not load reasoning override: %s", exc)
        return None


def _is_reasoning_unsupported_error(err: Exception) -> bool:
    err_str = str(err).lower()
    return any(needle in err_str for needle in _REASONING_UNSUPPORTED_NEEDLES)


def generate_title(user_message: str, assistant_response: str, timeout: float = 30.0) -> Optional[str]:
    """Generate a session title from the first exchange.

    Uses the auxiliary title task config (auxiliary.title.*). Returns the title
    string or None on failure.
    """
    # Truncate long messages to keep the request small
    user_snippet = user_message[:500] if user_message else ""
    assistant_snippet = assistant_response[:500] if assistant_response else ""

    messages = [
        {"role": "system", "content": _TITLE_PROMPT},
        {"role": "user", "content": f"User: {user_snippet}\n\nAssistant: {assistant_snippet}"},
    ]

    call_kwargs = {
        "task": _TITLE_TASK,
        "messages": messages,
        "max_tokens": 30,
        "temperature": 0.3,
        "timeout": timeout,
    }
    extra_body = _get_title_reasoning_extra_body()
    if extra_body:
        call_kwargs["extra_body"] = extra_body

    try:
        try:
            response = call_llm(**call_kwargs)
        except Exception as reasoning_err:
            if extra_body and _is_reasoning_unsupported_error(reasoning_err):
                logger.info(
                    "Title generation: reasoning override unsupported on configured title model (%s); retrying without reasoning",
                    reasoning_err,
                )
                call_kwargs.pop("extra_body", None)
                response = call_llm(**call_kwargs)
            else:
                raise
        title = (response.choices[0].message.content or "").strip()
        # Clean up: remove quotes, trailing punctuation, prefixes like "Title: "
        title = title.strip('"\'')
        if title.lower().startswith("title:"):
            title = title[6:].strip()
        # Enforce reasonable length
        if len(title) > 80:
            title = title[:77] + "..."
        return title if title else None
    except Exception as e:
        logger.debug("Title generation failed: %s", e)
        return None


def auto_title_session(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
) -> None:
    """Generate and set a session title if one doesn't already exist.

    Called in a background thread after the first exchange completes.
    Silently skips if:
    - session_db is None
    - session already has a title (user-set or previously auto-generated)
    - title generation fails
    """
    if not session_db or not session_id:
        return

    # Check if title already exists (user may have set one via /title before first response)
    try:
        existing = session_db.get_session_title(session_id)
        if existing:
            return
    except Exception:
        return

    title = generate_title(user_message, assistant_response)
    if not title:
        return

    try:
        session_db.set_session_title(session_id, title)
        logger.debug("Auto-generated session title: %s", title)
    except Exception as e:
        logger.debug("Failed to set auto-generated title: %s", e)


def maybe_auto_title(
    session_db,
    session_id: str,
    user_message: str,
    assistant_response: str,
    conversation_history: list,
) -> None:
    """Fire-and-forget title generation after the first exchange.

    Only generates a title when:
    - This appears to be the first user→assistant exchange
    - No title is already set
    """
    if not session_db or not session_id or not user_message or not assistant_response:
        return

    # Count user messages in history to detect first exchange.
    # conversation_history includes the exchange that just happened,
    # so for a first exchange we expect exactly 1 user message
    # (or 2 counting system). Be generous: generate on first 2 exchanges.
    user_msg_count = sum(1 for m in (conversation_history or []) if m.get("role") == "user")
    if user_msg_count > 2:
        return

    thread = threading.Thread(
        target=auto_title_session,
        args=(session_db, session_id, user_message, assistant_response),
        daemon=True,
        name="auto-title",
    )
    thread.start()
