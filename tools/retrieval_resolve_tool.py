#!/usr/bin/env python3
"""Metadata-only retrieval handle resolution for parent-safe recall."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from tools.registry import registry, tool_error

_MAX_SAFE_TEXT_CHARS = 500
_SAFE_RAW_METADATA_KEYS = {"raw_chars", "raw_bytes", "raw_size"}
_UNSAFE_KEY_TOKENS = (
    "raw",
    "body",
    "content",
    "searchable_text",
    "snippet",
    "preview",
    "stdout",
    "stderr",
    "output",
    "transcript",
    "text",
    "message",
)
_SENSITIVE_QUERY_TOKENS = (
    "token",
    "key",
    "secret",
    "signature",
    "sig",
    "auth",
    "credential",
    "password",
)
_SAFE_LOCATOR_KEYS = {
    "artifact_path",
    "sandbox_artifact_path",
    "path",
    "uri",
    "url",
    "line_start",
    "line_end",
    "byte_start",
    "byte_end",
    "offset",
    "length",
}


RETRIEVAL_RESOLVE_SCHEMA = {
    "name": "retrieval_resolve",
    "description": (
        "Resolve a session event or retrieval handle into parent-safe metadata and artifact "
        "locators. This tool never returns raw artifact contents, searchable text, snippets, "
        "stdout/stderr, previews, or message/body/content fields. Use it after session_search "
        "returns an event_id or handle_id and you need safe metadata. If artifact contents "
        "must be inspected, follow the returned artifact_inspection contract and delegate "
        "that work to a child/subagent rather than reading raw content into the parent context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "event_id": {
                "type": "string",
                "description": "Session event ID to resolve. Optional if handle_id is provided.",
            },
            "handle_id": {
                "type": "string",
                "description": "Retrieval handle ID to resolve. Optional if event_id is provided.",
            },
        },
        "required": [],
    },
}


def _is_unsafe_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in _SAFE_RAW_METADATA_KEYS:
        return False
    return any(token in lowered for token in _UNSAFE_KEY_TOKENS)


def _truncate_safe_value(value: Any, max_chars: int = _MAX_SAFE_TEXT_CHARS) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        safe = _redact_locator_string(value)
        return safe if len(safe) <= max_chars else safe[:max_chars] + "…[truncated]"
    if isinstance(value, list):
        return [_truncate_safe_value(item, max_chars=max_chars) for item in value[:20]]
    if isinstance(value, dict):
        return _sanitize_mapping(value, max_chars=max_chars)
    return str(value)[:max_chars]


def _redact_locator_string(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc or not parsed.query:
        return value
    redacted = []
    changed = False
    for key, item in parse_qsl(parsed.query, keep_blank_values=True):
        if any(token in key.lower() for token in _SENSITIVE_QUERY_TOKENS):
            redacted.append((key, "[redacted]"))
            changed = True
        else:
            redacted.append((key, item))
    if not changed:
        return value
    return urlunsplit((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        urlencode(redacted, doseq=True),
        parsed.fragment,
    ))


def _sanitize_mapping(value: Dict[str, Any], max_chars: int = _MAX_SAFE_TEXT_CHARS) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, item in value.items():
        key_text = str(key)
        if _is_unsafe_key(key_text):
            continue
        sanitized[key_text] = _truncate_safe_value(item, max_chars=max_chars)
    return sanitized


def _sanitize_locator(locator: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, item in locator.items():
        key_text = str(key)
        lowered = key_text.lower()
        if lowered not in _SAFE_LOCATOR_KEYS or _is_unsafe_key(key_text):
            continue
        sanitized[key_text] = _truncate_safe_value(item)
    return sanitized


def _safe_handle(handle: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "handle_id": _truncate_safe_value(handle.get("handle_id")),
        "source_type": _truncate_safe_value(handle.get("source_type")),
        "source_id": _truncate_safe_value(handle.get("source_id")),
        "locator": _sanitize_locator(handle.get("locator") or {}),
        "metadata": _sanitize_mapping(handle.get("metadata") or {}),
        "created_at": _truncate_safe_value(handle.get("created_at")),
    }
    return {key: value for key, value in result.items() if value not in (None, {}, [])}


def _safe_event(event: Dict[str, Any]) -> Dict[str, Any]:
    result = {
        "event_id": _truncate_safe_value(event.get("event_id")),
        "session_id": _truncate_safe_value(event.get("session_id")),
        "event_type": _truncate_safe_value(event.get("event_type")),
        "source_type": _truncate_safe_value(event.get("source_type")),
        "source_id": _truncate_safe_value(event.get("source_id")),
        "created_at": _truncate_safe_value(event.get("created_at")),
        "summary": _truncate_safe_value(event.get("summary")),
    }
    return {key: value for key, value in result.items() if value is not None}


def _matching_handle(event: Dict[str, Any], handle_id: str | None) -> Dict[str, Any] | None:
    handles = event.get("retrieval_handles") or []
    if not handle_id:
        return handles[0] if handles else None
    for handle in handles:
        if isinstance(handle, dict) and handle.get("handle_id") == handle_id:
            return handle
    return None


def _artifact_content_policy(handles: Iterable[Dict[str, Any]], payload: Dict[str, Any]) -> str | None:
    """Return the safest content-access policy carried by event metadata."""
    sources: list[Dict[str, Any]] = [payload]
    for handle in handles:
        if not isinstance(handle, dict):
            continue
        for key in ("metadata", "locator"):
            value = handle.get(key)
            if isinstance(value, dict):
                sources.append(value)

    for source in sources:
        policy = source.get("artifact_access_policy")
        if isinstance(policy, str) and policy.strip():
            return _truncate_safe_value(policy.strip())
        artifact_kind = source.get("artifact_kind")
        if artifact_kind == "delegation_child_final_response":
            return "delegate_only"
    return None


def _safe_artifacts(handles: Iterable[Dict[str, Any]], payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    handles = list(handles)
    policy = _artifact_content_policy(handles, payload)
    seen = set()
    for source in [payload, *(handle.get("locator") or {} for handle in handles)]:
        if not isinstance(source, dict):
            continue
        artifact = {}
        for key in ("artifact_path", "sandbox_artifact_path"):
            value = source.get(key)
            if isinstance(value, str):
                artifact[key] = _redact_locator_string(value)
        if artifact:
            if policy:
                artifact["artifact_access_policy"] = policy
            if policy == "delegate_only":
                artifact["parent_access"] = "metadata_only"
                artifact["inspection_route"] = "delegate_task"
            marker = tuple(sorted(artifact.items()))
            if marker not in seen:
                seen.add(marker)
                artifacts.append(artifact)
    return artifacts


def _artifact_inspection_contract(artifacts: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not artifacts:
        return None
    delegate_only = any(
        artifact.get("artifact_access_policy") == "delegate_only"
        for artifact in artifacts
    )
    return {
        "artifact_access_policy": "delegate_only" if delegate_only else "explicit_only",
        "parent_access": "metadata_only",
        "recommended_tool": "delegate_task",
        "recommended_toolsets": ["file"],
        "instruction": (
            "Do not read artifact contents into the parent context. Delegate a focused "
            "child/subagent with the artifact locator and request a concise summary, "
            "evidence counts, or validation result only."
        ),
    }


def _raw_size_metadata(event: Dict[str, Any], handles: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    sizes: Dict[str, Any] = {}
    payload = event.get("payload") or {}
    if isinstance(payload, dict):
        for key in _SAFE_RAW_METADATA_KEYS:
            if key in payload:
                sizes[key] = payload[key]
    for handle in handles:
        metadata = handle.get("metadata") if isinstance(handle, dict) else None
        if not isinstance(metadata, dict):
            continue
        for key in _SAFE_RAW_METADATA_KEYS:
            if key in metadata and key not in sizes:
                sizes[key] = metadata[key]
    return sizes


def _resolve_event(event_id: str | None, handle_id: str | None, db, current_session_id: str | None):
    event_by_event_id = None
    event_by_handle_id = None

    if event_id:
        get_by_event_id = getattr(db, "get_session_event_by_id", None)
        if not callable(get_by_event_id):
            return None, "Session database does not support event lookup."
        event_by_event_id = get_by_event_id(event_id, session_id=current_session_id)

    if handle_id:
        get_by_handle_id = getattr(db, "get_session_event_by_handle_id", None)
        if not callable(get_by_handle_id):
            return None, "Session database does not support handle lookup."
        event_by_handle_id = get_by_handle_id(handle_id, session_id=current_session_id)

    if event_id and not event_by_event_id:
        return None, "No matching retrieval metadata found."
    if handle_id and not event_by_handle_id:
        return None, "No matching retrieval metadata found."

    if event_by_event_id and event_by_handle_id:
        if event_by_event_id.get("event_id") != event_by_handle_id.get("event_id"):
            return None, "No matching retrieval metadata found."
        return event_by_event_id, None

    return event_by_event_id or event_by_handle_id, None


def retrieval_resolve(
    event_id: str | None = None,
    handle_id: str | None = None,
    db=None,
    current_session_id: str | None = None,
) -> str:
    """Resolve an event/handle into safe metadata and locators only."""
    event_id = event_id.strip() if isinstance(event_id, str) else None
    handle_id = handle_id.strip() if isinstance(handle_id, str) else None
    current_session_id = current_session_id.strip() if isinstance(current_session_id, str) else None
    if not event_id and not handle_id:
        return tool_error("Provide event_id, handle_id, or both.", success=False)
    if not current_session_id:
        return tool_error("Current session id required.", success=False)
    if db is None:
        return tool_error("Session database not available.", success=False)

    try:
        event, error = _resolve_event(event_id, handle_id, db, current_session_id)
        if error or not event:
            return tool_error(error or "No matching retrieval metadata found.", success=False)

        handles = [handle for handle in (event.get("retrieval_handles") or []) if isinstance(handle, dict)]
        matched_handle = _matching_handle(event, handle_id)
        if handle_id and not matched_handle:
            return tool_error("No matching retrieval metadata found.", success=False)

        safe_handles = [_safe_handle(handle) for handle in handles]
        payload_metadata = _sanitize_mapping(event.get("payload") or {})
        artifacts = _safe_artifacts(handles, event.get("payload") or {})
        raw_sizes = _raw_size_metadata(event, handles)

        result: Dict[str, Any] = {
            "success": True,
            "kind": "retrieval_resolution",
            "event": _safe_event(event),
            "retrieval_handles": safe_handles,
            "payload_metadata": payload_metadata,
            "guidance": (
                "Raw content is not returned. Inspect artifacts only through explicit "
                "delegation or targeted file/artifact tools when the user task requires it."
            ),
        }
        if matched_handle:
            result["matched_handle"] = _safe_handle(matched_handle)
        if artifacts:
            result["artifacts"] = artifacts
            artifact_contract = _artifact_inspection_contract(artifacts)
            if artifact_contract:
                result["artifact_inspection"] = artifact_contract
        if raw_sizes:
            result["raw_size"] = raw_sizes
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return tool_error(f"Failed to resolve retrieval metadata: {exc}", success=False)


def check_retrieval_resolve_requirements() -> bool:
    try:
        from hermes_state import DEFAULT_DB_PATH
        return DEFAULT_DB_PATH.parent.exists()
    except ImportError:
        return False


registry.register(
    name="retrieval_resolve",
    toolset="session_search",
    schema=RETRIEVAL_RESOLVE_SCHEMA,
    handler=lambda args, **kw: retrieval_resolve(
        event_id=args.get("event_id"),
        handle_id=args.get("handle_id"),
        db=kw.get("db"),
        current_session_id=kw.get("current_session_id"),
    ),
    check_fn=check_retrieval_resolve_requirements,
    emoji="🧭",
)
