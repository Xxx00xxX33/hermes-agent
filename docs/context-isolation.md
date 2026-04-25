# Hermes Context Isolation

Date: 2026-04-25
Repo: `/root/.hermes/hermes-agent`
Live runtime: `/root/.hermes/releases/hermes-agent-custom-20260418_115717-rebuild`
Status: implemented parent-safe context isolation baseline, with follow-up hardening listed below

## Goal

Hermes should keep the main agent's context focused on planning, synthesis, and validation. Dirty or high-volume work should happen in isolated child/subagent contexts, and raw data should not be hydrated into the parent prompt unless the user explicitly asks for a small, targeted result.

The durable invariant is:

> Parent context receives safe summaries, metadata, handles, locators, validation state, and decisions. Raw tool output, oversized child responses, logs, transcripts, file dumps, stdout/stderr, and artifact bodies stay outside the parent context and are inspected through explicit delegation or targeted tooling.

This document describes the current implementation boundaries and the rules future changes must preserve.

## Scope

This is a design snapshot for the Naranja Hermes context-isolation customization. It covers:

- `delegate_task` child/subagent isolation
- bounded child final responses
- persisted artifacts for omitted child responses
- parent-safe `SessionEvent` and `RetrievalHandle` metadata
- `session_search` recall behavior
- metadata-only `retrieval_resolve`
- artifact inspection policy for delegated child final-response artifacts
- verification requirements for maintenance and live runtime trees

It does not replace tests. Any change to these surfaces must include regression coverage for raw-data non-leakage.

## Parent vs. child responsibilities

### Parent agent

The parent agent is responsible for:

- understanding the user goal
- planning the work
- deciding what must be delegated
- synthesizing child summaries
- validating results and evidence
- returning the final answer

Parent-visible data should be compact and safe:

- short summaries
- status fields
- counts and byte/character sizes
- safe metadata
- safe artifact locators
- retrieval handles
- event IDs
- validation outcomes

### Child/subagent

A child agent is responsible for dirty work:

- tool-heavy exploration
- code inspection and edits
- tests and command execution
- raw artifact inspection
- large-output handling
- focused debugging or review

Children run with isolated conversation history and their own task/session context. Their intermediate tool calls and raw results are not appended to the parent's conversation history.

## Core data model

Implemented in `agent/context_references.py`.

### `RetrievalHandle`

A `RetrievalHandle` is a compact pointer to retrievable context. It is identified by a stable handle ID derived from safe identifiers and locators, not raw payloads.

Important fields:

- `handle_id`
- `source_type`
- `source_id`
- `locator`
- `metadata`
- `created_at`

Rules:

- `metadata` may contain safe size fields such as `raw_chars`, `raw_bytes`, and `raw_size`.
- `metadata` must not contain raw output, body/content text, transcripts, stdout/stderr, snippets, previews, or messages.
- `locator` may point to an artifact, but the parent must treat the locator as a pointer, not as permission to hydrate the artifact body into context.

### `SessionEvent`

A `SessionEvent` is a summary-level event that can be searched and resolved later.

Important fields:

- `event_id`
- `event_type`
- `session_id`
- `summary`
- `payload`
- `retrieval_handles`
- `created_at`

Rules:

- `summary` should be human-readable and safe.
- `payload` should hold safe metadata, status, counts, and locators.
- `retrieval_handles` should point to raw or bulky details indirectly.
- `searchable_text` recorded in the session database must remain a safe breadcrumb, not a raw payload copy.

## Delegation boundary

Implemented in `tools/delegate_tool.py`.

`delegate_task` spawns one or more child `AIAgent` instances with:

- a fresh conversation
- isolated task/session context
- restricted toolsets
- no parent conversation history
- `skip_context_files=True`
- `skip_memory=True`
- no `clarify` callback
- a depth limit that prevents recursive delegation beyond the configured boundary

Blocked child tools include:

- `delegate_task`
- `clarify`
- `memory`
- `send_message`
- `execute_code`

The child system prompt instructs subagents to provide concise summaries with evidence and validation status, and not to paste raw logs, large file contents, command dumps, or other bulky data.

## Child final-response handling

Implemented in `tools/delegate_tool.py`.

The parent-safe child summary limit is enforced by `_parent_safe_child_summary(...)`.

Current behavior:

- Child final responses at or below `_PARENT_CHILD_SUMMARY_MAX_CHARS` are returned to the parent as summaries.
- Oversized child final responses are omitted from the parent-visible result.
- The parent receives an omission notice plus safe size metadata.
- When the parent has a session ID and session DB, the omitted response is persisted as an artifact outside the parent prompt.

Parent-visible fields for omitted child responses may include:

- `summary_truncated`
- `summary_omitted_chars`
- `summary_chars`
- `summary_event_id`
- `summary_retrieval_handle`
- `raw_chars`
- `raw_bytes`

The raw child final response must not appear in:

- the `delegate_task` JSON result
- `SessionEvent.summary`
- `SessionEvent.payload`
- `searchable_text`
- `RetrievalHandle.metadata`
- final parent answer unless explicitly summarized in a small, targeted way

## Omitted child-response artifacts

Implemented in `tools/delegate_tool.py`.

Oversized child final responses are written under the active Hermes home:

```text
session_events/delegation_child_responses/<session_id>/task_<task_index>_<event_id>.txt
```

The artifact body contains the raw omitted child final response. The parent-visible event and handle contain only safe metadata and a locator.

Current delegated child final-response artifact metadata:

```text
artifact_kind = delegation_child_final_response
artifact_access_policy = delegate_only
parent_access = metadata_only
```

The safe search breadcrumb may mention:

```text
artifact_access_policy=delegate_only
```

It must not include the raw child response.

## Session search boundary

Implemented in `tools/session_search_tool.py` and `hermes_state.py`.

`session_search` supports two recall surfaces:

1. Session event search results
2. Conversation/session summary search results

Session event results are formatted with sanitized payloads and sanitized retrieval handles. Unsafe metadata keys are dropped, including keys containing:

- `raw`
- `body`
- `content`
- `searchable_text`
- `snippet`
- `preview`
- `stdout`
- `stderr`
- `output`
- `text`
- `message`
- `transcript`

Safe raw-size metadata keys are allowed:

- `raw_chars`
- `raw_bytes`
- `raw_size`

Known limitation: the legacy conversation summary path can fall back to a bounded raw preview when summarization is unavailable. That fallback is intentionally short, but it is not as strong as the session-event metadata-only contract. Future hardening should replace that fallback with a safer summary/error surface for contexts where raw-free recall is mandatory.

## Retrieval resolution boundary

Implemented in `tools/retrieval_resolve_tool.py`.

`retrieval_resolve` resolves an `event_id`, a `handle_id`, or both into safe metadata and artifact locators.

Hard rules:

- `current_session_id` is required before lookup.
- Lookup is scoped to the current session.
- The tool never returns raw artifact contents.
- The tool never returns `searchable_text`, snippets, previews, stdout/stderr, body/content/message fields, transcripts, or raw output fields.
- Safe raw-size metadata may be returned.

Returned result sections can include:

- `event`
- `retrieval_handles`
- `matched_handle`
- `payload_metadata`
- `artifacts`
- `artifact_inspection`
- `raw_size`
- `guidance`

Sanitization is key-name based. Do not add new parent-visible keys containing unsafe tokens unless the sanitizer and tests are updated deliberately.

Important naming rule:

- Use `artifact_access_policy` for safe policy metadata.
- Do not rename it to `content_access_policy` without changing sanitizer behavior, because keys containing `content` are intentionally treated as unsafe.

## Artifact inspection contract

Implemented in `tools/retrieval_resolve_tool.py`.

When `retrieval_resolve` sees delegated child final-response artifact metadata, it annotates artifact locators with:

```text
artifact_access_policy = delegate_only
parent_access = metadata_only
inspection_route = delegate_task
```

It also returns an `artifact_inspection` contract similar to:

```json
{
  "artifact_access_policy": "delegate_only",
  "parent_access": "metadata_only",
  "recommended_tool": "delegate_task",
  "recommended_toolsets": ["file"],
  "instruction": "Do not read artifact contents into the parent context. Delegate a focused child/subagent with the artifact locator and request a concise summary, evidence counts, or validation result only."
}
```

Rules:

- Parent should not read `delegate_only` artifact contents directly into its context.
- Parent should delegate artifact inspection to a child/subagent.
- Child should return a concise summary, evidence count, validation result, or minimal targeted excerpt only when required by the user task.
- The artifact locator is safe metadata, not raw-content permission.

Current limitation: this is a contract-level guard. It does not yet hard-block every possible parent-side file read of a `delegate_only` artifact. Future hardening should add a warning or enforcement layer at file/artifact read boundaries.

## Context-reference boundary

Implemented in `agent/context_references.py`.

`@file`, `@folder`, `@git`, `@diff`, `@staged`, and `@url` references can inject selected context into prompts. They are governed by:

- workspace-root checks
- sensitive path denial
- token budget checks
- binary-file rejection
- warning/block surfaces

This mechanism is separate from delegation artifacts. Future changes must avoid treating a `delegate_only` artifact path as an ordinary parent-side `@file` attachment without an explicit safety decision.

## Verification and regression coverage

Current relevant tests include:

- `tests/tools/test_delegate.py`
- `tests/tools/test_retrieval_resolve.py`
- `tests/tools/test_session_search.py`

The tests should cover these invariants:

- oversized child responses are omitted from parent-visible JSON
- raw child sentinel data is not present in safe event surfaces
- `searchable_text` remains a safe breadcrumb
- retrieval handle metadata exposes policy and size fields, not raw data
- `retrieval_resolve` requires current session scope
- `retrieval_resolve` returns metadata and locators only
- artifact inspection policy routes raw inspection through `delegate_task`

For Naranja Hermes work, repository tests alone are not enough. After changes that can affect live Hermes behavior, run the full live verifier from a neutral directory:

```bash
cd /tmp
/root/.local/bin/hermes-verify-live-runtime --full
```

The live runtime currently imports from:

```text
/root/.hermes/releases/hermes-agent-custom-20260418_115717-rebuild
```

Maintenance checkout changes under:

```text
/root/.hermes/hermes-agent
```

must be mirrored into the live runtime when the user-visible Hermes behavior is expected to change.

## Extension checklist

When adding or modifying context-bearing surfaces, answer these questions before implementation is considered complete:

1. Can raw text, stdout/stderr, body/content, transcript, snippet, preview, or message fields reach the parent context?
2. Is any raw or bulky data copied into `searchable_text`?
3. Are raw-size fields separated from raw content fields?
4. Does the surface expose a safe retrieval handle instead of raw payloads?
5. Are artifact locators marked with an access policy?
6. Does `delegate_only` content route through child/subagent inspection?
7. Are lookups scoped to the current session where applicable?
8. Do tests include a unique raw sentinel and assert it is absent from parent-visible results?
9. Does the live runtime verifier pass from `/tmp`?
10. If the change affects both maintenance and live runtime trees, are both updated and checked clean?

## Recommended next hardening phases

### 1. Dedicated context-isolation verifier

Add a focused verifier or test suite such as:

```text
tests/context_isolation/
```

or:

```text
hermes-verify-context-isolation
```

It should centralize sentinel-based tests for raw non-leakage across delegation, session search, retrieval resolution, callbacks, and error paths.

### 2. Direct-read guard for `delegate_only` artifacts

Add a soft warning or hard guard when parent-side file/artifact tools attempt to read artifacts marked:

```text
artifact_access_policy = delegate_only
```

Start with warnings or explicit guidance to avoid breaking legitimate user-requested file reads. Promote to enforcement only after tests cover expected workflows.

### 3. Batch delegation and callback hardening

Expand raw-sentinel tests to cover:

- multiple oversized child outputs
- batch aggregation
- progress callbacks
- display callbacks
- `_memory_manager.on_delegation(...)`
- child exception/error paths
- debug logging surfaces

### 4. Session recall fallback hardening

Replace or gate the bounded raw preview fallback in `session_search` when auxiliary summarization is unavailable. Prefer returning a safe error, event-only metadata, or a small synthetic summary that does not expose raw transcript text.

### 5. Release-chain safety

Document and/or automate the synchronization path between:

```text
/root/.hermes/hermes-agent
/root/.hermes/releases/hermes-agent-custom-20260418_115717-rebuild
```

Future release rebuilds should preserve the context-isolation commits or fail verification clearly.
