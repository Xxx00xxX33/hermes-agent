from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import re
import subprocess
import time
from typing import Any


_COMPLETION_CLAIM_PATTERNS = [
    r"\bfixed\b",
    r"\bdone\b",
    r"\bcompleted\b",
    r"\bverified\b",
    r"\bimplemented\b",
    r"已修复",
    r"已完成",
    r"已验证",
]


@dataclass
class VerificationCheck:
    id: str
    description: str
    kind: str
    required: bool = True
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class CriterionSpec:
    criterion_id: str
    label: str
    evidence_mode: str


@dataclass
class TaskContract:
    task_summary: str
    deliverable: str
    success_criteria: list[str] = field(default_factory=list)
    criteria: list[CriterionSpec] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    prohibited_actions: list[str] = field(default_factory=list)
    must_requirements: list[str] = field(default_factory=list)
    should_requirements: list[str] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    scope_boundaries: list[str] = field(default_factory=list)
    non_goals: list[str] = field(default_factory=list)
    verification_checks: list[VerificationCheck] = field(default_factory=list)
    completion_mode: str = "advisory"
    source: str = "agent_inferred"
    inference_quality: str = "medium"
    inference_warnings: list[str] = field(default_factory=list)


@dataclass
class CheckResult:
    check_id: str
    status: str
    evidence: list[str] = field(default_factory=list)
    details: str = ""


@dataclass
class CriterionEvaluation:
    criterion_id: str
    label: str
    evidence_mode: str
    status: str
    satisfied_by: str = ""
    evidence: list[str] = field(default_factory=list)
    details: str = ""
    check_id: str = ""


@dataclass
class AcceptanceResult:
    status: str
    checks_passed: int = 0
    checks_failed: int = 0
    checks_inconclusive: int = 0
    unmet_criteria: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    passed_criteria: list[str] = field(default_factory=list)
    violated_prohibited_actions: list[str] = field(default_factory=list)
    executable_evidence_checks: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    inconclusive_checks: list[str] = field(default_factory=list)
    gate_reason: str = ""
    confidence: str = "low"
    risk_level: str = "weak"
    risk_signals: list[str] = field(default_factory=list)
    summary_version: int = 1
    criterion_evaluations: list[CriterionEvaluation] = field(default_factory=list)
    check_results: list[CheckResult] = field(default_factory=list)
    final_assessment: str = ""



def task_contract_to_dict(contract: TaskContract) -> dict[str, Any]:
    return asdict(contract)



def acceptance_result_to_dict(result: AcceptanceResult) -> dict[str, Any]:
    return asdict(result)



def infer_minimal_task_contract(
    user_message: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> TaskContract:
    text = (user_message or "").strip()
    summary = text[:200] or "Current task"
    requirement_parts = _split_requirement_candidates(text)
    constraints = _extract_constraints(text, requirement_parts=requirement_parts)
    success_criteria = _infer_success_criteria(text, requirement_parts=requirement_parts)
    prohibited_actions = _extract_prohibited_actions(text)
    if not success_criteria and not prohibited_actions and summary:
        success_criteria = [summary]
    must_requirements = _extract_must_requirements(text, success_criteria)
    should_requirements = _extract_should_requirements(text)
    invariants = _extract_invariants(text, requirement_parts=requirement_parts)
    scope_boundaries = _extract_scope_boundaries(text, requirement_parts=requirement_parts)
    non_goals = _extract_non_goals(text)
    verification_checks = [
        VerificationCheck(
            id="transcript_coverage",
            description="Final response and transcript should cover the core task request.",
            kind="transcript_coverage",
        ),
        VerificationCheck(
            id="tool_evidence",
            description="Completion claims should be backed by tool evidence when tools were needed.",
            kind="tool_evidence",
        ),
    ]

    explicit_path = _extract_explicit_path(text)
    if explicit_path:
        verification_checks.append(
            VerificationCheck(
                id="explicit_path_exists",
                description="Explicitly referenced artifact path should exist.",
                kind="file_exists",
                payload={"path": explicit_path},
            )
        )

    inferred_output_checks = _infer_output_checks(text, explicit_path, success_criteria)
    verification_checks.extend(inferred_output_checks)
    verification_checks.extend(_infer_prohibited_action_checks(prohibited_actions))
    verification_checks.extend(_infer_command_exit_zero_checks(text, success_criteria, verification_checks))
    verification_checks.extend(_infer_process_running_checks(text, success_criteria, verification_checks))
    verification_checks.extend(_infer_negative_output_checks(text, success_criteria, verification_checks))
    verification_checks.extend(_infer_file_unchanged_checks(text, success_criteria, verification_checks, explicit_path))
    verification_checks.extend(_infer_file_not_exists_checks(text, verification_checks, explicit_path))
    deliverable = _infer_deliverable(summary, explicit_path, success_criteria)
    inference_quality, inference_warnings = _assess_inference_quality(
        success_criteria=success_criteria,
        verification_checks=verification_checks,
        explicit_path=explicit_path,
    )

    return TaskContract(
        task_summary=summary,
        deliverable=deliverable,
        success_criteria=success_criteria,
        criteria=_derive_criteria_specs(success_criteria),
        constraints=constraints,
        prohibited_actions=prohibited_actions,
        must_requirements=must_requirements,
        should_requirements=should_requirements,
        invariants=invariants,
        scope_boundaries=scope_boundaries,
        non_goals=non_goals,
        verification_checks=verification_checks,
        completion_mode="advisory",
        source="agent_inferred",
        inference_quality=inference_quality,
        inference_warnings=inference_warnings,
    )



def _assess_inference_quality(
    *,
    success_criteria: list[str],
    verification_checks: list[VerificationCheck],
    explicit_path: str | None,
) -> tuple[str, list[str]]:
    executable_kinds = {
        "command",
        "command_exit_zero",
        "command_stdout_regex",
        "file_exists",
        "directory_exists",
        "file_contains",
        "file_glob_count",
        "file_unchanged",
        "file_not_exists",
        "file_mtime_recent",
        "stdout_regex",
        "json_field_match",
        "json_field_exists",
        "json_field_contains",
        "process_running",
        "stdout_not_regex",
        "file_not_contains",
    }
    inferred_executable_checks = [
        check for check in verification_checks
        if check.kind in executable_kinds and check.id != "explicit_path_exists"
    ]

    if explicit_path and inferred_executable_checks:
        return "high", []

    warnings: list[str] = []
    if len(success_criteria) <= 1:
        warnings.append("single undifferentiated requirement")
        return "low", warnings

    if not inferred_executable_checks:
        warnings.append("no explicit executable verification inferred")
        return "medium", warnings

    return "medium", []



def _split_requirement_candidates(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"[，,；;]\s*|\s+且\s*|\s+并且\s*|\s*同时\s*|\s*另外\s*|\s*还要\s*|\s+and also\s+|\s+and\s+(?=ensure\s+)|\s+and\s+(?=make sure\s+)", text or "", flags=re.IGNORECASE)
        if part.strip()
    ]



def _classify_requirement_clauses(text: str, requirement_parts: list[str] | None = None) -> list[dict[str, str]]:
    parts = requirement_parts if requirement_parts is not None else _split_requirement_candidates(text)
    clauses: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for part in parts:
        candidate = str(part or "").strip()
        if not candidate:
            continue

        prohibited_actions = _extract_prohibited_actions(candidate)
        if prohibited_actions:
            leading_text = candidate
            lowered_leading = leading_text.lower()
            split_markers = [" but do not ", " do not ", " and do not ", "但不要", "不要"]
            split_index = -1
            matched_marker = ""
            for marker in split_markers:
                idx = lowered_leading.find(marker.lower())
                if idx != -1:
                    split_index = idx
                    matched_marker = marker
                    break
            if split_index > 0:
                leading_candidate = _clean_requirement_text(leading_text[:split_index].strip(" ，,；;"))
                normalized_leading = _normalize_requirement_key(leading_candidate)
                key = ("success", normalized_leading)
                if (
                    leading_candidate
                    and normalized_leading
                    and key not in seen
                    and not _looks_like_incomplete_positive_clause(leading_candidate)
                ):
                    seen.add(key)
                    clauses.append({"role": "success", "text": leading_candidate})
            for action in prohibited_actions:
                normalized = _normalize_requirement_key(action)
                key = ("prohibited", normalized)
                if not normalized or key in seen:
                    continue
                seen.add(key)
                clauses.append({"role": "prohibited", "text": action})
            trailing_text = leading_text[split_index + len(matched_marker):].strip(" ，,；;") if split_index != -1 else ""
            if trailing_text and all(
                _normalize_requirement_key(trailing_text) != _normalize_requirement_key(action)
                for action in prohibited_actions
            ):
                trailing_cleaned = _clean_requirement_text(trailing_text)
                trailing_normalized = _normalize_requirement_key(trailing_cleaned)
                if trailing_normalized in {"but", "and", "also", "同时", "另外", "还要"}:
                    trailing_cleaned = ""
                    trailing_normalized = ""
                trailing_role = "constraint" if _is_constraint_requirement(trailing_cleaned) else "success"
                trailing_key = (trailing_role, trailing_normalized)
                if trailing_cleaned and trailing_normalized and trailing_key not in seen:
                    seen.add(trailing_key)
                    clauses.append({"role": trailing_role, "text": trailing_cleaned})
            continue

        cleaned = _clean_requirement_text(candidate)
        if not cleaned:
            continue
        role = "constraint" if _is_constraint_requirement(cleaned) else "success"
        normalized = _normalize_requirement_key(candidate)
        key = (role, normalized)
        if not normalized or key in seen:
            continue
        seen.add(key)
        clauses.append({"role": role, "text": cleaned})
    return clauses



def _infer_deliverable(summary: str, explicit_path: str | None, success_criteria: list[str]) -> str:
    if explicit_path:
        return explicit_path
    if success_criteria:
        for criterion in success_criteria:
            if _looks_like_deliverable_requirement(criterion):
                return criterion
        return success_criteria[0]
    return summary



def _extract_constraints(text: str, requirement_parts: list[str] | None = None) -> list[str]:
    clauses = _classify_requirement_clauses(text, requirement_parts=requirement_parts)
    return [item["text"] for item in clauses if item.get("role") == "constraint"]



def _extract_must_requirements(text: str, success_criteria: list[str]) -> list[str]:
    if not success_criteria:
        return []
    lowered = (text or "").lower()
    if any(token in lowered for token in ("必须", "务必", "must", "required", "需要确保", "一定要")):
        return list(dict.fromkeys(str(item).strip() for item in success_criteria if str(item).strip()))
    return []



def _extract_should_requirements(text: str) -> list[str]:
    clauses = _split_requirement_candidates(text)
    shoulds: list[str] = []
    for clause in clauses:
        candidate = str(clause or "").strip()
        lowered = candidate.lower()
        if not candidate:
            continue
        if lowered.startswith("最好") or lowered.startswith("建议") or lowered.startswith("should ") or " ideally " in lowered:
            shoulds.append(_clean_requirement_text(candidate))
    return list(dict.fromkeys(item for item in shoulds if item))



def _extract_invariants(text: str, requirement_parts: list[str] | None = None) -> list[str]:
    clauses = _classify_requirement_clauses(text, requirement_parts=requirement_parts)
    invariants: list[str] = []
    for item in clauses:
        candidate = str(item.get("text") or "").strip()
        lowered = candidate.lower()
        if not candidate:
            continue
        if item.get("role") == "constraint" or candidate.startswith("保持") or lowered.startswith("keep ") or lowered.startswith("preserve "):
            invariants.append(candidate)
    return list(dict.fromkeys(invariants))



def _extract_scope_boundaries(text: str, requirement_parts: list[str] | None = None) -> list[str]:
    clauses = requirement_parts if requirement_parts is not None else _split_requirement_candidates(text)
    boundaries: list[str] = []
    for clause in clauses:
        candidate = str(clause or "").strip()
        lowered = candidate.lower()
        if not candidate:
            continue
        if "不要修改" in candidate or "仅修改" in candidate or "只改" in candidate or lowered.startswith("only change ") or lowered.startswith("do not modify "):
            boundaries.append(_clean_requirement_text(candidate))
    return list(dict.fromkeys(item for item in boundaries if item))



def _extract_non_goals(text: str) -> list[str]:
    clauses = _split_requirement_candidates(text)
    non_goals: list[str] = []
    for clause in clauses:
        candidate = str(clause or "").strip()
        lowered = candidate.lower()
        if not candidate:
            continue
        if candidate.startswith("不要求") or candidate.startswith("不用") or lowered.startswith("not required") or lowered.startswith("no need to"):
            non_goals.append(_clean_requirement_text(candidate))
    return list(dict.fromkeys(item for item in non_goals if item))



def _looks_like_incomplete_positive_clause(candidate: str) -> bool:
    lowered = (candidate or "").strip().lower()
    if not lowered:
        return False
    return lowered in {
        "确保输出",
        "输出",
        "确保显示",
        "显示",
        "ensure output",
        "output",
        "ensure show",
        "show",
        "print",
        "ensure print",
        "but",
        "and",
        "also",
        "同时",
        "另外",
        "还要",
    }



def _looks_like_deliverable_requirement(candidate: str) -> bool:
    lowered = (candidate or "").strip().lower()
    if not lowered:
        return False
    return any(
        token in candidate
        for token in ("修好", "更新", "修改", "生成", "创建", "新增", "fix", "update", "modify", "create", "generate")
    )






def _is_constraint_requirement(candidate: str) -> bool:
    lowered = (candidate or "").strip().lower()
    return bool(lowered) and (
        candidate.startswith("保持")
        or lowered.startswith("keep ")
        or lowered.startswith("preserve ")
    )



def _normalize_requirement_text(candidate: str) -> str:
    normalized = re.sub(r"^(?:且|并且|并|also)\s*", "", candidate or "", flags=re.IGNORECASE)
    normalized = re.sub(r"^(?:确保|请确保|verify|make sure)\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized



def _clean_requirement_text(candidate: str) -> str:
    cleaned = re.sub(r"^(?:且|并且|并|要|also)\s*", "", candidate or "", flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned



def _normalize_requirement_key(candidate: str) -> str:
    return _normalize_requirement_text(candidate).lower()



def _extract_expected_content(criterion: str) -> str | None:
    patterns = [
        r"stdout\s+contains?\s+([^，,；;]+)",
        r"(?:包含|含有|输出|打印|print)\s+([^，,；;]+)",
        r"contains?\s+([^，,；;]+)",
    ]
    for pattern in patterns:
        content_match = re.search(pattern, criterion, flags=re.IGNORECASE)
        if not content_match:
            continue
        expected = content_match.group(1).strip()
        if expected:
            return expected
    return None



def _criterion_targets_explicit_path(criterion: str) -> bool:
    return any(keyword in criterion for keyword in ("文件", "file", "路径", "path", "里包含", "中包含"))



def _criterion_targets_output(criterion: str) -> bool:
    return any(keyword in criterion for keyword in ("输出", "stdout", "打印", "print"))



def _evidence_mode_for_criterion(criterion: str) -> str:
    if _criterion_requires_artifact_evidence(criterion):
        return "executable_required"
    if _criterion_requires_behavioral_evidence(criterion):
        return "tool_backed"
    return "transcript_ok"



def _derive_criterion_metadata(success_criteria: list[str]) -> list[dict[str, str]]:
    metadata: list[dict[str, str]] = []
    for index, criterion in enumerate(success_criteria, start=1):
        metadata.append(
            {
                "criterion_id": f"criterion_{index}",
                "label": criterion,
                "evidence_mode": _evidence_mode_for_criterion(criterion),
            }
        )
    return metadata



def _derive_criteria_specs(success_criteria: list[str]) -> list[CriterionSpec]:
    return [CriterionSpec(**item) for item in _derive_criterion_metadata(success_criteria)]



def _contract_criteria(contract: TaskContract | None, success_criteria: list[str] | None = None) -> list[CriterionSpec]:
    criteria = list((contract.criteria if contract else []) or [])
    if criteria:
        return criteria
    return _derive_criteria_specs(success_criteria or (contract.success_criteria if contract else []))



def _build_file_contains_check(
    criterion_id: str,
    criterion: str,
    explicit_path: str,
    expected: str,
    index: int,
) -> VerificationCheck:
    return VerificationCheck(
        id=f"criterion_contains_{index}",
        description="Criterion-specific artifact path should contain expected text.",
        kind="file_contains",
        payload={
            "path": explicit_path,
            "contains": expected,
            "criterion_id": criterion_id,
            "criterion": criterion,
            "evidence_role": "primary",
            "target": "file",
            "target_path": explicit_path,
            "target_field": "content",
            "target_source": "explicit_path",
        },
    )



def _build_stdout_regex_check(
    criterion_id: str,
    criterion: str,
    expected: str,
    index: int,
) -> VerificationCheck:
    return VerificationCheck(
        id=f"criterion_stdout_{index}",
        description="Criterion-specific output should contain expected text.",
        kind="stdout_regex",
        payload={
            "pattern": re.escape(expected),
            "criterion_id": criterion_id,
            "criterion": criterion,
            "evidence_role": "supporting",
            "target": "stdout",
            "target_field": "content",
            "target_source": "command_stdout",
        },
    )



def _infer_success_criteria(text: str, requirement_parts: list[str] | None = None) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    clauses = _classify_requirement_clauses(text, requirement_parts=requirement_parts)
    return [item["text"] for item in clauses if item.get("role") == "success"]



def _extract_prohibited_actions(text: str) -> list[str]:
    prohibited: list[str] = []
    seen: set[str] = set()

    for match in re.findall(r"(?:但)?不要([^，,；;]+)", text or ""):
        candidate = match.strip()
        if not candidate:
            continue
        normalized = candidate.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        prohibited.append(candidate)

    for match in re.findall(r"\bdo not\s+([^,;.!?]+)", text or "", flags=re.IGNORECASE):
        candidate = match.strip()
        if not candidate:
            continue
        normalized = candidate.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        prohibited.append(candidate)

    return prohibited



def _infer_output_checks(
    text: str,
    explicit_path: str | None,
    success_criteria: list[str],
) -> list[VerificationCheck]:
    checks: list[VerificationCheck] = []
    criterion_metadata = _derive_criterion_metadata(success_criteria)
    for item in criterion_metadata:
        criterion = item["label"]
        criterion_id = item["criterion_id"]
        next_index = len(checks) + 1

        directory_exists_match = re.search(r"目录\s+((?:/|\./|\.\./|~/)[^\s，。,；;：:！!？?]+)\s+存在", criterion)
        if not directory_exists_match:
            directory_exists_match = re.search(
                r"directory\s+((?:/|\./|\.\./|~/)[^\s,;:!?]+)\s+exists",
                criterion,
                flags=re.IGNORECASE,
            )
        if directory_exists_match:
            directory_path = str(Path(directory_exists_match.group(1).strip()).expanduser())
            checks.append(
                VerificationCheck(
                    id=f"criterion_directory_exists_{next_index}",
                    description="Criterion-specific directory should exist.",
                    kind="directory_exists",
                    payload={
                        "path": directory_path,
                        "criterion_id": criterion_id,
                        "criterion": criterion,
                        "evidence_role": "primary",
                        "target": "directory",
                        "target_path": directory_path,
                        "target_field": "exists",
                        "target_source": "explicit_path",
                    },
                )
            )
            continue

        file_glob_count_match = re.search(
            r"目录\s+((?:/|\./|\.\./|~/)[^\s，。,；;：:！!？?]+)\s+下有\s+(\d+)\s+个\s+(\.[A-Za-z0-9_-]+)\s+文件",
            criterion,
        )
        if not file_glob_count_match:
            file_glob_count_match = re.search(
                r"directory\s+((?:/|\./|\.\./|~/)[^\s,;:!?]+)\s+has\s+(\d+)\s+(\.[A-Za-z0-9_-]+)\s+files",
                criterion,
                flags=re.IGNORECASE,
            )
        if file_glob_count_match:
            directory_path = str(Path(file_glob_count_match.group(1).strip()).expanduser())
            checks.append(
                VerificationCheck(
                    id=f"criterion_file_glob_count_{next_index}",
                    description="Criterion-specific directory should contain the expected file count.",
                    kind="file_glob_count",
                    payload={
                        "path": directory_path,
                        "glob": f"*{file_glob_count_match.group(3)}",
                        "min_count": int(file_glob_count_match.group(2)),
                        "criterion_id": criterion_id,
                        "criterion": criterion,
                        "evidence_role": "primary",
                        "target": "directory",
                        "target_path": directory_path,
                        "target_field": "glob_count",
                        "target_source": "explicit_path",
                    },
                )
            )
            continue

        json_field_exists_match = re.search(r"JSON\s+里存在\s+([A-Za-z0-9_.-]+)\s+字段", criterion, flags=re.IGNORECASE)
        if not json_field_exists_match:
            json_field_exists_match = re.search(r"JSON\s+has\s+field\s+([A-Za-z0-9_.-]+)", criterion, flags=re.IGNORECASE)
        if json_field_exists_match:
            payload: dict[str, Any] = {
                "field_path": json_field_exists_match.group(1).strip(),
                "criterion_id": criterion_id,
                "criterion": criterion,
                "evidence_role": "primary",
                "target": "json",
                "target_field": json_field_exists_match.group(1).strip(),
                "target_source": "response_json",
            }
            if explicit_path:
                payload["path"] = explicit_path
                payload["target_path"] = explicit_path
                payload["target_source"] = "explicit_path_json"
            checks.append(
                VerificationCheck(
                    id=f"criterion_json_field_exists_{next_index}",
                    description="Criterion-specific JSON field should exist.",
                    kind="json_field_exists",
                    payload=payload,
                )
            )
            continue

        expected = _extract_expected_content(criterion)
        if not expected:
            continue

        if explicit_path and _criterion_targets_explicit_path(criterion):
            checks.append(_build_file_contains_check(criterion_id, criterion, explicit_path, expected, next_index))
            continue

        if _criterion_targets_output(criterion):
            checks.append(_build_stdout_regex_check(criterion_id, criterion, expected, next_index))
    return checks



def _infer_prohibited_action_checks(prohibited_actions: list[str]) -> list[VerificationCheck]:
    checks: list[VerificationCheck] = []
    for action in prohibited_actions:
        file_match = re.search(r"文件\s+((?:/|\./|\.\./|~/)[^\s，。,；;：:！!？?]+)\s+包含\s+(.+)$", action)
        if file_match:
            raw_path = file_match.group(1).strip()
            path = str(Path(raw_path).expanduser())
            forbidden = file_match.group(2).strip()
            if forbidden:
                checks.append(
                    VerificationCheck(
                        id=f"prohibited_file_{len(checks) + 1}",
                        description="File should not contain prohibited text.",
                        kind="file_not_contains",
                        payload={
                            "path": path,
                            "contains": forbidden,
                            "prohibited_action": action,
                            "target": "file",
                            "target_path": path,
                            "target_field": "content",
                            "target_source": "explicit_path",
                        },
                    )
                )
                continue

        if any(keyword in action for keyword in ("显示", "输出", "stdout", "打印", "print")):
            subject = action
            for prefix in ("显示", "输出", "打印", "print", "do not print"):
                if subject.lower().startswith(prefix.lower()):
                    subject = subject[len(prefix):].strip()
                    break
            if not subject:
                continue
            checks.append(
                VerificationCheck(
                    id=f"prohibited_output_{len(checks) + 1}",
                    description="Output should not contain prohibited text.",
                    kind="stdout_not_regex",
                    payload={
                        "pattern": re.escape(subject),
                        "prohibited_action": action,
                    },
                )
            )
    return checks



def _criterion_specific_existing_check_kinds(
    success_criteria: list[str],
    verification_checks: list[VerificationCheck],
) -> dict[str, set[str]]:
    criteria = _derive_criterion_metadata(success_criteria)
    labels = {item["label"] for item in criteria}
    mapping: dict[str, set[str]] = {label: set() for label in labels}
    for check in verification_checks:
        bound_label = str(check.payload.get("criterion") or "").strip()
        if bound_label and bound_label in mapping:
            mapping[bound_label].add(str(check.kind or ""))
    return mapping



def _extract_shell_command(text: str) -> str | None:
    patterns = [
        r"命令\s+(.+?)\s+(?:执行成功|成功执行|exit\s*0)",
        r"command\s+(.+?)\s+(?:succeeds|succeeded|executed successfully|exit\s*0)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if not match:
            continue
        command = match.group(1).strip()
        if command:
            return command
    return None



def _extract_process_pattern(text: str) -> str | None:
    patterns = [
        r"进程\s+([^\s，。,；;：:！!？?]+)\s+正在运行",
        r"process\s+([^\s,;:!?]+)\s+is\s+running",
        r"service\s+([^\s,;:!?]+)\s+started\s+successfully",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if not match:
            continue
        subject = match.group(1).strip().strip("“”\"'` ")
        if subject:
            return subject
    return None



def _extract_negative_output_subject(text: str) -> str | None:
    patterns = [
        r"(?:输出|stdout)不要(?:显示|包含)?\s+([^，,；;。.!?]+)",
        r"output\s+must\s+not\s+contain\s+([^,;.!?]+)",
        r"do\s+not\s+(?:show|print|output)\s+([^,;.!?]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if not match:
            continue
        subject = match.group(1).strip().strip("“”\"'` ")
        if subject:
            return subject
    return None



def _read_text_if_small(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None



def _infer_command_exit_zero_checks(
    text: str,
    success_criteria: list[str],
    verification_checks: list[VerificationCheck],
) -> list[VerificationCheck]:
    command = _extract_shell_command(text)
    if not command:
        return []
    existing_kinds = _criterion_specific_existing_check_kinds(success_criteria, verification_checks)
    checks: list[VerificationCheck] = []
    for item in _derive_criterion_metadata(success_criteria):
        criterion = item["label"]
        lowered = criterion.lower()
        if criterion not in existing_kinds:
            continue
        if "command_exit_zero" in existing_kinds[criterion]:
            continue
        if "命令" not in criterion and "command" not in lowered:
            continue
        if not any(token in lowered for token in ("执行成功", "成功执行", "exit 0", "succeeds", "executed successfully")):
            continue
        checks.append(
            VerificationCheck(
                id=f"criterion_command_exit_zero_{len(checks) + 1}",
                description="Criterion-specific command should exit with code 0.",
                kind="command_exit_zero",
                payload={
                    "command": command,
                    "criterion_id": item["criterion_id"],
                    "criterion": criterion,
                    "evidence_role": "primary",
                    "target": "command",
                    "target_field": "exit_code",
                    "target_source": "explicit_command",
                },
            )
        )
    return checks



def _infer_process_running_checks(
    text: str,
    success_criteria: list[str],
    verification_checks: list[VerificationCheck],
) -> list[VerificationCheck]:
    pattern = _extract_process_pattern(text)
    if not pattern:
        return []
    existing_kinds = _criterion_specific_existing_check_kinds(success_criteria, verification_checks)
    checks: list[VerificationCheck] = []
    for item in _derive_criterion_metadata(success_criteria):
        criterion = item["label"]
        lowered = criterion.lower()
        if criterion not in existing_kinds:
            continue
        if "process_running" in existing_kinds[criterion]:
            continue
        if not any(token in criterion for token in ("进程", "服务")) and "process" not in lowered and "service" not in lowered:
            continue
        if not any(token in lowered for token in ("正在运行", "运行中", "is running", "started successfully")):
            continue
        checks.append(
            VerificationCheck(
                id=f"criterion_process_running_{len(checks) + 1}",
                description="Criterion-specific process should be running.",
                kind="process_running",
                payload={
                    "pattern": pattern,
                    "criterion_id": item["criterion_id"],
                    "criterion": criterion,
                    "evidence_role": "primary",
                    "target": "process",
                    "target_field": "running",
                    "target_source": "explicit_process_pattern",
                },
            )
        )
    return checks



def _infer_negative_output_checks(
    text: str,
    success_criteria: list[str],
    verification_checks: list[VerificationCheck],
) -> list[VerificationCheck]:
    subject = _extract_negative_output_subject(text)
    if not subject:
        return []
    existing_kinds = _criterion_specific_existing_check_kinds(success_criteria, verification_checks)
    checks: list[VerificationCheck] = []
    for item in _derive_criterion_metadata(success_criteria):
        criterion = item["label"]
        lowered = criterion.lower()
        if criterion not in existing_kinds:
            continue
        if "stdout_not_regex" in existing_kinds[criterion]:
            continue
        if not any(token in criterion for token in ("输出", "显示")) and not any(token in lowered for token in ("output", "show", "print", "stdout")):
            continue
        if not any(token in lowered for token in ("不要", "must not", "do not", "not contain")):
            continue
        checks.append(
            VerificationCheck(
                id=f"criterion_stdout_not_regex_{len(checks) + 1}",
                description="Criterion-specific output should not contain prohibited text.",
                kind="stdout_not_regex",
                payload={
                    "pattern": re.escape(subject),
                    "criterion_id": item["criterion_id"],
                    "criterion": criterion,
                    "evidence_role": "primary",
                    "target": "stdout",
                    "target_field": "content",
                    "target_source": "explicit_negative_output",
                },
            )
        )
    return checks



def _infer_file_unchanged_checks(
    text: str,
    success_criteria: list[str],
    verification_checks: list[VerificationCheck],
    explicit_path: str | None,
) -> list[VerificationCheck]:
    if not explicit_path:
        return []
    path = Path(explicit_path).expanduser()
    expected_content = _read_text_if_small(path)
    if expected_content is None:
        return []
    existing_kinds = _criterion_specific_existing_check_kinds(success_criteria, verification_checks)
    checks: list[VerificationCheck] = []
    for item in _derive_criterion_metadata(success_criteria):
        criterion = item["label"]
        lowered = criterion.lower()
        if criterion not in existing_kinds:
            continue
        if "file_unchanged" in existing_kinds[criterion]:
            continue
        if not any(token in criterion for token in ("保持", "不变", "不要修改")) and not any(token in lowered for token in ("keep", "unchanged", "do not modify")):
            continue
        if "文件" not in criterion and "file" not in lowered and "path" not in lowered and "路径" not in criterion:
            continue
        checks.append(
            VerificationCheck(
                id=f"criterion_file_unchanged_{len(checks) + 1}",
                description="Criterion-specific file should remain unchanged.",
                kind="file_unchanged",
                payload={
                    "path": str(path),
                    "content": expected_content,
                    "criterion_id": item["criterion_id"],
                    "criterion": criterion,
                    "evidence_role": "primary",
                    "target": "file",
                    "target_path": str(path),
                    "target_field": "content",
                    "target_source": "explicit_path",
                },
            )
        )
    if checks:
        return checks
    lowered_text = (text or "").lower()
    if any(token in text for token in ("保持", "不变", "不要修改")) or any(token in lowered_text for token in ("keep", "unchanged", "do not modify")):
        return [
            VerificationCheck(
                id="inferred_file_unchanged_1",
                description="Explicitly referenced file should remain unchanged.",
                kind="file_unchanged",
                payload={
                    "path": str(path),
                    "content": expected_content,
                    "target": "file",
                    "target_path": str(path),
                    "target_field": "content",
                    "target_source": "explicit_path",
                },
            )
        ]
    return checks



def _infer_file_not_exists_checks(
    text: str,
    verification_checks: list[VerificationCheck],
    explicit_path: str | None,
) -> list[VerificationCheck]:
    if not explicit_path:
        return []
    lowered = (text or "").lower()
    if not any(token in lowered for token in ("不要生成", "不应留下", "must not create", "do not create", "should not exist")):
        return []
    if any(check.kind == "file_not_exists" and str(check.payload.get("path") or "").strip() == explicit_path for check in verification_checks):
        return []
    return [
        VerificationCheck(
            id="inferred_file_not_exists_1",
            description="Explicit artifact path should not exist.",
            kind="file_not_exists",
            payload={
                "path": explicit_path,
                "target": "file",
                "target_path": explicit_path,
                "target_field": "exists",
                "target_source": "explicit_path",
            },
        )
    ]



def _run_file_not_exists_check(check: VerificationCheck) -> CheckResult:
    raw_path = str(check.payload.get("path") or "").strip()
    if not raw_path:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="File-not-exists check missing payload.path.",
        )

    path = Path(raw_path).expanduser()
    if path.exists():
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=[f"path exists: {path}"],
            details="Executable file absence check failed because the path exists.",
        )
    return CheckResult(
        check_id=check.id,
        status="pass",
        evidence=[f"path absent: {path}"],
        details="Executable file absence check passed.",
    )



def _run_file_mtime_recent_check(check: VerificationCheck) -> CheckResult:
    raw_path = str(check.payload.get("path") or "").strip()
    if not raw_path:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="File-mtime-recent check missing payload.path.",
        )
    within_seconds = check.payload.get("within_seconds")
    if within_seconds is None:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="File-mtime-recent check missing payload.within_seconds.",
        )
    try:
        threshold = float(within_seconds)
    except (TypeError, ValueError):
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="File-mtime-recent payload.within_seconds must be numeric.",
        )

    path = Path(raw_path).expanduser()
    if not path.exists():
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=[],
            details=f"File-mtime-recent check failed because path does not exist: {path}",
        )

    age_seconds = time.time() - path.stat().st_mtime
    evidence = [f"file age seconds: {age_seconds:.3f}", f"threshold seconds: {threshold:.3f}"]
    if age_seconds <= threshold:
        return CheckResult(
            check_id=check.id,
            status="pass",
            evidence=evidence,
            details="Executable file recency check passed.",
        )
    return CheckResult(
        check_id=check.id,
        status="fail",
        evidence=evidence,
        details=f"File recency check failed because age {age_seconds:.3f}s exceeded threshold {threshold:.3f}s.",
    )



def evaluate_claim_evidence_consistency(
    contract: TaskContract,
    messages: list[dict[str, Any]],
) -> AcceptanceResult:
    final_response = _extract_final_response(messages)
    tool_messages = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
    tool_names = [str(m.get("tool_name") or "tool") for m in tool_messages]
    completion_claimed = _contains_completion_claim(final_response)
    check_results: list[CheckResult] = []
    unmet_criteria: list[str] = []
    missing_evidence: list[str] = []
    criteria = _contract_criteria(contract)
    transcript_check = next((check for check in contract.verification_checks if check.id == "transcript_coverage"), None)
    other_checks = [check for check in contract.verification_checks if check.id != "transcript_coverage"]

    for check in other_checks:
        if check.id == "tool_evidence":
            has_executable_checks = any(
                candidate.kind in {
                    "command",
                    "command_exit_zero",
                    "file_exists",
                    "directory_exists",
                    "file_contains",
                    "file_glob_count",
                    "file_unchanged",
                    "file_not_exists",
                    "file_mtime_recent",
                    "stdout_regex",
                    "json_field_match",
                    "json_field_exists",
                    "json_field_contains",
                    "process_running",
                    "stdout_not_regex",
                    "file_not_contains",
                }
                for candidate in contract.verification_checks
            )
            if completion_claimed and not tool_messages and not has_executable_checks:
                missing_evidence.append("completion claim without tool evidence")
                check_results.append(
                    CheckResult(
                        check_id=check.id,
                        status="inconclusive",
                        evidence=[],
                        details="Assistant claimed completion without tool evidence.",
                    )
                )
            else:
                evidence = tool_names or (
                    ["executable acceptance checks provided evidence"]
                    if has_executable_checks
                    else ["no explicit completion claim"] if not completion_claimed else []
                )
                check_results.append(
                    CheckResult(
                        check_id=check.id,
                        status="pass",
                        evidence=evidence,
                        details="Tool-evidence consistency satisfied.",
                    )
                )
        elif check.kind == "command":
            check_results.append(_run_command_check(check))
        elif check.kind == "command_stdout_regex":
            check_results.append(_run_command_stdout_regex_check(check))
        elif check.kind == "command_exit_zero":
            check_results.append(_run_command_exit_zero_check(check))
        elif check.kind == "file_exists":
            check_results.append(_run_file_exists_check(check))
        elif check.kind == "directory_exists":
            check_results.append(_run_directory_exists_check(check))
        elif check.kind == "file_contains":
            check_results.append(_run_file_contains_check(check))
        elif check.kind == "file_glob_count":
            check_results.append(_run_file_glob_count_check(check))
        elif check.kind == "file_unchanged":
            check_results.append(_run_file_unchanged_check(check))
        elif check.kind == "file_not_exists":
            check_results.append(_run_file_not_exists_check(check))
        elif check.kind == "file_mtime_recent":
            check_results.append(_run_file_mtime_recent_check(check))
        elif check.kind == "stdout_regex":
            check_results.append(_run_stdout_regex_check(check))
        elif check.kind == "json_field_match":
            check_results.append(_run_json_field_match_check(check))
        elif check.kind == "json_field_exists":
            check_results.append(_run_json_field_exists_check(check))
        elif check.kind == "json_field_contains":
            check_results.append(_run_json_field_contains_check(check))
        elif check.kind == "process_running":
            check_results.append(_run_process_running_check(check))
        elif check.kind == "stdout_not_regex":
            check_results.append(_run_stdout_not_regex_check(check, final_response, tool_messages))
        elif check.kind == "file_not_contains":
            check_results.append(_run_file_not_contains_check(check))
        else:
            check_results.append(
                CheckResult(
                    check_id=check.id,
                    status="inconclusive",
                    evidence=[],
                    details="Unsupported MVP check type.",
                )
            )

    prohibited_results = _evaluate_prohibited_actions(
        contract.prohibited_actions,
        contract.verification_checks,
        check_results,
    )
    if prohibited_results:
        check_results.extend(prohibited_results)
        unmet_criteria.extend(
            _prohibited_label(result.check_id)
            for result in prohibited_results
            if result.status != "pass"
        )

    criterion_results = _evaluate_success_criteria(
        contract.success_criteria,
        final_response,
        tool_messages,
        verification_checks=contract.verification_checks,
        check_results=check_results,
        criteria=criteria,
    )
    criterion_evaluations = _criterion_evaluations_from_results(criteria, criterion_results)
    if transcript_check:
        check_results.extend(criterion_results)
        criterion_passes = sum(1 for result in criterion_results if result.status == "pass")
        criterion_total = len(criterion_results)
        if criterion_total == 0 or criterion_passes == criterion_total:
            check_results.append(
                CheckResult(
                    check_id=transcript_check.id,
                    status="pass",
                    evidence=["final response or bound checks reference all task criteria"],
                    details=f"Transcript coverage satisfied ({criterion_passes}/{criterion_total}).",
                )
            )
        else:
            unmet_criteria.extend(
                _criterion_label(result.check_id, contract.success_criteria)
                for result in criterion_results
                if result.status != "pass"
            )
            check_results.append(
                CheckResult(
                    check_id=transcript_check.id,
                    status="inconclusive",
                    evidence=[],
                    details=f"Transcript coverage incomplete ({criterion_passes}/{criterion_total}).",
                )
            )

    prohibited_bound_check_ids = {
        check.id
        for check in contract.verification_checks
        if str(check.payload.get("prohibited_action") or "").strip()
    }
    status_relevant_results = [
        check
        for check in check_results
        if check.check_id not in prohibited_bound_check_ids
    ]

    checks_failed = sum(1 for check in status_relevant_results if check.status == "fail")
    checks_inconclusive = sum(1 for check in status_relevant_results if check.status == "inconclusive")
    checks_passed = sum(1 for check in status_relevant_results if check.status == "pass")

    if checks_failed:
        status = "fail"
    elif checks_inconclusive:
        status = "inconclusive"
    else:
        status = "pass"

    final_assessment = _build_final_assessment(
        status=status,
        check_results=check_results,
        unmet_criteria=unmet_criteria,
        missing_evidence=missing_evidence,
        success_criteria=contract.success_criteria,
    )
    passed_criteria = list(dict.fromkeys(
        _criterion_label(check.check_id, contract.success_criteria)
        for check in criterion_results
        if check.status == "pass"
    ))
    violated_prohibited_actions = list(dict.fromkeys(
        _prohibited_label(check.check_id)
        for check in prohibited_results
        if check.status == "fail"
    ))
    executable_evidence_checks = list(dict.fromkeys(
        check.check_id
        for check in check_results
        if check.status == "pass"
        and not check.check_id.startswith(("criterion:", "prohibited:"))
        and check.check_id not in {"transcript_coverage", "tool_evidence"}
    ))
    failed_checks = list(dict.fromkeys(
        check.check_id
        for check in check_results
        if check.status == "fail"
        and not check.check_id.startswith(("criterion:", "prohibited:"))
    ))
    inconclusive_checks = list(dict.fromkeys(
        check.check_id
        for check in check_results
        if check.status == "inconclusive"
        and not check.check_id.startswith(("criterion:", "prohibited:"))
    ))
    tool_evidence_present = any(check.check_id == "tool_evidence" and check.status == "pass" for check in check_results)
    gate_reason = (
        "acceptance checks failed"
        if status == "fail"
        else "criterion evidence incomplete"
        if unmet_criteria or missing_evidence or inconclusive_checks
        else "executable-backed acceptance"
        if executable_evidence_checks
        else "tool-backed acceptance"
        if passed_criteria and tool_evidence_present
        else "transcript-only acceptance"
        if passed_criteria
        else "all acceptance checks passed"
    )
    confidence = _compute_acceptance_confidence(contract, criterion_results, check_results, executable_evidence_checks)
    contract._acceptance_final_response = final_response
    risk_level, risk_signals, gate_reason = _classify_acceptance_risk(
        contract=contract,
        status=status,
        gate_reason=gate_reason,
        executable_evidence_checks=executable_evidence_checks,
        confidence=confidence,
    )

    return AcceptanceResult(
        status=status,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
        checks_inconclusive=checks_inconclusive,
        unmet_criteria=unmet_criteria,
        missing_evidence=missing_evidence,
        passed_criteria=passed_criteria,
        violated_prohibited_actions=violated_prohibited_actions,
        executable_evidence_checks=executable_evidence_checks,
        failed_checks=failed_checks,
        inconclusive_checks=inconclusive_checks,
        gate_reason=gate_reason,
        confidence=confidence,
        risk_level=risk_level,
        risk_signals=risk_signals,
        summary_version=1,
        criterion_evaluations=criterion_evaluations,
        check_results=check_results,
        final_assessment=final_assessment,
    )



def _evaluate_success_criteria(
    success_criteria: list[str],
    final_response: str,
    tool_messages: list[dict[str, Any]],
    verification_checks: list[VerificationCheck] | None = None,
    check_results: list[CheckResult] | None = None,
    criteria: list[CriterionSpec] | None = None,
) -> list[CheckResult]:
    verification_checks = verification_checks or []
    check_results = check_results or []
    results: list[CheckResult] = []
    criteria = list(criteria or _derive_criteria_specs(success_criteria))
    for index, spec in enumerate(criteria, start=1):
        criterion = spec.label
        criterion_id = spec.criterion_id
        evidence_mode = spec.evidence_mode
        bound_checks = [
            check
            for check in verification_checks
            if str(check.payload.get("criterion_id") or "").strip() == criterion_id
            or str(check.payload.get("criterion") or "").strip() == criterion
        ]
        primary_bound_check_ids = {
            check.id
            for check in bound_checks
            if str(check.payload.get("evidence_role") or "").strip().lower() == "primary"
        }
        supporting_bound_check_ids = {
            check.id
            for check in bound_checks
            if check.id not in primary_bound_check_ids
        }
        failed_primary_results = [
            result
            for result in check_results
            if result.check_id in primary_bound_check_ids and result.status == "fail"
        ]
        passed_primary_results = [
            result
            for result in check_results
            if result.check_id in primary_bound_check_ids and result.status == "pass"
        ]
        passed_supporting_results = [
            result
            for result in check_results
            if result.check_id in supporting_bound_check_ids and result.status == "pass"
        ]
        inconclusive_primary_results = [
            result
            for result in check_results
            if result.check_id in primary_bound_check_ids and result.status == "inconclusive"
        ]

        if failed_primary_results:
            evidence = [f"primary bound check failed: {result.check_id}" for result in failed_primary_results]
            results.append(
                CheckResult(
                    check_id=f"criterion:{criterion_id}",
                    status="fail",
                    evidence=evidence,
                    details=f"Criterion {index} contradicted by primary bound evidence.",
                )
            )
            continue

        if passed_primary_results:
            evidence = [f"criterion covered: {criterion}"]
            evidence.extend(f"bound check passed: {result.check_id}" for result in passed_primary_results)
            results.append(
                CheckResult(
                    check_id=f"criterion:{criterion_id}",
                    status="pass",
                    evidence=evidence,
                    details=f"Criterion {index} covered by primary bound evidence.",
                )
            )
            continue

        if inconclusive_primary_results:
            results.append(
                CheckResult(
                    check_id=f"criterion:{criterion_id}",
                    status="inconclusive",
                    evidence=[],
                    details=f"Criterion {index} awaiting primary bound evidence.",
                )
            )
            continue

        covered_by_transcript = _criterion_covered([criterion], final_response, tool_messages)
        requires_artifact_evidence = _criterion_requires_artifact_evidence(criterion)
        requires_behavioral_evidence = _criterion_requires_behavioral_evidence(criterion)
        if covered_by_transcript and not requires_artifact_evidence:
            if requires_behavioral_evidence and not tool_messages:
                pass
            else:
                evidence = [f"criterion covered: {criterion}"]
                evidence.extend(f"supporting bound check passed: {result.check_id}" for result in passed_supporting_results)
                results.append(
                    CheckResult(
                        check_id=f"criterion:{criterion_id}",
                        status="pass",
                        evidence=evidence,
                        details=f"Criterion {index} covered.",
                    )
                )
                continue

        if passed_supporting_results and not primary_bound_check_ids:
            evidence = [f"criterion covered: {criterion}"]
            evidence.extend(f"supporting bound check passed: {result.check_id}" for result in passed_supporting_results)
            results.append(
                CheckResult(
                    check_id=f"criterion:{criterion_id}",
                    status="pass",
                    evidence=evidence,
                    details=f"Criterion {index} covered by supporting bound evidence.",
                )
            )
            continue

        results.append(
            CheckResult(
                check_id=f"criterion:{criterion_id}",
                status="inconclusive",
                evidence=[],
                details=f"Criterion {index} not clearly covered.",
            )
        )
    return results



def _criterion_label(check_id: str, success_criteria: list[str] | None = None) -> str:
    prefix = "criterion:"
    if not check_id.startswith(prefix):
        return check_id
    label = check_id[len(prefix):]
    if success_criteria and re.fullmatch(r"criterion_(\d+)", label):
        index = int(label.split("_", 1)[1]) - 1
        if 0 <= index < len(success_criteria):
            return success_criteria[index]
    return label



def _criterion_evaluations_from_results(
    criteria: list[CriterionSpec],
    criterion_results: list[CheckResult],
) -> list[CriterionEvaluation]:
    result_by_id = {result.check_id: result for result in criterion_results}
    evaluations: list[CriterionEvaluation] = []
    for spec in criteria:
        check_id = f"criterion:{spec.criterion_id}"
        result = result_by_id.get(check_id)
        if result is None:
            continue
        satisfied_by = ""
        lowered_evidence = "\n".join(result.evidence).lower()
        if "primary bound check failed" in lowered_evidence:
            satisfied_by = "blocked_by_primary_failure"
        elif "bound check passed" in lowered_evidence:
            satisfied_by = "executable" if spec.evidence_mode == "executable_required" else "tool"
        elif result.status == "pass":
            satisfied_by = "tool" if spec.evidence_mode == "tool_backed" else "transcript"
        evaluations.append(
            CriterionEvaluation(
                criterion_id=spec.criterion_id,
                label=spec.label,
                evidence_mode=spec.evidence_mode,
                status=result.status,
                satisfied_by=satisfied_by,
                evidence=list(result.evidence),
                details=result.details,
                check_id=result.check_id,
            )
        )
    return evaluations



def _build_acceptance_summary(result_json: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result_json, dict):
        result_json = {}

    final_assessment = str(result_json.get("final_assessment") or "").strip()
    gate_reason = str(result_json.get("gate_reason") or "").strip()
    confidence = str(result_json.get("confidence") or "").strip()

    missing_evidence = result_json.get("missing_evidence") or []
    if not isinstance(missing_evidence, list):
        missing_evidence = []

    criterion_evaluations = result_json.get("criterion_evaluations") or []
    if not isinstance(criterion_evaluations, list):
        criterion_evaluations = []

    passed_criteria = result_json.get("passed_criteria") or []
    if not isinstance(passed_criteria, list):
        passed_criteria = []

    unmet_criteria = result_json.get("unmet_criteria") or []
    if not isinstance(unmet_criteria, list):
        unmet_criteria = []

    violated_prohibited_actions = result_json.get("violated_prohibited_actions") or []
    if not isinstance(violated_prohibited_actions, list):
        violated_prohibited_actions = []

    if criterion_evaluations:
        if not passed_criteria:
            passed_criteria = [
                str(item.get("label") or "").strip()
                for item in criterion_evaluations
                if isinstance(item, dict) and str(item.get("status") or "").strip() == "pass" and str(item.get("label") or "").strip()
            ]
        if not unmet_criteria:
            unmet_criteria = [
                str(item.get("label") or "").strip()
                for item in criterion_evaluations
                if isinstance(item, dict)
                and str(item.get("status") or "").strip() in {"fail", "inconclusive"}
                and str(item.get("label") or "").strip()
            ]

    if not unmet_criteria:
        unmet_criteria = [str(item).strip() for item in violated_prohibited_actions if str(item).strip()]

    executable_evidence_checks = result_json.get("executable_evidence_checks") or []
    if not isinstance(executable_evidence_checks, list):
        executable_evidence_checks = []

    check_results = result_json.get("check_results") or []
    if not isinstance(check_results, list):
        check_results = []

    check_summary: list[str] = []
    for check in check_results[:4]:
        if not isinstance(check, dict):
            continue
        check_id = str(check.get("check_id") or "").strip()
        check_status = str(check.get("status") or "").strip()
        if check_id and check_status:
            check_summary.append(f"{check_id}={check_status}")

    return {
        "final_assessment": final_assessment,
        "gate_reason": gate_reason,
        "confidence": confidence,
        "missing_evidence": missing_evidence,
        "passed_criteria": passed_criteria,
        "unmet_criteria": unmet_criteria,
        "executable_evidence_checks": executable_evidence_checks,
        "check_summary": check_summary,
    }



def format_acceptance_status_lines(
    latest_acceptance: dict[str, Any] | None,
    *,
    markdown: bool = False,
) -> list[str]:
    if not isinstance(latest_acceptance, dict) or not latest_acceptance:
        return []

    result_status = (str(latest_acceptance.get("result_status") or "").strip() or "unknown")
    task_summary = str(latest_acceptance.get("task_summary") or "").strip()
    completion_mode = str(latest_acceptance.get("completion_mode") or "").strip()
    summary = _build_acceptance_summary(latest_acceptance.get("result_json") or {})

    def line(label: str, value: str) -> str:
        return f"**{label}:** {value}" if markdown else f"{label}: {value}"

    lines = [line("Latest Acceptance", result_status)]
    if task_summary:
        lines.append(line("Acceptance Task", task_summary))
    if completion_mode:
        lines.append(line("Acceptance Mode", completion_mode))
    if summary["final_assessment"]:
        lines.append(line("Acceptance Assessment", summary["final_assessment"]))
    if summary["gate_reason"]:
        lines.append(line("Acceptance Gate Reason", summary["gate_reason"]))
    if summary["confidence"]:
        lines.append(line("Acceptance Confidence", summary["confidence"]))
    if summary["passed_criteria"]:
        lines.append(line("Acceptance Passed Criteria", "; ".join(str(item) for item in summary["passed_criteria"][:3])))
    if summary["unmet_criteria"]:
        lines.append(line("Acceptance Unmet Criteria", "; ".join(str(item) for item in summary["unmet_criteria"][:3])))
    if summary["executable_evidence_checks"]:
        lines.append(line("Acceptance Evidence Checks", "; ".join(str(item) for item in summary["executable_evidence_checks"][:3])))
    if summary["missing_evidence"]:
        lines.append(line("Acceptance Missing Evidence", "; ".join(str(item) for item in summary["missing_evidence"][:3])))
    if summary["check_summary"]:
        lines.append(line("Acceptance Checks", "; ".join(summary["check_summary"])))
    return lines


def _evaluate_prohibited_actions(
    prohibited_actions: list[str],
    verification_checks: list[VerificationCheck],
    check_results: list[CheckResult],
) -> list[CheckResult]:
    results: list[CheckResult] = []
    for index, action in enumerate(prohibited_actions, start=1):
        bound_checks = [
            check
            for check in verification_checks
            if str(check.payload.get("prohibited_action") or "").strip() == action
        ]
        primary_bound_check_ids = {
            check.id
            for check in bound_checks
            if str(check.payload.get("evidence_role") or "").strip().lower() == "primary"
        }
        supporting_bound_check_ids = {
            check.id
            for check in bound_checks
            if check.id not in primary_bound_check_ids
        }
        failed_primary_results = [
            result
            for result in check_results
            if result.check_id in primary_bound_check_ids and result.status == "fail"
        ]
        passed_primary_results = [
            result
            for result in check_results
            if result.check_id in primary_bound_check_ids and result.status == "pass"
        ]
        inconclusive_primary_results = [
            result
            for result in check_results
            if result.check_id in primary_bound_check_ids and result.status == "inconclusive"
        ]
        failed_supporting_results = [
            result
            for result in check_results
            if result.check_id in supporting_bound_check_ids and result.status == "fail"
        ]
        passed_supporting_results = [
            result
            for result in check_results
            if result.check_id in supporting_bound_check_ids and result.status == "pass"
        ]

        if failed_primary_results:
            evidence = [f"prohibited action violated: {action}"]
            evidence.extend(f"primary bound check failed: {result.check_id}" for result in failed_primary_results)
            results.append(
                CheckResult(
                    check_id=f"prohibited:{action}",
                    status="fail",
                    evidence=evidence,
                    details=f"Prohibited action {index} violated by primary evidence.",
                )
            )
            continue

        if passed_primary_results:
            evidence = [f"prohibited action respected: {action}"]
            evidence.extend(f"primary bound check passed: {result.check_id}" for result in passed_primary_results)
            results.append(
                CheckResult(
                    check_id=f"prohibited:{action}",
                    status="pass",
                    evidence=evidence,
                    details=f"Prohibited action {index} respected by primary evidence.",
                )
            )
            continue

        if inconclusive_primary_results:
            results.append(
                CheckResult(
                    check_id=f"prohibited:{action}",
                    status="inconclusive",
                    evidence=[],
                    details=f"Prohibited action {index} awaiting primary evidence.",
                )
            )
            continue

        if failed_supporting_results:
            evidence = [f"prohibited action violated: {action}"]
            evidence.extend(f"supporting bound check failed: {result.check_id}" for result in failed_supporting_results)
            results.append(
                CheckResult(
                    check_id=f"prohibited:{action}",
                    status="fail",
                    evidence=evidence,
                    details=f"Prohibited action {index} violated by supporting evidence.",
                )
            )
            continue

        if passed_supporting_results:
            evidence = [f"prohibited action respected: {action}"]
            evidence.extend(f"supporting bound check passed: {result.check_id}" for result in passed_supporting_results)
            results.append(
                CheckResult(
                    check_id=f"prohibited:{action}",
                    status="pass",
                    evidence=evidence,
                    details=f"Prohibited action {index} respected by supporting evidence.",
                )
            )
            continue

        results.append(
            CheckResult(
                check_id=f"prohibited:{action}",
                status="inconclusive",
                evidence=[],
                details=f"Prohibited action {index} not directly verified.",
            )
        )
    return results



def _prohibited_label(check_id: str) -> str:
    prefix = "prohibited:"
    if check_id.startswith(prefix):
        return check_id[len(prefix):]
    return check_id



def _classify_acceptance_risk(
    *,
    contract: TaskContract,
    status: str,
    gate_reason: str,
    executable_evidence_checks: list[str],
    confidence: str,
) -> tuple[str, list[str], str]:
    normalized_status = str(status or "").strip().lower()
    normalized_gate_reason = str(gate_reason or "").strip().lower()
    normalized_confidence = str(confidence or "").strip().lower()
    inference_quality = str(getattr(contract, "inference_quality", "") or "").strip().lower()
    inference_warnings = [
        str(item).strip().lower()
        for item in getattr(contract, "inference_warnings", [])
        if str(item).strip()
    ]
    executable_backed = bool(executable_evidence_checks)
    signals: list[str] = []
    final_response = str(getattr(contract, "_acceptance_final_response", "") or "")
    covered_corpus = _normalize_text_for_criterion_matching(final_response)

    uncovered_must_requirements = [
        item for item in getattr(contract, "must_requirements", [])
        if _normalize_text_for_criterion_matching(item) not in covered_corpus
    ]
    changed_tokens_present = bool(re.search(r"\bchanged\b", final_response.lower())) or any(
        token in final_response.lower() for token in ("修改", "改了", "layout")
    )
    violated_invariants = [
        item for item in getattr(contract, "invariants", [])
        if (
            changed_tokens_present
            and any(token in str(item).lower() for token in ("保持", "不变", "unchanged", "preserve", "keep"))
        )
    ]
    completed_non_goals = [
        item for item in getattr(contract, "non_goals", [])
        if any(token in final_response.lower() for token in ("refactor", "重构", "entire status bar", "整个状态栏"))
    ]
    uncovered_should_requirements = [
        item for item in getattr(contract, "should_requirements", [])
        if _normalize_text_for_criterion_matching(item) not in covered_corpus
    ]

    if normalized_status in {"fail", "inconclusive"}:
        signals.append(f"acceptance status is {normalized_status}")
        return "blocked", signals, gate_reason

    if uncovered_must_requirements:
        signals.append("uncovered must requirements")
        if normalized_gate_reason == "all acceptance checks passed":
            gate_reason = "must requirements not clearly covered"
        return "risky", signals, gate_reason

    if violated_invariants:
        signals.append("invariants may be violated")
        if normalized_gate_reason == "all acceptance checks passed":
            gate_reason = "invariants may be violated"
        return "risky", signals, gate_reason

    if completed_non_goals:
        signals.append("non-goals may have been completed")
        if normalized_gate_reason == "all acceptance checks passed":
            gate_reason = "non-goals may have been completed"
        return "risky", signals, gate_reason

    if uncovered_should_requirements and normalized_confidence == "high":
        normalized_confidence = "medium"

    if normalized_gate_reason == "transcript-only acceptance":
        signals.append("transcript-only acceptance")

    if inference_quality == "low" and not executable_backed:
        if normalized_gate_reason != "transcript-only acceptance":
            gate_reason = "low-quality inferred contract"
        signals.append("low-quality inferred contract")
        return "risky", signals, gate_reason

    if (
        "no explicit executable verification inferred" in inference_warnings
        and not executable_backed
        and normalized_gate_reason != "transcript-only acceptance"
    ):
        if normalized_gate_reason != "inference warning: no explicit executable verification inferred":
            gate_reason = "inference warning: no explicit executable verification inferred"
        signals.append("no explicit executable verification inferred")
        return "risky", signals, gate_reason

    if executable_backed:
        signals.append("executable evidence present")
        if inference_quality == "high" or normalized_confidence == "high":
            return "verified", signals, gate_reason
        return "supported", signals, gate_reason

    if normalized_gate_reason == "tool-backed acceptance":
        signals.append("tool-backed acceptance")
        return "supported", signals, gate_reason

    if not signals:
        signals.append("transcript-only acceptance")
    return "weak", signals, gate_reason


def _compute_acceptance_confidence(
    contract: TaskContract,
    criterion_results: list[CheckResult],
    check_results: list[CheckResult],
    executable_evidence_checks: list[str],
) -> str:
    criterion_metadata = _derive_criterion_metadata(contract.success_criteria)
    criterion_result_by_id = {
        result.check_id[len("criterion:") :]: result
        for result in criterion_results
        if result.check_id.startswith("criterion:")
    }
    tool_evidence_present = any(result.check_id == "tool_evidence" and result.status == "pass" for result in check_results)
    has_executable_contract_checks = any(
        check.kind not in {"transcript_coverage", "tool_evidence"}
        for check in contract.verification_checks
    )

    if any(result.status == "fail" for result in criterion_results):
        return "medium" if executable_evidence_checks or has_executable_contract_checks or tool_evidence_present else "low"

    if any(result.status != "pass" for result in criterion_results):
        if executable_evidence_checks:
            return "medium"
        return "low"

    if not criterion_metadata:
        return "medium" if executable_evidence_checks or tool_evidence_present else "low"

    satisfied_modes: set[str] = set()
    for item in criterion_metadata:
        criterion_id = item["criterion_id"]
        result = criterion_result_by_id.get(criterion_id)
        if result and result.status == "pass":
            satisfied_modes.add(item["evidence_mode"])

    if "executable_required" in satisfied_modes:
        if "tool_backed" in satisfied_modes and tool_evidence_present:
            return "high"
        return "high" if len(satisfied_modes) == 1 else "medium"

    if "tool_backed" in satisfied_modes:
        return "medium" if tool_evidence_present else "low"

    if "transcript_ok" in satisfied_modes:
        return "low"

    if executable_evidence_checks:
        return "medium"
    return "low"



def _build_final_assessment(
    status: str,
    check_results: list[CheckResult],
    unmet_criteria: list[str],
    missing_evidence: list[str],
    success_criteria: list[str] | None = None,
) -> str:
    lines = [
        "Acceptance checks passed."
        if status == "pass"
        else "Acceptance is inconclusive because completion claims are not fully backed by evidence."
        if status == "inconclusive"
        else "Acceptance failed."
    ]

    passed_criteria = [
        _criterion_label(check.check_id, success_criteria)
        for check in check_results
        if check.check_id.startswith("criterion:") and check.status == "pass"
    ]
    violated_prohibited_actions = [
        _prohibited_label(check.check_id)
        for check in check_results
        if check.check_id.startswith("prohibited:") and check.status == "fail"
    ]
    executable_evidence = [
        check.check_id
        for check in check_results
        if check.status == "pass"
        and not check.check_id.startswith(("criterion:", "prohibited:"))
        and check.check_id not in {"transcript_coverage", "tool_evidence"}
    ]
    failed_checks = [
        check.check_id
        for check in check_results
        if check.status == "fail"
        and not check.check_id.startswith(("criterion:", "prohibited:"))
    ]

    if passed_criteria:
        lines.append(f"passed criteria: {', '.join(passed_criteria)}")
    unmet_items = list(dict.fromkeys(item for item in unmet_criteria if item))
    if unmet_items:
        lines.append(f"unmet criteria: {', '.join(unmet_items)}")
    if violated_prohibited_actions:
        lines.append(f"violated prohibited actions: {', '.join(violated_prohibited_actions)}")
    if executable_evidence:
        lines.append(f"executable evidence: {', '.join(executable_evidence)}")
    if failed_checks:
        lines.append(f"failed checks: {', '.join(failed_checks)}")
    if missing_evidence:
        lines.append(f"missing evidence: {', '.join(missing_evidence)}")

    return "\n".join(lines)



def _extract_final_response(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""



def _contains_completion_claim(text: str) -> bool:
    lowered = (text or "").lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _COMPLETION_CLAIM_PATTERNS)



def _normalize_text_for_criterion_matching(text: str) -> str:
    normalized = (text or "").lower()
    replacements = {
        "status bar": "status",
        "provider label": "provider name",
        "provider display": "provider name",
        "actual provider": "real provider",
        "actual provider name": "real provider",
        "named custom provider": "custom provider name",
        "selected custom provider": "custom provider name",
        "render": "show",
        "renders": "show",
        "display": "show",
        "displays": "show",
        "launch": "startup",
        "on launch": "startup",
        "startup": "startup",
        "starts": "startup",
        "it starts": "startup",
        "as soon as it starts": "startup",
        "init": "startup",
        "initial": "startup",
        "ui": "status",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"\bshows\b", "show", normalized)
    normalized = re.sub(r"\bselected\b", "", normalized)
    normalized = re.sub(r"\bthe\b|\ba\b|\ban\b|\bin\b|\bnow\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized



def _extract_criterion_tokens(text: str) -> list[str]:
    normalized = _normalize_text_for_criterion_matching(text)
    return [token for token in re.split(r"\W+", normalized) if len(token) >= 4]



def _criterion_covered(
    success_criteria: list[str],
    final_response: str,
    tool_messages: list[dict[str, Any]],
) -> bool:
    corpus_parts = [final_response or ""]
    corpus_parts.extend(str(m.get("content") or "") for m in tool_messages)
    corpus = _normalize_text_for_criterion_matching(" ".join(corpus_parts))
    for criterion in success_criteria:
        tokens = _extract_criterion_tokens(criterion)
        if not tokens:
            continue
        matched = [token for token in tokens if token in corpus]
        if matched and len(matched) == len(tokens):
            return True
    return False



def _criterion_requires_artifact_evidence(criterion: str) -> bool:
    lowered = (criterion or "").lower()
    return any(
        token in lowered
        for token in (
            "stdout",
            "output",
            "print",
            "file",
            "path",
            "directory",
            "json",
            "输出",
            "打印",
            "文件",
            "路径",
            "目录",
            "json",
        )
    )



def _criterion_requires_behavioral_evidence(criterion: str) -> bool:
    lowered = (criterion or "").lower()
    has_change_verb = any(
        token in lowered
        for token in (
            "fix",
            "update",
            "modify",
            "create",
            "generate",
            "修复",
            "修好",
            "修改",
            "更新",
            "创建",
            "生成",
            "新增",
        )
    )
    has_concrete_target = any(
        token in lowered
        for token in (
            "provider",
            "label",
            "status",
            "bar",
            "startup",
            "config",
            "command",
            "provider label",
            "status bar",
            "状态栏",
            "配置",
            "命令",
            "显示",
            "provider 显示",
        )
    )
    return has_change_verb and has_concrete_target



def _extract_explicit_path(text: str) -> str | None:
    text = (text or "").strip()
    path_match = re.search(r"((?:/|\./|\.\./|~/)[^\s，。,；;：:！!？?]+)", text)
    if path_match:
        raw = path_match.group(1).rstrip(".,;:!?)】」』’”")
        expanded = str(Path(raw).expanduser())
        return expanded
    return None



def _run_command_check(check: VerificationCheck) -> CheckResult:
    command = str(check.payload.get("command") or "").strip()
    if not command:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="Command check missing payload.command.",
        )

    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=[],
            details=f"Command check execution failed: {exc}",
        )

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    evidence = [f"command exited with code {completed.returncode}"]
    if stdout:
        evidence.append(f"stdout: {stdout[:160]}")
    if stderr:
        evidence.append(f"stderr: {stderr[:160]}")

    if completed.returncode == 0:
        return CheckResult(
            check_id=check.id,
            status="pass",
            evidence=evidence,
            details="Executable command check passed.",
        )
    return CheckResult(
        check_id=check.id,
        status="fail",
        evidence=evidence,
        details=f"Command check failed with exit code {completed.returncode}.",
    )



def _run_command_exit_zero_check(check: VerificationCheck) -> CheckResult:
    command = str(check.payload.get("command") or "").strip()
    if not command:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="Command-exit-zero check missing payload.command.",
        )

    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=[],
            details=f"Command-exit-zero execution failed: {exc}",
        )

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    evidence = [f"command exited with code {completed.returncode}"]
    if stdout:
        evidence.append(f"stdout: {stdout[:160]}")
    if stderr:
        evidence.append(f"stderr: {stderr[:160]}")

    if completed.returncode == 0:
        return CheckResult(
            check_id=check.id,
            status="pass",
            evidence=evidence,
            details="Executable command exit-zero check passed.",
        )
    return CheckResult(
        check_id=check.id,
        status="fail",
        evidence=evidence,
        details=f"Command exit-zero check failed with exit code {completed.returncode}.",
    )



def _run_file_exists_check(check: VerificationCheck) -> CheckResult:
    raw_path = str(check.payload.get("path") or "").strip()
    if not raw_path:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="File-exists check missing payload.path.",
        )

    path = Path(raw_path).expanduser()
    if path.exists():
        return CheckResult(
            check_id=check.id,
            status="pass",
            evidence=[f"path exists: {path}"],
            details="Executable file existence check passed.",
        )
    return CheckResult(
        check_id=check.id,
        status="fail",
        evidence=[],
        details=f"Required path does not exist: {path}",
    )



def _run_directory_exists_check(check: VerificationCheck) -> CheckResult:
    raw_path = str(check.payload.get("path") or "").strip()
    if not raw_path:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="Directory-exists check missing payload.path.",
        )

    path = Path(raw_path).expanduser()
    if path.is_dir():
        return CheckResult(
            check_id=check.id,
            status="pass",
            evidence=[f"directory exists: {path}"],
            details="Executable directory existence check passed.",
        )
    return CheckResult(
        check_id=check.id,
        status="fail",
        evidence=[],
        details=f"Required directory does not exist: {path}",
    )



def _run_file_glob_count_check(check: VerificationCheck) -> CheckResult:
    raw_path = str(check.payload.get("path") or "").strip()
    pattern = str(check.payload.get("glob") or "").strip()
    expected_count = check.payload.get("count")
    if not raw_path:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="File-glob-count check missing payload.path.",
        )
    if not pattern:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="File-glob-count check missing payload.glob.",
        )
    if not isinstance(expected_count, int):
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="File-glob-count check missing integer payload.count.",
        )

    path = Path(raw_path).expanduser()
    if not path.is_dir():
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=[],
            details=f"Required directory does not exist: {path}",
        )

    matches = sorted(path.glob(pattern))
    actual_count = len(matches)
    if actual_count == expected_count:
        return CheckResult(
            check_id=check.id,
            status="pass",
            evidence=[f"matched {actual_count} path(s): {', '.join(str(match) for match in matches[:5])}"],
            details="Executable file glob count check passed.",
        )
    return CheckResult(
        check_id=check.id,
        status="fail",
        evidence=[f"matched {actual_count} path(s) in {path} for glob {pattern}"] if matches else [],
        details=f"File glob count check expected {expected_count} but found {actual_count} for {pattern} in {path}.",
    )



def _run_file_unchanged_check(check: VerificationCheck) -> CheckResult:
    raw_path = str(check.payload.get("path") or "").strip()
    expected_content = check.payload.get("content")
    if not raw_path:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="File-unchanged check missing payload.path.",
        )
    if expected_content is None:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="File-unchanged check missing payload.content.",
        )

    path = Path(raw_path).expanduser()
    if not path.exists():
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=[],
            details=f"Required path does not exist: {path}",
        )

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=[],
            details=f"Failed to read file for unchanged check: {exc}",
        )

    expected_text = str(expected_content)
    if content == expected_text:
        return CheckResult(
            check_id=check.id,
            status="pass",
            evidence=[f"file content unchanged: {path}"],
            details="Executable file unchanged check passed.",
        )
    return CheckResult(
        check_id=check.id,
        status="fail",
        evidence=[f"path checked: {path}"],
        details="File content changed from expected content.",
    )



def _run_file_contains_check(check: VerificationCheck) -> CheckResult:
    raw_path = str(check.payload.get("path") or "").strip()
    needle = str(check.payload.get("contains") or "")
    if not raw_path:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="File-contains check missing payload.path.",
        )
    if not needle:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="File-contains check missing payload.contains.",
        )

    path = Path(raw_path).expanduser()
    if not path.exists():
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=[],
            details=f"Required path does not exist: {path}",
        )

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=[],
            details=f"Failed to read file for contains check: {exc}",
        )

    if needle in content:
        return CheckResult(
            check_id=check.id,
            status="pass",
            evidence=[f"path contains expected text: {needle}", f"path checked: {path}"],
            details="Executable file content check passed.",
        )
    return CheckResult(
        check_id=check.id,
        status="fail",
        evidence=[f"path checked: {path}"],
        details=f"Required text not found in file: {needle}",
    )



def _run_stdout_regex_check(check: VerificationCheck) -> CheckResult:
    command = str(check.payload.get("command") or "").strip()
    pattern = str(check.payload.get("pattern") or "")
    if not command:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="Stdout-regex check missing payload.command.",
        )
    if not pattern:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="Stdout-regex check missing payload.pattern.",
        )

    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=[],
            details=f"Stdout-regex command execution failed: {exc}",
        )

    stdout = completed.stdout or ""
    stderr = (completed.stderr or "").strip()
    evidence = [f"command exited with code {completed.returncode}"]
    if stderr:
        evidence.append(f"stderr: {stderr[:160]}")

    if completed.returncode != 0:
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=evidence,
            details=f"Stdout-regex command failed with exit code {completed.returncode}.",
        )

    if re.search(pattern, stdout, flags=re.MULTILINE):
        evidence.append(f"stdout matched regex: {pattern}")
        return CheckResult(
            check_id=check.id,
            status="pass",
            evidence=evidence,
            details="Executable stdout regex check passed.",
        )
    evidence.append(f"stdout sample: {stdout[:160].strip()}")
    return CheckResult(
        check_id=check.id,
        status="fail",
        evidence=evidence,
        details=f"Required stdout regex did not match: {pattern}",
    )



def _run_command_stdout_regex_check(check: VerificationCheck) -> CheckResult:
    command = str(check.payload.get("command") or "").strip()
    pattern = str(check.payload.get("pattern") or "")
    if not command:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="Command-stdout-regex check missing payload.command.",
        )
    if not pattern:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="Command-stdout-regex check missing payload.pattern.",
        )

    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=[],
            details=f"Command-stdout-regex execution failed: {exc}",
        )

    stdout = completed.stdout or ""
    stderr = (completed.stderr or "").strip()
    evidence = [f"command exited with code {completed.returncode}"]
    if stderr:
        evidence.append(f"stderr: {stderr[:160]}")

    if completed.returncode != 0:
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=evidence,
            details=f"Command-stdout-regex failed with exit code {completed.returncode}.",
        )

    if re.search(pattern, stdout, flags=re.MULTILINE):
        evidence.append(f"stdout matched regex: {pattern}")
        return CheckResult(
            check_id=check.id,
            status="pass",
            evidence=evidence,
            details="Executable command stdout regex check passed.",
        )

    evidence.append(f"stdout sample: {stdout[:160].strip()}")
    return CheckResult(
        check_id=check.id,
        status="fail",
        evidence=evidence,
        details=f"Command stdout regex did not match: {pattern}",
    )



def _run_json_command_payload(check: VerificationCheck, kind_label: str) -> tuple[dict[str, Any] | None, str, list[str], CheckResult | None]:
    command = str(check.payload.get("command") or "").strip()
    field_path = str(check.payload.get("field_path") or "").strip()
    if not command:
        return None, field_path, [], CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details=f"{kind_label} check missing payload.command.",
        )
    if not field_path:
        return None, field_path, [], CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details=f"{kind_label} check missing payload.field_path.",
        )

    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return None, field_path, [], CheckResult(
            check_id=check.id,
            status="fail",
            evidence=[],
            details=f"{kind_label} command execution failed: {exc}",
        )

    stdout = completed.stdout or ""
    stderr = (completed.stderr or "").strip()
    evidence = [f"command exited with code {completed.returncode}"]
    if stderr:
        evidence.append(f"stderr: {stderr[:160]}")
    if completed.returncode != 0:
        return None, field_path, evidence, CheckResult(
            check_id=check.id,
            status="fail",
            evidence=evidence,
            details=f"{kind_label} command failed with exit code {completed.returncode}.",
        )

    try:
        payload = json.loads(stdout)
    except Exception as exc:
        return None, field_path, evidence, CheckResult(
            check_id=check.id,
            status="fail",
            evidence=evidence,
            details=f"Failed to parse command stdout as JSON: {exc}",
        )

    return payload, field_path, evidence, None



def _lookup_json_field(payload: Any, field_path: str) -> tuple[bool, Any]:
    current = payload
    for part in field_path.split('.'):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False, None
    return True, current



def _run_json_field_match_check(check: VerificationCheck) -> CheckResult:
    payload, field_path, evidence, error_result = _run_json_command_payload(check, "Json-field-match")
    if error_result is not None:
        return error_result

    exists, current = _lookup_json_field(payload, field_path)
    if not exists:
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=evidence,
            details=f"JSON field path not found: {field_path}",
        )

    expected = check.payload.get("equals")
    if current == expected:
        evidence.append(f"json field matched expected value: {field_path}={expected!r}")
        return CheckResult(
            check_id=check.id,
            status="pass",
            evidence=evidence,
            details="Executable JSON field match check passed.",
        )
    evidence.append(f"json field actual value: {field_path}={current!r}")
    return CheckResult(
        check_id=check.id,
        status="fail",
        evidence=evidence,
        details=f"JSON field {field_path} expected {field_path}={expected!r} but got {current!r}.",
    )



def _run_json_field_exists_check(check: VerificationCheck) -> CheckResult:
    payload, field_path, evidence, error_result = _run_json_command_payload(check, "Json-field-exists")
    if error_result is not None:
        return error_result

    exists, current = _lookup_json_field(payload, field_path)
    if not exists:
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=evidence,
            details=f"JSON field path not found: {field_path}",
        )

    evidence.append(f"json field exists: {field_path}={current!r}")
    return CheckResult(
        check_id=check.id,
        status="pass",
        evidence=evidence,
        details="Executable JSON field exists check passed.",
    )



def _run_json_field_contains_check(check: VerificationCheck) -> CheckResult:
    payload, field_path, evidence, error_result = _run_json_command_payload(check, "Json-field-contains")
    if error_result is not None:
        return error_result

    exists, current = _lookup_json_field(payload, field_path)
    if not exists:
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=evidence,
            details=f"JSON field path not found: {field_path}",
        )

    expected_substring = str(check.payload.get("contains") or "")
    if not expected_substring:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=evidence,
            details="Json-field-contains check missing payload.contains.",
        )
    if not isinstance(current, str):
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=evidence,
            details=f"JSON field {field_path} must be a string for contains checks.",
        )
    if expected_substring in current:
        evidence.append(f"json field contained expected text: {field_path} contains {expected_substring!r}")
        return CheckResult(
            check_id=check.id,
            status="pass",
            evidence=evidence,
            details="Executable JSON field contains check passed.",
        )
    evidence.append(f"json field actual value: {field_path}={current!r}")
    return CheckResult(
        check_id=check.id,
        status="fail",
        evidence=evidence,
        details=f"JSON field {field_path} did not contain {expected_substring!r}.",
    )



def _run_process_running_check(check: VerificationCheck) -> CheckResult:
    pattern = str(check.payload.get("pattern") or "").strip()
    if not pattern:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="Process-running check missing payload.pattern.",
        )

    command = f"ps -eo pid=,comm=,args= | grep -i -- {subprocess.list2cmdline([pattern])} | grep -v grep"
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=[],
            details=f"Process-running check execution failed: {exc}",
        )

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    evidence = []
    if stderr:
        evidence.append(f"stderr: {stderr[:160]}")

    if completed.returncode == 0 and stdout:
        first_match = stdout.splitlines()[0][:160]
        evidence.append(f"running process matched pattern {pattern!r}: {first_match}")
        return CheckResult(
            check_id=check.id,
            status="pass",
            evidence=evidence,
            details="Executable process running check passed.",
        )

    return CheckResult(
        check_id=check.id,
        status="fail",
        evidence=evidence,
        details=f"No running process matched pattern {pattern!r}.",
    )



def _run_stdout_not_regex_check(
    check: VerificationCheck,
    final_response: str,
    tool_messages: list[dict[str, Any]],
) -> CheckResult:
    pattern = str(check.payload.get("pattern") or "")
    prohibited_action = str(check.payload.get("prohibited_action") or "").strip()
    if not pattern:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="Stdout-not-regex check missing payload.pattern.",
        )

    corpus_parts = [final_response or ""]
    corpus_parts.extend(str(m.get("content") or "") for m in tool_messages)
    corpus = "\n".join(corpus_parts)
    evidence = []
    if prohibited_action:
        evidence.append(f"prohibited action checked: {prohibited_action}")
    if re.search(pattern, corpus, flags=re.MULTILINE | re.IGNORECASE):
        evidence.append(f"prohibited pattern matched: {pattern}")
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=evidence,
            details=f"Prohibited output matched pattern: {pattern}",
        )
    evidence.append(f"prohibited pattern absent: {pattern}")
    return CheckResult(
        check_id=check.id,
        status="pass",
        evidence=evidence,
        details="Negative stdout regex check passed.",
    )



def _run_file_not_contains_check(check: VerificationCheck) -> CheckResult:
    raw_path = str(check.payload.get("path") or "").strip()
    needle = str(check.payload.get("contains") or "")
    prohibited_action = str(check.payload.get("prohibited_action") or "").strip()
    if not raw_path:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="File-not-contains check missing payload.path.",
        )
    if not needle:
        return CheckResult(
            check_id=check.id,
            status="inconclusive",
            evidence=[],
            details="File-not-contains check missing payload.contains.",
        )

    path = Path(raw_path).expanduser()
    if not path.exists():
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=[],
            details=f"Required path does not exist: {path}",
        )

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=[],
            details=f"Failed to read file for not-contains check: {exc}",
        )

    evidence = [f"path checked: {path}"]
    if prohibited_action:
        evidence.append(f"prohibited action checked: {prohibited_action}")
    if needle in content:
        evidence.append(f"prohibited text found: {needle}")
        return CheckResult(
            check_id=check.id,
            status="fail",
            evidence=evidence,
            details=f"Prohibited text found in file: {path}",
        )
    evidence.append(f"prohibited text absent: {needle}")
    return CheckResult(
        check_id=check.id,
        status="pass",
        evidence=evidence,
        details="Negative file content check passed.",
    )
