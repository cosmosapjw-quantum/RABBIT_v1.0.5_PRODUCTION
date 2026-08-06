from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ASSIGNMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
ADMISSION_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SHA256_FINGERPRINT_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
VALID_INDEPENDENCE_MODES = {"shared-core", "blind-results", "adjudication"}
VALID_DISCOVERY_MODES = {"targeted", "independent"}
VALID_RESULT_STATUS = {"pass", "fail", "inconclusive", "error"}
VALID_FINDING_VERDICTS = {"pass", "fail", "inconclusive", "error"}
RESULT_LIST_FIELDS = {
    "commands",
    "artifacts",
    "findings",
    "errors",
    "files_written",
}
RESULT_TEMPLATE_PATH = ".agent-harness/templates/RESULT_ENVELOPE.json"


def root() -> Path:
    try:
        text = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if text:
            return Path(text).resolve()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    here = Path(__file__).resolve()
    return here.parents[2]


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    for key, _value in pairs:
        if key in seen:
            raise json.JSONDecodeError(f"duplicate object key {key!r}", "", 0)
        seen.add(key)
    return dict(pairs)


def loads_strict(text: str) -> Any:
    """``json.loads`` that REFUSES an object with a repeated key.

    The stdlib parser keeps the last value silently, so
    ``{"assignment_id": "A-WRONG", ..., "assignment_id": "A-RIGHT"}`` presents
    one identity to this harness and a different one to any reader that keeps
    the first -- a jq filter, a JSON5 parser, a human reading top to bottom.
    Round 11 reproduced an admission envelope carrying two ``assignment_id``
    keys where the incorrect value came first and the registered one second:
    the Stop hook accepted it, because by the time any check ran there was only
    one key left to check.

    Every evidence-bearing document in the admission path is parsed through
    here: envelope, receipt, assignment, result, and each ADMISSIONS.jsonl row.
    Ambiguous identity bytes are refused rather than resolved, because the whole
    mechanism is an argument about which agent wrote which result, and a
    document that answers that differently depending on the parser is not
    evidence.
    """
    return json.loads(text, object_pairs_hook=_no_duplicate_keys)


def load_json(path: Path) -> Any:
    return loads_strict(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text_atomic(path: Path, text: str) -> None:
    """Write text so a reader never observes a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp.{os.getpid()}.{path.name}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def dump_json_atomic(path: Path, value: Any) -> None:
    """Same payload as dump_json, but a reader never observes a partial file."""
    write_text_atomic(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


PACK_PREAMBLE = (
    "This pack contains only the shared Tier-0 context. Assignment-specific "
    "context and sibling results are intentionally excluded."
)


def render_context_pack(
    repo: Path, version: str, built_at: str, entries: list[tuple[str, str]]
) -> str:
    """The canonical pack, as a pure function of the index and the shared files.

    One definition, used by the builder that writes the pack and by both
    verifiers that check it. Before BD623 the pack was rendered in one place and
    checked in two others by looking for the version string in its first six
    lines -- which meant the body was never checked at all. Measured: a pack
    truncated to 283 of 157,567 characters returned clean from both checks, and
    so did one whose body had been replaced with a forged frozen-decision row
    asserting that the harness freeze had been lifted.

    The board took the same argument and settled it the same way: a projection
    is verified by re-rendering it, not by storing a number about it. The pack
    is a verbatim concatenation rather than a projection, so it keeps its stored
    ``context_version`` for the "have the inputs changed" question -- but "is
    this what those inputs render to" is a different question, and only
    re-rendering answers it.
    """
    chunks = [
        "# Canonical Shared Context Pack",
        "",
        f"Context version: `{version}`",
        f"Built at: `{built_at}`",
        "",
        PACK_PREAMBLE,
    ]
    for rel, sha in entries:
        text = (repo / rel).read_text(encoding="utf-8")
        chunks.extend(["", f"---\n\n## Source: `{rel}`\n\nSHA-256: `{sha}`\n", text.rstrip()])
    return "\n".join(chunks).rstrip() + "\n"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def admission_path(harness: Path, run_id: str, assignment_id: str) -> Path | None:
    """Single-use parent-authenticated admission receipt for one assignment.

    SubagentStart never receives the spawn prompt (D-031/D-032), so no hook can
    infer which assignment an agent was launched for. The parent mints this
    receipt before spawning and SubagentStop admits the stopping agent only when
    it echoes a token whose digest matches. Returns None for unsafe path keys.
    """
    if not ADMISSION_KEY_RE.fullmatch(run_id or ""):
        return None
    if not ADMISSION_KEY_RE.fullmatch(assignment_id or ""):
        return None
    return harness / "admissions" / run_id / f"{assignment_id}.json"


def token_digest(token: str) -> str:
    return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()


def hash_files(repo: Path, files: list[str]) -> tuple[str, list[tuple[str, str]]]:
    digest = hashlib.sha256()
    entries: list[tuple[str, str]] = []
    for rel in files:
        path = repo / rel
        data = path.read_bytes()
        file_hash = hashlib.sha256(data).hexdigest()
        entries.append((rel, file_hash))
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest(), entries


def assignment_schema_version(assignment: dict[str, Any]) -> int:
    value = assignment.get("schema_version", 1)
    return value if isinstance(value, int) and not isinstance(value, bool) else -1


def assignment_runtime_agent_type(assignment: dict[str, Any]) -> str:
    return str(assignment.get("runtime_agent_type") or assignment.get("agent_type") or "")


def assignment_review_role(assignment: dict[str, Any]) -> str:
    return str(assignment.get("review_role") or assignment.get("agent_type") or "")


def validate_assignment_resource_hashes(
    repo: Path,
    assignment: dict[str, Any],
    *,
    role_files: dict[str, Any],
) -> list[str]:
    """Validate v2 role/template bytes without changing v1 historical artifacts."""
    if assignment_schema_version(assignment) < 2:
        return []

    errors: list[str] = []
    review_role = assignment_review_role(assignment)
    expected_role_files = role_files.get(review_role)
    if isinstance(expected_role_files, list):
        try:
            role_digest, _ = hash_files(repo, [str(item) for item in expected_role_files])
        except OSError as exc:
            errors.append(f"review role files are unreadable: {exc}")
        else:
            expected = "sha256:" + role_digest
            if assignment.get("review_role_sha256") != expected:
                errors.append(
                    "review_role_sha256 is stale: expected "
                    f"{expected!r}, got {assignment.get('review_role_sha256')!r}"
                )

    template_path = repo / RESULT_TEMPLATE_PATH
    try:
        template_digest = hashlib.sha256(template_path.read_bytes()).hexdigest()
    except OSError as exc:
        errors.append(f"result template is unreadable: {exc}")
    else:
        expected = "sha256:" + template_digest
        if assignment.get("result_template_sha256") != expected:
            errors.append(
                "result_template_sha256 is stale: expected "
                f"{expected!r}, got {assignment.get('result_template_sha256')!r}"
            )
    return errors


def active_run_id(repo: Path) -> str:
    path = repo / ".agent-harness" / "ACTIVE_RUN"
    if not path.exists():
        raise SystemExit("No active run. Use init_run.py first.")
    run_id = path.read_text(encoding="utf-8").strip()
    if not run_id:
        raise SystemExit("ACTIVE_RUN is empty.")
    return run_id


def validate_assignment_contract(
    assignment: Any,
    *,
    expected_run_id: str,
    expected_context_version: str,
    role_files: dict[str, Any],
) -> list[str]:
    """Return deterministic assignment-contract violations."""
    if not isinstance(assignment, dict):
        return ["assignment is not a JSON object"]

    errors: list[str] = []
    assignment_id = str(assignment.get("assignment_id") or "")
    if not ASSIGNMENT_ID_RE.fullmatch(assignment_id):
        errors.append("assignment_id has an invalid form")
    if str(assignment.get("run_id")) != expected_run_id:
        errors.append("run_id does not match the active run")
    if str(assignment.get("context_version")) != expected_context_version:
        errors.append("context_version is stale")

    schema_version = assignment_schema_version(assignment)
    if schema_version not in {1, 2}:
        errors.append(f"schema_version is unsupported: {schema_version!r}")

    agent_type = str(assignment.get("agent_type") or "")
    runtime_agent_type = assignment_runtime_agent_type(assignment)
    review_role = assignment_review_role(assignment)
    if schema_version == 1:
        if agent_type not in role_files:
            errors.append(f"agent_type has no registered role context: {agent_type!r}")
    else:
        if not agent_type:
            errors.append("agent_type must name the live runtime event type")
        if runtime_agent_type != agent_type:
            errors.append("runtime_agent_type must equal the agent_type compatibility alias")
        if review_role not in role_files:
            errors.append(f"review_role has no registered role context: {review_role!r}")
        expected_role_files = role_files.get(review_role)
        if assignment.get("review_role_files") != expected_role_files:
            errors.append("review_role_files does not match the registered review_role")
        review_hash = assignment.get("review_role_sha256")
        if not isinstance(review_hash, str) or not SHA256_FINGERPRINT_RE.fullmatch(
            review_hash
        ):
            errors.append("review_role_sha256 is not sha256:<64 hex>")
        if assignment.get("result_template") != RESULT_TEMPLATE_PATH:
            errors.append("result_template is not the canonical RESULT_ENVELOPE template")
        template_hash = assignment.get("result_template_sha256")
        if not isinstance(template_hash, str) or not SHA256_FINGERPRINT_RE.fullmatch(
            template_hash
        ):
            errors.append("result_template_sha256 is not sha256:<64 hex>")
    if assignment.get("independence_mode") not in VALID_INDEPENDENCE_MODES:
        errors.append("independence_mode is invalid")
    if assignment.get("discovery_mode") not in VALID_DISCOVERY_MODES:
        errors.append("discovery_mode is invalid")

    claim_ids = assignment.get("claim_ids")
    if (
        not isinstance(claim_ids, list)
        or not claim_ids
        or any(not isinstance(item, str) or not item for item in claim_ids)
    ):
        errors.append("claim_ids must be a non-empty list of strings")
    elif len(set(claim_ids)) != len(claim_ids):
        errors.append("claim_ids contains duplicates")

    result_path = str(assignment.get("result_path") or "")
    expected_result = (
        f".agent-harness/runs/{expected_run_id}/results/{assignment_id}.json"
    )
    if result_path != expected_result:
        errors.append("result_path is not the assignment's unique result path")

    required_inputs = assignment.get("required_inputs")
    own_assignment = (
        f".agent-harness/runs/{expected_run_id}/assignments/{assignment_id}.json"
    )
    required_core = {
        ".agent-harness/generated/CONTEXT_PACK.md",
        own_assignment,
    }
    if not isinstance(required_inputs, list):
        errors.append("required_inputs must be a list")
    elif not required_core.issubset(set(required_inputs)):
        errors.append("required_inputs omits the context pack or own assignment")
    elif schema_version >= 2:
        required_v2 = {
            RESULT_TEMPLATE_PATH,
            *[str(item) for item in role_files.get(review_role, [])],
        }
        if not required_v2.issubset(set(required_inputs)):
            errors.append("required_inputs omits the result template or review role files")

    allowed_tools = assignment.get("allowed_tools")
    if not isinstance(allowed_tools, list) or not allowed_tools:
        errors.append("allowed_tools must be a non-empty list")
    required_outputs = assignment.get("required_outputs")
    if required_outputs != [result_path]:
        errors.append("required_outputs must contain only result_path")

    delegable_claim_ids = assignment.get("delegable_claim_ids")
    if not isinstance(delegable_claim_ids, list):
        errors.append("delegable_claim_ids must be a list")
    elif assignment.get("may_spawn") and not delegable_claim_ids:
        errors.append("may_spawn=true requires non-empty delegable_claim_ids")
    elif not assignment.get("may_spawn") and delegable_claim_ids:
        errors.append("delegable_claim_ids requires may_spawn=true")

    sibling_results = assignment.get("allowed_sibling_results")
    if not isinstance(sibling_results, list):
        errors.append("allowed_sibling_results must be a list")
    elif assignment.get("independence_mode") == "blind-results" and sibling_results:
        errors.append("blind-results assignment may not read sibling results")
    return errors


def validate_result_contract(
    result: Any,
    assignment: dict[str, Any],
    *,
    expected_run_id: str,
    expected_context_version: str,
    expected_agent_id: str | None = None,
    expected_assignment_sha256: str | None = None,
) -> list[str]:
    """Return RESULT_ENVELOPE violations shared by live and static gates."""
    if not isinstance(result, dict):
        return ["result is not a JSON object"]

    errors: list[str] = []
    schema_version = assignment_schema_version(assignment)
    expected = {
        "schema_version": schema_version,
        "run_id": expected_run_id,
        "assignment_id": assignment.get("assignment_id"),
        "context_version": expected_context_version,
        "agent_type": assignment.get("agent_type"),
        "result_path": assignment.get("result_path"),
    }
    if schema_version >= 2:
        expected.update(
            {
                "runtime_agent_type": assignment_runtime_agent_type(assignment),
                "review_role": assignment_review_role(assignment),
            }
        )
    for key, expected_value in expected.items():
        if result.get(key) != expected_value:
            errors.append(
                f"{key} mismatch: expected {expected_value!r}, got {result.get(key)!r}"
            )

    agent_id = result.get("agent_id")
    if not isinstance(agent_id, str) or not agent_id:
        errors.append("agent_id must be a non-empty string")
    elif expected_agent_id is not None and agent_id != expected_agent_id:
        errors.append(
            f"agent_id mismatch: expected {expected_agent_id!r}, got {agent_id!r}"
        )

    spawn_contract = result.get("spawn_contract")
    if not isinstance(spawn_contract, dict):
        errors.append("spawn_contract must be an object")
    else:
        expected_spawn = {
            "run_id": expected_run_id,
            "assignment_id": assignment.get("assignment_id"),
            "context_version": expected_context_version,
            "independence_mode": assignment.get("independence_mode"),
            "prompt_header_verified": True,
            "subagent_start_injected": True,
            "subagent_start_preflight": "PASS",
        }
        if schema_version >= 2:
            expected_spawn.update(
                {
                    "runtime_agent_type": assignment_runtime_agent_type(assignment),
                    "review_role": assignment_review_role(assignment),
                    "review_role_verified": True,
                    "review_role_sha256": assignment.get("review_role_sha256"),
                    "result_template_verified": True,
                    "result_template_sha256": assignment.get(
                        "result_template_sha256"
                    ),
                }
            )
        if expected_assignment_sha256 is not None:
            expected_spawn["assignment_sha256"] = expected_assignment_sha256
        for key, expected_value in expected_spawn.items():
            if spawn_contract.get(key) != expected_value:
                errors.append(
                    "spawn_contract."
                    f"{key} mismatch: expected {expected_value!r}, "
                    f"got {spawn_contract.get(key)!r}"
                )

    status = result.get("status")
    if status not in VALID_RESULT_STATUS:
        errors.append(f"status is invalid: {status!r}")
    for key in ("started_at", "completed_at"):
        if not isinstance(result.get(key), str) or not result[key]:
            errors.append(f"{key} must be a non-empty string")
    if not isinstance(result.get("tool_versions"), dict):
        errors.append("tool_versions must be an object")
    for key in RESULT_LIST_FIELDS:
        if not isinstance(result.get(key), list):
            errors.append(f"{key} must be a list")

    result_path = str(assignment.get("result_path") or "")
    files_written = result.get("files_written")
    if isinstance(files_written, list) and files_written != [result_path]:
        errors.append("files_written must contain only the declared result_path")

    findings = result.get("findings")
    assigned_claims = set(assignment.get("claim_ids") or [])
    if isinstance(findings, list):
        if status != "error" and not findings:
            errors.append("non-error result must contain at least one finding")
        covered_claims: set[str] = set()
        for index, finding in enumerate(findings):
            prefix = f"findings[{index}]"
            if not isinstance(finding, dict):
                errors.append(f"{prefix} is not an object")
                continue
            for key in (
                "finding_id",
                "claim_id",
                "verdict",
                "severity",
                "statement",
                "evidence_fingerprint",
            ):
                if not isinstance(finding.get(key), str) or not finding[key]:
                    errors.append(f"{prefix}.{key} must be a non-empty string")
            for key in (
                "assumptions_used",
                "evidence_refs",
                "counterevidence_refs",
                "reproduction",
                "unresolved",
            ):
                if not isinstance(finding.get(key), list):
                    errors.append(f"{prefix}.{key} must be a list")
            claim_id = finding.get("claim_id")
            if isinstance(claim_id, str):
                covered_claims.add(claim_id)
                if claim_id not in assigned_claims:
                    errors.append(f"{prefix}.claim_id is not owned by the assignment")
            if finding.get("verdict") not in VALID_FINDING_VERDICTS:
                errors.append(f"{prefix}.verdict is invalid")
            fingerprint = finding.get("evidence_fingerprint")
            if not isinstance(fingerprint, str) or not SHA256_FINGERPRINT_RE.fullmatch(
                fingerprint
            ):
                errors.append(f"{prefix}.evidence_fingerprint is not sha256:<64 hex>")
            confidence = finding.get("confidence")
            if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
                errors.append(f"{prefix}.confidence must be numeric")
            elif not 0 <= float(confidence) <= 1:
                errors.append(f"{prefix}.confidence must be between 0 and 1")
        if status != "error" and assigned_claims - covered_claims:
            missing = ", ".join(sorted(assigned_claims - covered_claims))
            errors.append(f"findings omit assigned claim_ids: {missing}")
        verdicts = {
            finding.get("verdict")
            for finding in findings
            if isinstance(finding, dict)
        }
        if status == "pass" and verdicts != {"pass"}:
            errors.append("status=pass requires every finding verdict to be pass")
        elif status == "fail" and "fail" not in verdicts:
            errors.append("status=fail requires at least one fail finding")
        elif status == "inconclusive" and "inconclusive" not in verdicts:
            errors.append(
                "status=inconclusive requires at least one inconclusive finding"
            )
    return errors
