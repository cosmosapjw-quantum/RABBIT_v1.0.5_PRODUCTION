from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
HOOKS_DIR = REPO / ".codex" / "hooks"
HARNESS_SCRIPTS = REPO / ".agent-harness" / "scripts"
sys.path.insert(0, str(HOOKS_DIR))
sys.path.insert(0, str(HARNESS_SCRIPTS))

import subagent_start_context  # noqa: E402
import validate_harness  # noqa: E402
from _harness import (  # noqa: E402
    RESULT_TEMPLATE_PATH,
    validate_assignment_contract,
    validate_assignment_resource_hashes,
    validate_result_contract,
)


RUN_ID = "run-fixture"
ASSIGNMENT_ID = "A-HOOK-FIXTURE"
RESULT_PATH = f".agent-harness/runs/{RUN_ID}/results/{ASSIGNMENT_ID}.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def make_harness(repo: Path) -> tuple[str, dict[str, object]]:
    shared_rel = ".agent-harness/context/SHARED_CONTEXT.md"
    shared = b"fixture shared context\n"
    shared_path = repo / shared_rel
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    shared_path.write_bytes(shared)
    digest = hashlib.sha256()
    digest.update(shared_rel.encode("utf-8"))
    digest.update(b"\0")
    digest.update(shared)
    digest.update(b"\0")
    version = digest.hexdigest()
    role_rel = ".agent-harness/context/roles/context_mapper.md"
    role_path = repo / role_rel
    role_path.parent.mkdir(parents=True, exist_ok=True)
    role_path.write_text("fixture context-mapper role\n", encoding="utf-8")
    index = {
        "context_version": version,
        "max_injected_chars": 24000,
        "shared_files": [shared_rel],
        "file_hashes": {shared_rel: hashlib.sha256(shared).hexdigest()},
        "role_files": {"default": [], "context_mapper": [role_rel]},
    }
    write_json(repo / ".agent-harness/context/CONTEXT_INDEX.json", index)
    pack = f"# Fixture pack\n\nContext version: `{version}`\n\nfixture\n"
    pack_path = repo / ".agent-harness/generated/CONTEXT_PACK.md"
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    pack_path.write_text(pack, encoding="utf-8")
    active = repo / ".agent-harness/ACTIVE_RUN"
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text(RUN_ID + "\n", encoding="utf-8")
    result_template = repo / RESULT_TEMPLATE_PATH
    result_template.parent.mkdir(parents=True, exist_ok=True)
    result_template.write_text(
        (REPO / RESULT_TEMPLATE_PATH).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assignment_path = (
        f".agent-harness/runs/{RUN_ID}/assignments/{ASSIGNMENT_ID}.json"
    )
    assignment: dict[str, object] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "assignment_id": ASSIGNMENT_ID,
        "parent_assignment_id": None,
        "depth": 1,
        "agent_type": "context_mapper",
        "context_version": version,
        "independence_mode": "shared-core",
        "discovery_mode": "targeted",
        "may_spawn": False,
        "delegable_claim_ids": [],
        "claim_ids": ["C-HARNESS-INTEGRITY"],
        "task": "Exercise the hook contract.",
        "required_inputs": [
            ".agent-harness/generated/CONTEXT_PACK.md",
            assignment_path,
        ],
        "allowed_sibling_results": [],
        "allowed_tools": ["exec_command", "apply_patch"],
        "required_outputs": [RESULT_PATH],
        "result_path": RESULT_PATH,
        "status": "registered",
    }
    write_json(repo / assignment_path, assignment)
    return version, assignment


def make_harness_v2(repo: Path) -> tuple[str, dict[str, object]]:
    version, assignment = make_harness(repo)
    index = json.loads(
        (repo / ".agent-harness/context/CONTEXT_INDEX.json").read_text(
            encoding="utf-8"
        )
    )
    role_files = index["role_files"]["context_mapper"]
    role_digest = hashlib.sha256()
    for rel in role_files:
        data = (repo / rel).read_bytes()
        role_digest.update(rel.encode("utf-8"))
        role_digest.update(b"\0")
        role_digest.update(data)
        role_digest.update(b"\0")
    template_digest = hashlib.sha256((repo / RESULT_TEMPLATE_PATH).read_bytes()).hexdigest()
    assignment.update(
        {
            "schema_version": 2,
            "agent_type": "default",
            "runtime_agent_type": "default",
            "review_role": "context_mapper",
            "review_role_files": role_files,
            "review_role_sha256": "sha256:" + role_digest.hexdigest(),
            "result_template": RESULT_TEMPLATE_PATH,
            "result_template_sha256": "sha256:" + template_digest,
        }
    )
    assignment["required_inputs"] = [
        *assignment["required_inputs"],
        RESULT_TEMPLATE_PATH,
        *role_files,
    ]
    write_json(
        repo
        / f".agent-harness/runs/{RUN_ID}/assignments/{ASSIGNMENT_ID}.json",
        assignment,
    )
    return version, assignment


def assignment_sha256(repo: Path, assignment_id: str = ASSIGNMENT_ID) -> str:
    path = (
        repo
        / f".agent-harness/runs/{RUN_ID}/assignments/{assignment_id}.json"
    )
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def valid_result(
    version: str,
    assignment_hash: str,
    assignment_id: str = ASSIGNMENT_ID,
    agent_id: str = "agent-fixture",
) -> dict[str, object]:
    result_path = f".agent-harness/runs/{RUN_ID}/results/{assignment_id}.json"
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "assignment_id": assignment_id,
        "context_version": version,
        "agent_id": agent_id,
        "agent_type": "context_mapper",
        "spawn_contract": {
            "run_id": RUN_ID,
            "assignment_id": assignment_id,
            "context_version": version,
            "independence_mode": "shared-core",
            "prompt_header_verified": True,
            "subagent_start_injected": True,
            "subagent_start_preflight": "PASS",
            "assignment_sha256": assignment_hash,
        },
        "status": "pass",
        "result_path": result_path,
        "started_at": "2026-07-22T00:00:00+00:00",
        "completed_at": "2026-07-22T00:01:00+00:00",
        "tool_versions": {"python": "3"},
        "commands": ["fixture"],
        "artifacts": [],
        "findings": [
            {
                "finding_id": "F-HOOK-FIXTURE",
                "claim_id": "C-HARNESS-INTEGRITY",
                "verdict": "pass",
                "severity": "high",
                "statement": "The fixture contract passed.",
                "assumptions_used": [],
                "evidence_refs": [result_path],
                "evidence_fingerprint": "sha256:" + "0" * 64,
                "counterevidence_refs": [],
                "reproduction": ["pytest fixture"],
                "confidence": 1.0,
                "unresolved": [],
            }
        ],
        "errors": [],
        "files_written": [result_path],
    }


def valid_result_v2(
    version: str, assignment_hash: str, assignment: dict[str, object]
) -> dict[str, object]:
    result = valid_result(version, assignment_hash)
    result.update(
        {
            "schema_version": 2,
            "agent_type": "default",
            "runtime_agent_type": "default",
            "review_role": "context_mapper",
        }
    )
    result["spawn_contract"].update(
        {
            "runtime_agent_type": "default",
            "review_role": "context_mapper",
            "review_role_verified": True,
            "review_role_sha256": assignment["review_role_sha256"],
            "result_template_verified": True,
            "result_template_sha256": assignment["result_template_sha256"],
        }
    )
    return result


def test_subagentstart_injects_identity_and_pass(tmp_path: Path, capsys) -> None:
    make_harness(tmp_path)
    subagent_start_context.inject_subagent_context(
        {"agent_id": "agent-fixture", "agent_type": "context_mapper"}, tmp_path
    )
    output = json.loads(capsys.readouterr().out)
    context = output["hookSpecificOutput"]["additionalContext"]
    assert "[MANDATORY SUBAGENT BOOTSTRAP]" in context
    assert "Agent ID: agent-fixture" in context
    assert "Hook preflight: PASS" in context
    assert "VS Code collaboration does not expose `spawn_agent`" in context
    assert "spawn_contract" in context


def test_shared_assignment_and_result_contracts(tmp_path: Path) -> None:
    version, assignment = make_harness(tmp_path)
    assignment_hash = assignment_sha256(tmp_path)
    assert not validate_assignment_contract(
        assignment,
        expected_run_id=RUN_ID,
        expected_context_version=version,
        role_files={"context_mapper": []},
    )
    assert not validate_result_contract(
        valid_result(version, assignment_hash),
        assignment,
        expected_run_id=RUN_ID,
        expected_context_version=version,
        expected_agent_id="agent-fixture",
        expected_assignment_sha256=assignment_hash,
    )

    invalid = valid_result(version, assignment_hash)
    invalid["spawn_contract"]["prompt_header_verified"] = False
    errors = validate_result_contract(
        invalid,
        assignment,
        expected_run_id=RUN_ID,
        expected_context_version=version,
        expected_agent_id="agent-fixture",
        expected_assignment_sha256=assignment_hash,
    )
    assert any("prompt_header_verified mismatch" in error for error in errors)


def test_v2_contract_separates_runtime_type_and_review_role(tmp_path: Path) -> None:
    version, assignment = make_harness_v2(tmp_path)
    assignment_hash = assignment_sha256(tmp_path)
    index = json.loads(
        (tmp_path / ".agent-harness/context/CONTEXT_INDEX.json").read_text(
            encoding="utf-8"
        )
    )
    assert not validate_assignment_contract(
        assignment,
        expected_run_id=RUN_ID,
        expected_context_version=version,
        role_files=index["role_files"],
    )
    assert not validate_assignment_resource_hashes(
        tmp_path,
        assignment,
        role_files=index["role_files"],
    )
    assert not validate_result_contract(
        valid_result_v2(version, assignment_hash, assignment),
        assignment,
        expected_run_id=RUN_ID,
        expected_context_version=version,
        expected_agent_id="agent-fixture",
        expected_assignment_sha256=assignment_hash,
    )


def test_subagentstop_retry_remains_fail_closed_then_accepts_valid_result(
    tmp_path: Path,
) -> None:
    version, _ = make_harness(tmp_path)
    assignment_hash = assignment_sha256(tmp_path)
    result_path = tmp_path / RESULT_PATH
    result = valid_result(version, assignment_hash)
    result.pop("result_path")
    write_json(result_path, result)
    write_lease(tmp_path, version)
    write_admission(tmp_path, version)
    event = dict(stop_event(version))
    event["stop_hook_active"] = True
    command = [sys.executable, str(HOOKS_DIR / "subagent_stop_validate.py")]
    blocked = subprocess.run(
        command,
        cwd=tmp_path,
        input=json.dumps(event),
        text=True,
        capture_output=True,
        check=False,
    )
    assert blocked.returncode == 0
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "retry" in json.loads(blocked.stdout)["reason"]

    write_json(result_path, valid_result(version, assignment_hash))
    accepted = subprocess.run(
        command,
        cwd=tmp_path,
        input=json.dumps(event),
        text=True,
        capture_output=True,
        check=False,
    )
    assert accepted.returncode == 0
    assert accepted.stdout == ""


def test_subagentstop_accepts_default_runtime_for_distinct_review_role(
    tmp_path: Path,
) -> None:
    version, assignment = make_harness_v2(tmp_path)
    assignment_hash = assignment_sha256(tmp_path)
    write_json(
        tmp_path / RESULT_PATH,
        valid_result_v2(version, assignment_hash, assignment),
    )
    write_lease(tmp_path, version)
    admission = write_admission(tmp_path, version)
    event = dict(stop_event(version))
    event.pop("stop_hook_active")
    event["agent_type"] = "default"
    command = [sys.executable, str(HOOKS_DIR / "subagent_stop_validate.py")]
    accepted = subprocess.run(
        command,
        cwd=tmp_path,
        input=json.dumps(event),
        text=True,
        capture_output=True,
        check=False,
    )
    assert accepted.returncode == 0
    assert accepted.stdout == ""

    assert json.loads(admission.read_text(encoding="utf-8"))["state"] == "consumed"

    assignment_path = (
        tmp_path
        / f".agent-harness/runs/{RUN_ID}/assignments/{ASSIGNMENT_ID}.json"
    )
    invalid_assignment = dict(assignment)
    invalid_assignment["review_role"] = "unregistered_role"
    write_json(assignment_path, invalid_assignment)
    # Re-arm lease and receipt *against the edited bytes* so this probe reaches
    # the contract check rather than the earlier tamper/digest guards.
    write_lease(tmp_path, version)
    write_admission(tmp_path, version)
    blocked_assignment = subprocess.run(
        command,
        cwd=tmp_path,
        input=json.dumps(event),
        text=True,
        capture_output=True,
        check=False,
    )
    assert json.loads(blocked_assignment.stdout)["decision"] == "block"
    assert "assignment contract" in json.loads(blocked_assignment.stdout)["reason"]
    assert "review_role" in json.loads(blocked_assignment.stdout)["reason"]
    write_json(assignment_path, assignment)
    write_lease(tmp_path, version)
    write_admission(tmp_path, version)

    event["agent_type"] = "context_mapper"
    blocked = subprocess.run(
        command,
        cwd=tmp_path,
        input=json.dumps(event),
        text=True,
        capture_output=True,
        check=False,
    )
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "runtime_agent_type" in json.loads(blocked.stdout)["reason"]


def test_new_assignment_prints_exact_header_first(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    subprocess.run(
        ["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    write_json(
        tmp_path / f".agent-harness/runs/{RUN_ID}/RUN_PLAN.json",
        {
            "context_version": version,
            "budget": {"max_total": 8, "max_depth": 2},
        },
    )
    template = json.loads(
        (REPO / ".agent-harness/templates/ASSIGNMENT.json").read_text(
            encoding="utf-8"
        )
    )
    write_json(tmp_path / ".agent-harness/templates/ASSIGNMENT.json", template)
    completed = subprocess.run(
        [
            sys.executable,
            str(HARNESS_SCRIPTS / "new_assignment.py"),
            "--assignment-id",
            "A-NEW",
            "--agent-type",
            "default",
            "--review-role",
            "context_mapper",
            "--task",
            "fixture",
            "--claim-id",
            "C-NEW",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert lines[:4] == [
        f"RUN_ID={RUN_ID}",
        "ASSIGNMENT_ID=A-NEW",
        f"CONTEXT_VERSION={version}",
        "INDEPENDENCE_MODE=shared-core",
    ]
    assert lines[4] == f".agent-harness/runs/{RUN_ID}/assignments/A-NEW.json"
    created = json.loads(
        (
            tmp_path
            / f".agent-harness/runs/{RUN_ID}/assignments/A-NEW.json"
        ).read_text(encoding="utf-8")
    )
    assert created["schema_version"] == 2
    assert created["agent_type"] == "default"
    assert created["runtime_agent_type"] == "default"
    assert created["review_role"] == "context_mapper"
    assert created["review_role_files"] == [
        ".agent-harness/context/roles/context_mapper.md"
    ]
    assert RESULT_TEMPLATE_PATH in created["required_inputs"]
    index = json.loads(
        (tmp_path / ".agent-harness/context/CONTEXT_INDEX.json").read_text(
            encoding="utf-8"
        )
    )
    assert not validate_assignment_resource_hashes(
        tmp_path,
        created,
        role_files=index["role_files"],
    )


ADMISSION_TOKEN = "fixture-admission-token-0001"


def write_lease(
    repo: Path,
    version: str,
    agent_id: str = "agent-fixture",
    assignment_ids: tuple[str, ...] = (ASSIGNMENT_ID,),
) -> Path:
    lease = repo / ".agent-harness" / "leases" / f"{agent_id}.json"
    write_json(
        lease,
        {
            "schema_version": 1,
            "agent_id": agent_id,
            "agent_type": "context_mapper",
            "run_id": RUN_ID,
            "context_version": version,
            "created_at": "2026-07-28T00:00:00+00:00",
            "assignment_digests": {
                aid: assignment_sha256(repo, aid) for aid in assignment_ids
            },
        },
    )
    return lease


def write_admission(
    repo: Path,
    version: str,
    assignment_id: str = ASSIGNMENT_ID,
    token: str = ADMISSION_TOKEN,
    state: str = "open",
    expected_agent_id: str | None = None,
) -> Path:
    """Stand in for `admit_agent.py`: the parent-minted single-use receipt.

    `expected_agent_id` is written only when supplied, in the same field
    position `admit_agent.py` uses, so callers that do not pass it keep minting
    exactly the receipt they minted before this parameter existed.
    """
    path = (
        repo / ".agent-harness" / "admissions" / RUN_ID / f"{assignment_id}.json"
    )
    receipt: dict[str, object] = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "assignment_id": assignment_id,
        "assignment_sha256": assignment_sha256(repo, assignment_id),
        "runtime_agent_type": "context_mapper",
        "context_version": version,
        "token_digest": "sha256:"
        + hashlib.sha256(token.encode("utf-8")).hexdigest(),
    }
    if expected_agent_id is not None:
        receipt["expected_agent_id"] = expected_agent_id
    receipt["state"] = state
    receipt["created_at"] = "2026-07-28T00:00:00+00:00"
    write_json(path, receipt)
    return path


def add_second_assignment(repo: Path, assignment_id: str) -> dict[str, object]:
    """A sibling assignment of the same runtime type, for substitution tests."""
    source = (
        repo / f".agent-harness/runs/{RUN_ID}/assignments/{ASSIGNMENT_ID}.json"
    )
    assignment = json.loads(source.read_text(encoding="utf-8"))
    result_path = f".agent-harness/runs/{RUN_ID}/results/{assignment_id}.json"
    assignment_path = (
        f".agent-harness/runs/{RUN_ID}/assignments/{assignment_id}.json"
    )
    assignment.update(
        {
            "assignment_id": assignment_id,
            "result_path": result_path,
            "required_outputs": [result_path],
            "required_inputs": [
                ".agent-harness/generated/CONTEXT_PACK.md",
                assignment_path,
            ],
        }
    )
    write_json(repo / assignment_path, assignment)
    return assignment


def stop_event(
    version: str,
    assignment_id: str = ASSIGNMENT_ID,
    proof: str | None = ADMISSION_TOKEN,
    agent_id: str = "agent-fixture",
) -> dict[str, object]:
    marker: dict[str, object] = {
        "assignment_id": assignment_id,
        "context_version": version,
        "status": "pass",
        "result_path": f".agent-harness/runs/{RUN_ID}/results/{assignment_id}.json",
    }
    if proof is not None:
        marker["admission_proof"] = proof
    return {
        "hook_event_name": "SubagentStop",
        "stop_hook_active": False,
        "agent_id": agent_id,
        "agent_type": "context_mapper",
        "last_assistant_message": "HARNESS_RESULT: " + json.dumps(marker),
    }


def run_stop_hook(repo: Path, event: dict[str, object]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(HOOKS_DIR / "subagent_stop_validate.py")],
        cwd=repo,
        input=json.dumps(event),
        text=True,
        capture_output=True,
        check=False,
    )


def test_subagentstop_accepts_leased_run_after_pointer_moves(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    assignment_hash = assignment_sha256(tmp_path)
    write_json(tmp_path / RESULT_PATH, valid_result(version, assignment_hash))
    lease = write_lease(tmp_path, version)
    write_admission(tmp_path, version)
    (tmp_path / ".agent-harness/ACTIVE_RUN").write_text(
        "run-decoy\n", encoding="utf-8"
    )
    accepted = run_stop_hook(tmp_path, stop_event(version))
    assert accepted.returncode == 0
    assert accepted.stdout == ""
    consumed_lease = json.loads(lease.read_text(encoding="utf-8"))
    assert consumed_lease["state"] == "consumed"
    assert consumed_lease["consumed_assignment_id"] == ASSIGNMENT_ID


def test_subagentstop_blocks_on_unparseable_lease(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    assignment_hash = assignment_sha256(tmp_path)
    write_json(tmp_path / RESULT_PATH, valid_result(version, assignment_hash))
    lease = tmp_path / ".agent-harness/leases/agent-fixture.json"
    lease.parent.mkdir(parents=True, exist_ok=True)
    lease.write_text("{corrupt", encoding="utf-8")
    blocked = run_stop_hook(tmp_path, stop_event(version))
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "cannot be parsed" in json.loads(blocked.stdout)["reason"]
    assert lease.exists(), "corrupt lease evidence must be preserved"


def test_subagentstop_without_lease_blocks_on_moved_pointer(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    assignment_hash = assignment_sha256(tmp_path)
    write_json(tmp_path / RESULT_PATH, valid_result(version, assignment_hash))
    write_admission(tmp_path, version)
    (tmp_path / ".agent-harness/ACTIVE_RUN").write_text(
        "run-decoy\n", encoding="utf-8"
    )
    blocked = run_stop_hook(tmp_path, stop_event(version))
    assert blocked.returncode == 0
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "No Start-time run lease" in json.loads(blocked.stdout)["reason"]


def test_subagentstop_blocks_leaseless_agent_even_with_matching_pointer(
    tmp_path: Path,
) -> None:
    """D-067: a lease-write failure at Start must be fail-closed at Stop.

    Before D-067 an agent with no lease was resolved through the mutable
    ACTIVE_RUN pointer, so a Start that silently failed to record its lease still
    produced an admissible result. The pointer here is correct and everything
    else is valid; only the lease is missing.
    """
    version, _ = make_harness(tmp_path)
    assignment_hash = assignment_sha256(tmp_path)
    write_json(tmp_path / RESULT_PATH, valid_result(version, assignment_hash))
    write_admission(tmp_path, version)
    blocked = run_stop_hook(tmp_path, stop_event(version))
    assert json.loads(blocked.stdout)["decision"] == "block"
    reason = json.loads(blocked.stdout)["reason"]
    assert "No Start-time run lease" in reason
    assert "ACTIVE_RUN" in reason


def test_subagentstop_blocks_when_assignment_tampered_after_start(
    tmp_path: Path,
) -> None:
    version, _ = make_harness(tmp_path)
    assignment_hash = assignment_sha256(tmp_path)
    write_json(tmp_path / RESULT_PATH, valid_result(version, assignment_hash))
    lease = write_lease(tmp_path, version)
    admission = write_admission(tmp_path, version)
    assignment_path = (
        tmp_path / f".agent-harness/runs/{RUN_ID}/assignments/{ASSIGNMENT_ID}.json"
    )
    assignment_path.write_bytes(assignment_path.read_bytes() + b"\n")
    blocked = run_stop_hook(tmp_path, stop_event(version))
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "changed after SubagentStart" in json.loads(blocked.stdout)["reason"]
    assert lease.exists(), "lease must survive a blocked stop for the retry"
    assert (
        json.loads(admission.read_text(encoding="utf-8"))["state"] == "open"
    ), "a blocked stop must not consume the admission receipt"


def test_subagentstart_writes_atomic_lease(tmp_path: Path, capsys) -> None:
    version, _ = make_harness(tmp_path)
    subagent_start_context.inject_subagent_context(
        {"agent_id": "agent-fixture", "agent_type": "context_mapper"}, tmp_path
    )
    output = json.loads(capsys.readouterr().out)
    context = output["hookSpecificOutput"]["additionalContext"]
    assert f"Run lease: recorded for run {RUN_ID}" in context
    assert "verify_assignment.py" in context
    lease_path = tmp_path / ".agent-harness/leases/agent-fixture.json"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    assert lease["run_id"] == RUN_ID
    assert lease["context_version"] == version
    assert lease["assignment_digests"][ASSIGNMENT_ID] == assignment_sha256(tmp_path)
    assert not list(lease_path.parent.glob(".tmp.*")), "atomic write left temp files"


def test_verify_assignment_matches_sealed_hashes(tmp_path: Path) -> None:
    make_harness_v2(tmp_path)
    subprocess.run(
        ["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    assignment_rel = f".agent-harness/runs/{RUN_ID}/assignments/{ASSIGNMENT_ID}.json"
    command = [
        sys.executable,
        str(HARNESS_SCRIPTS / "verify_assignment.py"),
        assignment_rel,
    ]
    passed = subprocess.run(
        command, cwd=tmp_path, text=True, capture_output=True, check=False
    )
    assert passed.returncode == 0, passed.stderr
    assert "VERIFY: PASS" in passed.stdout
    assert f"assignment_sha256={assignment_sha256(tmp_path)}" in passed.stdout
    role_path = tmp_path / ".agent-harness/context/roles/context_mapper.md"
    role_path.write_bytes(role_path.read_bytes() + b"tampered\n")
    failed = subprocess.run(
        command, cwd=tmp_path, text=True, capture_output=True, check=False
    )
    assert failed.returncode == 1
    assert "review_role_sha256" in failed.stdout
    assert "VERIFY: FAIL" in failed.stdout


# --- D-067: exact agent-to-assignment admission ------------------------------


def test_subagentstop_blocks_same_run_assignment_substitution(tmp_path: Path) -> None:
    """The D-065 controlled negative, now fail-closed.

    An agent admitted for one assignment submits a valid result for a different,
    same-runtime assignment in the same run. The Start-time lease seals both, so
    D-058 accepted this; only the parent-minted receipt distinguishes them.
    """
    version, _ = make_harness(tmp_path)
    other_id = "A-HOOK-FIXTURE-OTHER"
    add_second_assignment(tmp_path, other_id)
    write_json(
        tmp_path / f".agent-harness/runs/{RUN_ID}/results/{other_id}.json",
        valid_result(version, assignment_sha256(tmp_path, other_id), other_id),
    )
    write_lease(tmp_path, version, assignment_ids=(ASSIGNMENT_ID, other_id))
    admitted = write_admission(tmp_path, version, ASSIGNMENT_ID, token="token-for-fixture")
    other = write_admission(tmp_path, version, other_id, token="token-for-other")

    blocked = run_stop_hook(
        tmp_path, stop_event(version, assignment_id=other_id, proof="token-for-fixture")
    )
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "admission_proof does not match" in json.loads(blocked.stdout)["reason"]
    assert json.loads(admitted.read_text(encoding="utf-8"))["state"] == "open"
    assert json.loads(other.read_text(encoding="utf-8"))["state"] == "open"


def test_subagentstop_blocks_missing_admission_proof(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version)
    write_admission(tmp_path, version)
    blocked = run_stop_hook(tmp_path, stop_event(version, proof=None))
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "admission_proof" in json.loads(blocked.stdout)["reason"]


def test_subagentstop_blocks_replayed_consumed_admission(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version)
    write_admission(tmp_path, version, state="consumed")
    blocked = run_stop_hook(tmp_path, stop_event(version))
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "single-use" in json.loads(blocked.stdout)["reason"]


def test_subagentstop_blocks_when_no_admission_exists(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version)
    blocked = run_stop_hook(tmp_path, stop_event(version))
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "admission receipt" in json.loads(blocked.stdout)["reason"]
    assert "admit_agent.py" in json.loads(blocked.stdout)["reason"]


def test_subagentstop_records_write_attribution_on_acceptance(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    result_file = tmp_path / RESULT_PATH
    write_json(result_file, valid_result(version, assignment_sha256(tmp_path)))
    write_lease(tmp_path, version)
    admission = write_admission(tmp_path, version)
    accepted = run_stop_hook(tmp_path, stop_event(version))
    assert accepted.returncode == 0
    assert accepted.stdout == ""
    receipt = json.loads(admission.read_text(encoding="utf-8"))
    assert receipt["state"] == "consumed"
    assert receipt["consumed_by_agent_id"] == "agent-fixture"
    assert receipt["result_sha256"] == (
        "sha256:" + hashlib.sha256(result_file.read_bytes()).hexdigest()
    )
    assert receipt["consumed_at"]
    assert not list(admission.parent.glob(".tmp.*")), "atomic write left temp files"


def test_two_agents_in_one_run_bind_to_distinct_assignments(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    other_id = "A-HOOK-FIXTURE-OTHER"
    add_second_assignment(tmp_path, other_id)
    write_json(
        tmp_path / RESULT_PATH,
        valid_result(version, assignment_sha256(tmp_path), agent_id="agent-one"),
    )
    write_json(
        tmp_path / f".agent-harness/runs/{RUN_ID}/results/{other_id}.json",
        valid_result(
            version,
            assignment_sha256(tmp_path, other_id),
            other_id,
            agent_id="agent-two",
        ),
    )
    ids = (ASSIGNMENT_ID, other_id)
    write_lease(tmp_path, version, agent_id="agent-one", assignment_ids=ids)
    write_lease(tmp_path, version, agent_id="agent-two", assignment_ids=ids)
    first = write_admission(tmp_path, version, ASSIGNMENT_ID, token="token-one")
    second = write_admission(tmp_path, version, other_id, token="token-two")

    accepted_one = run_stop_hook(
        tmp_path, stop_event(version, proof="token-one", agent_id="agent-one")
    )
    accepted_two = run_stop_hook(
        tmp_path,
        stop_event(
            version, assignment_id=other_id, proof="token-two", agent_id="agent-two"
        ),
    )
    assert accepted_one.stdout == ""
    assert accepted_two.stdout == ""
    first_receipt = json.loads(first.read_text(encoding="utf-8"))
    second_receipt = json.loads(second.read_text(encoding="utf-8"))
    assert first_receipt["consumed_by_agent_id"] == "agent-one"
    assert second_receipt["consumed_by_agent_id"] == "agent-two"
    assert first_receipt["result_sha256"] != second_receipt["result_sha256"]


def test_subagentstart_hard_fails_when_lease_cannot_be_written(
    tmp_path: Path, capsys
) -> None:
    version, _ = make_harness(tmp_path)
    leases = tmp_path / ".agent-harness" / "leases"
    leases.mkdir(parents=True, exist_ok=True)
    leases.chmod(0o500)
    try:
        subagent_start_context.inject_subagent_context(
            {"agent_id": "agent-fixture", "agent_type": "context_mapper"}, tmp_path
        )
        output = json.loads(capsys.readouterr().out)
    finally:
        leases.chmod(0o700)
    context = output["hookSpecificOutput"]["additionalContext"]
    assert "Run lease: NOT RECORDED" in context
    assert "Hook preflight: FAIL" in context
    assert "run lease could not be written" in context
    assert output["systemMessage"] == (
        "SubagentStart preflight failed; do not perform substantive work"
    )
    assert not (leases / "agent-fixture.json").exists()
    assert version


START_FAIL_MESSAGE = "SubagentStart preflight failed; do not perform substantive work"


def run_start_hook(
    repo: Path, capsys, agent_id: str = "agent-fixture"
) -> tuple[str, dict[str, object]]:
    """Drive SubagentStart in-process; returns (injected context, hook output)."""
    subagent_start_context.inject_subagent_context(
        {"agent_id": agent_id, "agent_type": "context_mapper"}, repo
    )
    output = json.loads(capsys.readouterr().out)
    return output["hookSpecificOutput"]["additionalContext"], output


def bootstrap_line(context: str, prefix: str) -> str:
    """The one `<prefix>...` line of the injected bootstrap block.

    Asserting against this line rather than the whole context keeps each check
    pinned to the field it is about: a `Run lease:` string cannot satisfy an
    `Admission receipt:` assertion, and neither can the injected pack.
    """
    lines = [line for line in context.splitlines() if line.startswith(prefix)]
    assert len(lines) == 1, f"expected exactly one {prefix!r} line, found {lines}"
    return lines[0]


def test_subagentstart_reports_an_open_bound_admission_receipt(
    tmp_path: Path, capsys
) -> None:
    """D-065 obligation 2 (receipt half): the admitted case is reported, not fatal."""

    version, _ = make_harness(tmp_path)
    receipt = write_admission(tmp_path, version, expected_agent_id="agent-fixture")
    before = receipt.read_bytes()
    context, output = run_start_hook(tmp_path, capsys)
    admission = bootstrap_line(context, "Admission receipt: ")
    assert "Admission receipt: open, bound to assignment" in admission
    assert repr(ASSIGNMENT_ID) in admission
    assert repr(RUN_ID) in admission
    assert "Hook preflight: PASS" in context
    assert "systemMessage" not in output
    assert receipt.read_bytes() == before, "Start must never mutate a receipt"
    assert [p.name for p in receipt.parent.iterdir()] == [f"{ASSIGNMENT_ID}.json"], (
        "Start must not claim or lock the receipt; consumption stays at Stop"
    )


def test_subagentstart_hard_fails_on_a_consumed_admission_receipt(
    tmp_path: Path, capsys
) -> None:
    """A receipt naming this agent that cannot admit it is a hard Start failure."""

    version, _ = make_harness(tmp_path)
    receipt = write_admission(
        tmp_path, version, state="consumed", expected_agent_id="agent-fixture"
    )
    before = receipt.read_bytes()
    context, output = run_start_hook(tmp_path, capsys)
    admission = bootstrap_line(context, "Admission receipt: ")
    assert "Admission receipt: UNUSABLE" in admission
    assert "'consumed'" in admission
    preflight = bootstrap_line(context, "Hook preflight: ")
    assert preflight.startswith("Hook preflight: FAIL")
    assert "not 'open'" in preflight
    assert output["systemMessage"] == START_FAIL_MESSAGE
    # The lease succeeded: the FAIL is carried by the receipt check alone.
    assert f"Run lease: recorded for run {RUN_ID}" in context
    assert receipt.read_bytes() == before, "Start must never mutate a receipt"


def test_subagentstart_hard_fails_on_an_unparseable_admission_receipt(
    tmp_path: Path, capsys
) -> None:
    """An unreadable receipt cannot be proven not to be this agent's: FAIL, not skip."""

    version, _ = make_harness(tmp_path)
    receipt = (
        tmp_path / ".agent-harness" / "admissions" / RUN_ID / f"{ASSIGNMENT_ID}.json"
    )
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text("{corrupt", encoding="utf-8")
    context, output = run_start_hook(tmp_path, capsys)
    admission = bootstrap_line(context, "Admission receipt: ")
    assert "Admission receipt: AMBIGUOUS" in admission
    assert f"{ASSIGNMENT_ID}.json" in admission
    preflight = bootstrap_line(context, "Hook preflight: ")
    assert preflight.startswith("Hook preflight: FAIL")
    assert "ambiguous" in preflight
    assert output["systemMessage"] == START_FAIL_MESSAGE
    assert receipt.read_text(encoding="utf-8") == "{corrupt", (
        "corrupt receipt evidence must be preserved"
    )
    assert version


def test_subagentstart_reports_an_absent_admission_receipt_without_failing(
    tmp_path: Path, capsys
) -> None:
    """Most spawned agents hold no assignment, so absence is reported, not fatal."""

    version, _ = make_harness(tmp_path)
    context, output = run_start_hook(tmp_path, capsys)
    admission = bootstrap_line(context, "Admission receipt: ")
    assert "Admission receipt: none found" in admission
    assert repr("agent-fixture") in admission
    assert "Hook preflight: PASS" in context
    assert "systemMessage" not in output

    # A receipt minted for a different agent is still "none found" for this one.
    write_admission(tmp_path, version, expected_agent_id="agent-other")
    context, output = run_start_hook(tmp_path, capsys)
    assert "Admission receipt: none found" in bootstrap_line(
        context, "Admission receipt: "
    )
    assert "Hook preflight: PASS" in context
    assert "systemMessage" not in output


def test_subagentstart_hard_fails_on_a_receipt_bound_to_another_run(
    tmp_path: Path, capsys
) -> None:
    """Receipt is open, in the active run's directory, but claims a different run."""

    version, _ = make_harness(tmp_path)
    receipt_path = write_admission(
        tmp_path, version, expected_agent_id="agent-fixture"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["run_id"] = "run-decoy"
    write_json(receipt_path, receipt)
    context, output = run_start_hook(tmp_path, capsys)
    admission = bootstrap_line(context, "Admission receipt: ")
    assert "Admission receipt: UNUSABLE" in admission
    assert "'run-decoy'" in admission
    preflight = bootstrap_line(context, "Hook preflight: ")
    assert preflight.startswith("Hook preflight: FAIL")
    assert "bound to a different run" in preflight
    assert output["systemMessage"] == START_FAIL_MESSAGE
    assert json.loads(receipt_path.read_text(encoding="utf-8"))["state"] == "open", (
        "an open receipt for another run must stay open; Start only reads"
    )


def test_subagentstart_hard_fails_on_ambiguous_admission_receipts(
    tmp_path: Path, capsys
) -> None:
    """Two open receipts claim this agent_id: each usable alone, ambiguous together."""

    version, _ = make_harness(tmp_path)
    other_id = "A-HOOK-FIXTURE-OTHER"
    add_second_assignment(tmp_path, other_id)
    write_admission(tmp_path, version, expected_agent_id="agent-fixture")
    write_admission(
        tmp_path,
        version,
        other_id,
        token="token-two",
        expected_agent_id="agent-fixture",
    )
    context, output = run_start_hook(tmp_path, capsys)
    admission = bootstrap_line(context, "Admission receipt: ")
    assert "Admission receipt: AMBIGUOUS" in admission
    assert ASSIGNMENT_ID in admission
    assert other_id in admission
    preflight = bootstrap_line(context, "Hook preflight: ")
    assert preflight.startswith("Hook preflight: FAIL")
    assert "2 admission receipts" in preflight
    assert output["systemMessage"] == START_FAIL_MESSAGE


def test_subagentstart_separates_a_lease_failure_from_a_healthy_receipt(
    tmp_path: Path, capsys
) -> None:
    """The two Start checks are independent: neither masks nor implies the other."""

    version, _ = make_harness(tmp_path)
    write_admission(tmp_path, version, expected_agent_id="agent-fixture")
    leases = tmp_path / ".agent-harness" / "leases"
    leases.mkdir(parents=True, exist_ok=True)
    leases.chmod(0o500)
    try:
        context, output = run_start_hook(tmp_path, capsys)
    finally:
        leases.chmod(0o700)
    assert "Run lease: NOT RECORDED" in bootstrap_line(context, "Run lease: ")
    assert "open, bound to assignment" in bootstrap_line(context, "Admission receipt: ")
    preflight = bootstrap_line(context, "Hook preflight: ")
    assert preflight.startswith("Hook preflight: FAIL")
    assert "run lease could not be written" in preflight
    assert "receipt" not in preflight.lower(), (
        "a healthy receipt must not be blamed for the lease failure"
    )
    assert output["systemMessage"] == START_FAIL_MESSAGE
    assert not (leases / "agent-fixture.json").exists()


def test_subagentstart_skips_the_receipt_check_for_an_unsafe_agent_id(
    tmp_path: Path, capsys
) -> None:
    """An unsafe agent_id is never used as a path key: reported, not traversed."""

    version, _ = make_harness(tmp_path)
    context, output = run_start_hook(tmp_path, capsys, agent_id="agent/../escape")
    assert (
        "Admission receipt: not checked (agent_id is not a safe admission key)"
        in bootstrap_line(context, "Admission receipt: ")
    )
    assert "Run lease: not recorded (agent_id is not a safe lease key)" in (
        bootstrap_line(context, "Run lease: ")
    )
    preflight = bootstrap_line(context, "Hook preflight: ")
    assert preflight.startswith("Hook preflight: FAIL")
    assert "not a safe lease key" in preflight
    assert "receipt" not in preflight.lower(), (
        "the skipped receipt check must contribute no error"
    )
    assert output["systemMessage"] == START_FAIL_MESSAGE
    assert not (tmp_path / ".agent-harness" / "leases").exists()
    assert not (tmp_path / ".agent-harness" / "admissions").exists()
    assert version


def test_admit_agent_mints_single_use_receipt(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    subprocess.run(
        ["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    command = [
        sys.executable,
        str(HARNESS_SCRIPTS / "admit_agent.py"),
        "--assignment-id",
        ASSIGNMENT_ID,
        "--expect-agent-id",
        "agent-fixture",
    ]
    minted = subprocess.run(
        command, cwd=tmp_path, text=True, capture_output=True, check=False
    )
    assert minted.returncode == 0, minted.stderr
    lines = minted.stdout.splitlines()
    assert lines[0] == f"RUN_ID={RUN_ID}"
    assert lines[1] == f"ASSIGNMENT_ID={ASSIGNMENT_ID}"
    assert lines[2] == f"CONTEXT_VERSION={version}"
    assert lines[3] == "INDEPENDENCE_MODE=shared-core"
    token = lines[4].split("=", 1)[1]
    assert len(token) >= 32

    receipt_path = (
        tmp_path / ".agent-harness" / "admissions" / RUN_ID / f"{ASSIGNMENT_ID}.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["state"] == "open"
    assert receipt["assignment_sha256"] == assignment_sha256(tmp_path)
    assert receipt["token_digest"] == (
        "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()
    )
    assert token not in receipt_path.read_text(encoding="utf-8")

    reminted = subprocess.run(
        command, cwd=tmp_path, text=True, capture_output=True, check=False
    )
    assert reminted.returncode != 0
    assert "single-use" in reminted.stderr

    ledger = (
        tmp_path / f".agent-harness/runs/{RUN_ID}/ADMISSIONS.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(ledger) == 1
    minted_line = json.loads(ledger[0])
    assert minted_line["event"] == "minted"
    assert minted_line["assignment_id"] == ASSIGNMENT_ID
    assert token not in ledger[0]


def test_admit_agent_end_to_end_accepts_and_ledgers(tmp_path: Path) -> None:
    """The only test that drives the real minting path into the real Stop hook."""

    version, _ = make_harness(tmp_path)
    subprocess.run(
        ["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    minted = subprocess.run(
        [
            sys.executable,
            str(HARNESS_SCRIPTS / "admit_agent.py"),
            "--assignment-id",
            ASSIGNMENT_ID,
            "--expect-agent-id",
            "agent-fixture",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert minted.returncode == 0, minted.stderr
    token = minted.stdout.splitlines()[4].split("=", 1)[1]

    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version)
    accepted = run_stop_hook(tmp_path, stop_event(version, proof=token))
    assert accepted.returncode == 0
    assert accepted.stdout == ""

    ledger = [
        json.loads(line)
        for line in (tmp_path / f".agent-harness/runs/{RUN_ID}/ADMISSIONS.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["event"] for row in ledger] == ["minted", "consumed"]
    assert ledger[1]["agent_id"] == "agent-fixture"
    assert ledger[1]["token_digest"] == ledger[0]["token_digest"]


def test_admission_bound_to_expected_agent_id(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version, agent_id="agent-thief")
    receipt_path = write_admission(tmp_path, version)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["expected_agent_id"] = "agent-fixture"
    write_json(receipt_path, receipt)
    event = stop_event(version, agent_id="agent-thief")
    blocked = run_stop_hook(tmp_path, event)
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "was minted for agent" in json.loads(blocked.stdout)["reason"]


def test_resumed_agent_can_restop_after_acceptance(tmp_path: Path) -> None:
    """Acceptance must not livelock a resumed subagent (D-067 review F-D067-03)."""

    version, _ = make_harness(tmp_path)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version)
    write_admission(tmp_path, version)
    for attempt in range(3):
        again = run_stop_hook(tmp_path, stop_event(version))
        assert again.returncode == 0, attempt
        assert again.stdout == "", (attempt, again.stdout)


def test_consumed_receipt_still_blocks_a_different_agent(tmp_path: Path) -> None:
    """The single-use receipt check, not the result contract, must be the refusal.

    Strengthened in the D-065 round-6 lane: the body asserted only
    `decision == "block"`, and this stop is refused by `validate_result_contract`
    anyway -- the admitted artifact names `agent-fixture` as its writer, so
    `agent-other` fails the `agent_id` check whatever the receipt says. Deleting
    the single-use guard therefore left the fixture green while a consumed
    receipt admitted a second agent's result. The reason string is now pinned to
    the receipt state that is supposed to do the refusing, and the ledger is
    checked for the second consume row the guard exists to prevent.
    """

    version, _ = make_harness(tmp_path)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version)
    receipt = write_admission(tmp_path, version)
    assert run_stop_hook(tmp_path, stop_event(version)).stdout == ""
    write_lease(tmp_path, version, agent_id="agent-other")
    blocked = run_stop_hook(tmp_path, stop_event(version, agent_id="agent-other"))
    payload = json.loads(blocked.stdout)
    assert payload["decision"] == "block"
    assert "single-use" in payload["reason"], payload["reason"]
    assert "'consumed'" in payload["reason"], payload["reason"]
    assert repr(ASSIGNMENT_ID) in payload["reason"], payload["reason"]
    assert "RESULT_ENVELOPE" not in payload["reason"], (
        "the result contract must not be what refuses this; the receipt state is"
    )
    # The attribution already on the record must survive the refused stop, and
    # no second one may join it.
    body = json.loads(receipt.read_text(encoding="utf-8"))
    assert body["state"] == "consumed"
    assert body["consumed_by_agent_id"] == "agent-fixture"
    consumed = [row for row in _ledger_rows(tmp_path) if row["event"] == "consumed"]
    assert [(row["agent_id"], row["assignment_id"]) for row in consumed] == [
        ("agent-fixture", ASSIGNMENT_ID)
    ], consumed


def test_concurrent_stops_consume_receipt_once(tmp_path: Path) -> None:
    """O_EXCL claim, not the state read, is what makes the receipt single-use.

    Strengthened in the D-065 round-6 lane on two counts. The losers used to be
    judged on stdout alone, with neither stderr nor the exit status captured, so
    a child that died on an import error produced exactly the same empty stdout
    as one that was idempotently accepted -- the fixture could not tell a
    working harness from a broken one. And nothing in it proved the four stops
    ever contended for anything: on a machine that happened to serialise them,
    every assertion still held. The deterministic contention arm at the end
    holds the per-(run, agent_id) lock from the test process, so the mutual
    exclusion is observed rather than hoped for.
    """

    version, _ = make_harness(tmp_path)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_admission(tmp_path, version)
    write_lease(tmp_path, version)
    payload = json.dumps(stop_event(version))
    procs = [
        subprocess.Popen(
            [sys.executable, str(HOOKS_DIR / "subagent_stop_validate.py")],
            cwd=tmp_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    # Feed every child before reading any of them: communicate() in a loop
    # serialises the children and tests nothing (round-2 review F-15).
    for proc in procs:
        assert proc.stdin is not None
        proc.stdin.write(payload)
        proc.stdin.close()
    finished: list[tuple[int, str, str]] = []
    for proc in procs:
        assert proc.stdout is not None and proc.stderr is not None
        out = proc.stdout.read()
        err = proc.stderr.read()
        proc.wait()
        finished.append((proc.returncode, out, err))
    # A hook that crashed is not a hook that fell through to the idempotent
    # path, and only the exit status and stderr can tell them apart.
    for returncode, out, err in finished:
        assert returncode == 0, (returncode, out, err)
        assert err == "", err
    outs = [out for _, out, _ in finished]
    # Timing decides how many losers see a consumed receipt (accept idempotently)
    # versus only the claim or the agent lock (fail closed); all are correct.
    # What must hold regardless is that exactly one of them wrote the
    # attribution, and that no loser silently produced a second one.
    assert any(out == "" for out in outs), outs
    for out in outs:
        if out:
            assert json.loads(out)["decision"] == "block", out
    ledger = (
        tmp_path / f".agent-harness/runs/{RUN_ID}/ADMISSIONS.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    consumed = [line for line in ledger if json.loads(line)["event"] == "consumed"]
    assert len(consumed) == 1, ledger

    # The timing-free arm. Holding the per-(run, agent_id) flock from here makes
    # a concurrent stop by that agent observable without racing anything: it is
    # refused while the lock is held, and admitted the moment the descriptor
    # closes. Without this the four stops above could all have run back to back
    # and the fixture would still have passed.
    lock_file = _agent_lock_path(tmp_path, "agent-fixture")
    assert lock_file.is_file(), "the consume path must have taken the agent lock"
    held = os.open(lock_file, os.O_RDWR)
    try:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contended = run_stop_hook(tmp_path, stop_event(version))
    finally:
        os.close(held)
    assert contended.returncode == 0, contended.stderr
    reason = json.loads(contended.stdout)["reason"]
    assert "Another SubagentStop for agent_id 'agent-fixture'" in reason, reason
    assert "serialised rather than interleaved" in reason, reason
    # Released with the descriptor, so the same stop goes through immediately
    # afterwards and still produces no second attribution row.
    assert run_stop_hook(tmp_path, stop_event(version)).stdout == ""
    assert len(
        [row for row in _ledger_rows(tmp_path) if row["event"] == "consumed"]
    ) == 1


def test_planted_claim_cannot_skip_attribution(tmp_path: Path) -> None:
    """A claim naming the stopping agent must not admit an unattributed result.

    The first fix let the idempotent re-stop path default a missing
    ``result_sha256`` to the current one, so planting a claim produced an
    accepted result with no attribution record at all (round-2 review F-11).
    """

    version, _ = make_harness(tmp_path)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version)
    receipt = write_admission(tmp_path, version)
    receipt.with_name(receipt.name + ".claim").write_text(
        "agent-fixture\n", encoding="utf-8"
    )
    blocked = run_stop_hook(tmp_path, stop_event(version))
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "never attributed" in json.loads(blocked.stdout)["reason"]
    assert json.loads(receipt.read_text(encoding="utf-8"))["state"] == "open"
    assert not (tmp_path / f".agent-harness/runs/{RUN_ID}/ADMISSIONS.jsonl").exists()


def test_restop_with_changed_bytes_is_refused(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    result_file = tmp_path / RESULT_PATH
    write_json(result_file, valid_result(version, assignment_sha256(tmp_path)))
    write_lease(tmp_path, version)
    write_admission(tmp_path, version)
    assert run_stop_hook(tmp_path, stop_event(version)).stdout == ""
    edited = valid_result(version, assignment_sha256(tmp_path))
    edited["commands"] = ["fixture", "edited after admission"]
    write_json(result_file, edited)
    blocked = run_stop_hook(tmp_path, stop_event(version))
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "changed after this agent's result was admitted" in (
        json.loads(blocked.stdout)["reason"]
    )


def test_admit_agent_fails_closed_when_ledger_cannot_be_written(
    tmp_path: Path,
) -> None:
    """Ledger-append-failure fixture: no receipt may outlive a failed mint.

    Named accurately in the D-065 round-6 lane; the old docstring called this
    the receipt-write fixture, which it is not. What is chmod'ed here is the
    *run* directory, and `admit_agent.py` writes the ledger first, so this
    aborts in the ledger append and returns before the receipt write is ever
    reached -- proved by the `ledger could not be appended` assertion below.
    The receipt half lives in its own directory and has its own fixture,
    `test_admit_agent_fails_closed_when_the_receipt_cannot_be_written`.
    """

    version, _ = make_harness(tmp_path)
    subprocess.run(
        ["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    run_dir = tmp_path / ".agent-harness/runs" / RUN_ID
    run_dir.chmod(0o500)
    try:
        minted = subprocess.run(
            [
                sys.executable,
                str(HARNESS_SCRIPTS / "admit_agent.py"),
                "--assignment-id",
                ASSIGNMENT_ID,
                "--expect-agent-id",
                "agent-fixture",
            ],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        run_dir.chmod(0o700)
    assert minted.returncode != 0
    assert "ledger could not be appended" in minted.stderr
    assert not (
        tmp_path / ".agent-harness/admissions" / RUN_ID / f"{ASSIGNMENT_ID}.json"
    ).exists()
    assert version


def test_scrub_admission_proof_replaces_token_with_digest(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    subprocess.run(
        ["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    log_dir = tmp_path / ".agent-harness/runs" / RUN_ID / "raw_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    event = stop_event(version)
    write_json(log_dir / "stop_event.json", event)
    completed = subprocess.run(
        [
            sys.executable,
            str(HARNESS_SCRIPTS / "scrub_admission_proof.py"),
            "--run",
            RUN_ID,
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    text = (log_dir / "stop_event.json").read_text(encoding="utf-8")
    assert ADMISSION_TOKEN not in text
    assert hashlib.sha256(ADMISSION_TOKEN.encode("utf-8")).hexdigest() in text
    # Idempotent: a second pass finds nothing left to scrub.
    again = subprocess.run(
        [
            sys.executable,
            str(HARNESS_SCRIPTS / "scrub_admission_proof.py"),
            "--run",
            RUN_ID,
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert "total tokens scrubbed: 0" in again.stdout


def test_validate_harness_cross_checks_the_admission_ledger(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    subprocess.run(
        ["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    write_json(
        tmp_path / f".agent-harness/runs/{RUN_ID}/RUN_PLAN.json",
        {
            "context_version": version,
            "budget": {"max_concurrent": 1, "max_total": 4, "max_depth": 1},
        },
    )
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version)
    write_admission(tmp_path, version)
    assert run_stop_hook(tmp_path, stop_event(version)).stdout == ""

    def validate() -> dict:
        completed = subprocess.run(
            [sys.executable, str(HARNESS_SCRIPTS / "validate_harness.py")],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )
        return json.loads(completed.stdout)

    assert validate()["ok"] is True
    ledger = tmp_path / f".agent-harness/runs/{RUN_ID}/ADMISSIONS.jsonl"
    ledger.unlink()
    payload = validate()
    assert payload["ok"] is False
    assert any("no admission ledger" in error for error in payload["errors"])


def _validate(tmp_path: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, str(HARNESS_SCRIPTS / "validate_harness.py")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    return json.loads(completed.stdout)


def _minimal_run_plan(tmp_path: Path, version: str) -> None:
    write_json(
        tmp_path / f".agent-harness/runs/{RUN_ID}/RUN_PLAN.json",
        {
            "context_version": version,
            "budget": {"max_concurrent": 1, "max_total": 4, "max_depth": 1},
        },
    )


def test_reopen_cannot_clear_the_tamper_detector(tmp_path: Path) -> None:
    """`--reopen` must not launder an edited artifact into `pending_results`.

    The receipt-driven check alone is defeated by replacing a consumed receipt
    with an open one; the append-only ledger is what makes an admitted digest a
    permanent statement (D-067 round-3 review F-R3-10).
    """

    version, _ = make_harness(tmp_path)
    subprocess.run(
        ["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    _minimal_run_plan(tmp_path, version)
    result_file = tmp_path / RESULT_PATH
    write_json(result_file, valid_result(version, assignment_sha256(tmp_path)))
    write_lease(tmp_path, version)
    write_admission(tmp_path, version)
    assert run_stop_hook(tmp_path, stop_event(version)).stdout == ""
    assert _validate(tmp_path)["ok"] is True

    edited = valid_result(version, assignment_sha256(tmp_path))
    edited["commands"] = ["fixture", "edited after admission"]
    write_json(result_file, edited)
    assert _validate(tmp_path)["ok"] is False

    reopened = subprocess.run(
        [
            sys.executable,
            str(HARNESS_SCRIPTS / "admit_agent.py"),
            "--assignment-id",
            ASSIGNMENT_ID,
            "--expect-agent-id",
            "agent-fixture",
            "--reopen",
            "--reason",
            "round-3 fixture",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert reopened.returncode == 0, reopened.stderr
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "differs from the bytes recorded in the admission ledger" in error
        for error in payload["errors"]
    ), payload["errors"]


def test_consume_ledger_failure_leaves_the_receipt_open(tmp_path: Path) -> None:
    """A failed ledger append must not yield an accepted, unattributed result."""

    version, _ = make_harness(tmp_path)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version)
    receipt = write_admission(tmp_path, version)
    run_dir = tmp_path / ".agent-harness/runs" / RUN_ID
    run_dir.chmod(0o500)
    try:
        blocked = run_stop_hook(tmp_path, stop_event(version))
    finally:
        run_dir.chmod(0o700)
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert json.loads(receipt.read_text(encoding="utf-8"))["state"] == "open"
    assert not receipt.with_name(receipt.name + ".claim").exists()
    # The retry now succeeds and produces exactly one attribution row.
    assert run_stop_hook(tmp_path, stop_event(version)).stdout == ""
    ledger = (run_dir / "ADMISSIONS.jsonl").read_text(encoding="utf-8").splitlines()
    assert len([r for r in ledger if json.loads(r)["event"] == "consumed"]) == 1


def test_admit_agent_requires_an_agent_binding(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    subprocess.run(
        ["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    base = [
        sys.executable,
        str(HARNESS_SCRIPTS / "admit_agent.py"),
        "--assignment-id",
        ASSIGNMENT_ID,
    ]
    refused = subprocess.run(
        base, cwd=tmp_path, text=True, capture_output=True, check=False
    )
    assert refused.returncode != 0
    assert "--expect-agent-id is required" in refused.stderr
    allowed = subprocess.run(
        base + ["--agent-id-unknown"], cwd=tmp_path, text=True,
        capture_output=True, check=False,
    )
    assert allowed.returncode == 0, allowed.stderr
    receipt = json.loads(
        (tmp_path / ".agent-harness/admissions" / RUN_ID / f"{ASSIGNMENT_ID}.json")
        .read_text(encoding="utf-8")
    )
    assert receipt["expected_agent_id"] == ""
    assert version


def test_scrubber_reaches_escaped_and_non_json_evidence(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    subprocess.run(
        ["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    logs = tmp_path / ".agent-harness/runs" / RUN_ID / "raw_logs"
    logs.mkdir(parents=True, exist_ok=True)
    # A marker embedded inside another JSON string: escaped quoting.
    write_json(logs / "nested.json", {"transcript": json.dumps(stop_event(version))})
    (logs / "console.log").write_text(
        'HARNESS_RESULT: {"admission_proof":"' + ADMISSION_TOKEN + '"}\n',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(HARNESS_SCRIPTS / "scrub_admission_proof.py"),
            "--run",
            RUN_ID,
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for name in ("nested.json", "console.log"):
        assert ADMISSION_TOKEN not in (logs / name).read_text(encoding="utf-8"), name
    digest = hashlib.sha256(ADMISSION_TOKEN.encode("utf-8")).hexdigest()
    assert digest in (logs / "console.log").read_text(encoding="utf-8")


def test_claim_holder_blocks_a_second_agent(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    receipt = write_admission(tmp_path, version)
    receipt.with_name(receipt.name + ".claim").write_text(
        "agent-first\n", encoding="utf-8"
    )
    write_lease(tmp_path, version)
    blocked = run_stop_hook(tmp_path, stop_event(version))
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "already claimed by agent-first" in json.loads(blocked.stdout)["reason"]


def test_stop_blocks_when_active_run_is_deleted(tmp_path: Path) -> None:
    """Deleting ACTIVE_RUN must not switch validation off (review F-D067-06)."""

    version, _ = make_harness(tmp_path)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    (tmp_path / ".agent-harness/ACTIVE_RUN").unlink()
    blocked = run_stop_hook(tmp_path, stop_event(version))
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "ACTIVE_RUN is missing or empty" in json.loads(blocked.stdout)["reason"]

    (tmp_path / ".agent-harness/ACTIVE_RUN").write_text("   \n", encoding="utf-8")
    blanked = run_stop_hook(tmp_path, stop_event(version))
    assert json.loads(blanked.stdout)["decision"] == "block"


def test_symlinked_result_path_is_rejected(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    write_lease(tmp_path, version)
    write_admission(tmp_path, version)
    real = tmp_path / ".agent-harness/runs" / RUN_ID / "elsewhere.json"
    write_json(real, valid_result(version, assignment_sha256(tmp_path)))
    link = tmp_path / RESULT_PATH
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(real)
    blocked = run_stop_hook(tmp_path, stop_event(version))
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "symlink" in json.loads(blocked.stdout)["reason"]


def test_validate_harness_flags_unattributed_result(tmp_path: Path) -> None:
    version, _ = make_harness(tmp_path)
    subprocess.run(
        ["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    write_json(
        tmp_path / f".agent-harness/runs/{RUN_ID}/RUN_PLAN.json",
        {
            "context_version": version,
            "budget": {"max_concurrent": 1, "max_total": 4, "max_depth": 1},
        },
    )
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    completed = subprocess.run(
        [sys.executable, str(HARNESS_SCRIPTS / "validate_harness.py")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert any(
        "no admission receipt at all" in error for error in payload["errors"]
    ), payload["errors"]


# ---------------------------------------------------------------------------
# D-067 round-4 review, defect 1: attribution has to cover *every* run
# directory, not just the active one, and the boundary at which the admission
# mechanism was introduced has to be pinned rather than exempted.
# ---------------------------------------------------------------------------

LEGACY_RUN_ID = "run-fixture-legacy"
LEGACY_ASSIGNMENT_ID = "A-LEGACY-FIXTURE"
LEGACY_MANIFEST_REL = ".agent-harness/context/LEGACY_RESULTS_MANIFEST.json"


def _git_init(tmp_path: Path) -> None:
    subprocess.run(
        ["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    # Mirror the real repo, where `/.agent-harness/runs/` is gitignored in full
    # (.gitignore:62) and individual evidence files are force-added. Without
    # this the fixture would not reproduce the condition that makes defect 1
    # invisible: a new file under runs/ produces no git status entry at all,
    # not even `??`.
    (tmp_path / ".gitignore").write_text(
        "/.agent-harness/runs/\n/.agent-harness/leases/\n"
        "/.agent-harness/admissions/\n/.agent-harness/ACTIVE_RUN\n",
        encoding="utf-8",
    )
    # `--emit-legacy-manifest` records the HEAD it was generated at, so the
    # fixture needs one commit before the boundary can be pinned.
    _git_commit_all(tmp_path, "fixture baseline")


def _git_commit_all(tmp_path: Path, message: str) -> None:
    subprocess.run(
        ["git", "add", "-A"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "user.name=harness fixture",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            message,
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def _admitted_active_run(tmp_path: Path) -> str:
    """A fixture whose active-run result is properly admitted and ledgered."""
    version, _ = make_harness(tmp_path)
    _git_init(tmp_path)
    _minimal_run_plan(tmp_path, version)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version)
    write_admission(tmp_path, version)
    assert run_stop_hook(tmp_path, stop_event(version)).stdout == ""
    return version


def _legacy_result_path(tmp_path: Path) -> Path:
    """A result in a second run directory that predates the admission ledger."""
    path = (
        tmp_path
        / ".agent-harness/runs"
        / LEGACY_RUN_ID
        / "results"
        / f"{LEGACY_ASSIGNMENT_ID}.json"
    )
    write_json(path, {"schema_version": 2, "summary": "pre-admission history"})
    return path


def _emit_legacy_manifest(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(HARNESS_SCRIPTS / "validate_harness.py"),
            "--emit-legacy-manifest",
            *args,
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )


def test_fabricated_result_in_a_non_active_run_is_flagged(tmp_path: Path) -> None:
    """The reported defect: a result nobody ever assigned, outside the active run.

    Before the fix, attribution ran only inside `.agent-harness/runs/<ACTIVE_RUN>/`
    and only in the ledger->result direction, so a wholly fabricated artifact
    dropped into any other run directory left the validator reporting `ok: true`.
    """

    version = _admitted_active_run(tmp_path)
    assert _validate(tmp_path)["ok"] is True

    fabricated = (
        tmp_path
        / ".agent-harness/runs"
        / LEGACY_RUN_ID
        / "results"
        / "A-NEVER-ASSIGNED.json"
    )
    write_json(fabricated, {"schema_version": 2, "agent_id": "ghost", "status": "pass"})
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "not attributed" in error and "A-NEVER-ASSIGNED" in error
        for error in payload["errors"]
    ), payload["errors"]
    assert version


def test_fabricated_result_in_a_ledgered_run_is_flagged(tmp_path: Path) -> None:
    """Dropping the file next to genuinely admitted results does not help either."""

    _admitted_active_run(tmp_path)
    fabricated = (
        tmp_path / ".agent-harness/runs" / RUN_ID / "results" / "A-SMUGGLED.json"
    )
    write_json(fabricated, {"schema_version": 2, "status": "pass"})
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "not attributed" in error and "A-SMUGGLED" in error
        for error in payload["errors"]
    ), payload["errors"]


def test_legacy_manifest_pins_rather_than_exempts_preadmission_results(
    tmp_path: Path,
) -> None:
    """The boundary is pinned by digest, so ledger-less runs are not a free pass.

    Skipping runs without an `ADMISSIONS.jsonl` would reintroduce the hole. The
    manifest instead records what was there when the mechanism landed: genuine
    history passes, an edit to it fails, and a newly planted file fails.
    """

    _admitted_active_run(tmp_path)
    legacy = _legacy_result_path(tmp_path)
    pinned_bytes = legacy.read_bytes()

    # Unpinned, the legacy result is unattributed and therefore an error.
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any("not attributed" in error for error in payload["errors"])

    emitted = _emit_legacy_manifest(tmp_path)
    assert emitted.returncode == 0, emitted.stderr
    manifest = json.loads((tmp_path / LEGACY_MANIFEST_REL).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["entry_count"] == 1
    assert manifest["entries"] == [
        {
            "run_id": LEGACY_RUN_ID,
            "assignment_id": LEGACY_ASSIGNMENT_ID,
            "sha256": "sha256:" + hashlib.sha256(pinned_bytes).hexdigest(),
        }
    ]
    # Reproducible and self-describing rather than hand-written.
    assert "--emit-legacy-manifest" in manifest["generated_by"]
    assert manifest["generated_at"] and manifest["generated_at_head"]
    assert "ADMISSIONS.jsonl" in manifest["selection_criterion"]
    # The active run is ledgered, so it must not be grandfathered.
    assert all(entry["run_id"] != RUN_ID for entry in manifest["entries"])

    payload = _validate(tmp_path)
    assert payload["ok"] is True, payload
    # Uncommitted, the pin is real but not yet git-bound; say so out loud.
    assert payload["legacy_results_manifest"] == (
        "ok (1 pinned); UNCOMMITTED, so the pin is not yet git-bound"
    )
    _git_commit_all(tmp_path, "pin the legacy boundary")
    assert _validate(tmp_path)["legacy_results_manifest"] == "ok (1 pinned)"

    # (a) editing a pinned legacy result fails, even though git cannot see the
    #     file at all: `.agent-harness/runs/` is gitignored in full.
    assert (
        subprocess.run(
            ["git", "status", "--porcelain", "--", str(legacy)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        == ""
    )
    write_json(legacy, {"schema_version": 2, "summary": "rewritten history"})
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "changed since the legacy manifest pinned it" in error
        for error in payload["errors"]
    ), payload["errors"]
    legacy.write_bytes(pinned_bytes)
    assert _validate(tmp_path)["ok"] is True

    # (b) a newly planted file in the same ledger-less run still fails.
    write_json(legacy.with_name("A-PLANTED.json"), {"schema_version": 2})
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "not attributed" in error and "A-PLANTED" in error
        for error in payload["errors"]
    ), payload["errors"]


def test_uncommitted_legacy_manifest_edit_or_deletion_is_an_error(
    tmp_path: Path,
) -> None:
    """Blessing a fabrication by editing the manifest is a working-tree error.

    Renamed in the D-070 round-5 review. The old name,
    `test_committed_legacy_manifest_cannot_be_edited_or_deleted`, claimed more
    than the body proves: `git status --porcelain` reports a tracked file that
    differs from HEAD, so this only holds while the edit is UNCOMMITTED. One
    `git commit` cleared it. The committed case is
    `test_legacy_manifest_pin_survives_a_commit` below.
    """

    _admitted_active_run(tmp_path)
    legacy = _legacy_result_path(tmp_path)
    assert _emit_legacy_manifest(tmp_path).returncode == 0
    _git_commit_all(tmp_path, "pin the legacy boundary")
    assert _validate(tmp_path)["ok"] is True

    manifest_path = tmp_path / LEGACY_MANIFEST_REL
    planted = legacy.with_name("A-BLESSED.json")
    write_json(planted, {"schema_version": 2, "summary": "fabricated"})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"].append(
        {
            "run_id": LEGACY_RUN_ID,
            "assignment_id": "A-BLESSED",
            "sha256": "sha256:" + hashlib.sha256(planted.read_bytes()).hexdigest(),
        }
    )
    manifest["entry_count"] = len(manifest["entries"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "Legacy results manifest modified after commit" in error
        for error in payload["errors"]
    ), payload["errors"]

    # Deleting it does not help: the deletion is visible, and with nothing
    # grandfathered every legacy result becomes unattributed.
    planted.unlink()
    manifest_path.unlink()
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "Legacy results manifest modified after commit" in error
        for error in payload["errors"]
    ), payload["errors"]
    assert any("not attributed" in error for error in payload["errors"])


def test_legacy_manifest_fails_closed_when_it_cannot_be_trusted(
    tmp_path: Path,
) -> None:
    """An unparseable or malformed manifest is an error, never a skip."""

    _admitted_active_run(tmp_path)
    _legacy_result_path(tmp_path)
    assert _emit_legacy_manifest(tmp_path).returncode == 0
    assert _validate(tmp_path)["ok"] is True
    manifest_path = tmp_path / LEGACY_MANIFEST_REL

    manifest_path.write_text("{not json", encoding="utf-8")
    payload = _validate(tmp_path)
    assert payload["ok"] is False
    assert any("not valid JSON" in error for error in payload["errors"])
    assert any("not attributed" in error for error in payload["errors"])

    manifest_path.write_text(
        json.dumps({"schema_version": 99, "entries": []}), encoding="utf-8"
    )
    payload = _validate(tmp_path)
    assert payload["ok"] is False
    assert any("unsupported schema_version" in error for error in payload["errors"])

    manifest_path.write_text(
        json.dumps({"schema_version": 1, "entries": [{"run_id": LEGACY_RUN_ID}]}),
        encoding="utf-8",
    )
    payload = _validate(tmp_path)
    assert payload["ok"] is False
    assert any("entry 1 is malformed" in error for error in payload["errors"])

    # A re-emit must not silently overwrite a pinned boundary.
    refused = _emit_legacy_manifest(tmp_path)
    assert refused.returncode != 0
    assert "already exists" in refused.stderr
    assert _emit_legacy_manifest(tmp_path, "--force").returncode == 0


# ---------------------------------------------------------------------------
# D-067 round-4 review, defect 2: an admitted digest is only permanently pinned
# if superseding it must be *declared*.
# ---------------------------------------------------------------------------


def _ledger_rows(tmp_path: Path) -> list[dict]:
    ledger = tmp_path / f".agent-harness/runs/{RUN_ID}/ADMISSIONS.jsonl"
    return [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_ledger_rows(tmp_path: Path, rows: list[dict]) -> None:
    ledger = tmp_path / f".agent-harness/runs/{RUN_ID}/ADMISSIONS.jsonl"
    ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _edit_admitted_result(tmp_path: Path, version: str) -> None:
    edited = valid_result(version, assignment_sha256(tmp_path))
    edited["commands"] = ["fixture", "edited after admission"]
    write_json(tmp_path / RESULT_PATH, edited)


def test_undeclared_second_consume_is_refused(tmp_path: Path) -> None:
    """Deleting the gitignored `.claim` produces a second consume row.

    Measured route (b): the receipt is already `consumed` by this agent, so the
    stop hook's idempotent path is taken -- but that path is only reached when
    the O_EXCL claim exists. With the claim removed the hook takes the fresh
    acceptance path instead, never compares the new bytes against the admitted
    ones, and appends a second consume row. The digest check then passed because
    the validator kept only the last row. It no longer does.
    """

    version = _admitted_active_run(tmp_path)
    receipt = tmp_path / ".agent-harness/admissions" / RUN_ID / f"{ASSIGNMENT_ID}.json"
    claim = receipt.with_name(receipt.name + ".claim")
    assert claim.is_file()
    assert json.loads(receipt.read_text(encoding="utf-8"))["state"] == "consumed"
    admitted_sha = _ledger_rows(tmp_path)[-1]["result_sha256"]

    _edit_admitted_result(tmp_path, version)
    assert _validate(tmp_path)["ok"] is False

    claim.unlink()
    assert run_stop_hook(tmp_path, stop_event(version)).stdout == ""
    rows = _ledger_rows(tmp_path)
    consumes = [row for row in rows if row["event"] == "consumed"]
    assert len(consumes) == 2, rows
    assert not any(row["event"] == "reopened" for row in rows)

    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    conflict = [
        error
        for error in payload["errors"]
        if "second consume with no declared reopen" in error
    ]
    assert conflict, payload["errors"]
    assert admitted_sha in conflict[0]
    assert consumes[1]["result_sha256"] in conflict[0]
    assert f"{RUN_ID}/{ASSIGNMENT_ID}" in conflict[0]


def test_declared_reopen_licenses_a_second_consume(tmp_path: Path) -> None:
    """`--reopen` now records a distinct, digest-naming supersession row."""

    version = _admitted_active_run(tmp_path)
    admitted_sha = _ledger_rows(tmp_path)[-1]["result_sha256"]
    _edit_admitted_result(tmp_path, version)
    assert _validate(tmp_path)["ok"] is False

    reopened = subprocess.run(
        [
            sys.executable,
            str(HARNESS_SCRIPTS / "admit_agent.py"),
            "--assignment-id",
            ASSIGNMENT_ID,
            "--expect-agent-id",
            "agent-fixture",
            "--reopen",
            "--reason",
            "first agent died before stopping",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert reopened.returncode == 0, reopened.stderr
    token = next(
        line.split("=", 1)[1]
        for line in reopened.stdout.splitlines()
        if line.startswith("ADMISSION_TOKEN=")
    )

    row = _ledger_rows(tmp_path)[-1]
    assert row["event"] == "reopened", row
    assert row["superseded_result_sha256"] == admitted_sha
    assert row["superseded_state"] == "consumed"
    assert row["reason"] == "first agent died before stopping"

    assert run_stop_hook(tmp_path, stop_event(version, proof=token)).stdout == ""
    rows = _ledger_rows(tmp_path)
    assert [entry["event"] for entry in rows] == ["consumed", "reopened", "consumed"]
    payload = _validate(tmp_path)
    assert payload["ok"] is True, payload
    # The superseded digest is still on the permanent record.
    assert rows[0]["result_sha256"] == admitted_sha
    assert rows[2]["result_sha256"] != admitted_sha


def test_reopen_row_must_match_the_digest_it_claims_to_supersede(
    tmp_path: Path,
) -> None:
    """A reopen row that names other bytes, or carries no reason, licenses nothing."""

    version = _admitted_active_run(tmp_path)
    _edit_admitted_result(tmp_path, version)
    reopened = subprocess.run(
        [
            sys.executable,
            str(HARNESS_SCRIPTS / "admit_agent.py"),
            "--assignment-id",
            ASSIGNMENT_ID,
            "--expect-agent-id",
            "agent-fixture",
            "--reopen",
            "--reason",
            "fixture",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert reopened.returncode == 0, reopened.stderr
    token = next(
        line.split("=", 1)[1]
        for line in reopened.stdout.splitlines()
        if line.startswith("ADMISSION_TOKEN=")
    )
    assert run_stop_hook(tmp_path, stop_event(version, proof=token)).stdout == ""
    good_rows = _ledger_rows(tmp_path)
    assert _validate(tmp_path)["ok"] is True

    for mutation, field, value in (
        ("wrong digest", "superseded_result_sha256", "sha256:" + "0" * 64),
        ("blank reason", "reason", "   "),
    ):
        rows = [dict(row) for row in good_rows]
        rows[1][field] = value
        _write_ledger_rows(tmp_path, rows)
        payload = _validate(tmp_path)
        assert payload["ok"] is False, (mutation, payload)
        assert any(
            "second consume with no declared reopen" in error
            for error in payload["errors"]
        ), (mutation, payload["errors"])

    # An ordinary mint row cannot stand in for a declared reopen either.
    rows = [dict(row) for row in good_rows]
    rows[1]["event"] = "minted"
    _write_ledger_rows(tmp_path, rows)
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "second consume with no declared reopen" in error
        for error in payload["errors"]
    ), payload["errors"]


def test_admission_ledger_garbage_is_an_error_not_a_skip(tmp_path: Path) -> None:
    """A non-object ledger row used to be silently ignored."""

    _admitted_active_run(tmp_path)
    rows = _ledger_rows(tmp_path)
    ledger = tmp_path / f".agent-harness/runs/{RUN_ID}/ADMISSIONS.jsonl"
    ledger.write_text(
        json.dumps(rows[0], sort_keys=True) + "\n[1, 2, 3]\n", encoding="utf-8"
    )
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "Admission ledger line is not an object" in error for error in payload["errors"]
    ), payload["errors"]


# ---------------------------------------------------------------------------
# Regression guards for checks the round-4 panel confirmed already working.
# ---------------------------------------------------------------------------


def test_validate_harness_fails_closed_when_git_is_unavailable(tmp_path: Path) -> None:
    """No git means no integrity evidence, which is a failure, not a pass.

    The scripts are copied into the fixture so `_harness.root()` falls back to
    the fixture tree rather than the real repository when `git rev-parse` cannot
    run; otherwise the check would silently validate the wrong tree.
    """

    version, _ = make_harness(tmp_path)
    _minimal_run_plan(tmp_path, version)
    scripts = tmp_path / ".agent-harness" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for name in ("validate_harness.py", "_harness.py"):
        (scripts / name).write_bytes((HARNESS_SCRIPTS / name).read_bytes())

    completed = subprocess.run(
        [sys.executable, str(scripts / "validate_harness.py")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env={"PATH": str(tmp_path / "no-such-bin")},
    )
    assert completed.returncode == 1, completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    required = (
        "Tracked run evidence integrity check could not run",
        "Legacy results manifest integrity check could not run",
    )
    for prefix in required:
        matched = [error for error in payload["errors"] if error.startswith(prefix)]
        assert matched, (prefix, payload["errors"])
        assert "git is required for validation" in matched[0]


def test_validate_harness_still_surfaces_ssot_divergence(tmp_path: Path) -> None:
    """The registry-vs-prose consistency check stays wired in and fail-closed."""

    _admitted_active_run(tmp_path)
    _emit_legacy_manifest(tmp_path)
    assert _validate(tmp_path)["ssot_consistency"].startswith("not applicable")

    registry = tmp_path / ".agent-harness/context/GATE_REGISTRY.json"
    write_json(registry, {"gates": []})
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        error.startswith("SSOT") for error in payload["errors"]
    ), payload["errors"]

    # Deleting a tracked registry must not switch the check off.
    _git_commit_all(tmp_path, "track the gate registry")
    registry.unlink()
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "cannot be disabled by deleting its input" in error
        for error in payload["errors"]
    ), payload["errors"]


# ---------------------------------------------------------------------------
# D-070 round-5 review, finding 1: the manifest's git binding stopped at the
# working tree, so one `git commit` laundered a fabricated result plus the
# manifest entry blessing it. Its identity is now pinned on two further
# cross-checked surfaces.
# ---------------------------------------------------------------------------

SECOND_ASSIGNMENT_ID = "A-HOOK-FIXTURE-2"
OTHER_RUN_ID = "run-fixture-other"


def _canonical_manifest_digest(entries: list[dict]) -> str:
    """The documented canonical form, written out independently of the checker.

    Reimplemented here on purpose: if `validate_harness.canonical_legacy_digest`
    ever changes shape, this fixture disagrees with it instead of silently
    following it.
    """
    payload = "\n".join(
        "{run_id}\0{assignment_id}\0{sha256}".format(**entry)
        for entry in sorted(
            entries, key=lambda entry: (entry["run_id"], entry["assignment_id"])
        )
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_fixture_facts(tmp_path: Path, entry_count: int) -> None:
    """A minimal declared-facts surface; validate_harness reads only the value."""
    write_json(
        tmp_path / ".agent-harness/context/SSOT_FACTS.json",
        {
            "schema_version": 1,
            "purpose": "fixture",
            "coverage_policy": {"claims": {"exempt": []}},
            "assertion_exemptions": [],
            "facts": [
                {
                    "fact_id": "legacy_results_manifest_entry_count",
                    "description": "fixture",
                    "measurement": "fixture",
                    "value": entry_count,
                    "as_of_commit": "fixture",
                    "assertions": [],
                }
            ],
        },
    )


def _repinned_validator(tmp_path: Path, canonical_sha256: str) -> Path:
    """Copy the checker into the fixture with its pinned digest re-aimed.

    The pinned digest is a constant of the real repository, so a fixture can
    only exercise it by substituting its own. The substitution is a plain
    replacement of the constant's value and is asserted to have happened.
    """
    scripts = tmp_path / ".agent-harness" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "_harness.py").write_bytes((HARNESS_SCRIPTS / "_harness.py").read_bytes())
    source = (HARNESS_SCRIPTS / "validate_harness.py").read_text(encoding="utf-8")
    pinned = validate_harness.LEGACY_MANIFEST_PINNED_SHA256
    assert source.count(pinned) == 1, "the pinned digest is not a single literal"
    patched = source.replace(pinned, canonical_sha256)
    target = scripts / "validate_harness.py"
    target.write_text(patched, encoding="utf-8")
    return target


def _validate_with(script: Path, tmp_path: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.stdout, completed.stderr
    return json.loads(completed.stdout)


def _pinned_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """An admitted run plus a committed, fully pinned legacy manifest."""
    _admitted_active_run(tmp_path)
    legacy = _legacy_result_path(tmp_path)
    assert _emit_legacy_manifest(tmp_path).returncode == 0
    manifest_path = tmp_path / LEGACY_MANIFEST_REL
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _write_fixture_facts(tmp_path, manifest["entry_count"])
    script = _repinned_validator(
        tmp_path, _canonical_manifest_digest(manifest["entries"])
    )
    _git_commit_all(tmp_path, "pin the legacy boundary")
    return legacy, script


def test_legacy_manifest_pin_survives_a_commit(tmp_path: Path) -> None:
    """The reported defect: `git commit` used to make an edited manifest clean.

    Reproduced before the fix as `ok: true, "ok (2 pinned)"`. The porcelain
    check only ever saw a tracked file differing from HEAD, and
    `.agent-harness/runs/` is gitignored in full, so committing the manifest
    left nothing at all to notice.
    """

    legacy, script = _pinned_fixture(tmp_path)
    payload = _validate_with(script, tmp_path)
    assert payload["ok"] is True, payload
    assert payload["legacy_results_manifest"] == (
        "ok (1 pinned); pinned by declared fact and digest (1)"
    )

    manifest_path = tmp_path / LEGACY_MANIFEST_REL
    planted = legacy.with_name("A-BLESSED.json")
    write_json(planted, {"schema_version": 2, "summary": "fabricated"})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"].append(
        {
            "run_id": LEGACY_RUN_ID,
            "assignment_id": "A-BLESSED",
            "sha256": "sha256:" + hashlib.sha256(planted.read_bytes()).hexdigest(),
        }
    )
    manifest["entry_count"] = len(manifest["entries"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _git_commit_all(tmp_path, "launder: commit the edited manifest")

    payload = _validate_with(script, tmp_path)
    assert payload["ok"] is False, payload
    assert not any(
        "modified after commit" in error for error in payload["errors"]
    ), "the porcelain check is clean again; the pin is what has to catch this"
    assert any(
        "declared-facts file pins 1" in error for error in payload["errors"]
    ), payload["errors"]
    assert any(
        "do not match the digest pinned" in error for error in payload["errors"]
    ), payload["errors"]

    # Moving the declared count too is still not enough: the digest pins WHICH
    # artifacts, not merely how many.
    _write_fixture_facts(tmp_path, 2)
    _git_commit_all(tmp_path, "launder: move the declared count as well")
    payload = _validate_with(script, tmp_path)
    assert payload["ok"] is False, payload
    assert not any("declared-facts file pins" in error for error in payload["errors"])
    assert any(
        "do not match the digest pinned" in error for error in payload["errors"]
    ), payload["errors"]

    # A swap at an unchanged count fails on the digest alone.
    planted.unlink()
    manifest["entries"] = manifest["entries"][1:]
    manifest["entry_count"] = 1
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_fixture_facts(tmp_path, 1)
    _git_commit_all(tmp_path, "launder: swap one pinned entry for another")
    payload = _validate_with(script, tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "do not match the digest pinned" in error for error in payload["errors"]
    ), payload["errors"]

    # A manifest that miscounts itself is refused outright: check_ssot_consistency
    # reads the `entry_count` field while this checker counts the entries, and
    # the cross-check only means something while those are the same number.
    manifest["entry_count"] = 99
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _git_commit_all(tmp_path, "launder: lie about the entry count")
    payload = _validate_with(script, tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "declares entry_count 99 but carries 1 entries" in error
        for error in payload["errors"]
    ), payload["errors"]


def test_legacy_manifest_pin_cannot_be_dropped_by_deleting_its_declaration(
    tmp_path: Path,
) -> None:
    """Deleting the tracked declared-facts file is an error, not a bypass."""

    _, script = _pinned_fixture(tmp_path)
    assert _validate_with(script, tmp_path)["ok"] is True

    (tmp_path / ".agent-harness/context/SSOT_FACTS.json").unlink()
    payload = _validate_with(script, tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "Declared-facts file is tracked in git but missing" in error
        for error in payload["errors"]
    ), payload["errors"]

    # Emptying it of the declaration is an error too, rather than a silent pass.
    write_json(
        tmp_path / ".agent-harness/context/SSOT_FACTS.json",
        {"schema_version": 1, "facts": []},
    )
    payload = _validate_with(script, tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "declares no 'legacy_results_manifest_entry_count'" in error
        for error in payload["errors"]
    ), payload["errors"]

    # And removing the manifest while its pin stands is refused as well.
    _write_fixture_facts(tmp_path, 1)
    (tmp_path / LEGACY_MANIFEST_REL).unlink()
    payload = _validate_with(script, tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "cannot be removed or broken to drop the pin" in error
        for error in payload["errors"]
    ), payload["errors"]


def test_repository_legacy_manifest_pin_is_self_consistent() -> None:
    """The real pin must describe the real manifest, or it has rotted.

    Read-only against the repository: the checker's constant, the declared fact
    and the manifest are three surfaces that only mean something while they
    agree.
    """

    manifest = json.loads(
        (REPO / LEGACY_MANIFEST_REL).read_text(encoding="utf-8")
    )
    entries = manifest["entries"]
    assert manifest["entry_count"] == len(entries)
    assert _canonical_manifest_digest(entries) == (
        validate_harness.LEGACY_MANIFEST_PINNED_SHA256
    )
    facts = json.loads(
        (REPO / ".agent-harness/context/SSOT_FACTS.json").read_text(encoding="utf-8")
    )
    declared = [
        fact
        for fact in facts["facts"]
        if fact["fact_id"] == validate_harness.LEGACY_MANIFEST_PIN_FACT_ID
    ]
    assert len(declared) == 1, "the manifest pin must be declared exactly once"
    assert declared[0]["value"] == len(entries)


# ---------------------------------------------------------------------------
# D-070 round-5 review, finding F-R5-04: the open-receipt carve-out was
# unbounded in time and in scope. It must stay wide enough that a live run is
# never wedged, and no wider.
# ---------------------------------------------------------------------------


def _mint(tmp_path: Path, assignment_id: str, agent_id: str) -> str:
    minted = subprocess.run(
        [
            sys.executable,
            str(HARNESS_SCRIPTS / "admit_agent.py"),
            "--assignment-id",
            assignment_id,
            "--expect-agent-id",
            agent_id,
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert minted.returncode == 0, minted.stderr
    return next(
        line.split("=", 1)[1]
        for line in minted.stdout.splitlines()
        if line.startswith("ADMISSION_TOKEN=")
    )


def _in_flight_second_agent(tmp_path: Path) -> tuple[str, str, Path]:
    """A second assignment, admitted through the real interface, result written.

    This is exactly the shape of the reported attack -- no forgery, only the
    legitimate `admit_agent.py` interface -- and also exactly the shape of a
    genuinely live agent that has not stopped yet. The two are indistinguishable
    at this instant, which is why the carve-out is bounded rather than removed.
    """
    version = _admitted_active_run(tmp_path)
    add_second_assignment(tmp_path, SECOND_ASSIGNMENT_ID)
    token = _mint(tmp_path, SECOND_ASSIGNMENT_ID, "agent-fixture-2")
    result_file = (
        tmp_path / f".agent-harness/runs/{RUN_ID}/results/{SECOND_ASSIGNMENT_ID}.json"
    )
    write_json(
        result_file,
        valid_result(
            version,
            assignment_sha256(tmp_path, SECOND_ASSIGNMENT_ID),
            assignment_id=SECOND_ASSIGNMENT_ID,
            agent_id="agent-fixture-2",
        ),
    )
    return version, token, result_file


def test_live_run_with_an_open_admission_is_not_wedged(tmp_path: Path) -> None:
    """The control for round-2 finding F-12: the carve-out still carves out.

    A genuinely in-flight admission in the active run keeps the validator green,
    now reports the pending artifact with its digest and a count, and the run
    completes normally when the agent stops.
    """

    version, token, result_file = _in_flight_second_agent(tmp_path)
    payload = _validate(tmp_path)
    assert payload["ok"] is True, payload
    assert payload["pending_result_count"] == 1
    pending = payload["pending_results"][0]
    assert pending["run_id"] == RUN_ID
    assert pending["assignment_id"] == SECOND_ASSIGNMENT_ID
    assert pending["sha256"] == (
        "sha256:" + hashlib.sha256(result_file.read_bytes()).hexdigest()
    )
    assert pending["receipt_created_at"]

    write_lease(
        tmp_path,
        version,
        agent_id="agent-fixture-2",
        assignment_ids=(ASSIGNMENT_ID, SECOND_ASSIGNMENT_ID),
    )
    accepted = run_stop_hook(
        tmp_path,
        stop_event(
            version,
            assignment_id=SECOND_ASSIGNMENT_ID,
            proof=token,
            agent_id="agent-fixture-2",
        ),
    )
    assert accepted.stdout == "", accepted.stdout
    payload = _validate(tmp_path)
    assert payload["ok"] is True, payload
    assert payload["pending_result_count"] == 0


def test_open_admission_carve_out_lapses_when_the_admission_is_stale(
    tmp_path: Path,
) -> None:
    """The tolerance is bounded in time, so "never stop" is no longer a strategy.

    The receipt is back-dated rather than waited out; waiting is what an
    attacker would actually do, and the clock the validator reads is the same
    either way.
    """

    _in_flight_second_agent(tmp_path)
    assert _validate(tmp_path)["ok"] is True

    receipt = (
        tmp_path / ".agent-harness/admissions" / RUN_ID / f"{SECOND_ASSIGNMENT_ID}.json"
    )
    body = json.loads(receipt.read_text(encoding="utf-8"))
    body["created_at"] = "2020-01-01T00:00:00+00:00"
    write_json(receipt, body)
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "carve-out is refused" in error and "in-flight ceiling" in error
        for error in payload["errors"]
    ), payload["errors"]

    # Dating the receipt forward buys nothing either.
    body["created_at"] = "2099-01-01T00:00:00+00:00"
    write_json(receipt, body)
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "in the future" in error for error in payload["errors"]
    ), payload["errors"]

    # Nor does removing the timestamp the bound is measured against.
    body.pop("created_at")
    write_json(receipt, body)
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "no parseable UTC created_at" in error for error in payload["errors"]
    ), payload["errors"]


def test_open_admission_carve_out_needs_a_registered_assignment(
    tmp_path: Path,
) -> None:
    """Nothing can be in flight for an assignment the run does not have."""

    _in_flight_second_agent(tmp_path)
    assert _validate(tmp_path)["ok"] is True
    (
        tmp_path
        / f".agent-harness/runs/{RUN_ID}/assignments/{SECOND_ASSIGNMENT_ID}.json"
    ).unlink()
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "carve-out is refused" in error and "no assignment is registered" in error
        for error in payload["errors"]
    ), payload["errors"]


def test_open_admission_carve_out_is_bound_to_the_active_run(
    tmp_path: Path,
) -> None:
    """The global forward sweep must not make the carve-out global too.

    Reproduced before the fix: a receipt minted while `ACTIVE_RUN` pointed at a
    second run directory excused arbitrary bytes there for ever, even after the
    pointer moved back -- and outside the active run not even the result
    contract is applied to them.
    """

    version = _admitted_active_run(tmp_path)
    add_second_assignment(tmp_path, SECOND_ASSIGNMENT_ID)
    assignment = json.loads(
        (
            tmp_path
            / f".agent-harness/runs/{RUN_ID}/assignments/{SECOND_ASSIGNMENT_ID}.json"
        ).read_text(encoding="utf-8")
    )
    result_rel = (
        f".agent-harness/runs/{OTHER_RUN_ID}/results/{SECOND_ASSIGNMENT_ID}.json"
    )
    assignment.update(
        {
            "run_id": OTHER_RUN_ID,
            "result_path": result_rel,
            "required_outputs": [result_rel],
            "required_inputs": [
                ".agent-harness/generated/CONTEXT_PACK.md",
                f".agent-harness/runs/{OTHER_RUN_ID}/assignments/"
                f"{SECOND_ASSIGNMENT_ID}.json",
            ],
        }
    )
    write_json(
        tmp_path
        / f".agent-harness/runs/{OTHER_RUN_ID}/assignments/{SECOND_ASSIGNMENT_ID}.json",
        assignment,
    )
    write_json(
        tmp_path / f".agent-harness/runs/{OTHER_RUN_ID}/RUN_PLAN.json",
        {
            "context_version": version,
            "budget": {"max_concurrent": 1, "max_total": 4, "max_depth": 1},
        },
    )
    active = tmp_path / ".agent-harness/ACTIVE_RUN"
    active.write_text(OTHER_RUN_ID + "\n", encoding="utf-8")
    _mint(tmp_path, SECOND_ASSIGNMENT_ID, "agent-fixture-3")
    write_json(tmp_path / result_rel, {"anything": "at all"})
    active.write_text(RUN_ID + "\n", encoding="utf-8")

    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "carve-out is refused" in error
        and OTHER_RUN_ID in error
        and "not the active run" in error
        for error in payload["errors"]
    ), payload["errors"]


# ---------------------------------------------------------------------------
# D-070 round-5 review, finding F-R5-08: an unusable active run crashed the
# validator instead of reporting an error. Confirmed present at a1cdd8a.
# ---------------------------------------------------------------------------


def test_missing_active_run_directory_is_a_clear_error(tmp_path: Path) -> None:
    """A crash is fail-closed but unreadable: stdout was empty, JSON unparseable."""

    _admitted_active_run(tmp_path)
    (tmp_path / ".agent-harness/ACTIVE_RUN").write_text(
        "run-that-never-existed\n", encoding="utf-8"
    )
    completed = subprocess.run(
        [sys.executable, str(HARNESS_SCRIPTS / "validate_harness.py")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1, completed.stderr
    assert "Traceback" not in completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert any(
        "ACTIVE_RUN names a run directory that does not exist" in error
        for error in payload["errors"]
    ), payload["errors"]


def test_unusable_active_run_plan_is_a_clear_error(tmp_path: Path) -> None:
    """The same for a run plan that is absent, not an object, or budget-less."""

    _admitted_active_run(tmp_path)
    plan_path = tmp_path / f".agent-harness/runs/{RUN_ID}/RUN_PLAN.json"
    version = json.loads(plan_path.read_text(encoding="utf-8"))["context_version"]

    plan_path.unlink()
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "Active run plan is missing or unreadable" in error
        for error in payload["errors"]
    ), payload["errors"]

    plan_path.write_text("[]", encoding="utf-8")
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "Active run plan is not a JSON object" in error for error in payload["errors"]
    ), payload["errors"]

    write_json(plan_path, {"context_version": version, "budget": {}})
    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    assert any(
        "no integer budget.max_total" in error for error in payload["errors"]
    ), payload["errors"]


# ---------------------------------------------------------------------------
# D-065 obligation 1, round-5 finding: one *agent_id* binds to one assignment,
# not merely one token. What was enforced before was token-to-assignment, so a
# single agent_id could hold -- and consume -- two receipts for two assignments
# in one run. admit_agent.py refuses to mint the second receipt;
# subagent_stop_validate.py refuses to consume it. Both halves are exercised
# here, together with the false-positive controls that keep an ordinary
# multi-agent run unwedged.
# ---------------------------------------------------------------------------


def _admit(
    tmp_path: Path, assignment_id: str, agent_id: str, *extra: str
) -> subprocess.CompletedProcess:
    """`admit_agent.py` as a subprocess, whether it succeeds or refuses.

    `_mint` asserts success and returns the token; the mint-time binding guard
    needs the failing invocations too, so this returns the process instead.
    """
    return subprocess.run(
        [
            sys.executable,
            str(HARNESS_SCRIPTS / "admit_agent.py"),
            "--assignment-id",
            assignment_id,
            "--expect-agent-id",
            agent_id,
            *extra,
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )


def _receipt_path(tmp_path: Path, assignment_id: str) -> Path:
    return (
        tmp_path / ".agent-harness" / "admissions" / RUN_ID / f"{assignment_id}.json"
    )


def _two_assignment_fixture(tmp_path: Path) -> str:
    """Two registered assignments in the active run, mintable through the CLI."""
    version, _ = make_harness(tmp_path)
    subprocess.run(
        ["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True, text=True
    )
    add_second_assignment(tmp_path, SECOND_ASSIGNMENT_ID)
    return version


def test_admit_agent_refuses_second_assignment_for_same_agent_id(
    tmp_path: Path,
) -> None:
    """The reported gap: the same agent_id minted onto a second assignment.

    Nothing about the second mint is malformed -- the assignment is registered,
    the run is active, the token is fresh. The only thing wrong with it is that
    `agent-fixture` is already bound elsewhere in this run, which is precisely
    the obligation the token-only check never expressed.
    """

    _two_assignment_fixture(tmp_path)
    first = _admit(tmp_path, ASSIGNMENT_ID, "agent-fixture")
    assert first.returncode == 0, first.stderr
    receipt_before = _receipt_path(tmp_path, ASSIGNMENT_ID).read_bytes()

    refused = _admit(tmp_path, SECOND_ASSIGNMENT_ID, "agent-fixture")
    assert refused.returncode != 0, refused.stdout
    assert "already holds an open or consumed admission receipt" in refused.stderr
    # The refusal has to name the assignment already held, not just complain:
    # the operator's next move is to reopen it or pick another agent_id.
    assert repr(ASSIGNMENT_ID) in refused.stderr, refused.stderr
    assert repr(SECOND_ASSIGNMENT_ID) in refused.stderr, refused.stderr

    assert not _receipt_path(tmp_path, SECOND_ASSIGNMENT_ID).exists()
    assert _receipt_path(tmp_path, ASSIGNMENT_ID).read_bytes() == receipt_before, (
        "a refused mint must not disturb the receipt it refused on behalf of"
    )
    assert json.loads(receipt_before.decode("utf-8"))["state"] == "open"
    assert [
        (row["event"], row["assignment_id"]) for row in _ledger_rows(tmp_path)
    ] == [("minted", ASSIGNMENT_ID)]


def test_admit_agent_reopen_same_assignment_is_exempt_from_the_binding_guard(
    tmp_path: Path,
) -> None:
    """The carve-out is deliberate and must stay narrow.

    `--reopen` re-mints for the SAME assignment and agent, which is the
    supersession path `main` already governs. Exempting it must not exempt the
    agent_id itself: a second assignment is still refused afterwards.
    """

    _two_assignment_fixture(tmp_path)
    assert _admit(tmp_path, ASSIGNMENT_ID, "agent-fixture").returncode == 0
    first_receipt = json.loads(
        _receipt_path(tmp_path, ASSIGNMENT_ID).read_text(encoding="utf-8")
    )

    reopened = _admit(
        tmp_path,
        ASSIGNMENT_ID,
        "agent-fixture",
        "--reopen",
        "--reason",
        "binding-guard carve-out fixture",
    )
    assert reopened.returncode == 0, reopened.stderr
    assert "already holds an open or consumed" not in reopened.stderr
    reopened_receipt = json.loads(
        _receipt_path(tmp_path, ASSIGNMENT_ID).read_text(encoding="utf-8")
    )
    assert reopened_receipt["state"] == "open"
    assert reopened_receipt["expected_agent_id"] == "agent-fixture"
    assert reopened_receipt["token_digest"] != first_receipt["token_digest"], (
        "the carve-out must be a real re-mint, not a no-op that hid a refusal"
    )

    refused = _admit(tmp_path, SECOND_ASSIGNMENT_ID, "agent-fixture")
    assert refused.returncode != 0, refused.stdout
    assert "already holds an open or consumed admission receipt" in refused.stderr
    assert repr(ASSIGNMENT_ID) in refused.stderr, refused.stderr
    assert not _receipt_path(tmp_path, SECOND_ASSIGNMENT_ID).exists()


def test_admit_agent_binding_guard_allows_distinct_agent_ids(tmp_path: Path) -> None:
    """The false-positive control: two agents, two assignments, one run.

    A guard that keyed off the receipt rather than off the agent_id would refuse
    the second mint here and wedge every multi-agent run, which is a worse
    failure than the one being fixed.
    """

    _two_assignment_fixture(tmp_path)
    first = _admit(tmp_path, ASSIGNMENT_ID, "agent-one")
    assert first.returncode == 0, first.stderr
    second = _admit(tmp_path, SECOND_ASSIGNMENT_ID, "agent-two")
    assert second.returncode == 0, second.stderr

    receipts = {
        assignment_id: json.loads(
            _receipt_path(tmp_path, assignment_id).read_text(encoding="utf-8")
        )
        for assignment_id in (ASSIGNMENT_ID, SECOND_ASSIGNMENT_ID)
    }
    assert receipts[ASSIGNMENT_ID]["expected_agent_id"] == "agent-one"
    assert receipts[SECOND_ASSIGNMENT_ID]["expected_agent_id"] == "agent-two"
    assert [r["state"] for r in receipts.values()] == ["open", "open"]
    assert (
        receipts[ASSIGNMENT_ID]["token_digest"]
        != receipts[SECOND_ASSIGNMENT_ID]["token_digest"]
    )
    assert sorted(
        (row["event"], row["assignment_id"]) for row in _ledger_rows(tmp_path)
    ) == [("minted", ASSIGNMENT_ID), ("minted", SECOND_ASSIGNMENT_ID)]


def test_admit_agent_binding_guard_fails_closed_on_unreadable_sibling_receipt(
    tmp_path: Path,
) -> None:
    """A receipt that cannot be read cannot be proven not to bind this agent.

    The guard therefore fails on any unparseable sibling, not only on ones that
    turn out to name this agent_id -- otherwise corrupting a receipt would be a
    way to mint past the binding.
    """

    _two_assignment_fixture(tmp_path)
    corrupt = _receipt_path(tmp_path, SECOND_ASSIGNMENT_ID)
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_text("{corrupt", encoding="utf-8")

    refused = _admit(tmp_path, ASSIGNMENT_ID, "agent-fixture")
    assert refused.returncode != 0, refused.stdout
    assert "agent-to-assignment binding cannot be verified" in refused.stderr
    assert corrupt.name in refused.stderr, refused.stderr
    assert not _receipt_path(tmp_path, ASSIGNMENT_ID).exists()
    assert not (
        tmp_path / f".agent-harness/runs/{RUN_ID}/ADMISSIONS.jsonl"
    ).exists(), "a refused mint must not leave a minted row behind"
    assert corrupt.read_text(encoding="utf-8") == "{corrupt", (
        "corrupt receipt evidence must be preserved"
    )


def test_subagentstop_blocks_agent_id_already_consumed_a_different_assignment(
    tmp_path: Path,
) -> None:
    """The Stop-time backstop, which is the half that actually consumes.

    The two receipts are written directly rather than minted, which is the
    point: the mint guard only covers receipts created after it landed, so the
    ledger check has to stand on its own for anything minted before the fix or
    written around it. The second stop presents its OWN valid token for its own
    registered assignment -- the only defect is that one agent_id is doing both.
    """

    version, _ = make_harness(tmp_path)
    add_second_assignment(tmp_path, SECOND_ASSIGNMENT_ID)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    second_result = (
        tmp_path / f".agent-harness/runs/{RUN_ID}/results/{SECOND_ASSIGNMENT_ID}.json"
    )
    write_json(
        second_result,
        valid_result(
            version,
            assignment_sha256(tmp_path, SECOND_ASSIGNMENT_ID),
            SECOND_ASSIGNMENT_ID,
        ),
    )
    write_lease(
        tmp_path, version, assignment_ids=(ASSIGNMENT_ID, SECOND_ASSIGNMENT_ID)
    )
    first = write_admission(
        tmp_path,
        version,
        ASSIGNMENT_ID,
        token="token-first",
        expected_agent_id="agent-fixture",
    )
    second = write_admission(
        tmp_path,
        version,
        SECOND_ASSIGNMENT_ID,
        token="token-second",
        expected_agent_id="agent-fixture",
    )

    accepted = run_stop_hook(tmp_path, stop_event(version, proof="token-first"))
    assert accepted.stdout == "", accepted.stdout
    assert json.loads(first.read_text(encoding="utf-8"))["state"] == "consumed"
    assert json.loads(first.read_text(encoding="utf-8"))["consumed_by_agent_id"] == (
        "agent-fixture"
    )

    blocked = run_stop_hook(
        tmp_path,
        stop_event(version, assignment_id=SECOND_ASSIGNMENT_ID, proof="token-second"),
    )
    payload = json.loads(blocked.stdout)
    assert payload["decision"] == "block", payload
    assert "already consumed" in payload["reason"]
    assert repr(ASSIGNMENT_ID) in payload["reason"], payload["reason"]
    assert repr(SECOND_ASSIGNMENT_ID) in payload["reason"], payload["reason"]

    assert json.loads(second.read_text(encoding="utf-8"))["state"] == "open", (
        "a blocked stop must leave the receipt it was refused unspent"
    )
    consumed = [
        row for row in _ledger_rows(tmp_path) if row["event"] == "consumed"
    ]
    assert [(row["agent_id"], row["assignment_id"]) for row in consumed] == [
        ("agent-fixture", ASSIGNMENT_ID)
    ], consumed


def test_subagentstop_ledger_conflict_check_ignores_same_assignment_restop(
    tmp_path: Path,
) -> None:
    """The conflict filter is (agent_id AND assignment_id), not either alone.

    Another agent's consume row must not block this one, and this agent's own
    consume row must not block its idempotent re-stop. That re-stop path has
    already been broken once by a well-meant check (round-2 review F-11), so a
    guard added above it needs its own control.
    """

    version, _ = make_harness(tmp_path)
    add_second_assignment(tmp_path, SECOND_ASSIGNMENT_ID)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_json(
        tmp_path / f".agent-harness/runs/{RUN_ID}/results/{SECOND_ASSIGNMENT_ID}.json",
        valid_result(
            version,
            assignment_sha256(tmp_path, SECOND_ASSIGNMENT_ID),
            SECOND_ASSIGNMENT_ID,
            agent_id="agent-other",
        ),
    )
    write_lease(tmp_path, version)
    write_lease(
        tmp_path,
        version,
        agent_id="agent-other",
        assignment_ids=(SECOND_ASSIGNMENT_ID,),
    )
    write_admission(
        tmp_path,
        version,
        ASSIGNMENT_ID,
        token="token-mine",
        expected_agent_id="agent-fixture",
    )
    write_admission(
        tmp_path,
        version,
        SECOND_ASSIGNMENT_ID,
        token="token-theirs",
        expected_agent_id="agent-other",
    )

    foreign = run_stop_hook(
        tmp_path,
        stop_event(
            version,
            assignment_id=SECOND_ASSIGNMENT_ID,
            proof="token-theirs",
            agent_id="agent-other",
        ),
    )
    assert foreign.stdout == "", foreign.stdout

    for attempt in range(3):
        again = run_stop_hook(tmp_path, stop_event(version, proof="token-mine"))
        assert again.returncode == 0, (attempt, again.stderr)
        assert again.stdout == "", (attempt, again.stdout)

    consumed = [
        (row["agent_id"], row["assignment_id"])
        for row in _ledger_rows(tmp_path)
        if row["event"] == "consumed"
    ]
    assert sorted(consumed) == [
        ("agent-fixture", ASSIGNMENT_ID),
        ("agent-other", SECOND_ASSIGNMENT_ID),
    ], consumed


def test_subagentstop_fails_closed_on_unreadable_admission_ledger(
    tmp_path: Path,
) -> None:
    """The ledger is the enforcement point, so an unprovable negative blocks.

    Defaulting an unparseable ledger to "no conflict" would make garbling one
    line the way past the binding check.
    """

    version, _ = make_harness(tmp_path)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version)
    write_admission(tmp_path, version, expected_agent_id="agent-fixture")
    accepted = run_stop_hook(tmp_path, stop_event(version))
    assert accepted.stdout == "", accepted.stdout  # control: the same stop, readable

    ledger = tmp_path / f".agent-harness/runs/{RUN_ID}/ADMISSIONS.jsonl"
    ledger.write_text(
        ledger.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8"
    )

    blocked = run_stop_hook(tmp_path, stop_event(version))
    payload = json.loads(blocked.stdout)
    assert payload["decision"] == "block", payload
    assert "line 2 is malformed" in payload["reason"], payload["reason"]
    assert "binding cannot be verified" in payload["reason"], payload["reason"]
    assert repr(RUN_ID) in payload["reason"], payload["reason"]


# ---------------------------------------------------------------------------
# D-065 obligation 3, round-6 mutation finding. A mint performs TWO writes, in
# two different directories: the append-only ledger under
# `.agent-harness/runs/<run>/`, then the receipt under
# `.agent-harness/admissions/<run>/`. Only the ledger one had a fixture --
# `test_admit_agent_fails_closed_when_ledger_cannot_be_written` chmods the run
# directory and so returns before the receipt is ever attempted -- and round 6
# proved the gap by mutation: replacing the receipt-write `except OSError:
# fail(...)` with `pass`, which prints a live ADMISSION_TOKEN for a receipt
# that is not on disk, left 74 passed / 0 failed.
# ---------------------------------------------------------------------------


def _git_init_bare(tmp_path: Path) -> None:
    """`git init` and nothing else, so `_harness.root()` sees the fixture tree.

    `root()` falls back to the script's own `parents[2]` -- the REAL repository
    -- when `git rev-parse` fails, so every fixture that runs anything out of
    `.agent-harness/scripts/` has to be a git repository of its own or it
    validates, and can write to, the live tree. `_git_init` further down also
    writes a .gitignore and commits, which mint-only fixtures do not need.
    """
    subprocess.run(
        ["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True, text=True
    )


def _admissions_dir(tmp_path: Path, run_id: str = RUN_ID) -> Path:
    return tmp_path / ".agent-harness" / "admissions" / run_id


def _agent_lock_path(
    tmp_path: Path, agent_id: str, run_id: str = RUN_ID
) -> Path:
    """The per-(run, agent_id) consume lock `subagent_stop_validate.py` takes.

    Spelled out here rather than imported so the fixture disagrees with the
    hook if either moves, instead of silently following it.
    """
    return _admissions_dir(tmp_path, run_id) / ".agent-locks" / f"{agent_id}.lock"


def _token_of(completed: subprocess.CompletedProcess) -> str:
    return next(
        line.split("=", 1)[1]
        for line in completed.stdout.splitlines()
        if line.startswith("ADMISSION_TOKEN=")
    )


def _admit_unbound(
    tmp_path: Path, assignment_id: str, *extra: str
) -> subprocess.CompletedProcess:
    """`admit_agent.py --agent-id-unknown`: a receipt bound to its token only.

    The mode in which the mint-time binding guard is not merely skipped but not
    constructible, so the whole of D-065 obligation 1 rests on the Stop-time
    per-agent lock.
    """
    return subprocess.run(
        [
            sys.executable,
            str(HARNESS_SCRIPTS / "admit_agent.py"),
            "--assignment-id",
            assignment_id,
            "--agent-id-unknown",
            *extra,
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )


def _mint_with_unwritable_receipt_dir(
    tmp_path: Path, agent_id: str = "agent-fixture", *extra: str
) -> subprocess.CompletedProcess:
    """Mint with the RECEIPT directory read-only and the ledger writable.

    This is what isolates obligation 3's second write. The ledger lives under
    `.agent-harness/runs/<run>/` and the receipt under
    `.agent-harness/admissions/<run>/`, so revoking write on the latter alone
    lets the append succeed and fails `dump_json_atomic` -- the exact split the
    round-6 reproduction used, and the reason a surviving `minted` row is a
    load-bearing assertion rather than a detail.
    """
    admissions = _admissions_dir(tmp_path)
    admissions.mkdir(parents=True, exist_ok=True)
    admissions.chmod(0o500)
    try:
        return _admit(tmp_path, ASSIGNMENT_ID, agent_id, *extra)
    finally:
        admissions.chmod(0o700)


def test_admit_agent_fails_closed_when_the_receipt_cannot_be_written(
    tmp_path: Path,
) -> None:
    """Obligation 3's missing half: the RECEIPT write failing, hard.

    A mint that prints its token but leaves no receipt is the worst of both
    states: the parent pastes a live ADMISSION_TOKEN into a spawn prompt, and
    the agent's stop is then refused hours later for a receipt that never
    existed. Nothing in the ledger row distinguishes it from a healthy mint.
    """

    version, _ = make_harness(tmp_path)
    _git_init_bare(tmp_path)
    minted = _mint_with_unwritable_receipt_dir(tmp_path)

    assert minted.returncode != 0, minted.stdout
    # The error has to name the receipt, not the ledger: they are separate
    # failure surfaces and the operator's next move differs.
    assert "admission receipt could not be written" in minted.stderr, minted.stderr
    assert "PermissionError" in minted.stderr, minted.stderr
    assert "ledger could not be appended" not in minted.stderr, minted.stderr
    # Nothing may reach stdout. stdout is what the parent pastes into the spawn
    # prompt, and the token is the only thing that can consume a receipt.
    assert minted.stdout == "", minted.stdout
    assert "ADMISSION_TOKEN" not in minted.stdout

    assert not _receipt_path(tmp_path, ASSIGNMENT_ID).exists()
    assert list(_admissions_dir(tmp_path).iterdir()) == [], (
        "a failed atomic write must not leave a .tmp.* file behind either"
    )
    # ...and the ledger row IS there. Without this assertion the fixture would
    # not have proven which of the mint's two writes it made fail, which is
    # exactly how the existing chmod-the-run-directory fixture came to be
    # docstringed as this one.
    assert [
        (row["event"], row["assignment_id"]) for row in _ledger_rows(tmp_path)
    ] == [("minted", ASSIGNMENT_ID)], _ledger_rows(tmp_path)
    assert version


# ---------------------------------------------------------------------------
# D-065 obligation 1, round-6 finding: the per-(run, agent_id) consume lock.
# The round-5 fix put a plain ledger READ in front of a consume whose only
# atomic primitive was the per-*assignment* O_EXCL claim, so two concurrent
# stops by one agent_id on two assignments took two different claim files and
# raced -- both admitted in 126 of 200 measured trials. The lock is the missing
# mutual exclusion; these fixtures are its detector.
# ---------------------------------------------------------------------------

AGENT_LOCK_TRIALS = 12


def _feed_stop_hooks(
    tmp_path: Path, events: list[dict[str, object]]
) -> list[tuple[int, str, str]]:
    """Run every stop hook concurrently; return `(returncode, stdout, stderr)`.

    Every child is fed before any is read, for the reason
    `test_concurrent_stops_consume_receipt_once` records: `communicate()` in a
    loop serialises them and tests nothing.
    """
    procs = [
        subprocess.Popen(
            [sys.executable, str(HOOKS_DIR / "subagent_stop_validate.py")],
            cwd=tmp_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in events
    ]
    for proc, event in zip(procs, events):
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(event))
        proc.stdin.close()
    finished: list[tuple[int, str, str]] = []
    for proc in procs:
        assert proc.stdout is not None and proc.stderr is not None
        out = proc.stdout.read()
        err = proc.stderr.read()
        proc.wait()
        finished.append((proc.returncode, out, err))
    return finished


def _token_only_two_assignment_trial(tree: Path) -> tuple[str, str, str]:
    """One trial tree: two assignments minted `--agent-id-unknown`, results written.

    Minted through the real CLI in the real mode, because that is the mode in
    which no mint-time guard exists at all: both receipts are agent-unbound, so
    a single agent_id holding both tokens is refused by nothing except the
    Stop-time lock.
    """
    version, _ = make_harness(tree)
    _git_init_bare(tree)
    add_second_assignment(tree, SECOND_ASSIGNMENT_ID)
    first = _admit_unbound(tree, ASSIGNMENT_ID, "--reason", "runtime assigns agent ids")
    second = _admit_unbound(
        tree, SECOND_ASSIGNMENT_ID, "--reason", "runtime assigns agent ids"
    )
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    write_json(tree / RESULT_PATH, valid_result(version, assignment_sha256(tree)))
    write_json(
        tree / f".agent-harness/runs/{RUN_ID}/results/{SECOND_ASSIGNMENT_ID}.json",
        valid_result(
            version,
            assignment_sha256(tree, SECOND_ASSIGNMENT_ID),
            SECOND_ASSIGNMENT_ID,
        ),
    )
    write_lease(tree, version, assignment_ids=(ASSIGNMENT_ID, SECOND_ASSIGNMENT_ID))
    return version, _token_of(first), _token_of(second)


def test_concurrent_stops_by_one_agent_consume_exactly_one_assignment(
    tmp_path: Path,
) -> None:
    """The reported race, over enough trials to be a detector rather than a hope.

    One `agent_id`, two agent-unbound receipts, two concurrent stops. The defect
    reproduced in roughly 63% of trials, so a single trial would miss it more
    than a third of the time; at twelve, a guard that is absent survives with
    probability 0.37**12, about six in a million. Each trial gets its own tree
    because the property being measured is per-run state.
    """

    for trial in range(AGENT_LOCK_TRIALS):
        tree = tmp_path / f"trial-{trial:02d}"
        tree.mkdir()
        version, first_token, second_token = _token_only_two_assignment_trial(tree)
        finished = _feed_stop_hooks(
            tree,
            [
                stop_event(version, proof=first_token),
                stop_event(
                    version, assignment_id=SECOND_ASSIGNMENT_ID, proof=second_token
                ),
            ],
        )
        for returncode, out, err in finished:
            assert returncode == 0, (trial, returncode, out, err)
            assert err == "", (trial, err)
        consumed = [
            row for row in _ledger_rows(tree) if row["event"] == "consumed"
        ]
        assert len(consumed) == 1, (trial, consumed)
        assert consumed[0]["agent_id"] == "agent-fixture", (trial, consumed)
        assert len({row["assignment_id"] for row in consumed}) == 1, (trial, consumed)
        # The loser must say why, and it must be the binding, not a coincidence.
        losers = [out for _, out, _ in finished if out]
        assert len(losers) == 1, (trial, [out for _, out, _ in finished])
        reason = json.loads(losers[0])["reason"]
        assert "D-065 obligation 1" in reason, (trial, reason)
        assert "one assignment per run" in reason, (trial, reason)
        # Both receipts still exist; exactly one of them was spent.
        states = sorted(
            json.loads(
                _receipt_path(tree, assignment_id).read_text(encoding="utf-8")
            )["state"]
            for assignment_id in (ASSIGNMENT_ID, SECOND_ASSIGNMENT_ID)
        )
        assert states == ["consumed", "open"], (trial, states)


def test_concurrent_stops_by_distinct_agents_are_both_admitted(
    tmp_path: Path,
) -> None:
    """The false-positive control: a per-agent lock must not serialise the run.

    A lock keyed on the run, or one never released, would refuse one of these
    two perfectly ordinary stops and wedge every multi-agent run -- a worse
    failure than the race it is there to close. Run over the same number of
    trials as the positive case so a lock that is merely usually-wrong is not
    mistaken for a correct one.
    """

    for trial in range(AGENT_LOCK_TRIALS):
        tree = tmp_path / f"trial-{trial:02d}"
        tree.mkdir()
        version, first_token, second_token = _token_only_two_assignment_trial(tree)
        for assignment_id, agent_id in (
            (ASSIGNMENT_ID, "agent-one"),
            (SECOND_ASSIGNMENT_ID, "agent-two"),
        ):
            result_rel = (
                f".agent-harness/runs/{RUN_ID}/results/{assignment_id}.json"
            )
            write_json(
                tree / result_rel,
                valid_result(
                    version,
                    assignment_sha256(tree, assignment_id),
                    assignment_id,
                    agent_id=agent_id,
                ),
            )
            write_lease(
                tree,
                version,
                agent_id=agent_id,
                assignment_ids=(ASSIGNMENT_ID, SECOND_ASSIGNMENT_ID),
            )
        finished = _feed_stop_hooks(
            tree,
            [
                stop_event(version, proof=first_token, agent_id="agent-one"),
                stop_event(
                    version,
                    assignment_id=SECOND_ASSIGNMENT_ID,
                    proof=second_token,
                    agent_id="agent-two",
                ),
            ],
        )
        for returncode, out, err in finished:
            assert returncode == 0, (trial, returncode, out, err)
            assert err == "", (trial, err)
            assert out == "", (trial, out)
        consumed = sorted(
            (row["agent_id"], row["assignment_id"])
            for row in _ledger_rows(tree)
            if row["event"] == "consumed"
        )
        assert consumed == [
            ("agent-one", ASSIGNMENT_ID),
            ("agent-two", SECOND_ASSIGNMENT_ID),
        ], (trial, consumed)
        # Two agents, two locks: the lock is keyed on (run, agent_id), not run.
        assert sorted(
            path.name
            for path in _agent_lock_path(tree, "agent-one").parent.iterdir()
        ) == ["agent-one.lock", "agent-two.lock"]


def test_agent_lock_is_released_after_a_blocked_stop(tmp_path: Path) -> None:
    """A refused stop must not wedge its own agent_id for the rest of the run.

    The lock is an flock on a descriptor precisely so that "released on every
    failure path" holds structurally, including the ~20 `block(); return` paths
    inside the guarded region. The retry here is the same agent, in the same
    run, immediately after a hard refusal.
    """

    version, _ = make_harness(tmp_path)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version)
    receipt = write_admission(tmp_path, version)

    blocked = run_stop_hook(tmp_path, stop_event(version, proof="not-the-token"))
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "admission_proof does not match" in json.loads(blocked.stdout)["reason"]
    lock_file = _agent_lock_path(tmp_path, "agent-fixture")
    assert lock_file.is_file(), "the blocked stop must still have taken the lock"

    accepted = run_stop_hook(tmp_path, stop_event(version))
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout == "", accepted.stdout
    assert json.loads(receipt.read_text(encoding="utf-8"))["state"] == "consumed"
    assert lock_file.is_file(), (
        "the lock file is deliberately left in place; unlinking a flocked path "
        "lets a racing process take a lock on a different inode of the same name"
    )


def test_agent_lock_directory_failure_is_fail_closed(tmp_path: Path) -> None:
    """An unusable lock directory is an error, never a skip.

    Skipping the lock when it cannot be created reopens exactly the window it
    closes, and does so silently -- the consume would look completely ordinary
    from the outside.
    """

    version, _ = make_harness(tmp_path)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version)
    receipt = write_admission(tmp_path, version)
    admissions = _admissions_dir(tmp_path)
    admissions.chmod(0o500)
    try:
        blocked = run_stop_hook(tmp_path, stop_event(version))
    finally:
        admissions.chmod(0o700)

    assert blocked.returncode == 0, blocked.stderr
    reason = json.loads(blocked.stdout)["reason"]
    assert "agent consume lock" in reason, reason
    assert "lock directory could not be created" in reason, reason
    assert "PermissionError" in reason, reason
    assert "D-065 obligation 1" in reason, reason
    assert json.loads(receipt.read_text(encoding="utf-8"))["state"] == "open", (
        "a stop that never took the lock must not have consumed anything"
    )
    assert not (
        tmp_path / f".agent-harness/runs/{RUN_ID}/ADMISSIONS.jsonl"
    ).exists(), "no attribution row may be written without the lock"
    assert not receipt.with_name(receipt.name + ".claim").exists()


def test_planted_symlink_at_the_agent_lock_path_blocks(tmp_path: Path) -> None:
    """`O_NOFOLLOW` turns a planted lock symlink into ELOOP, not a lock elsewhere.

    Without it, planting a symlink at the lock path aims the flock at some other
    inode, so two stops for one agent take "the lock" on two different files and
    the mutual exclusion evaporates while looking entirely healthy.
    """

    version, _ = make_harness(tmp_path)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version)
    receipt = write_admission(tmp_path, version)
    lock_file = _agent_lock_path(tmp_path, "agent-fixture")
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    decoy = lock_file.parent / "decoy.lock"
    decoy.write_text("", encoding="utf-8")
    lock_file.symlink_to(decoy)

    blocked = run_stop_hook(tmp_path, stop_event(version))
    assert blocked.returncode == 0, blocked.stderr
    reason = json.loads(blocked.stdout)["reason"]
    assert "agent consume lock" in reason, reason
    assert "lock could not be opened" in reason, reason
    assert json.loads(receipt.read_text(encoding="utf-8"))["state"] == "open"
    assert not (
        tmp_path / f".agent-harness/runs/{RUN_ID}/ADMISSIONS.jsonl"
    ).exists()
    assert lock_file.is_symlink(), "the hook must not have followed or replaced it"
    assert decoy.read_text(encoding="utf-8") == "", (
        "nothing may be written through the planted symlink"
    )

    # Control: the same stop is admitted once the symlink is gone, so the block
    # is attributable to the symlink and not to the rest of the fixture.
    lock_file.unlink()
    assert run_stop_hook(tmp_path, stop_event(version)).stdout == ""


def test_agent_binding_mode_is_recorded_on_the_receipt_and_the_ledger(
    tmp_path: Path,
) -> None:
    """`agent_binding` is a recorded fact, not something inferred from a blank.

    An empty `expected_agent_id` could equally mean "minted with no agent
    binding" or "field written by an older tool". The receipt is gitignored
    working state, so the ledger row has to carry it too or an auditor cannot
    tell after the fact which receipts never bound an agent.
    """

    version = _two_assignment_fixture(tmp_path)
    unbound = _admit_unbound(
        tmp_path, ASSIGNMENT_ID, "--reason", "runtime assigns agent ids"
    )
    assert unbound.returncode == 0, unbound.stderr
    # The mode is warned about on stderr, never silently, and never on stdout.
    assert "--agent-id-unknown" in unbound.stderr
    assert "bound to its token only" in unbound.stderr
    assert "NOT guaranteed" in unbound.stderr
    assert "WARNING" not in unbound.stdout

    bound = _admit(tmp_path, SECOND_ASSIGNMENT_ID, "agent-two")
    assert bound.returncode == 0, bound.stderr

    receipts = {
        assignment_id: json.loads(
            _receipt_path(tmp_path, assignment_id).read_text(encoding="utf-8")
        )
        for assignment_id in (ASSIGNMENT_ID, SECOND_ASSIGNMENT_ID)
    }
    assert receipts[ASSIGNMENT_ID]["agent_binding"] == "token-only"
    assert receipts[ASSIGNMENT_ID]["expected_agent_id"] == ""
    assert receipts[SECOND_ASSIGNMENT_ID]["agent_binding"] == "agent-id"
    assert receipts[SECOND_ASSIGNMENT_ID]["expected_agent_id"] == "agent-two"

    rows = {row["assignment_id"]: row for row in _ledger_rows(tmp_path)}
    assert rows[ASSIGNMENT_ID]["agent_binding"] == "token-only"
    assert rows[ASSIGNMENT_ID]["agent_binding_reason"] == "runtime assigns agent ids"
    assert rows[SECOND_ASSIGNMENT_ID]["agent_binding"] == "agent-id"
    assert "agent_binding_reason" not in rows[SECOND_ASSIGNMENT_ID]

    # A second unbound mint names the other open, agent-unbound receipts, which
    # is what keeps the residual ambiguity of the mode concrete rather than
    # stated once in a docstring.
    add_second_assignment(tmp_path, "A-HOOK-FIXTURE-3")
    also = _admit_unbound(tmp_path, "A-HOOK-FIXTURE-3")
    assert also.returncode == 0, also.stderr
    assert f"other open, agent-unbound receipt(s): {ASSIGNMENT_ID}" in also.stderr
    assert "no --reason was given" in also.stderr
    assert version


# ---------------------------------------------------------------------------
# Every fixture above this line runs inside a single hardcoded RUN_ID, so the
# one shape none of them can express is an agent legitimately reused across two
# DIFFERENT runs. That matters now: the guards D-065 obligation 1 rests on are
# all keyed per-run -- the mint scans `admissions/<run>/`, the Stop-time ledger
# check reads `runs/<run>/ADMISSIONS.jsonl`, and the consume lock is
# per-(run, agent_id) -- so an over-broad version of any of them would look
# correct in every other fixture here and wedge the second run of a session.
# ---------------------------------------------------------------------------

CROSS_RUN_ID = "run-fixture-cross"


def _register_assignment_in_run(
    repo: Path, run_id: str, assignment_id: str = ASSIGNMENT_ID
) -> dict[str, object]:
    """Clone the fixture assignment into another run directory.

    `add_second_assignment` gives a sibling inside RUN_ID; this gives the same
    assignment in a different run, which is the axis the module-level RUN_ID
    constant makes otherwise unreachable.
    """
    source = repo / f".agent-harness/runs/{RUN_ID}/assignments/{ASSIGNMENT_ID}.json"
    assignment = json.loads(source.read_text(encoding="utf-8"))
    result_rel = f".agent-harness/runs/{run_id}/results/{assignment_id}.json"
    assignment_rel = f".agent-harness/runs/{run_id}/assignments/{assignment_id}.json"
    assignment.update(
        {
            "run_id": run_id,
            "assignment_id": assignment_id,
            "result_path": result_rel,
            "required_outputs": [result_rel],
            "required_inputs": [
                ".agent-harness/generated/CONTEXT_PACK.md",
                assignment_rel,
            ],
        }
    )
    write_json(repo / assignment_rel, assignment)
    return assignment


def _run_assignment_sha256(
    repo: Path, run_id: str, assignment_id: str = ASSIGNMENT_ID
) -> str:
    path = repo / f".agent-harness/runs/{run_id}/assignments/{assignment_id}.json"
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_result_in_run(
    repo: Path,
    run_id: str,
    version: str,
    assignment_id: str = ASSIGNMENT_ID,
    agent_id: str = "agent-fixture",
) -> Path:
    """`valid_result` re-aimed at another run, every run-bearing field included."""
    digest = _run_assignment_sha256(repo, run_id, assignment_id)
    result = valid_result(version, digest, assignment_id, agent_id)
    result_rel = f".agent-harness/runs/{run_id}/results/{assignment_id}.json"
    result["run_id"] = run_id
    result["result_path"] = result_rel
    result["files_written"] = [result_rel]
    result["findings"][0]["evidence_refs"] = [result_rel]
    result["spawn_contract"]["run_id"] = run_id
    result["spawn_contract"]["assignment_sha256"] = digest
    write_json(repo / result_rel, result)
    return repo / result_rel


def _write_lease_in_run(
    repo: Path,
    run_id: str,
    version: str,
    agent_id: str = "agent-fixture",
    assignment_ids: tuple[str, ...] = (ASSIGNMENT_ID,),
) -> Path:
    lease = repo / ".agent-harness" / "leases" / f"{agent_id}.json"
    write_json(
        lease,
        {
            "schema_version": 1,
            "agent_id": agent_id,
            "agent_type": "context_mapper",
            "run_id": run_id,
            "context_version": version,
            "created_at": "2026-07-28T00:00:00+00:00",
            "assignment_digests": {
                aid: _run_assignment_sha256(repo, run_id, aid)
                for aid in assignment_ids
            },
        },
    )
    return lease


def _stop_event_in_run(
    run_id: str,
    version: str,
    assignment_id: str = ASSIGNMENT_ID,
    proof: str = ADMISSION_TOKEN,
    agent_id: str = "agent-fixture",
) -> dict[str, object]:
    marker = {
        "assignment_id": assignment_id,
        "context_version": version,
        "status": "pass",
        "result_path": f".agent-harness/runs/{run_id}/results/{assignment_id}.json",
        "admission_proof": proof,
    }
    return {
        "hook_event_name": "SubagentStop",
        "stop_hook_active": False,
        "agent_id": agent_id,
        "agent_type": "context_mapper",
        "last_assistant_message": "HARNESS_RESULT: " + json.dumps(marker),
    }


def test_one_agent_bound_in_one_run_is_still_admissible_in_another(
    tmp_path: Path,
) -> None:
    """One agent_id, one assignment PER RUN -- not one for its whole existence.

    The obligation is scoped to a run, and reusing an agent id across runs is
    ordinary: a session initialises a second run and spawns the same named agent
    again. Nothing above this line can catch a guard that dropped the run from
    its key, because every other fixture lives in the one hardcoded RUN_ID. The
    second run deliberately uses a DIFFERENT assignment id, so both run-scoped
    guards are actually reached: a mint-time scan that swept every run's
    receipts, and a Stop-time ledger read that swept every run's rows, would each
    see the first binding and refuse. Both consumes go through the real mint and
    the real Stop hook.
    """

    version, _ = make_harness(tmp_path)
    _git_init_bare(tmp_path)

    first_token = _mint(tmp_path, ASSIGNMENT_ID, "agent-roaming")
    _write_result_in_run(tmp_path, RUN_ID, version, agent_id="agent-roaming")
    _write_lease_in_run(tmp_path, RUN_ID, version, agent_id="agent-roaming")
    first_stop = run_stop_hook(
        tmp_path,
        _stop_event_in_run(
            RUN_ID, version, proof=first_token, agent_id="agent-roaming"
        ),
    )
    assert first_stop.returncode == 0, first_stop.stderr
    assert first_stop.stdout == "", first_stop.stdout

    # A second run, same agent id, different assignment.
    _register_assignment_in_run(tmp_path, CROSS_RUN_ID, SECOND_ASSIGNMENT_ID)
    (tmp_path / ".agent-harness/ACTIVE_RUN").write_text(
        CROSS_RUN_ID + "\n", encoding="utf-8"
    )
    second_mint = _admit(tmp_path, SECOND_ASSIGNMENT_ID, "agent-roaming")
    assert second_mint.returncode == 0, (
        "an agent bound in a previous run must still be mintable in a new one: "
        + second_mint.stderr
    )
    second_token = _token_of(second_mint)
    _write_result_in_run(
        tmp_path,
        CROSS_RUN_ID,
        version,
        assignment_id=SECOND_ASSIGNMENT_ID,
        agent_id="agent-roaming",
    )
    _write_lease_in_run(
        tmp_path,
        CROSS_RUN_ID,
        version,
        agent_id="agent-roaming",
        assignment_ids=(SECOND_ASSIGNMENT_ID,),
    )
    second_stop = run_stop_hook(
        tmp_path,
        _stop_event_in_run(
            CROSS_RUN_ID,
            version,
            assignment_id=SECOND_ASSIGNMENT_ID,
            proof=second_token,
            agent_id="agent-roaming",
        ),
    )
    assert second_stop.returncode == 0, second_stop.stderr
    assert second_stop.stdout == "", second_stop.stdout

    for run_id, assignment_id in (
        (RUN_ID, ASSIGNMENT_ID),
        (CROSS_RUN_ID, SECOND_ASSIGNMENT_ID),
    ):
        rows = [
            json.loads(line)
            for line in (
                tmp_path / f".agent-harness/runs/{run_id}/ADMISSIONS.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert [row["event"] for row in rows] == ["minted", "consumed"], (run_id, rows)
        assert rows[1]["agent_id"] == "agent-roaming", (run_id, rows)
        assert rows[1]["run_id"] == run_id, (run_id, rows)
        assert rows[1]["assignment_id"] == assignment_id, (run_id, rows)
        # Each run's detector sees one agent on one assignment, as it should.
        assert validate_harness.check_one_agent_one_assignment(run_id, rows) == []
    # Two runs, two lock files: the lock is keyed on (run, agent_id), so the
    # first run's lock cannot exclude the second run's stop.
    for run_id in (RUN_ID, CROSS_RUN_ID):
        assert _agent_lock_path(tmp_path, "agent-roaming", run_id).is_file(), run_id

    # ...and the per-run ceiling still bites INSIDE the second run.
    _register_assignment_in_run(tmp_path, CROSS_RUN_ID, "A-HOOK-FIXTURE-3")
    refused = _admit(tmp_path, "A-HOOK-FIXTURE-3", "agent-roaming")
    assert refused.returncode != 0, refused.stdout
    assert "already holds an open or consumed admission receipt" in refused.stderr
    assert repr(SECOND_ASSIGNMENT_ID) in refused.stderr, refused.stderr
    assert repr(CROSS_RUN_ID) in refused.stderr, refused.stderr


# ---------------------------------------------------------------------------
# D-070 round-6 finding F-R6-02: `--reopen` used to restamp `created_at` with
# `utc_now()` unconditionally, so one documented, non-forged reopen of a
# never-consumed receipt reset the in-flight clock on the same never-verified
# bytes, indefinitely -- the ceiling was not a bound. admit_agent.py now asks
# the append-only ledger the same question validate_harness asks before it
# stamps anything.
# ---------------------------------------------------------------------------


def _backdate_open_chain(
    tmp_path: Path, assignment_id: str, hours: float, run_id: str = RUN_ID
) -> str:
    """Back-date the row that opened this assignment's currently-open chain.

    Walks the rows the way `open_chain_started_at` and
    `open_admission_chain_start` both do -- a consume closes the chain, the
    first mint/reopen after that opens it -- so the fixture moves exactly the
    timestamp those functions read, and no other. Back-dating rather than
    waiting is not a shortcut: it is the same surface either function measures,
    and it is the surface an operator would have to forge.
    """
    stamp = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(
        timespec="seconds"
    )
    ledger = tmp_path / f".agent-harness/runs/{run_id}/ADMISSIONS.jsonl"
    rows = [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    opened_at: int | None = None
    for position, row in enumerate(rows):
        if str(row.get("assignment_id")) != assignment_id:
            continue
        if row.get("event") == "consumed":
            opened_at = None
        elif row.get("event") in ("minted", "reopened") and opened_at is None:
            opened_at = position
    assert opened_at is not None, "no open admission chain to back-date"
    rows[opened_at]["at"] = stamp
    rows[opened_at]["chain_started_at"] = stamp
    ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return stamp


def _minted_and_admitted_run(tmp_path: Path) -> str:
    """`_admitted_active_run`, but through the real mint so the ledger has a mint row.

    `_admitted_active_run` writes the receipt directly with `write_admission`,
    which is deliberate there -- the Stop-side checks have to stand on their own
    for receipts nobody minted -- but it leaves the ledger with a lone `consumed`
    row and no `minted` one, so nothing about mint-time chain bookkeeping can be
    asserted against it.
    """
    version, _ = make_harness(tmp_path)
    _git_init(tmp_path)
    _minimal_run_plan(tmp_path, version)
    token = _mint(tmp_path, ASSIGNMENT_ID, "agent-fixture")
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version)
    accepted = run_stop_hook(tmp_path, stop_event(version, proof=token))
    assert accepted.stdout == "", accepted.stdout
    return version


def test_reopening_a_never_consumed_receipt_carries_the_chain_start_forward(
    tmp_path: Path,
) -> None:
    """The receipt must stop claiming an age the append-only ledger disagrees with.

    `created_at` was `now` on every mint, so a reopened-but-never-consumed
    receipt read as freshly minted to any human opening the file, and to the
    validator's receipt-side bound. Both timestamps now carry the ledger's true
    chain start, and the difference is named loudly on stderr every time.
    """

    version, _ = make_harness(tmp_path)
    _git_init_bare(tmp_path)
    assert _admit(tmp_path, ASSIGNMENT_ID, "agent-fixture").returncode == 0
    chain_start = _backdate_open_chain(tmp_path, ASSIGNMENT_ID, 30)

    reopened = _admit(
        tmp_path,
        ASSIGNMENT_ID,
        "agent-fixture",
        "--reopen",
        "--reason",
        "first agent died before stopping",
    )
    assert reopened.returncode == 0, reopened.stderr

    receipt = json.loads(
        _receipt_path(tmp_path, ASSIGNMENT_ID).read_text(encoding="utf-8")
    )
    assert receipt["created_at"] == chain_start, receipt
    assert receipt["chain_started_at"] == chain_start, receipt
    assert receipt["state"] == "open"

    row = _ledger_rows(tmp_path)[-1]
    assert row["event"] == "reopened"
    assert row["chain_started_at"] == chain_start, row
    assert row["at"] != chain_start, (
        "the reopen still records WHEN it happened; only the chain start is "
        "carried forward"
    )
    assert receipt["created_at"] != row["at"], (
        "created_at must not be restamped to the moment of the reopen"
    )

    # The warning is the check that cannot be silenced, so it has to name both
    # the true start and how long it has been open.
    assert "reopening a receipt that was never consumed" in reopened.stderr
    assert chain_start in reopened.stderr, reopened.stderr
    assert "(30.0h)" in reopened.stderr, reopened.stderr
    assert "does not restart that clock" in reopened.stderr
    assert "WARNING" not in reopened.stdout
    assert version


def test_chain_started_at_is_recorded_on_minted_and_reopened_rows(
    tmp_path: Path,
) -> None:
    """The field is on every mint row, and a consume legitimately restarts it.

    A reopen after a genuine consume supersedes bytes that really were
    admitted, which is a fresh admission: its clock starts over. Only a reopen
    of a chain no consume has closed carries the old start forward. Both shapes
    have to be visible on the row itself, or the intent is only recoverable by
    replaying the whole ledger.
    """

    version = _minted_and_admitted_run(tmp_path)
    rows = _ledger_rows(tmp_path)
    assert [row["event"] for row in rows] == ["minted", "consumed"]
    assert rows[0]["chain_started_at"] == rows[0]["at"], (
        "a first mint opens its own chain"
    )

    _edit_admitted_result(tmp_path, version)
    reopened = _admit(
        tmp_path,
        ASSIGNMENT_ID,
        "agent-fixture",
        "--reopen",
        "--reason",
        "superseding the admitted bytes",
    )
    assert reopened.returncode == 0, reopened.stderr
    row = _ledger_rows(tmp_path)[-1]
    assert row["event"] == "reopened"
    assert row["chain_started_at"] == row["at"], (
        "the consume closed the previous chain, so this reopen opens a new one"
    )
    receipt = json.loads(
        _receipt_path(tmp_path, ASSIGNMENT_ID).read_text(encoding="utf-8")
    )
    assert receipt["chain_started_at"] == row["at"]
    assert receipt["created_at"] == row["at"]
    # ...and no never-consumed warning, because this reopen is not that shape.
    assert "reopening a receipt that was never consumed" not in reopened.stderr


def test_reopen_of_a_never_consumed_receipt_declares_no_supersession(
    tmp_path: Path,
) -> None:
    """An empty declaration must be distinguishable from a real one.

    `superseded_result_sha256: ""` alone cannot say whether nothing was ever
    admitted or whether the lookup merely defaulted, so
    `supersedes_admitted_result` states which, explicitly.
    """

    version, _ = make_harness(tmp_path)
    _git_init_bare(tmp_path)
    _minimal_run_plan(tmp_path, version)
    assert _admit(tmp_path, ASSIGNMENT_ID, "agent-fixture").returncode == 0

    first_reopen = _admit(
        tmp_path,
        ASSIGNMENT_ID,
        "agent-fixture",
        "--reopen",
        "--reason",
        "nothing was ever admitted for this assignment",
    )
    assert first_reopen.returncode == 0, first_reopen.stderr
    row = _ledger_rows(tmp_path)[-1]
    assert row["event"] == "reopened"
    assert row["supersedes_admitted_result"] is False, row
    assert row["superseded_result_sha256"] == "", row
    assert row["superseded_state"] == "open", row

    # Now admit something for real, and reopen again.
    token = _token_of(first_reopen)
    result_file = tmp_path / RESULT_PATH
    write_json(result_file, valid_result(version, assignment_sha256(tmp_path)))
    write_lease(tmp_path, version)
    assert run_stop_hook(tmp_path, stop_event(version, proof=token)).stdout == ""
    admitted_sha = "sha256:" + hashlib.sha256(result_file.read_bytes()).hexdigest()

    second_reopen = _admit(
        tmp_path,
        ASSIGNMENT_ID,
        "agent-fixture",
        "--reopen",
        "--reason",
        "superseding bytes that really were admitted",
    )
    assert second_reopen.returncode == 0, second_reopen.stderr
    row = _ledger_rows(tmp_path)[-1]
    assert row["supersedes_admitted_result"] is True, row
    assert row["superseded_result_sha256"] == admitted_sha, row
    assert row["superseded_state"] == "consumed", row
    assert row["superseded_consumed_by_agent_id"] == "agent-fixture", row


def test_nested_reopen_separates_chain_age_from_supersession(
    tmp_path: Path,
) -> None:
    """`chain_started_at` and `supersedes_admitted_result` answer different questions.

    Consumed, then reopened without being consumed, then reopened again. The
    chain start moves once (the consume closed the old chain) and then stops
    moving (the second reopen re-mints an admission nothing ever consumed),
    while the superseded digest is the same admitted digest throughout: the
    last bytes EVER admitted, across every past chain. Collapsing the two into
    one field -- in either direction -- is wrong for one of these three rows.
    """

    version = _admitted_active_run(tmp_path)
    admitted_sha = _ledger_rows(tmp_path)[-1]["result_sha256"]

    first = _admit(
        tmp_path, ASSIGNMENT_ID, "agent-fixture", "--reopen", "--reason", "reopen one"
    )
    assert first.returncode == 0, first.stderr
    first_row = _ledger_rows(tmp_path)[-1]
    assert first_row["chain_started_at"] == first_row["at"], first_row
    assert first_row["supersedes_admitted_result"] is True, first_row
    assert first_row["superseded_result_sha256"] == admitted_sha, first_row

    # Back-date the chain this reopen opened, so the carry-forward is provable
    # rather than being hidden by two mints inside the same second.
    chain_start = _backdate_open_chain(tmp_path, ASSIGNMENT_ID, 6)
    second = _admit(
        tmp_path, ASSIGNMENT_ID, "agent-fixture", "--reopen", "--reason", "reopen two"
    )
    assert second.returncode == 0, second.stderr
    second_row = _ledger_rows(tmp_path)[-1]

    # The chain age question: this reopen did NOT start a new chain.
    assert second_row["chain_started_at"] == chain_start, second_row
    assert second_row["at"] != chain_start, second_row
    assert "(6.0h)" in second.stderr, second.stderr
    # The supersession question: unchanged, because no second consume happened.
    assert second_row["supersedes_admitted_result"] is True, second_row
    assert second_row["superseded_result_sha256"] == admitted_sha, second_row
    assert second_row["superseded_state"] == "open", second_row
    # ...and the two really are different answers on the same row.
    assert second_row["chain_started_at"] != first_row["chain_started_at"]
    assert (
        second_row["superseded_result_sha256"]
        == first_row["superseded_result_sha256"]
    )
    assert version


def test_backdated_admission_chain_refuses_the_carve_out_end_to_end(
    tmp_path: Path,
) -> None:
    """The whole loop: reopen an old chain, write a result, get refused.

    No forgery anywhere -- `admit_agent.py --reopen --reason '...'` is this
    module's own documented recovery for a dead agent. Before F-R6-02 the
    reopen restamped `created_at` and the artifact was excused for another
    24 hours, repeatable for ever. The refusal now names the RECEIPT-side age,
    because the receipt itself carries the carried-forward chain start.
    """

    version, _ = make_harness(tmp_path)
    _git_init_bare(tmp_path)
    _minimal_run_plan(tmp_path, version)
    assert _admit(tmp_path, ASSIGNMENT_ID, "agent-fixture").returncode == 0
    _backdate_open_chain(tmp_path, ASSIGNMENT_ID, 30)
    reopened = _admit(
        tmp_path,
        ASSIGNMENT_ID,
        "agent-fixture",
        "--reopen",
        "--reason",
        "the agent is taking its time",
    )
    assert reopened.returncode == 0, reopened.stderr
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )

    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    refusal = [
        error for error in payload["errors"] if "carve-out is refused" in error
    ]
    assert refusal, payload["errors"]
    assert "open admission receipt has been open for 30.0h" in refusal[0], refusal[0]
    assert "24h in-flight ceiling" in refusal[0], refusal[0]
    assert f"{RUN_ID}/{ASSIGNMENT_ID}" in refusal[0], refusal[0]


# ---------------------------------------------------------------------------
# D-065 obligation 2, round-6 correction. Start used to have two states -- a
# receipt for this agent_id, or none -- on the argument that a failed mint and
# a legitimately unregistered helper agent were indistinguishable from inside
# the hook. They are not: `admit_agent.py` writes the ledger BEFORE the
# receipt, so a `minted` row naming this agent_id with no receipt on disk is a
# failed mint, and Start can read it. The rule now has a third state.
# ---------------------------------------------------------------------------


def test_subagentstart_hard_fails_on_an_orphaned_mint(
    tmp_path: Path, capsys
) -> None:
    """The obligation-2 case that used to report `none found` / PASS.

    Driven through the real `admit_agent.py` against a `chmod 500` admissions
    directory rather than by hand-writing a ledger row, because the whole point
    of the round-6 correction is that this state is REACHABLE: the ledger
    append lands under `runs/`, the receipt write under `admissions/`, and one
    can fail while the other succeeds.
    """

    version, _ = make_harness(tmp_path)
    _git_init_bare(tmp_path)
    minted = _mint_with_unwritable_receipt_dir(tmp_path)
    assert minted.returncode != 0, minted.stdout
    assert not _receipt_path(tmp_path, ASSIGNMENT_ID).exists()

    context, output = run_start_hook(tmp_path, capsys)
    admission = bootstrap_line(context, "Admission receipt: ")
    assert "Admission receipt: MINT FAILED" in admission, admission
    assert repr("agent-fixture") in admission, admission
    assert ASSIGNMENT_ID in admission, admission
    assert repr(RUN_ID) in admission, admission
    preflight = bootstrap_line(context, "Hook preflight: ")
    assert preflight.startswith("Hook preflight: FAIL"), preflight
    assert "did not produce a usable receipt" in preflight, preflight
    assert output["systemMessage"] == START_FAIL_MESSAGE
    # The lease succeeded, so the FAIL is carried by the receipt check alone.
    assert f"Run lease: recorded for run {RUN_ID}" in context
    assert version


def test_subagentstart_passes_for_an_unregistered_agent_despite_ledger_activity(
    tmp_path: Path, capsys
) -> None:
    """The false-positive control that decides whether the third state is usable.

    Most spawned agents hold no registered assignment. If a busy ledger were
    enough to fail them, the common case would become indistinguishable from
    the dangerous one and the whole check would be turned off within a day.
    Only a mint naming THIS agent_id counts.
    """

    version = _minted_and_admitted_run(tmp_path)
    add_second_assignment(tmp_path, SECOND_ASSIGNMENT_ID)
    assert _admit(tmp_path, SECOND_ASSIGNMENT_ID, "agent-two").returncode == 0
    rows = _ledger_rows(tmp_path)
    assert {row["event"] for row in rows} == {"minted", "consumed"}, rows

    context, output = run_start_hook(tmp_path, capsys, agent_id="agent-helper")
    admission = bootstrap_line(context, "Admission receipt: ")
    assert "Admission receipt: none found" in admission, admission
    assert repr("agent-helper") in admission, admission
    assert "Hook preflight: PASS" in context
    assert "systemMessage" not in output
    assert version


def test_subagentstart_does_not_resurrect_a_consumed_mint(
    tmp_path: Path, capsys
) -> None:
    """A `consumed` row proves a receipt existed, so its absence is not a failure.

    Stop only appends `consumed` after claiming and validating an on-disk
    admission file. Reading a missing receipt back as a failed mint would turn
    every completed admission into a phantom Start failure the moment the
    gitignored working state was cleaned up.
    """

    _minted_and_admitted_run(tmp_path)
    receipt = _receipt_path(tmp_path, ASSIGNMENT_ID)
    receipt.with_name(receipt.name + ".claim").unlink()
    receipt.unlink()

    context, output = run_start_hook(tmp_path, capsys)
    assert "Admission receipt: none found" in bootstrap_line(
        context, "Admission receipt: "
    )
    assert "Hook preflight: PASS" in context
    assert "systemMessage" not in output

    # The control that makes the assertion above mean something: with the
    # consume row dropped, the very same tree IS a failed mint.
    _write_ledger_rows(
        tmp_path, [row for row in _ledger_rows(tmp_path) if row["event"] != "consumed"]
    )
    context, output = run_start_hook(tmp_path, capsys)
    assert "Admission receipt: MINT FAILED" in bootstrap_line(
        context, "Admission receipt: "
    )
    assert output["systemMessage"] == START_FAIL_MESSAGE


def test_subagentstart_lets_a_reopen_supersede_a_stale_mint_for_the_old_agent(
    tmp_path: Path, capsys
) -> None:
    """A later `reopened` row is a more specific statement about the assignment.

    Reopening for a different agent leaves the old agent's `minted` row in the
    append-only ledger for ever. If that row still counted, the first agent
    would fail Start permanently for a receipt that was deliberately reassigned.
    """

    version, _ = make_harness(tmp_path)
    _git_init_bare(tmp_path)
    assert _admit(tmp_path, ASSIGNMENT_ID, "agent-one").returncode == 0
    reopened = _admit(
        tmp_path,
        ASSIGNMENT_ID,
        "agent-two",
        "--reopen",
        "--reason",
        "reassigned to a second agent",
    )
    assert reopened.returncode == 0, reopened.stderr
    events = [(row["event"], row["expected_agent_id"]) for row in _ledger_rows(tmp_path)]
    assert events == [("minted", "agent-one"), ("reopened", "agent-two")], events

    context, output = run_start_hook(tmp_path, capsys, agent_id="agent-one")
    assert "Admission receipt: none found" in bootstrap_line(
        context, "Admission receipt: "
    ), context
    assert "Hook preflight: PASS" in context
    assert "systemMessage" not in output

    context, output = run_start_hook(tmp_path, capsys, agent_id="agent-two")
    assert "Admission receipt: open, bound to assignment" in bootstrap_line(
        context, "Admission receipt: "
    )
    assert "Hook preflight: PASS" in context
    assert "systemMessage" not in output
    assert version


def test_subagentstart_flags_the_new_agent_when_a_reopen_fails(
    tmp_path: Path, capsys
) -> None:
    """A failed reopen is a failed mint for the agent it was reopened FOR.

    The ledger row lands, the receipt write does not, and the previous agent's
    receipt is still sitting on disk -- so the directory scan finds a perfectly
    healthy receipt that simply names somebody else. Start must fail the agent
    the reopen was for, and say why.
    """

    version, _ = make_harness(tmp_path)
    _git_init_bare(tmp_path)
    assert _admit(tmp_path, ASSIGNMENT_ID, "agent-one").returncode == 0
    stale = json.loads(
        _receipt_path(tmp_path, ASSIGNMENT_ID).read_text(encoding="utf-8")
    )

    failed = _mint_with_unwritable_receipt_dir(
        tmp_path, "agent-two", "--reopen", "--reason", "reassigning after a hang"
    )
    assert failed.returncode != 0, failed.stdout
    assert "admission receipt could not be written" in failed.stderr
    surviving = json.loads(
        _receipt_path(tmp_path, ASSIGNMENT_ID).read_text(encoding="utf-8")
    )
    assert surviving == stale, "the failed reopen must not have half-written a receipt"

    context, output = run_start_hook(tmp_path, capsys, agent_id="agent-two")
    admission = bootstrap_line(context, "Admission receipt: ")
    assert "Admission receipt: MINT FAILED" in admission, admission
    assert repr("agent-two") in admission, admission
    assert ASSIGNMENT_ID in admission, admission
    preflight = bootstrap_line(context, "Hook preflight: ")
    assert preflight.startswith("Hook preflight: FAIL"), preflight
    assert "overwritten by a failed reopen" in preflight, preflight
    assert output["systemMessage"] == START_FAIL_MESSAGE

    # Honest about the limit: agent-one's receipt is still open on disk, so
    # Start reports it as usable. Only Stop can refuse the token itself.
    context, output = run_start_hook(tmp_path, capsys, agent_id="agent-one")
    assert "Admission receipt: open, bound to assignment" in bootstrap_line(
        context, "Admission receipt: "
    )
    assert "Hook preflight: PASS" in context
    assert version


def test_subagentstart_fails_on_a_malformed_admission_ledger_line(
    tmp_path: Path, capsys
) -> None:
    """An unparseable ledger line cannot be proven irrelevant to this agent_id.

    The same unprovable-negative rule the receipt scan already applies. Dropping
    the line silently would make garbling one row the way to hide a failed mint.
    """

    version, _ = make_harness(tmp_path)
    ledger = tmp_path / f".agent-harness/runs/{RUN_ID}/ADMISSIONS.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("{not json\n", encoding="utf-8")

    context, output = run_start_hook(tmp_path, capsys)
    admission = bootstrap_line(context, "Admission receipt: ")
    assert "Admission receipt: AMBIGUOUS" in admission, admission
    assert "line 1 is malformed" in admission, admission
    preflight = bootstrap_line(context, "Hook preflight: ")
    assert preflight.startswith("Hook preflight: FAIL"), preflight
    assert "cannot be ruled out" in preflight, preflight
    assert output["systemMessage"] == START_FAIL_MESSAGE
    assert ledger.read_text(encoding="utf-8") == "{not json\n", (
        "malformed ledger evidence must be preserved"
    )

    # A non-object row is refused on the same rule, not only invalid JSON.
    ledger.write_text("[1, 2, 3]\n", encoding="utf-8")
    context, output = run_start_hook(tmp_path, capsys)
    assert "line 1 is not an object" in bootstrap_line(
        context, "Admission receipt: "
    )
    assert output["systemMessage"] == START_FAIL_MESSAGE
    assert version


# ---------------------------------------------------------------------------
# The after-the-fact detectors. A live guard that is raced or bypassed leaves no
# trace unless something reads the append-only record afterwards, so
# `check_one_agent_one_assignment` is exercised on row sequences directly:
# every arm below is a shape the ledger can genuinely hold, and the clean ones
# decide whether the detector is usable at all.
# ---------------------------------------------------------------------------

LEDGER_STAMP = "2026-07-28T00:00:00+00:00"


def _consume_row(
    agent_id: str, assignment_id: str, sha: str = "sha256:" + "1" * 64
) -> dict:
    return {
        "event": "consumed",
        "run_id": RUN_ID,
        "assignment_id": assignment_id,
        "agent_id": agent_id,
        "result_sha256": sha,
        "at": LEDGER_STAMP,
    }


def _reopen_row(
    assignment_id: str, released_agent: str = "", reason: str = ""
) -> dict:
    return {
        "event": "reopened",
        "run_id": RUN_ID,
        "assignment_id": assignment_id,
        "superseded_consumed_by_agent_id": released_agent,
        "reason": reason,
        "at": LEDGER_STAMP,
    }


def test_check_one_agent_one_assignment_over_ledger_row_sequences() -> None:
    """One agent_id binds to one assignment per run, released only by declaration.

    The five flagged shapes are the race itself plus the four ways a reopen can
    look like a release without being one; the three clean shapes are what stop
    the detector from failing ordinary runs. A reopen releases a binding only
    when it names the assignment that agent holds, names the agent, and carries
    a reason -- the same declared-supersession gesture
    `check_declared_supersession` requires, read for a different question.
    """

    second = SECOND_ASSIGNMENT_ID
    flagged: list[tuple[str, list[dict], str]] = [
        (
            "the race: one agent consumes two assignments",
            [_consume_row("agent-one", ASSIGNMENT_ID), _consume_row("agent-one", second)],
            "consuming two",
        ),
        (
            "a reopen with no reason releases nothing",
            [
                _consume_row("agent-one", ASSIGNMENT_ID),
                _reopen_row(ASSIGNMENT_ID, released_agent="agent-one", reason="   "),
                _consume_row("agent-one", second),
            ],
            "consuming two",
        ),
        (
            "a reopen naming another agent releases nothing",
            [
                _consume_row("agent-one", ASSIGNMENT_ID),
                _reopen_row(ASSIGNMENT_ID, released_agent="agent-two", reason="why"),
                _consume_row("agent-one", second),
            ],
            "consuming two",
        ),
        (
            "a reopen of another assignment releases nothing",
            [
                _consume_row("agent-one", ASSIGNMENT_ID),
                _reopen_row(second, released_agent="agent-one", reason="why"),
                _consume_row("agent-one", second),
            ],
            "consuming two",
        ),
        (
            "a consume with no agent_id is unattributable",
            [_consume_row("", ASSIGNMENT_ID)],
            "consume with no agent_id",
        ),
    ]
    for label, rows, expected in flagged:
        errors = validate_harness.check_one_agent_one_assignment(RUN_ID, rows)
        assert errors, label
        assert any(expected in error for error in errors), (label, errors)
        assert any(RUN_ID in error for error in errors), (label, errors)

    # The race error has to name the operator's way out, or it is a dead end.
    race = validate_harness.check_one_agent_one_assignment(
        RUN_ID,
        [_consume_row("agent-one", ASSIGNMENT_ID), _consume_row("agent-one", second)],
    )
    assert len(race) == 1, race
    assert "'agent-one'" in race[0], race[0]
    assert ASSIGNMENT_ID in race[0] and second in race[0], race[0]
    assert "D-065 obligation 1" in race[0], race[0]
    assert "--reopen --reason" in race[0], race[0]

    clean: list[tuple[str, list[dict]]] = [
        (
            "two distinct agents on two assignments",
            [_consume_row("agent-one", ASSIGNMENT_ID), _consume_row("agent-two", second)],
        ),
        (
            "a declared release rebinds the same agent",
            [
                _consume_row("agent-one", ASSIGNMENT_ID),
                _reopen_row(
                    ASSIGNMENT_ID, released_agent="agent-one", reason="agent died"
                ),
                _consume_row("agent-one", second),
            ],
        ),
        (
            "the same agent re-consuming the same assignment",
            [
                _consume_row("agent-one", ASSIGNMENT_ID),
                _consume_row("agent-one", ASSIGNMENT_ID),
            ],
        ),
    ]
    for label, rows in clean:
        assert validate_harness.check_one_agent_one_assignment(RUN_ID, rows) == [], label


def test_reopen_does_not_restart_the_in_flight_ceiling(tmp_path: Path) -> None:
    """The ledger bound alone must refuse a chain the receipt claims is fresh.

    F-R6-02 has two surfaces and they are load-bearing in opposite directions.
    Here the receipt is stamped `now` -- exactly what `admit_agent.py` did
    before the fix, and what an operator gets by rewriting the gitignored
    working-state file -- so the receipt-side bound passes and only the
    append-only chain start can refuse. Its twin,
    `test_backdated_admission_chain_refuses_the_carve_out_end_to_end`, asserts
    the receipt half with the ledger untouched.
    """

    version, _ = make_harness(tmp_path)
    _git_init_bare(tmp_path)
    _minimal_run_plan(tmp_path, version)
    assert _admit(tmp_path, ASSIGNMENT_ID, "agent-fixture").returncode == 0
    chain_start = _backdate_open_chain(tmp_path, ASSIGNMENT_ID, 30)
    assert _admit(
        tmp_path,
        ASSIGNMENT_ID,
        "agent-fixture",
        "--reopen",
        "--reason",
        "still working on it",
    ).returncode == 0
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )

    receipt_path = _receipt_path(tmp_path, ASSIGNMENT_ID)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    receipt["chain_started_at"] = receipt["created_at"]
    write_json(receipt_path, receipt)

    payload = _validate(tmp_path)
    assert payload["ok"] is False, payload
    refusal = [
        error for error in payload["errors"] if "carve-out is refused" in error
    ]
    assert refusal, payload["errors"]
    assert "its admission chain has been open since" in refusal[0], refusal[0]
    assert chain_start in refusal[0], refusal[0]
    assert "30.0h" in refusal[0], refusal[0]
    assert "not from the receipt's created_at" in refusal[0], refusal[0]
    assert "open admission receipt has been open for" not in refusal[0], (
        "the receipt-side bound must have passed, so this refusal is the "
        "ledger's alone"
    )


def test_pending_carve_out_is_refused_without_a_readable_ledger(
    tmp_path: Path,
) -> None:
    """The in-flight ceiling is measured from a record the holder cannot refresh.

    With that record unreadable the age cannot be established at all, and an
    unestablished age is not a young one -- otherwise garbling the ledger would
    excuse every open receipt in the run indefinitely.
    """

    _in_flight_second_agent(tmp_path)
    assert _validate(tmp_path)["ok"] is True  # control: the same tree, readable

    ledger = tmp_path / f".agent-harness/runs/{RUN_ID}/ADMISSIONS.jsonl"
    ledger.chmod(0o000)
    try:
        payload = _validate(tmp_path)
    finally:
        ledger.chmod(0o600)
    assert payload["ok"] is False, payload
    assert any(
        "Admission ledger is unreadable" in error for error in payload["errors"]
    ), payload["errors"]
    refusal = [
        error
        for error in payload["errors"]
        if "carve-out is refused" in error and SECOND_ASSIGNMENT_ID in error
    ]
    assert refusal, payload["errors"]
    assert "no readable admission ledger" in refusal[0], refusal[0]
    assert "in-flight ceiling is measured from" in refusal[0], refusal[0]
    assert _validate(tmp_path)["ok"] is True, "the refusal must lapse with the cause"


# ---------------------------------------------------------------------------
# D-070 round-6 review, finding 2: merge_results.py globbed `results/*.json`
# and reduced whatever it found, with no reference to admissions, receipts, the
# ledger or the legacy manifest -- a second, unguarded path to treating a result
# as real, and the one whose output the adjudicator actually reads. It now
# imports the validator's own attribution pass rather than restating the rule.
# ---------------------------------------------------------------------------


def _merge_results(
    tmp_path: Path,
    script: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script or (HARNESS_SCRIPTS / "merge_results.py"))],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        **({"env": env} if env is not None else {}),
    )


def _merged_results_path(tmp_path: Path) -> Path:
    return tmp_path / f".agent-harness/runs/{RUN_ID}/MERGED_RESULTS.json"


def _merge_refusal(completed: subprocess.CompletedProcess) -> list[str]:
    """The refusal payload, insisting it is readable JSON rather than a crash."""
    assert completed.returncode != 0, completed.stdout
    assert "Traceback" not in completed.stderr, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False, payload
    assert payload["wrote"] is None, payload
    return payload["errors"]


def _copy_harness_scripts(tmp_path: Path, *names: str) -> Path:
    """Copy scripts into the fixture so `root()` cannot fall back to the real repo.

    `_harness.root()` degrades to the script's own `parents[2]` when git is
    unavailable, which for the installed scripts is the LIVE repository. Any
    fixture that removes git therefore has to run a copy, or it validates -- and
    could write to -- the real tree.
    """
    scripts = tmp_path / ".agent-harness" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for name in names:
        (scripts / name).write_bytes((HARNESS_SCRIPTS / name).read_bytes())
    return scripts


def test_merge_refuses_an_unattributed_result(tmp_path: Path) -> None:
    """The reported defect: unattributed bytes reaching MERGED_RESULTS.json."""

    _admitted_active_run(tmp_path)
    merged = _merge_results(tmp_path)
    assert merged.returncode == 0, merged.stdout + merged.stderr
    baseline = _merged_results_path(tmp_path).read_bytes()
    payload = json.loads(baseline.decode("utf-8"))
    assert payload["result_count"] == 1
    assert payload["attribution"]["results"] == [
        {
            "assignment_id": ASSIGNMENT_ID,
            "sha256": "sha256:"
            + hashlib.sha256((tmp_path / RESULT_PATH).read_bytes()).hexdigest(),
            "attribution": "attributed",
        }
    ], payload["attribution"]

    smuggled = tmp_path / f".agent-harness/runs/{RUN_ID}/results/A-SMUGGLED.json"
    write_json(smuggled, {"schema_version": 2, "status": "pass"})
    errors = _merge_refusal(_merge_results(tmp_path))
    assert any(
        "A-SMUGGLED" in error and "'unattributed'" in error for error in errors
    ), errors
    assert _merged_results_path(tmp_path).read_bytes() == baseline, (
        "a refused merge must leave the previous output untouched, not replace "
        "it with a weaker one"
    )


def test_merge_refuses_a_still_pending_result(tmp_path: Path) -> None:
    """A carve-out that keeps a live run unwedged must not survive into the merge.

    Validation runs continuously during a run, so it has to tolerate the window
    between the artifact being written and SubagentStop consuming the receipt.
    Merging is an operator step whose output feeds adjudication, and the wait is
    one Stop dispatch -- so refusing wedges nothing, which the second half here
    shows by simply letting the agent stop.
    """

    version, token, _ = _in_flight_second_agent(tmp_path)
    assert _validate(tmp_path)["ok"] is True
    assert _validate(tmp_path)["pending_result_count"] == 1

    errors = _merge_refusal(_merge_results(tmp_path))
    assert any(
        SECOND_ASSIGNMENT_ID in error and "'pending'" in error for error in errors
    ), errors
    assert not _merged_results_path(tmp_path).exists()

    write_lease(
        tmp_path,
        version,
        agent_id="agent-fixture-2",
        assignment_ids=(ASSIGNMENT_ID, SECOND_ASSIGNMENT_ID),
    )
    accepted = run_stop_hook(
        tmp_path,
        stop_event(
            version,
            assignment_id=SECOND_ASSIGNMENT_ID,
            proof=token,
            agent_id="agent-fixture-2",
        ),
    )
    assert accepted.stdout == "", accepted.stdout
    merged = _merge_results(tmp_path)
    assert merged.returncode == 0, merged.stdout + merged.stderr
    payload = json.loads(_merged_results_path(tmp_path).read_text(encoding="utf-8"))
    assert payload["result_count"] == 2
    assert sorted(
        entry["assignment_id"] for entry in payload["attribution"]["results"]
    ) == sorted([ASSIGNMENT_ID, SECOND_ASSIGNMENT_ID])
    assert {
        entry["attribution"] for entry in payload["attribution"]["results"]
    } == {"attributed"}


def test_merge_follows_a_declared_supersession(tmp_path: Path) -> None:
    """Superseded bytes must not keep being merged after a declared reopen.

    The merge reads the same attribution pass the validator does, so the digest
    it records has to move with the last consume row -- otherwise the durable
    artifact the adjudicator reads would still describe the replaced bytes.
    """

    version = _admitted_active_run(tmp_path)
    assert _merge_results(tmp_path).returncode == 0
    first = json.loads(_merged_results_path(tmp_path).read_text(encoding="utf-8"))
    first_sha = first["attribution"]["results"][0]["sha256"]

    _edit_admitted_result(tmp_path, version)
    assert _validate(tmp_path)["ok"] is False
    assert _merge_refusal(_merge_results(tmp_path))

    reopened = _admit(
        tmp_path,
        ASSIGNMENT_ID,
        "agent-fixture",
        "--reopen",
        "--reason",
        "first agent died before stopping",
    )
    assert reopened.returncode == 0, reopened.stderr
    accepted = run_stop_hook(
        tmp_path, stop_event(version, proof=_token_of(reopened))
    )
    assert accepted.stdout == "", accepted.stdout
    assert _validate(tmp_path)["ok"] is True

    merged = _merge_results(tmp_path)
    assert merged.returncode == 0, merged.stdout + merged.stderr
    second = json.loads(_merged_results_path(tmp_path).read_text(encoding="utf-8"))
    second_sha = second["attribution"]["results"][0]["sha256"]
    assert second_sha != first_sha
    assert second_sha == "sha256:" + hashlib.sha256(
        (tmp_path / RESULT_PATH).read_bytes()
    ).hexdigest()
    assert second["result_count"] == 1
    assert second["errors"] == [], (
        "the retained `errors` list is empty by construction now; anything that "
        "would land in it refuses the merge instead"
    )


def test_merge_fails_closed_when_git_is_unavailable(tmp_path: Path) -> None:
    """No git means no integrity evidence, and merging is not exempt from that.

    The scripts are copied into the fixture for the reason `_copy_harness_scripts`
    records: without git, `_harness.root()` resolves to the real repository.
    """

    _admitted_active_run(tmp_path)
    assert _merge_results(tmp_path).returncode == 0
    baseline = _merged_results_path(tmp_path).read_bytes()
    scripts = _copy_harness_scripts(
        tmp_path, "merge_results.py", "validate_harness.py", "_harness.py"
    )

    completed = _merge_results(
        tmp_path,
        script=scripts / "merge_results.py",
        env={"PATH": str(tmp_path / "no-such-bin")},
    )
    errors = _merge_refusal(completed)
    assert any("git is required for validation" in error for error in errors), errors
    assert json.loads(completed.stdout)["run_id"] == RUN_ID, (
        "the refusal must be about the fixture run, not the live repository"
    )
    assert _merged_results_path(tmp_path).read_bytes() == baseline


def test_merge_fails_closed_on_a_garbage_admission_ledger(tmp_path: Path) -> None:
    """An unparseable ledger line is what would vouch for these bytes."""

    _admitted_active_run(tmp_path)
    assert _merge_results(tmp_path).returncode == 0
    baseline = _merged_results_path(tmp_path).read_bytes()

    ledger = tmp_path / f".agent-harness/runs/{RUN_ID}/ADMISSIONS.jsonl"
    ledger.write_text(
        ledger.read_text(encoding="utf-8") + "[1, 2, 3]\n", encoding="utf-8"
    )
    errors = _merge_refusal(_merge_results(tmp_path))
    assert any(
        "Admission ledger line is not an object" in error for error in errors
    ), errors
    assert _merged_results_path(tmp_path).read_bytes() == baseline


def test_merge_reports_an_unreadable_result_artifact_instead_of_crashing(
    tmp_path: Path,
) -> None:
    """A crash is fail-closed but unreadable: stdout empty, JSON unparseable.

    Every caller of this script parses its stdout, so an artifact that cannot be
    read has to come out as a refusal payload naming the file, not as a
    traceback (the D-070 F-R5-08 class).
    """

    _admitted_active_run(tmp_path)
    assert _merge_results(tmp_path).returncode == 0
    baseline = _merged_results_path(tmp_path).read_bytes()

    victim = tmp_path / RESULT_PATH
    victim.chmod(0o000)
    try:
        completed = _merge_results(tmp_path)
    finally:
        victim.chmod(0o600)
    errors = _merge_refusal(completed)
    assert any(
        "could not be read (PermissionError)" in error and ASSIGNMENT_ID in error
        for error in errors
    ), errors
    assert any("'unreadable'" in error for error in errors), errors
    assert _merged_results_path(tmp_path).read_bytes() == baseline
    # And the tree merges again once the artifact is readable, so the refusal
    # tracked the cause rather than latching.
    assert _merge_results(tmp_path).returncode == 0
