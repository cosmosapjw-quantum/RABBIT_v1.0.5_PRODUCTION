from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
HOOKS_DIR = REPO / ".codex" / "hooks"
HARNESS_SCRIPTS = REPO / ".agent-harness" / "scripts"
sys.path.insert(0, str(HOOKS_DIR))
sys.path.insert(0, str(HARNESS_SCRIPTS))

import subagent_start_context  # noqa: E402
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


def assignment_sha256(repo: Path) -> str:
    path = (
        repo
        / f".agent-harness/runs/{RUN_ID}/assignments/{ASSIGNMENT_ID}.json"
    )
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def valid_result(version: str, assignment_hash: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "assignment_id": ASSIGNMENT_ID,
        "context_version": version,
        "agent_id": "agent-fixture",
        "agent_type": "context_mapper",
        "spawn_contract": {
            "run_id": RUN_ID,
            "assignment_id": ASSIGNMENT_ID,
            "context_version": version,
            "independence_mode": "shared-core",
            "prompt_header_verified": True,
            "subagent_start_injected": True,
            "subagent_start_preflight": "PASS",
            "assignment_sha256": assignment_hash,
        },
        "status": "pass",
        "result_path": RESULT_PATH,
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
                "evidence_refs": [RESULT_PATH],
                "evidence_fingerprint": "sha256:" + "0" * 64,
                "counterevidence_refs": [],
                "reproduction": ["pytest fixture"],
                "confidence": 1.0,
                "unresolved": [],
            }
        ],
        "errors": [],
        "files_written": [RESULT_PATH],
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
    marker = {
        "assignment_id": ASSIGNMENT_ID,
        "context_version": version,
        "status": "pass",
        "result_path": RESULT_PATH,
    }
    event = {
        "hook_event_name": "SubagentStop",
        "stop_hook_active": True,
        "agent_id": "agent-fixture",
        "agent_type": "context_mapper",
        "last_assistant_message": "HARNESS_RESULT: " + json.dumps(marker),
    }
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
    marker = {
        "assignment_id": ASSIGNMENT_ID,
        "context_version": version,
        "status": "pass",
        "result_path": RESULT_PATH,
    }
    event = {
        "hook_event_name": "SubagentStop",
        "agent_id": "agent-fixture",
        "agent_type": "default",
        "last_assistant_message": "HARNESS_RESULT: " + json.dumps(marker),
    }
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

    assignment_path = (
        tmp_path
        / f".agent-harness/runs/{RUN_ID}/assignments/{ASSIGNMENT_ID}.json"
    )
    invalid_assignment = dict(assignment)
    invalid_assignment["review_role"] = "unregistered_role"
    write_json(assignment_path, invalid_assignment)
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


def write_lease(repo: Path, version: str, agent_id: str = "agent-fixture") -> Path:
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
            "assignment_digests": {ASSIGNMENT_ID: assignment_sha256(repo)},
        },
    )
    return lease


def stop_event(version: str) -> dict[str, object]:
    marker = {
        "assignment_id": ASSIGNMENT_ID,
        "context_version": version,
        "status": "pass",
        "result_path": RESULT_PATH,
    }
    return {
        "hook_event_name": "SubagentStop",
        "stop_hook_active": False,
        "agent_id": "agent-fixture",
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
    (tmp_path / ".agent-harness/ACTIVE_RUN").write_text(
        "run-decoy\n", encoding="utf-8"
    )
    accepted = run_stop_hook(tmp_path, stop_event(version))
    assert accepted.returncode == 0
    assert accepted.stdout == ""
    assert not lease.exists(), "lease must be consumed on acceptance"


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
    (tmp_path / ".agent-harness/ACTIVE_RUN").write_text(
        "run-decoy\n", encoding="utf-8"
    )
    blocked = run_stop_hook(tmp_path, stop_event(version))
    assert blocked.returncode == 0
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "Registered assignment does not exist" in json.loads(blocked.stdout)["reason"]


def test_subagentstop_blocks_when_assignment_tampered_after_start(
    tmp_path: Path,
) -> None:
    version, _ = make_harness(tmp_path)
    assignment_hash = assignment_sha256(tmp_path)
    write_json(tmp_path / RESULT_PATH, valid_result(version, assignment_hash))
    lease = write_lease(tmp_path, version)
    assignment_path = (
        tmp_path / f".agent-harness/runs/{RUN_ID}/assignments/{ASSIGNMENT_ID}.json"
    )
    assignment_path.write_bytes(assignment_path.read_bytes() + b"\n")
    blocked = run_stop_hook(tmp_path, stop_event(version))
    assert json.loads(blocked.stdout)["decision"] == "block"
    assert "changed after SubagentStart" in json.loads(blocked.stdout)["reason"]
    assert lease.exists(), "lease must survive a blocked stop for the retry"


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
