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
) -> Path:
    """Stand in for `admit_agent.py`: the parent-minted single-use receipt."""
    path = (
        repo / ".agent-harness" / "admissions" / RUN_ID / f"{assignment_id}.json"
    )
    write_json(
        path,
        {
            "schema_version": 1,
            "run_id": RUN_ID,
            "assignment_id": assignment_id,
            "assignment_sha256": assignment_sha256(repo, assignment_id),
            "runtime_agent_type": "context_mapper",
            "context_version": version,
            "token_digest": "sha256:"
            + hashlib.sha256(token.encode("utf-8")).hexdigest(),
            "state": state,
            "created_at": "2026-07-28T00:00:00+00:00",
        },
    )
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
    version, _ = make_harness(tmp_path)
    write_json(
        tmp_path / RESULT_PATH, valid_result(version, assignment_sha256(tmp_path))
    )
    write_lease(tmp_path, version)
    write_admission(tmp_path, version)
    assert run_stop_hook(tmp_path, stop_event(version)).stdout == ""
    write_lease(tmp_path, version, agent_id="agent-other")
    blocked = run_stop_hook(tmp_path, stop_event(version, agent_id="agent-other"))
    assert json.loads(blocked.stdout)["decision"] == "block"


def test_concurrent_stops_consume_receipt_once(tmp_path: Path) -> None:
    """O_EXCL claim, not the state read, is what makes the receipt single-use."""

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
    outs = []
    for proc in procs:
        assert proc.stdout is not None
        outs.append(proc.stdout.read())
        proc.wait()
    # Timing decides how many losers see a consumed receipt (accept idempotently)
    # versus only the claim (fail closed); both are correct. What must hold
    # regardless is that exactly one of them wrote the attribution, and that no
    # loser silently produced a second one.
    assert any(out == "" for out in outs), outs
    for out in outs:
        if out:
            assert json.loads(out)["decision"] == "block", out
    ledger = (
        tmp_path / f".agent-harness/runs/{RUN_ID}/ADMISSIONS.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    consumed = [line for line in ledger if json.loads(line)["event"] == "consumed"]
    assert len(consumed) == 1, ledger


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
    """Receipt-write-failure fixture: no receipt may outlive a failed mint."""

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
