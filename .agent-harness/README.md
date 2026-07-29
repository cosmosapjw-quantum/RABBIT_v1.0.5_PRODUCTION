# Shared-context subagent harness

This harness does not make separate subagents share a hidden model state or free KV cache. It replaces that unavailable assumption with a deterministic, versioned context contract:

1. durable policy in `AGENTS.md`;
2. compact Tier-0 context in `context/`;
3. main-owned registered launch admission and pre/post write hashing;
4. a generated context pack and agent identity injected by `SubagentStart`;
5. registered assignment slices;
6. sibling-result isolation for blind cross-validation;
7. fail-closed `SubagentStop` validation of the result and spawn contract.

## Install

1. Merge `AGENTS.md.fragment` into the repository root `AGENTS.md`.
2. Merge `.codex/config.toml` into the existing project config.
3. Copy `.codex/hooks/`, `.codex/agents/`, `.agents/skills/`, and `.agent-harness/` into the repo.
4. Start Codex in the trusted repository and run `/hooks`; review and trust the project hooks.
   After changing `.codex/hooks.json`, reload the VS Code window or start a new Codex
   session before treating the new event registration as live.
5. Edit the shared context templates.

## Start a run

```bash
python3 .agent-harness/scripts/build_context_pack.py
python3 .agent-harness/scripts/init_run.py \
  --spec-ref docs/specs/current.md \
  --base-ref main \
  --head-ref HEAD

python3 .agent-harness/scripts/new_assignment.py \
  --assignment-id A-WX \
  --agent-type default \
  --review-role cas_wolfram_xact \
  --independence-mode blind-results \
  --claim-id C-001 \
  --task 'Verify C-001 with Wolfram Language and xAct under CAS-001.'

python3 .agent-harness/scripts/validate_harness.py
```

Before spawning, mint the assignment's single-use admission receipt with
`python3 .agent-harness/scripts/admit_agent.py --assignment-id <ASSIGNMENT_ID>`
(add `--expect-agent-id <id>` when the parent chooses the agent id). Paste the
first five lines it prints — `RUN_ID`, `ASSIGNMENT_ID`, `CONTEXT_VERSION`,
`INDEPENDENCE_MODE`, `ADMISSION_TOKEN` — verbatim at the start of the subagent
spawn prompt; the trailing `receipt:` line is not part of the prompt. The main
agent validates the harness immediately before launch and records the non-run
diff/status hashes. The assignment JSON supplies the unique result path.

The token is what binds one agent to one assignment: `SubagentStart` never
receives the spawn prompt, so without it `SubagentStop` can only trust whichever
assignment the stopping agent names for itself. The receipt is consumed exactly
once, recording the writing `agent_id` and the result SHA-256 into the run's
append-only `ADMISSIONS.jsonl`. This defeats confusion, races, and prompt-level
substitution; it is not a defence against an agent that deliberately writes
outside its declared result path, because agent and harness share one OS user.

VS Code collaboration `spawn_agent` does not currently traverse project
`PreToolUse`; live R3 allow-rewrite and block-only probes both demonstrated that gap.
Do not claim automatic launch interception. Completion remains fail-closed because
`SubagentStart` injects current context and identity, while `SubagentStop` validates
the agent's structured `spawn_contract`, assignment-file SHA-256, result envelope,
and event identity. The main agent separately verifies the pre/post write boundary.

`agent_type`/`runtime_agent_type` name the actual Codex runtime reported by the live
SubagentStart and SubagentStop events. `review_role` independently selects the logical
role context. This separation is mandatory when a dedicated role launcher is unavailable
and a supported `default` runtime performs a CAS or adjudication assignment. New v2
assignments seal the exact role files and RESULT_ENVELOPE template by SHA-256; Stop
validates both hashes and both identities without pretending the runtime is a dedicated
agent type. Historical v1 assignments retain their original single-field semantics.

## Spawn prompt template

```text
RUN_ID=<run-id>
ASSIGNMENT_ID=<assignment-id>
CONTEXT_VERSION=<sha256>
INDEPENDENCE_MODE=shared-core|blind-results|adjudication
ADMISSION_TOKEN=<single-use token from admit_agent.py>

Execute only the registered assignment. Load the canonical pack and assignment file before analysis. Do not inspect sibling results unless the assignment permits it. Write the declared result artifact and end with HARNESS_RESULT, including admission_proof set to the ADMISSION_TOKEN above.
```

## Finish

```bash
python3 .agent-harness/scripts/merge_results.py
python3 .agent-harness/scripts/validate_harness.py
```

The adjudicator should consume `MERGED_RESULTS.json` plus only the disputed evidence needed for a targeted decision.

`SubagentStop` validates every completion attempt, including a retry after an earlier
blocked stop. A controlled negative stop test may therefore be followed by a corrected
RESULT_ENVELOPE in the same subagent turn. The result artifact must include the
`agent_id` and runtime `agent_type` injected by `SubagentStart`, its registered
`runtime_agent_type`, `review_role`, and `result_path`, the verified role/template hashes
in `spawn_contract`, canonical lowercase-enum `findings`, and `files_written` containing
only that result path.
