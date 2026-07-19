# Shared-context subagent harness

This harness does not make separate subagents share a hidden model state or free KV cache. It replaces that unavailable assumption with a deterministic, versioned context contract:

1. durable policy in `AGENTS.md`;
2. compact Tier-0 context in `context/`;
3. a generated context pack injected by `SubagentStart`;
4. registered assignment slices;
5. sibling-result isolation for blind cross-validation;
6. machine-readable result envelopes and deduplication.

## Install

1. Merge `AGENTS.md.fragment` into the repository root `AGENTS.md`.
2. Merge `.codex/config.toml` into the existing project config.
3. Copy `.codex/hooks/`, `.codex/agents/`, `.agents/skills/`, and `.agent-harness/` into the repo.
4. Start Codex in the trusted repository and run `/hooks`; review and trust the project hooks.
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
  --agent-type cas_wolfram_xact \
  --independence-mode blind-results \
  --claim-id C-001 \
  --task 'Verify C-001 with Wolfram Language and xAct under CAS-001.'

python3 .agent-harness/scripts/validate_harness.py
```

Paste the header printed by `new_assignment.py` at the start of the subagent spawn prompt. The assignment JSON supplies the unique result path.

## Spawn prompt template

```text
RUN_ID=<run-id>
ASSIGNMENT_ID=<assignment-id>
CONTEXT_VERSION=<sha256>
INDEPENDENCE_MODE=shared-core|blind-results|adjudication

Execute only the registered assignment. Load the canonical pack and assignment file before analysis. Do not inspect sibling results unless the assignment permits it. Write the declared result artifact and end with HARNESS_RESULT.
```

## Finish

```bash
python3 .agent-harness/scripts/merge_results.py
python3 .agent-harness/scripts/validate_harness.py
```

The adjudicator should consume `MERGED_RESULTS.json` plus only the disputed evidence needed for a targeted decision.
