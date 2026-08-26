# PR #1 R1C Contract Amendment R3 — Corrected Baseline Patch

This amendment is the controlling bootstrap authority for resuming the existing
PR #1 closure run after `CONTRACT_AMENDMENT_REQUIRED`.

It changes only the frozen baseline patch container and its component identity.
It does not change the implementation base, mathematical contract, must-close
catalogue, deferred typed blockers, allowed production scope, tolerances, golden
bits, or claim ceiling.

## Canonical identities

```text
repository:
cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION

existing PR:
#1

implementation branch:
fix/ode-r1c-certified-edge-outcomes-20260824

required implementation HEAD:
50d3bc5b8093bc33e9311f94505c5ee0711ce51b

contract branch:
external-audit/pr1-r1c-dynamic-compiled-contract-v3-20260826

frozen corrected release commit:
8d310d27938ffd2e934c580fc5ec036342c0c13b

superseded release commit:
635939939295122b05e6e086b87ee1a128f9afcd

corrected patch blob:
527a907b2736310d46bd4d84020b8af0e752d479

corrected manifest blob:
2003d2361a2e3c79afb7fa5a16c67bf7565d3025
```

The former patch was syntactically corrupt because its hunk declared 29
new-side lines while containing 30. R3 changes only the hunk header from
`+704,29` to `+704,30`; the Rust test body and all scientific/numerical values
are byte-identical.

## Resume policy

Continue the already-created local implementation worktree. Do not reset,
rebase, amend, clean, or discard production work. The reported production diff
is empty and the only dirty paths are materialized untracked contract files.

Before resuming, require:

```bash
set -euo pipefail

cd /home/cosmosapjw/Dropbox/rabbit/rabbit-pr1-closure-v2

test "$(git branch --show-current)" = "local/pr1-r1c-closure-v2"
test "$(git rev-parse HEAD)" = "50d3bc5b8093bc33e9311f94505c5ee0711ce51b"
test -z "$(git diff --name-only)"
test -z "$(git diff --cached --name-only)"

CONTRACT_COMMIT=8d310d27938ffd2e934c580fc5ec036342c0c13b

for path in \
  .codex/audit/pr1-r1c-closure/CONTRACT_MANIFEST.json \
  .codex/audit/pr1-r1c-closure/AUDIT_COMPILED_PACKAGE.json \
  .codex/audit/pr1-r1c-closure/pr1_r1c_numeric_reproducer.py \
  .codex/audit/pr1-r1c-closure/BASELINE_KNOWN_BAD.patch \
  docs/audit/PR1_R1C_DYNAMIC_ADVERSARIAL_REAUDIT_2026-08-25.md; do
  mkdir -p "$(dirname "$path")"
  git show "$CONTRACT_COMMIT:$path" > "$path"
done

python3 -m json.tool \
  .codex/audit/pr1-r1c-closure/CONTRACT_MANIFEST.json >/dev/null
python3 -m json.tool \
  .codex/audit/pr1-r1c-closure/AUDIT_COMPILED_PACKAGE.json >/dev/null

python3 - <<'PY'
import json
import subprocess
from pathlib import Path
manifest = json.loads(Path(
    '.codex/audit/pr1-r1c-closure/CONTRACT_MANIFEST.json'
).read_text(encoding='utf-8'))
assert manifest['package_id'] == \
    'rabbit-pr1-r1c-dynamic-compiled-contract-v3-20260826'
assert manifest['audited_implementation_head'] == \
    '50d3bc5b8093bc33e9311f94505c5ee0711ce51b'
for item in manifest['components']:
    path = item['path'].lstrip('/')
    actual = subprocess.check_output(
        ['git', 'hash-object', path], text=True
    ).strip()
    assert actual == item['blob_sha'], (path, actual, item['blob_sha'])
print('CONTRACT_COMPONENT_IDENTITIES_OK')
PY

patch=.codex/audit/pr1-r1c-closure/BASELINE_KNOWN_BAD.patch
git apply --check "$patch"
```

`git apply --check` must succeed before any Rust source mutation. If it fails,
stop with `CONTRACT_AMENDMENT_DIVERGENCE` and report the exact output; do not
edit the frozen component locally.

After the check succeeds, continue from section 5 of
`docs/audit/PR1_R1C_CODEX_HANDOFF_2026-08-25.md`, using
`CONTRACT_COMMIT=8d310d27938ffd2e934c580fc5ec036342c0c13b` wherever the superseded release
commit appears.

Run the exact PRE-002B apply/test/reverse sequence. The expected baseline fact
remains:

```text
PauliEdgeApplicationKind::Solved
extent bits = 0xbcce_ff08_07bf_a264
```

Only after that real known-bad Rust observation may RED/GREEN implementation
work begin.

## Unchanged scope and completion semantics

```text
must_close P0: 3
must_close P1: 6
deferred typed P1: unchanged
claim ceiling: focused local Pauli primitive only
PR body update: only after implementation/evidence completion
merge: not authorized
```

Do not count the prior corrupt-patch attempt as a scientific or production
failure. It is a deterministic contract-artifact formatting defect.

Return the existing closeout schema. Add these fields:

```text
contract_amendment: R3
contract_release_commit: 8d310d27938ffd2e934c580fc5ec036342c0c13b
superseded_release_commit: 635939939295122b05e6e086b87ee1a128f9afcd
baseline_patch_apply_check: PASS|FAIL
```
