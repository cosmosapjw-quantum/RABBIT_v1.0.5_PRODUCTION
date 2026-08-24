# Rust ODE reconstruction external-audit packet

Branch target: `external-audit/ode-rust-reconstruction-complete-20260824`  
Delivery base: `0879e61c660d446ffc33d9024ad73faacc44327e`  
Reconstruction development base: `78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`

This branch is a complete evidence handoff for external audit. It is not a
production candidate, endpoint-authority result, or solver-promotion decision.
Independent review is intentionally pending; publication of the branch enables
that review.

## Inherited research

The branch descends from both earlier external-audit publications:

1. `external-audit/ode-four-loop-complete-20260823` at
   `9c05a65eaa5fbb86bec5d131c0d300689217de16`; and
2. `external-audit/ode-four-loop-adversarial-review-20260824` at
   `0879e61c660d446ffc33d9024ad73faacc44327e`.

The first ancestor already contains all six prior ODE research run trees, both
user-supplied harness ZIPs, the rejected C1 candidate, raw trajectories and logs,
the integrated final report, and its full 24,408-file task-attributable `/tmp`
snapshot. The second ancestor adds the independent adversarial directive. Those
bytes are inherited here without duplicating a second 462 MB archive.

## Current reconstruction surface

The current candidate adds or changes exactly these eight research files:

- `docs/audit/RUST_ODE_COLLISION_RECONSTRUCTION_2026-08-24.md`
- `native/rabbit_cpu/src/electron_event.rs`
- `native/rabbit_cpu/src/electron_event_falsifiers.rs`
- `native/rabbit_cpu/src/electron_spectral.rs`
- `native/rabbit_cpu/src/electron_supplied.rs`
- `native/rabbit_cpu/src/isotropic_boltzmann.rs`
- `native/rabbit_cpu/src/lib.rs`
- `native/rabbit_cpu/src/pauli_edge_step.rs`

The cumulative reconstruction diff is 1,710 added and 49 deleted lines, net
+1,661. `RECONSTRUCTION_SOURCE_SHA256SUMS` binds each final file;
`RECONSTRUCTION_WORKTREE_DIFF.patch` preserves a standalone full-index patch,
including both new files. The evidence note records the P0 discriminator, edge
gain/loss reconstruction, transactional collision substep, energy closure,
parallel event construction, previously executed focused runs, and the remaining
endpoint blockers.

## Current temporary and ignored snapshot

`TEMP_ARCHIVE_PATHS.txt` freezes the eight attributable paths captured before
the publication branch was assembled:

1. the complete isolated RABBIT reconstruction worktree, including ignored
   `native/rabbit_cpu/target/` and Python `__pycache__/` trees;
2. the measured Python environment at `/tmp/rabbit-ode-p0-measured-venv`;
3. the separate Cargo target at `/tmp/rabbit-ode-native-target`;
4. the VigilODE research clone at `/tmp/vigilode-recon-20260824.pUiXaC`;
5. the retained exact-HEAD raw trajectory;
6. the corrected P0 discriminator script; and
7. both P0 result JSON files.

The snapshot contains 11,806 regular files, 4 symlinks, and 13,683 tar entries.
Its uncompressed POSIX tar is 2,556,252,160 bytes with SHA-256
`eb250e4a85e983b00ac93012841777025da8f2fb8956b33c02609be3e6e5d967`.
The zstd stream is 544,174,962 bytes with SHA-256
`d051b9469a0845f0557c95416a6a2c06706f8ae929c396614ce12fbc5479f460`.
It is force-added as seven chunks, each below GitHub's 100 MB single-blob limit.

Reconstruct and inspect it only in a disposable directory; the snapshot contains
build products, a Python environment, Git metadata, and a linked-worktree `.git`
pointer whose original absolute target is not portable:

```bash
cd external_audit/ode_rust_reconstruction_20260824
sha256sum -c TMP_ARCHIVE_CHUNK_SHA256SUMS
cat tmp_archives/task_owned_tmp.tar.zst.part-* > task_owned_tmp.tar.zst
sha256sum task_owned_tmp.tar.zst
zstd -t task_owned_tmp.tar.zst
zstd -d task_owned_tmp.tar.zst -o task_owned_tmp.tar
mkdir audit-extract
tar -xf task_owned_tmp.tar -C audit-extract
```

`TEMP_SOURCE_SHA256SUMS` records every original regular file by absolute path,
`TEMP_SOURCE_SYMLINKS.txt` records each link target, and
`TEMP_ARCHIVE_CONTENTS.txt` records archive metadata. Extraction was compared
against the source inventory: 11,806/11,806 regular-file hashes and 4/4 symlink
targets matched. Reassembly of the seven chunks reproduced the whole zstd hash
and passed zstd stream validation.

The four high-value P0 files are also exposed directly under
`tmp_materialized/tmp/`, with `MATERIALIZED_SHA256SUMS`, so an auditor need not
unpack 544 MB before inspecting the discriminator and raw data.

The uncompressed tar and unsplit zstd file are not separately committed because
the seven tracked chunks reconstruct the exact stream; committing all three forms
would only duplicate identical packaging bytes. The ephemeral bounded-publication
contract is also not a research artifact and is removed under its governing rule.

## Validation and claim boundary

No new test, benchmark, solver, endpoint, package, JAX, Diffrax, or scientific
command was run to create or publish this branch. The only new checks were
packaging checks: source hashing, archive listing, zstd integrity, extracted-file
hash comparison, staged-byte inspection, Git ancestry/ref inspection, and remote
ref equality.

All cargo commands and numerical values in the reconstruction evidence note were
executed before this publication request. They remain focused candidate evidence;
they do not establish full-suite, endpoint, cold-wall, same-physics parity, or
production authority.

The original user checkout remains `diagnosis_report` at
`78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b` with its pre-existing modified and
untracked files untouched. No pull request, merge, tag, release, or force-push is
part of this delivery.
