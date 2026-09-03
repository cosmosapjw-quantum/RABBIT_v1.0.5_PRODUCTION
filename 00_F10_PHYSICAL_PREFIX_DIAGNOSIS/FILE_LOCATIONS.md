# Complete file-location map

All links are repository-relative and resolve on `diagnosis_report`. SHA-256 is
over the linked file's exact bytes. [PROVENANCE_INDEX.json](PROVENANCE_INDEX.json)
adds Git blob/source-commit metadata; [SHA256SUMS](SHA256SUMS) is the complete
machine-verifiable digest list. Files are linked at canonical locations and are
not copied into this directory.

## Requirement routing

| Requirement | Open first | Boundary |
|---|---|---|
| `REQ-SOURCE` | [SOURCE_BUNDLE.json](SOURCE_BUNDLE.json), [solver ZIP](../RABBIT_F10_SolverAlgorithm_Blocker_Research_Loop_2026-08-06.zip), [math/physics ZIP](../RABBIT_F10_MathPhysics_Blocker_Research_Loop_2026-08-06.zip) | Exact base tree plus retained archive/internal-bundle bytes. |
| `REQ-CHECKPOINTS` | [PREFIX_INPUTS.json](PREFIX_INPUTS.json), [state 1200](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_1200.npz), [state 2000](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_2000.npz), [state 3000](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_3000.npz) | Original order-60 / `y_max=30` retained states and complete raw V3a domain evidence. |
| `REQ-INPUTS` | [initial NPZ](initial_state_order60_ymax30.npz), [quadrature/catalog manifest](QUADRATURE_CATALOG_MANIFEST.json) | Initial state is `DERIVED`, not retained-run evidence. |
| `REQ-RHS-JVP` | [active receipt](receipts_v2/PHYSICAL_RHS_JVP_RECEIPTS.json), [raw vectors](receipts_v2/PHYSICAL_RHS_JVP_VECTORS.npz) | Direct physical calls; historical `obs_jac_*` files are not used as JVP receipts. |
| `REQ-RECEIPTS` | [RECEIPT_INDEX.json](RECEIPT_INDEX.json) | First-law, occupation, rejection/roundoff, and finite-domain-tail selectors for all states. |
| `REQ-CONTRACT` | [PREFIX_CONTRACT.json](PREFIX_CONTRACT.json), [contract digest](PREFIX_CONTRACT.sha256) | Active recovery seal `acb5641e8008f0c8305e8e83db4d7269ba9e1cd6`. |
| `REQ-ENTRYPOINT` | [diagnosis README](README.md), [root README](../README.md) | Standalone branch; no `main` merge or gate movement. |

## Derived fixture, contracts, and receipts

| Location | SHA-256 | Role / status |
|---|---|---|
| [SOURCE_BUNDLE.json](SOURCE_BUNDLE.json) | `2fa75fc782fb512ebd66395a5b1951f1fe9966f36cd25ab25741b7254ebe8e8f` | Exact source/archive manifest; `VALIDATED`. |
| [PREFIX_INPUTS.json](PREFIX_INPUTS.json) | `927cd39f58d1414da45f80e4c8bc39692533d8741684f14e3ca760a3582e594f` | Initial/checkpoint identity; `VALIDATED`. |
| [QUADRATURE_CATALOG_MANIFEST.json](QUADRATURE_CATALOG_MANIFEST.json) | `3dd211c9abf4701b0d527f58d170a8873ee3b044238a3972cad147d01c6d4d5d` | Value-level grid/catalog identity; `VALIDATED`. |
| [initial_state_order60_ymax30.npz](initial_state_order60_ymax30.npz) | `6df77abea95b71ba1c26953d6595197c4da689e1761a6f7186349c318504698e` | Deterministic source-derived initial state; `DERIVED`. |
| [PREFIX_CONTRACT.json](PREFIX_CONTRACT.json) | `c26ef7b9c6e9ba1f8fd0a57b1f9f068c89ea336f3aa58751d7e708d2becd88bc` | Active prospective recovery contract; `SPECIFIED`. |
| [PREFIX_CONTRACT.sha256](PREFIX_CONTRACT.sha256) | `6220c709dd29de3a049683ffb35a018be41b83c2fd5ba814b720301679cb486c` | Exact contract checksum record. |
| [first-attempt receipt](receipts/PHYSICAL_RHS_JVP_RECEIPTS.json) | `21bb4162a7cd2d1816c3f0965520dacfe2d4e2feb17165b32dc65c0ce4889b8a` | Preserved v1 failure output; exposed base-evidence retention defect. |
| [first-attempt vectors](receipts/PHYSICAL_RHS_JVP_VECTORS.npz) | `7a709729fa3d3039c5f5c72aa8e6322984e9fd1f4a30b67575e64b1ceaf4cb1c` | Preserved unchanged v1 vectors. |
| [first-attempt run log](receipts/RECEIPT_RUN_LOG.json) | `d23c6ee0165c835d2fb6273b3df92a36e9eb3d7e90d226b7fea83a5d81813b1a` | Binds v1 seal/contract/output hashes. |
| [active v2 receipt](receipts_v2/PHYSICAL_RHS_JVP_RECEIPTS.json) | `7db72888d007d0ff736d6aaf0257b039a02508eab7b4905fd81af0ec5f024864` | Four base receipts and direct-JVP attempts; negative at `creep_1200`. |
| [active v2 vectors](receipts_v2/PHYSICAL_RHS_JVP_VECTORS.npz) | `30d9dd1912291c2ff70a665dd7976e239975be66689b44053238cb7fe79d3289` | Raw base/shifted/collision/direction/JVP/Arnoldi arrays. |
| [active v2 run log](receipts_v2/RECEIPT_RUN_LOG.json) | `f5e75857766f759c645650b1dc67c98fdaf8b0d362a3cd50cf0f803ef8b782e1` | Binds recovery seal, contract and v2 output hashes. |
| [JSON-normalized final verifier](verify_final_json_normalized.py) | `0054904403eb59bcd3cd6be5092a1e90d5270e5f5a9f46b67e45b9f3730a0509` | Post-seal wrapper for the sealed tuple/list comparison defect; all substantive checks remain delegated. |

Final generated machine files are [BRANCH_SCOPE.json](BRANCH_SCOPE.json),
[PROVENANCE_INDEX.json](PROVENANCE_INDEX.json),
[RECEIPT_INDEX.json](RECEIPT_INDEX.json), [READINESS.json](READINESS.json),
[VALIDATION_LEDGER.json](VALIDATION_LEDGER.json), and
[SHA256SUMS](SHA256SUMS). Their own exact hashes are recorded in `SHA256SUMS`
to avoid a self-referential Markdown digest.

## Exact runtime and audit source lock

| Canonical location | SHA-256 |
|---|---|
| [src/rabbit/decoupling/_independent_noqke.py](../src/rabbit/decoupling/_independent_noqke.py) | `760a7c044081e507fae9d5695b301bd44f6466d96322c46f53b77161e32b558a` |
| [scripts/audit/_trajectory_core.py](../scripts/audit/_trajectory_core.py) | `3ec42baa678a0f36baa79dab7ed6fcd136272be1634b50e78bc4b41b7e480cda` |
| [scripts/audit/d069_independent_trajectory_r4.py](../scripts/audit/d069_independent_trajectory_r4.py) | `670bbbda461be5ac10d03671d47d89acdaca329640bb152676c5a76acc11cb0c` |
| [native/rabbit_cpu/src/isotropic_boltzmann.rs](../native/rabbit_cpu/src/isotropic_boltzmann.rs) | `bc61b942d4d616f022a2681c79ba6e052c54caa659ef2da7840b57dbbb01e236` |
| [native/rabbit_cpu/src/electron_catalog.rs](../native/rabbit_cpu/src/electron_catalog.rs) | `930066ca47d729386eebe91091680fc2f6e2c7039f0760166f2c88c3af54b805` |
| [native/rabbit_cpu/src/quadrature.rs](../native/rabbit_cpu/src/quadrature.rs) | `356f4e92b3fcc270c822c6cc7bd785c681011af6de374cf4b88b3a895b904786` |
| [tests/test_independent_noqke_comparator.py](../tests/test_independent_noqke_comparator.py) | `8f6036fbe4db303e84c0c76f740f3f0ffd72411e504c5b282431e6592c9d9ad8` |
| [fixture runner](../scripts/audit/f10_physical_prefix_fixture.py) | `a9e0cf1bb5bcd0dbd286be3e6ad28a5d696aaff8bfe99a7797be449ee82b08b6` |
| [fixture tests](../tests/test_f10_physical_prefix_fixture.py) | `226e4b4e2ea82734183d86f470377a73e7fd2ce9f0693c3a678c38172b6b86c4` |
| [final-verifier normalization test](../tests/test_f10_physical_prefix_final_verify.py) | `4c35b3b422d323cb1471c343cb7bd7a398025bf46b0ddf7309d176866b919447` |
| [D-071 trajectory closure](../docs/audit/BD622_D071_trajectory_closure_2026-08-04.md) | `b732283c836797a917d7a43aca10b11aaf132a7669b8fe4e0b3fd09d1a481c7d` |
| [V2 option-3 protocol](../docs/audit/BD622_V2_option3_closed_and_protocol_2026-08-04.md) | `3933180b16fc9baf5bf0f871b4d2e99ff3d151445252fb2249c996b958954e9c` |
| [V2 result](../docs/audit/BD622_V2_result_2026-08-05.md) | `770e7de8f81886b2a56a204fbf6db5aa90ad185bff77b6faf4e7fb17abc529fc` |
| [V3 protocol](../docs/audit/BD622_V3_protocol_2026-08-05.md) | `83df9d0edc085fe5a0a58008d24b29f795541be10fe8a5618db6592618456c77` |
| [V3 report](../docs/audit/BD622_V3_report_2026-08-06.md) | `8b028acdbf4a4c3644f4fcdd0b1b8e112d635cb6979e8b9dac03b1516a17c78f` |

The full exact base blob/tree inventory, including Git blob OIDs and the
reconstruction command, is in [SOURCE_BUNDLE.json](SOURCE_BUNDLE.json).

## Retained archives and campaign provenance

| Canonical location | SHA-256 |
|---|---|
| [solver research ZIP](../RABBIT_F10_SolverAlgorithm_Blocker_Research_Loop_2026-08-06.zip) | `8ffb9c34019e4bc9e431985df9fe69a347ced5da11f68308a1943187e3829fd8` |
| [math/physics research ZIP](../RABBIT_F10_MathPhysics_Blocker_Research_Loop_2026-08-06.zip) | `bb3ca057d1ecee6b11e33bba5dbcd8325a23d95dfe925bb5a235866d05ed4fb0` |
| [V1 instrumented RHS](../.agent-harness/runs/run-20260804-f10-v1-diagnostic/instrument/instrumented_rhs.py) | `ad85f2731ff21bacc781c2a5cb96c7b6021d2f7ceb59c188bd5ac993c95a4cdb` |
| [V1 instrumented BDF](../.agent-harness/runs/run-20260804-f10-v1-diagnostic/instrument/instrumented_bdf.py) | `19f96e121f89c60b9e0380bf4a53b0b08c4e07b69bba7be295f4370d8a5ccf06` |
| [V3 runner](../.agent-harness/runs/run-20260805-f10-v3-campaign/run_v3.py) | `78d124d850360d1aed723f2b3e3218a39e4c6c56a3f2544424fb4d2fbc83e307` |
| [V3 analyser](../.agent-harness/runs/run-20260805-f10-v3-campaign/analyse_v3.py) | `9c4275bde13d1539ee098e6b25ec2420c292aa9334176f0856223a5255303fb6` |
| [V3 renderer](../.agent-harness/runs/run-20260805-f10-v3-campaign/render_v3.py) | `acc1b9bab83b61749042f25e358b5020803350a60a6f62a84461fe75a359ab26` |
| [V3 analysis](../.agent-harness/runs/run-20260805-f10-v3-campaign/ANALYSIS_V3.json) | `0f5ca81f7003c1f484d6ef0deab83f4c3202e561d13f727c2a5d544f832b8276` |
| [V3 report verification](../.agent-harness/runs/run-20260805-f10-v3-campaign/report_verification_output.json) | `c4796e00034a1079fefe20c65b54ad7cde704fbd5f26eaa42cb8bc6d32d8b0c4` |
| [V3 r4 reference](../.agent-harness/runs/run-20260805-f10-v3-campaign/r4_reference.json) | `18c92316f914c6e184472875fa7564d077397a3c5136244e6859ca0d0cc28f5c` |
| [V3a pins](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/pins_verified.json) | `c86640d620cb6daf183bbf6321a26b104f221fd17df9bc25e8192cd7ab9bede6` |
| [V3a self-test](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/selftest_result.json) | `16d12378225ef8b1fe0948ccbd09cb18a3155ef24efbfeec07dc342a13423193` |
| [V3a driver log](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/driver.log) | `05124f03df2e19c3d10609dee93a5e1b30a1e86a6b42080fbbca37996500172e` |
| [V3a nohup log](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/nohup.log) | `05124f03df2e19c3d10609dee93a5e1b30a1e86a6b42080fbbca37996500172e` |

## Complete retained V3a domain directory

The `obs_jac_*.npz` files below are historical finite-difference observation
Jacobians only. The active direct-JVP receipt does not consume them.

| Canonical location | SHA-256 |
|---|---|
| [accepted.jsonl](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/accepted.jsonl) | `3d1077381bcff291a60343e556e76cf3eb20fe3324fd8961281eeb62baa5c7d1` |
| [jac_factor.jsonl](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/jac_factor.jsonl) | `57ff04827629688b693d2162c465a292109e3afb7de31b32974b48520f879cb7` |
| [jacobian_events.jsonl](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/jacobian_events.jsonl) | `6faf9cc66fa1563f5fe7620252f039a973a7b650292e2af363fd4fe18dea0de2` |
| [newton_calls.jsonl](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/newton_calls.jsonl) | `f6e524959af67b21de0c6c52871c35cb282d2d4383679d3df2b890a3558b0871` |
| [obs_jac_1200.npz](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/obs_jac_1200.npz) | `bd904e4150fbf009cfd3f0918c78d68618769d10d98ae3c4b681132503f6ca49` |
| [obs_jac_2000.npz](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/obs_jac_2000.npz) | `9eb9ac920646a037ecfc4ebcb66de8d6e04599d4ac3fc7f73b883569a31d0f5f` |
| [obs_jac_3000.npz](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/obs_jac_3000.npz) | `595b107ea98ed9244a6199b2ea94d038a5aa3b7d84d9db5e13231cef520adf66` |
| [ratcheted_cols_1200.npz](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/ratcheted_cols_1200.npz) | `93dba4914e08de49be4466e9b8973e2763c524c684c19860efa29a9f415c8d6e` |
| [ratcheted_cols_2000.npz](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/ratcheted_cols_2000.npz) | `ac33162bca5a3b6c357a972d5ef2f5aed0748a68bb2a3effb3e3597c8f6a087e` |
| [ratcheted_cols_3000.npz](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/ratcheted_cols_3000.npz) | `3399b945fab9469c9997dd7e950992a2d323e7ddc7efa93c1d9338f42a065e6f` |
| [rhs_calls.jsonl](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/rhs_calls.jsonl) | `2ce10b05aec15c47b278de7279f495106430301eaacc44de099094b1cb778e7e` |
| [state_1200.npz](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_1200.npz) | `c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380` |
| [state_2000.npz](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_2000.npz) | `780ad7c1388caec23f02012781717d43ffb85d96d4d501c40c504939e7c9a44d` |
| [state_3000.npz](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_3000.npz) | `dbf760acfde72c2617192592995d6593043309d4c0de7dc6fab99d805f9504ad` |
| [step_events.jsonl](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/step_events.jsonl) | `5344b9420c2a172d42714aa3775d5662f0ab820415f411bb31b57f2c63da8334` |
| [summary.json](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/summary.json) | `eae659168c8f6269db62a74729336837e9f00ca37a2b5b5a5f9b906ec6f67683` |
| [trials.jsonl](../.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/trials.jsonl) | `c0998aeb5bd8a9cdad6e440acbfb554fbb27b2d7bfd17d02761a5b40603f3c3a` |

## Governance and validation locations

- [Approved branch design](../docs/superpowers/specs/2026-08-11-f10-diagnosis-report-branch-design.md)
- [Executed implementation plan](../docs/superpowers/plans/2026-08-11-f10-physical-prefix-diagnosis-implementation.md)
- [Project state](../docs/harness/PROJECT_STATE.md)
- [Claim ledger](../docs/harness/CLAIM_LEDGER.md)
- [Validation ledger](../docs/harness/VALIDATION_LEDGER.md)
- [Decision log](../docs/harness/DECISION_LOG.md)
- [Next-session preservation prompt](../docs/harness/NEXT_SESSION_PROMPT.md)

The diagnosis indexes do not feed the gate registry or generated status board.
