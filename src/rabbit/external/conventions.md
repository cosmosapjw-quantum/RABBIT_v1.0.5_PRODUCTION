# External BBN Code Conventions

This document is the single source of truth for the unit and parameter
conventions used to call out to external BBN codes from
`rabbit.external.*`. Changes here ripple to every wrapper.

| Quantity | RABBIT convention | External code conversion |
|---|---|---|
| `eta` (baryon-to-photon ratio) | absolute, ≈ 6.104e-10 | NUDEC_BSM accepts absolute eta. AlterBBN takes eta_10 = eta×1e10. PArthENoPE 3.0 accepts both via flag. |
| `Y_p` (helium-4 mass fraction) | mass fraction | AlterBBN, PArthENoPE 3.0, PRIMAT-AC2024 already mass; older AlterBBN ≤ 1.4 reports number — convert via Y_mass = 4·Y_num / (1+3·Y_num). |
| `D/H` | number ratio | All modern codes already number. |
| `N_eff` | effective ν number including incomplete-decoupling heating | Standard-model benchmark 3.044 (Mangano 2005, Froustey 2020). |
| `tau_n` (neutron lifetime) | seconds, PDG 2024 = 878.4 ± 0.5 | NUDEC_BSM accepts seconds. AlterBBN takes seconds. |

## Reference cross-code values (for offline parity)

| Code | Y_p | D/H | N_eff | Source |
|---|---|---|---|---|
| Mangano 2005 | — | — | 3.044 | Mangano et al. 2005, Nucl. Phys. B729 221 |
| Froustey 2020 | — | — | 3.044 | Froustey, Volpe, Pisanti 2020 |
| LASAGNA | 0.24700 | 2.531e-5 | 3.044 | Akita & Yamaguchi 2020 |
| FortEPiaNO | 0.24700 | 2.531e-5 | 3.044 | Bennett+ 2021 |
| PRIMAT-AC2024 | 0.24750 | 2.560e-5 | 3.044 | Pitrou+ 2024 update |
| AlterBBN 2.4 | 0.24735 | 2.547e-5 | 3.044 | Arbey+ 2018 |

These are the **fiducial** values at η = 6.104e-10, τ_n = 878.4 s,
N_eff_SM = 3.044. They are bundled into the static
``tests/fixtures/tier3_cross_code.json`` for offline comparison; the
``rabbit.external.*`` wrappers replace them with **live** subprocess
calls when the corresponding backend is installed.

## Test skip convention

A test using one of these wrappers must be decorated with
``@pytest.mark.cross_code`` and use ``pytest.importorskip`` or call
``has_<backend>()`` for graceful skip when offline. The static fixture
test (``test_pr_t3d_cross_code_skeleton.py``) remains as the always-on
baseline.
