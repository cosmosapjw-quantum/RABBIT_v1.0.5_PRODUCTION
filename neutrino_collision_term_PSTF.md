짧은 요약:
가장 깔끔한 형태는 **Pauli blocking까지 포함한 gain–loss factor를 먼저 (f_i) polynomial로 정리한 뒤**, 모든 (f_i(E_i,e_i))를 PSTF multipole (F^s_{A_\ell}(p_i))로 치환하는 방식이야. 그러면 collision hierarchy는 코드에서 바로 쓰기 좋은

[
C_{A_\ell}
==========

## K_{34}FF

K_{12}FF
+
K_{123}FFF
+
K_{124}FFF
----------

## K_{134}FFF

K_{234}FFF
]

꼴로 떨어진다.

핵심은 **quartic (f_1f_2f_3f_4) 항이 정확히 cancel된다는 것**이야. 그래서 fermionic (2\leftrightarrow2) collision term은 최대 cubic monomial만 남는다. 이게 코드 이식에서 꽤 큰 이점이 된다.

---

## 1. 기본 변수와 PSTF expansion

metric/signature는

[
g_{ab}=(-,+,+,+),\qquad u^au_a=-c^2,
]

[
h_{ab}=g_{ab}+\frac{u_au_b}{c^2}.
]

각 입자 momentum은 local tetrad에서

[
P_i^a=\frac{E_i}{c^2}u^a+p_i e_i^a,
\qquad
h_{ab}e_i^ae_i^b=1,\qquad u_ae_i^a=0,
]

[
E_i(p_i)=\sqrt{m_i^2c^4+p_i^2c^2}.
]

PSTF basis는

[
O_{A_\ell}(e):=e_{\langle a_1}\cdots e_{a_\ell\rangle},
\qquad
A_\ell=a_1\cdots a_\ell .
]

distribution function을

[
\boxed{
f^s(p,e)
========

\sum_{\ell=0}^{L_{\max}}
F^s_{A_\ell}(p)O^{A_\ell}(e)
}
]

로 둔다. inversion은

[
\boxed{
F^s_{A_\ell}(p)
===============

\Delta_\ell^{-1}
\int d\Omega,
O_{A_\ell}(e)f^s(p,e)
}
]

[
\Delta_\ell
===========

\frac{4\pi}{2\ell+1}
\frac{2^\ell(\ell!)^2}{(2\ell)!}.
]

이건 (1+3) covariant PSTF multipole formalism의 표준 정규화다. PSTF multipole algebra와 mode/multipole relation은 Gebbie–Ellis의 (1+3) covariant CMB formalism에서 체계적으로 정리되어 있다. ([arXiv][1])

코드에서는 Cartesian STF tensor 전체를 저장하지 말고, 독립 basis index

[
\lambda=(\ell,\sigma),\qquad \sigma=1,\ldots,2\ell+1
]

를 쓰는 게 낫다. 그럼

[
f^s(p,e)=F^s_\lambda(p),O^\lambda(e)
]

처럼 쓸 수 있다. 아래에서는 tensor notation과 compressed mode notation을 같이 쓸게.

---

## 2. HM collision integral의 출발점

채널을

[
r:\quad s_1(1)+s_2(2)\leftrightarrow s_3(3)+s_4(4)
]

라고 쓰자. HM isotropic kernel은 이 (2\to2) invariant collision integral에서 isotropy를 써서 줄인 형태다. HM 계산은 full Fermi–Dirac statistics와 electron mass dependence를 포함한 neutrino decoupling 계산으로 제시되어 있다. ([arXiv][2])

anisotropic case에서는 scalar HM (F(p_1,p_2,p_3))를 그대로 쓰지 말고, 다음 invariant integral을 PSTF projection해야 한다.

[
C_1^r
=====

\frac{S_r}{2E_1}
\int
d\widetilde P_2d\widetilde P_3d\widetilde P_4,
(2\pi\hbar)^4
\delta^{(4)}(P_1+P_2-P_3-P_4)
|\mathcal M_r|^2
\Lambda_r .
]

여기서

[
d\widetilde P_i
===============

\frac{g_i,d^3p_i}{(2\pi\hbar)^3,2E_i}.
]

정확한 전역 normalization은 네 기존 JAX HM convention에 맞추면 된다. 특히 (C)를 (P^a\nabla_a f=C)의 우변으로 둘지, local-time equation (\dot f=C_{\rm time})의 우변으로 둘지에 따라 (E_1), (c) factor가 한 번 더 이동할 수 있다. 아래 구조는 그 normalization 선택과 독립적이다.

---

## 3. Gain–loss polynomial 정리

fermion blocking factor는

[
\Lambda_r
=========

## (1-f_1)(1-f_2)f_3f_4

f_1f_2(1-f_3)(1-f_4).
]

이걸 전개하면 quartic term이 정확히 사라져서

[
\boxed{
\Lambda_r
=========

## f_3f_4

f_1f_2
+
f_1f_2f_3
+
f_1f_2f_4
---------

## f_1f_3f_4

f_2f_3f_4 .
}
]

즉 subset notation으로

[
\Lambda_r
=========

\sum_{S\in\mathcal S}\chi_S\prod_{i\in S}f_i
]

[
\mathcal S
==========

{34,12,123,124,134,234},
]

[
\chi_{34}=+1,\qquad
\chi_{12}=-1,
]

[
\chi_{123}=+1,\qquad
\chi_{124}=+1,\qquad
\chi_{134}=-1,\qquad
\chi_{234}=-1.
]

이게 코드용으로 제일 중요한 compactification이다. ((1-F)) multipole을 따로 만들 필요가 없다.

단, 이 단순 cancel은 scalar occupation number에 대한 이야기야. flavour density matrix를 full non-commutative matrix로 진화시키면 ordering 문제가 생길 수 있고, 이 polynomial을 그대로 쓰면 안 된다.

---

## 4. PSTF collision multipole

입자 (1), species (s_1)에 대한 collision multipole은

[
\boxed{
C^{s_1,r}*{A*\ell}(p_1)
=======================

\Delta_\ell^{-1}
\int d\Omega_1,
O_{A_\ell}(e_1)
C_1^r(p_1,e_1).
}
]

위 polynomial을 PSTF expansion에 넣으면 최종적으로

[
\boxed{
\begin{aligned}
C^{s_1,r}*{A*\ell}(p_1)
=======================

\int dp_2dp_3
\sum_{S\in\mathcal S}
\chi_S
,
\mathcal K^{r;S}*{A*\ell|{B_i}*{i\in S}}
(p_1,p_2,p_3)
\prod*{i\in S}
F^{s_i}_{B_i}(p_i)
\end{aligned}
}
]

가 된다.

여기서 (p_4)는 energy conservation으로 결정된다:

[
E_4=E_1+E_2-E_3,
]

[
p_4=
\frac{1}{c}
\sqrt{E_4^2-m_4^2c^4}.
]

만약 (E_4<m_4c^2)이면 해당 kernel은 0이다.

좀 더 펼쳐 쓰면,

[
\boxed{
\begin{aligned}
C^{s_1,r}*{A*\ell}
=&
\int dp_2dp_3
\Big[
\mathcal K^{r;34}*{A*\ell|D E}
F^{s_3}*{D}(p_3)F^{s_4}*{E}(p_4)
\
&-
\mathcal K^{r;12}*{A*\ell|B C}
F^{s_1}*{B}(p_1)F^{s_2}*{C}(p_2)
\
&+
\mathcal K^{r;123}*{A*\ell|B C D}
F^{s_1}*{B}(p_1)F^{s_2}*{C}(p_2)F^{s_3}*{D}(p_3)
\
&+
\mathcal K^{r;124}*{A_\ell|B C E}
F^{s_1}*{B}(p_1)F^{s_2}*{C}(p_2)F^{s_4}*{E}(p_4)
\
&-
\mathcal K^{r;134}*{A_\ell|B D E}
F^{s_1}*{B}(p_1)F^{s_3}*{D}(p_3)F^{s_4}*{E}(p_4)
\
&-
\mathcal K^{r;234}*{A_\ell|C D E}
F^{s_2}*{C}(p_2)F^{s_3}*{D}(p_3)F^{s_4}_{E}(p_4)
\Big].
\end{aligned}
}
]

여기서 (B,C,D,E)는 각각 어떤 rank (n_i)의 PSTF multi-index 전체를 의미한다. 즉

[
B\equiv B_{n_1},\quad
C\equiv C_{n_2},\quad
D\equiv D_{n_3},\quad
E\equiv E_{n_4},
]

이고 각 rank에 대해 (n_i=0,\ldots,L_{\max})를 sum한다.

이 식은 weak anisotropy expansion이 아니다. (F_{A_\ell})들의 nonlinear product를 그대로 유지한다. 유일한 근사는 (L_{\max}) truncation뿐이다.

---

## 5. Kernel definition

이제 모든 복잡한 angular/kinematic structure를 (\mathcal K^{r;S})에 넣는다.

[
\sigma_1=\sigma_2=+1,
\qquad
\sigma_3=\sigma_4=-1.
]

[
\mathbf Q
=========

p_1e_1+p_2e_2-p_3e_3-p_4e_4.
]

그러면

[
\boxed{
\begin{aligned}
\mathcal K^{r;S}*{A*\ell|{B_i}*{i\in S}}
(p_1,p_2,p_3)
=&
\mathcal P_r(p_1,p_2,p_3,p_4)
\
&\times
\int
\prod*{j=1}^4d\Omega_j,
\frac{O_{A_\ell}(e_1)}{\Delta_\ell}
\prod_{i\in S}O_{B_i}(e_i)
\
&\times
\delta^{(3)}(\mathbf Q)
|\mathcal M_r|^2 .
\end{aligned}
}
]

여기서 radial prefactor는, 하나의 흔한 invariant convention에서는,

[
\mathcal P_r
============

\Theta(E_4-m_4c^2)
\frac{S_r(2\pi\hbar)^4}{2E_1}
\left[
\frac{g_2p_2^2}{(2\pi\hbar)^3,2E_2}
\right]
\left[
\frac{g_3p_3^2}{(2\pi\hbar)^3,2E_3}
\right]
\left[
\frac{g_4p_4}{(2\pi\hbar)^3,2c^2}
\right].
]

마지막 factor는

[
\int dp_4,
\frac{p_4^2}{2E_4}
\delta(E_1+E_2-E_3-E_4)
=======================

\frac{p_4}{2c^2}
]

에서 온다.

네 기존 HM implementation에서 prefactor convention이 다르면 이 (\mathcal P_r)만 교체하면 된다. 아래의 multipole algebra는 그대로다.

---

## 6. HM matrix element를 코드용 descriptor로 쓰기

HM류 (2\to2) weak matrix element는 보통 다음 구조로 쓸 수 있다:

[
\boxed{
|\mathcal M_r|^2
================

G_F^2
\left[
\sum_q\eta_{rq}
\Pi_{i_qj_q}\Pi_{k_ql_q}
+
m_e^2c^2
\sum_q\zeta_{rq}
\Pi_{i_qj_q}
\right].
}
]

우리 부호 convention에서는

[
\boxed{
\Pi_{ij}:=-P_i^aP_{ja}
======================

## \frac{E_iE_j}{c^2}

p_ip_j\mu_{ij},
}
]

[
\mu_{ij}:=e_i^ae_{ja}.
]

따라서 channel (r)는 코드에서 다음 descriptor로 표현하면 된다.

```text
channel r:
    species = (s1, s2, s3, s4)
    masses  = (m1, m2, m3, m4)
    symmetry_factor = S_r
    terms_bilinear:
        eta_q, (i_q, j_q), (k_q, l_q)
    terms_mass:
        zeta_q, (i_q, j_q)
    terms_mass_quartic:
        chi_q
```

Implementation note (2026-05-14): `src/rabbit/collisions/pstf_process_catalog.py`
now instantiates this descriptor form for finite-mass HM elastic `nu-e`,
finite-mass HM pair annihilation/creation via the in-tree crossing convention
`M2_nu_nubar_to_ee(s,t,u) = M2_nu_e_elastic(u,t,s)`, and pairwise diagonal
no-QKE `nu-nu` channels.  The finite-mass pair descriptor uses `Pi_ij Pi_kl`,
`m_e^2 Pi_ij`, and the newly explicit `m_e^4` constant term needed by the
crossed HM form.  The default staged supported catalog enumerates the
`{nue,nuebar,nux}` banks using separate finite-mass elastic `nu-e_minus` and
`nu-e_plus` entries plus finite-mass pair entries, while the UR catalog helper
remains available as an explicit compatibility/reference mode.  The `nu-nu`
default remains limited to
the six ordered off-diagonal pairs used by the staged pairwise bridge.  The
pairwise descriptor helper also exposes explicit reference Fierz factors: `2`
for identical target/partner banks and `1` for off-diagonal banks.
`pstf_process_particle_mode_labels(...)` maps those descriptors onto the live
neutrino and electron/positron distribution labels used by the AP6 radial
provider.  The provider supports fixed bath modes, dynamic ultra-relativistic
Fermi-Dirac bath callbacks, and dynamic finite-mass zero-chemical-potential
Fermi-Dirac bath callbacks that use the live AP18 `T_gamma_MeV` payload and the
`T_nu_e_MeV` energy scale.  The LRS `pstf_radial` route now builds
descriptor-label-aware radial grids instead of smoke-only placeholder momentum
fractions: neutrino legs use `p=q`, electron/positron legs use
`p=sqrt(max(q^2-(m_e/T_nu_e0)^2,0))` on the frozen dimensionless solver grid,
and the finite-mass electron/positron bath is evaluated on the corresponding
total-energy grid.  The finite-mass provider also
supports signed electron/positron chemical-potential inputs.  The LRS
`pstf_radial` route can now pass an explicit fixed
`electron_chemical_potential_MeV`, using the opposite signs for `e_minus` and
`e_plus` bath labels, but the default route remains zero-chemical-potential
and does not evolve charge asymmetry.  A deterministic AP6 sensitivity
artifact evaluates the same radial source directly across fixed `mu_e` values
and records concrete `dQ_nue_pair_N`, `dQ_nux_bank_N`, radial-energy,
radial-number, and `radial_max_abs_C_mode` deltas against the zero-`mu_e`
row.  The finite-mass EOS module also provides a bounded charge-neutrality
root solve for `n_e- - n_e+`, and the LRS route can opt into
`electron_chemical_potential_mode="charge_neutrality"` to derive `mu_e` from
the current network mass fractions, `T_gamma`, and `eta`; the corresponding
artifact records the solved `mu_e` and source deltas against the default
zero-`mu_e` row.  This is an algebraic bath closure, not an evolved
charge-asymmetry state.  The LRS route now
evaluates all 15 default supported descriptors with finite-mass elastic and
pair-annihilation matrix elements plus a temperature-updated finite-mass
electron/positron bath; this is still a staged smoke route, not QED-corrected
EOS feedback, live charge-asymmetry evolution, or a full electron
thermodynamics solve.
`evaluate_pstf_process_radial_collision_source(...)` also composes
the descriptor, radial grid, and six-monomial contraction into concrete
`C_modes` values for deterministic smoke-scale inputs.  The follow-up
`compute_pstf_process_radial_moments(...)` integrates a selected returned mode
into raw-quadrature number and energy moments, `sum_i w_i E_i^2 C_i` and
`sum_i w_i E_i^3 C_i`, so the next bridge can carry concrete collision
feedback numbers instead of only structural contracts.  The AP18 adapter
`build_augmented_pstf_radial_moment_thermo_source(...)` then groups those
moments by neutrino species and maps energy moments into
`dQ_nue_pair_N`/`dQ_nux_bank_N` for explicit 3T callback use.  The live
provider `build_augmented_pstf_radial_moment_provider(...)` reconstructs
current occupation modes from the AP18 `A_modes`/`q_nodes` payload, evaluates
configured radial process descriptors, and feeds those moment objects into the
same thermo-source adapter.  The LRS and non-LRS AP18 source evaluators now
have focused regressions showing that this live radial bridge returns finite
`dQ_nue_pair_N`/`dQ_nux_bank_N` values with nonzero radial `C_modes`
diagnostics for pairwise diagonal no-QKE `nu-nu` configurations.  The LRS
collision-feedback validation artifact/candidate gate can now select
`source_variant="pstf_radial"`, which routes the configured radial provider
through the AP18 3T source callback and uses a frozen-initial-state source by
default for smoke-scale stability.  The unbudgeted `live_rhs` policy is
rejected for that variant until repeated radial contractions have a dedicated
runtime gate.  A first tiny-span budgeted live-RHS diagnostic artifact now runs
that same radial provider inside the 3T RHS with an explicit source-evaluation
budget and records either concrete radial source diagnostics or a fail-closed
budget-exceeded result.  A companion source-policy artifact runs that live RHS
row against the frozen-initial-state route at matched smoke settings and records
live-minus-frozen observable/source-diagnostic deltas.  This is still a no-QKE deterministic
collision-reference layer and staged validation route, not a promoted
collision-coupled BBN runtime.

Implementation note (2026-05-15): the AP6 radial provider now applies
conserved-moment projections to all default diagonal `nu-nu` sources before
thermo/hierarchy feedback.  Identical-bank self-scattering remains
number/energy-neutral, while off-diagonal `nu_alpha+nu_beta` elastic
scattering is number-neutral for each output species and energy-neutral over
the complete unordered pair while preserving the relative raw species
energy-transfer difference.  The AP55 source-budget artifact now exposes
all-nine diagnostics:
`n_radial_nunu_sources=9`, `n_radial_nunu_number_projected_sources=9`,
`n_radial_offdiagonal_nunu_number_projected_sources=6`,
`n_radial_offdiagonal_nunu_pair_energy_projected_sources=6`, and
`n_radial_offdiagonal_nunu_pair_energy_projected_pairs=3`; a real LRS
source-budget smoke had `radial_nunu_max_abs_number_moment =
4.828087799349512e-20`,
`radial_offdiagonal_nunu_max_abs_number_moment = 2.498747194400186e-20`, and
`radial_offdiagonal_nunu_pair_max_abs_energy_residual =
9.317362419797304e-20`.

그러면 kernel은

[
\boxed{
\mathcal K^{r;S}
================

\mathcal P_rG_F^2
\left[
\sum_q\eta_{rq},
\mathcal D^{S;(i_qj_q)(k_ql_q)}
+
m_e^2c^2
\sum_q\zeta_{rq},
\mathcal D^{S;(i_qj_q)}
\right].
}
]

이제 (\mathcal D)는 universal geometric kernels다. reaction channel마다 새로 derivation할 필요가 없다.

---

## 7. Universal geometric kernels

먼저

[
\boxed{
\begin{aligned}
\mathcal G^{S;Q}*{A*\ell|{B_i}*{i\in S}}
:=
\int
\prod*{j=1}^4d\Omega_j,
\frac{O_{A_\ell}(e_1)}{\Delta_\ell}
\prod_{i\in S}O_{B_i}(e_i)
\delta^{(3)}(\mathbf Q)
Q .
\end{aligned}
}
]

필요한 (Q)는 세 종류뿐이다:

[
Q=1,\qquad
Q=\mu_{ij},\qquad
Q=\mu_{ij}\mu_{kl}.
]

따라서

[
\mathcal G^{S;0}
\equiv
\mathcal G^{S;1},
]

[
\mathcal G^{S;ij}
\equiv
\mathcal G^{S;\mu_{ij}},
]

[
\mathcal G^{S;ij,kl}
\equiv
\mathcal G^{S;\mu_{ij}\mu_{kl}}.
]

그러면

[
\boxed{
\mathcal D^{S;(ij)}
===================

\frac{E_iE_j}{c^2}
\mathcal G^{S;0}
----------------

p_ip_j
\mathcal G^{S;ij}.
}
]

그리고

[
\boxed{
\begin{aligned}
\mathcal D^{S;(ij)(kl)}
=&
\frac{E_iE_jE_kE_l}{c^4}
\mathcal G^{S;0}
\
&-
\frac{E_iE_jp_kp_l}{c^2}
\mathcal G^{S;kl}
-----------------

\frac{p_ip_jE_kE_l}{c^2}
\mathcal G^{S;ij}
\
&+
p_ip_jp_kp_l
\mathcal G^{S;ij,kl}.
\end{aligned}
}
]

이게 가장 compact한 kernel factorization이다.

즉 구현에서는 HM channel coefficient (\eta,\zeta)만 바꿔 끼우고, 나머지는 전부 universal (\mathcal G) table을 재사용하면 된다.

---

## 8. 최종 코드형 수식

compressed PSTF mode index를 쓰자.

[
\lambda=(\ell,\sigma),
\qquad
F^s_\lambda[p_i]\equiv F^s_\lambda(p_i).
]

입자 1의 output mode를 (a)라고 쓰면, 한 채널 (r)의 contribution은

[
\boxed{
\begin{aligned}
C^{s_1,r}_{a}[i_1]
==================

\sum_{i_2,i_3}
w_{i_2}w_{i_3}
\Big[
&
K^{r;34}*{a d e}[i_1,i_2,i_3],
F^{s_3}*{d}[i_3],
\widetilde F^{s_4}_{e}[i_1,i_2,i_3]
\
&
-

K^{r;12}*{a b c}[i_1,i_2,i_3],
F^{s_1}*{b}[i_1],
F^{s_2}*{c}[i_2]
\
&
+
K^{r;123}*{a b c d}[i_1,i_2,i_3],
F^{s_1}*{b}[i_1],
F^{s_2}*{c}[i_2],
F^{s_3}*{d}[i_3]
\
&
+
K^{r;124}*{a b c e}[i_1,i_2,i_3],
F^{s_1}*{b}[i_1],
F^{s_2}*{c}[i_2],
\widetilde F^{s_4}_{e}[i_1,i_2,i_3]
\
&
-

K^{r;134}*{a b d e}[i_1,i_2,i_3],
F^{s_1}*{b}[i_1],
F^{s_3}*{d}[i_3],
\widetilde F^{s_4}*{e}[i_1,i_2,i_3]
\
&
-

K^{r;234}*{a c d e}[i_1,i_2,i_3],
F^{s_2}*{c}[i_2],
F^{s_3}*{d}[i_3],
\widetilde F^{s_4}*{e}[i_1,i_2,i_3]
\Big].
\end{aligned}
}
]

여기서

[
\widetilde F^{s_4}_{e}[i_1,i_2,i_3]
===================================

F^{s_4}_{e}(p_4(i_1,i_2,i_3)).
]

만약 (p_4)가 grid point가 아니면 interpolation으로

[
\widetilde F^{s_4}_{e}[i_1,i_2,i_3]
===================================

\sum_{j_4}
I[i_1,i_2,i_3,j_4],
F^{s_4}_{e}[j_4]
]

라고 둔다.

이 interpolation matrix (I)를 kernel에 흡수하면

[
K[i_1,i_2,i_3,j_4]
]

형태가 되고, 완전히 tensor contraction으로 바뀐다. 다만 memory가 커질 수 있으니 JAX에서는 보통 (i_2,i_3)를 chunking하거나 `lax.scan`으로 돌리는 게 안전하다.

---

## 9. JAX pseudocode 형태

대략 이런 구조가 된다.

```python
def collision_channel(F, channel, K, p4_interp):
    # F[species, ip, mode]
    # K[S][i1, i2, i3, out_mode, in_modes...]
    # p4_interp returns F4_tilde[i1, i2, i3, mode]

    s1, s2, s3, s4 = channel.species

    F1 = F[s1]  # [Np, Nm]
    F2 = F[s2]
    F3 = F[s3]
    F4 = F[s4]

    # output C1: [Np, Nm]
    C1 = 0

    for i1 in range(Np):
        for i2 in range(Np):
            for i3 in range(Np):
                f1 = F1[i1]                       # [Nm]
                f2 = F2[i2]
                f3 = F3[i3]
                f4 = interp_p4(F4, i1, i2, i3)    # [Nm]

                C1[i1] += w[i2] * w[i3] * (
                    einsum("ade,d,e->a",    K["34"][i1,i2,i3],  f3, f4)
                  - einsum("abc,b,c->a",    K["12"][i1,i2,i3],  f1, f2)
                  + einsum("abcd,b,c,d->a", K["123"][i1,i2,i3], f1, f2, f3)
                  + einsum("abce,b,c,e->a", K["124"][i1,i2,i3], f1, f2, f4)
                  - einsum("abde,b,d,e->a", K["134"][i1,i2,i3], f1, f3, f4)
                  - einsum("acde,c,d,e->a", K["234"][i1,i2,i3], f2, f3, f4)
                )

    return C1
```

실제 JAX에서는 Python loop를 쓰면 안 되고, (i_2,i_3) 방향은 `vmap` 또는 `lax.scan`으로 chunking하는 게 맞다. 핵심 구조만 보면 위와 같다.

---

## 10. Kernel precomputation strategy

실제로는 (K^{r;S})를 바로 저장하지 말고, 다음 순서로 쪼개는 게 좋아.

먼저 universal angular table:

[
\mathcal G^{S;0},
\qquad
\mathcal G^{S;ij},
\qquad
\mathcal G^{S;ij,kl}
]

를 만든다.

그다음 channel descriptor에서

[
\mathcal D^{S;(ij)},
\qquad
\mathcal D^{S;(ij)(kl)}
]

를 조합한다.

마지막으로

[
\mathcal K^{r;S}
================

\mathcal P_rG_F^2
\left[
\sum_q\eta_{rq}\mathcal D^{S;q}
+
m_e^2c^2
\sum_q\zeta_{rq}\mathcal D^{S;q}
\right]
]

를 만든다.

즉 코드 구조는

```text
PSTF basis
    ↓
universal G kernels
    ↓
D kernels for Pi_ij and Pi_ij Pi_kl
    ↓
channel-specific K kernels
    ↓
six monomial contraction
```

이렇게 가는 게 제일 깨끗하다.

---

## 11. Isotropic limit check

isotropic limit에서는

[
F^s_{A_\ell}=0,\qquad \ell\ge1.
]

그러면 모든 input mode가 monopole이고, rotational invariance 때문에

[
C_{A_\ell}=0,\qquad \ell\ge1.
]

남는 것은

[
C_\varnothing(p_1)
]

뿐이다. 이게 HM scalar collision integral로 돌아가야 한다. 이걸 첫 번째 unit test로 삼으면 된다.

두 번째 unit test는 equilibrium detailed balance다.

[
f_i=f_{\rm FD}(E_i)
]

이면

[
\Lambda_r=0
]

이어야 하므로

[
C_{A_\ell}=0
]

가 나와야 한다.

세 번째 unit test는 (\nu\nu) elastic scattering에서 total particle number와 total four-momentum moment가 보존되는지다. 특히 truncation이나 interpolation이 들어가면 이 test가 제일 먼저 깨진다.

---

## 12. 차원과 부호 체크

[
F^s_{A_\ell}
]

는 distribution function의 multipole이므로 dimensionless다.

[
\Pi_{ij}
========

## \frac{E_iE_j}{c^2}

p_ip_j\mu_{ij}
]

는 momentum squared 차원을 갖는다.

따라서

[
\Pi_{ij}\Pi_{kl}
]

와

[
m_e^2c^2\Pi_{ij}
]

는 둘 다 momentum(^4) 차원이다. HM matrix element의 두 종류 term이 같은 차원을 갖는다는 점에서 consistency check가 된다.

최종 (C_{A_\ell})의 차원은 네가 Boltzmann equation을 어떻게 쓰느냐에 따라 달라진다.

[
P^a\nabla_a f=C[f]
]

로 쓰면 (C)는 momentum/time 계열 차원을 갖고, local-time hierarchy

[
\dot F_{A_\ell}=C_{A_\ell}^{\rm time}+\cdots
]

로 쓰면

[
[C_{A_\ell}^{\rm time}]=\mathrm{s}^{-1}.
]

따라서 prefactor (\mathcal P_r)에서 (E_1) 또는 (c) factor를 네 기존 Rodas5P RHS convention과 반드시 맞춰야 한다.

---

## 최종 압축형

구현용으로는 이 한 줄을 핵심 식으로 보면 돼.

[
\boxed{
C^{s_1,r}_{a}
=============

## K^{r;34}_{ade}F^3_dF^4_e

K^{r;12}*{abc}F^1_bF^2_c
+
K^{r;123}*{abcd}F^1_bF^2_cF^3_d
+
K^{r;124}_{abce}F^1_bF^2_cF^4_e
-------------------------------

## K^{r;134}_{abde}F^1_bF^3_dF^4_e

K^{r;234}_{acde}F^2_cF^3_dF^4_e .
}
]

여기서 모든 momentum integration, (p_4) interpolation, matrix element, angular momentum conservation, PSTF projection은 (K^{r;S}) 안에 들어간다.

이 형태가 JAX에 제일 잘 맞는다. `K`를 static array 또는 chunked generator로 두고, (F)만 runtime state로 두면 된다. 가장 중요한 이점은 **blocking factor를 (\bar F=1-F)로 직접 다루지 않고, six-monomial polynomial로 처리한다는 것**이야. 이렇게 하면 코드가 짧아지고, kernel storage도 four-(F) product가 아니라 최대 three-(F) contraction으로 끝난다.

[1]: https://arxiv.org/pdf/astro-ph/9804316?utm_source=chatgpt.com "1+3 Covariant Cosmic Microwave Background anisotropies I"
[2]: https://arxiv.org/abs/astro-ph/9506015?utm_source=chatgpt.com "Neutrino decoupling in the early Universe"
