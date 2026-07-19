# Class A Promotion Packet

**Feature**: Class A orthogonal diagonal Bianchi BBN (6 types)
**Promotion target**: Candidate (reduced-transport, documented approximation)
**Date**: 2026-04-05

---

## 1. Capability Spec

### Supported
- 6 orthogonal diagonal types: I, II, VI₀, VII₀, VIII, IX
- Exact background geometry (curvature source S±, Gauss curvature K)
- Reduced κ-cascade transport (NOT exact curved PSTF)
- CL0–CL2 correction levels
- N_q = 6–20

### NOT supported (explicitly)
- Exact curved PSTF transport with ℓ→ℓ±1 coupling matrices
- Class B types (IV, V, VIh, VIIh)
- Tilted + Class A combined
- Teff + Class A combined

### Critical approximation contract
**Geometry is exact. Transport is approximate.** The curved hierarchy uses
a reduced κ-cascade closure, NOT the full K_{ℓℓ'} coupling matrices.
This means:
- ℓ_max = 2 closure is exact ONLY for Type I (κ=0)
- For curved types, the transport truncation error grows with κ
- The proxy is accurate to ~1% for |κ| < 0.1

---

## 2. Physics Contract

### Type I limit recovery
- N1=N2=N3=0 → curvature source S±=0, K=0 (exact)
- Geometry RHS reduces exactly to Type I RHS
- Friedmann constraint: Ω=1-Σ² (exact)

### Curvature sources (per type)
- Type I: S=0, K=0
- Type II: S+ = -N₁²/3, K = N₁²/4
- Type VII₀: S = 0 (isotropic curvature cancellation), K=0
- Type IX: S = 0, K = -(N₁N₂+N₂N₃+N₃N₁)/4

---

## 3. Validation Matrix

| Test | Status |
|---|---|
| Type I limit (S=0, K=0 at N_i=0) | ✅ PASS |
| Distinct curvature by type | ✅ PASS |
| Smooth curvature deformation | ✅ PASS (dΣ+ varies <4% for N1∈[0,0.1]) |
| Friedmann constraint | ✅ PASS |
| Geometry RHS functional | ✅ PASS |

---

## 4. Support Matrix

| Config | Status |
|---|---|
| Type I via Class A driver | **Production** |
| Type II, small curvature | **Production** (reduced transport) |
| Type VII₀, IX | **Beta** (larger curvature, proxy accuracy TBD) |
| Exact curved PSTF | **Disabled** (not implemented) |

---

## 5. Permitted / Forbidden Claims

### Permitted
- ✅ "Class A BBN with exact background geometry and reduced κ-cascade transport"
- ✅ "Type I limit recovery verified"
- ✅ "6 orthogonal diagonal types supported"

### Forbidden
- ❌ "Exact curved Bianchi transport solver"
- ❌ "Full curved PSTF hierarchy with K_{ℓℓ'} coupling"
- ❌ "Publication-grade all-type Bianchi BBN"
