# AD Gradients Promotion Packet

**Feature**: Differentiable BBN forward model via custom_vjp
**Promotion target**: Candidate (diagnostic AD, restricted parameter set)
**Date**: 2026-04-05

## 1. Capability Spec

### Supported
- JAX backend only (custom_vjp mechanism)
- Parameters: Sigma_H, eta (primary differentiable parameters)
- Gradient verification: built-in FD cross-check via gradient_check()
- Sensitivity analysis and diagnostic use

### NOT supported
- Publication-grade NUTS/HMC inference with native gradients
- Gradients through phase boundaries (discontinuous)
- SciPy backend (no AD path)
- Gradients w.r.t. N_q, correction_level, or discrete parameters

### Mechanism
custom_vjp with FD fallback for verification. NOT pure forward-mode AD.

## 2. Permitted / Forbidden Claims
- ✅ "Diagnostic AD gradients for sensitivity analysis"
- ✅ "FD-verified custom_vjp bridge"
- ❌ "Full differentiable Bianchi-BBN solver"
- ❌ "Gradient-based inference ready"

---

# Inference Promotion Packet

**Feature**: Bayesian inference pipeline
**Promotion target**: Candidate (exploratory parameter estimation only)
**Date**: 2026-04-05

## 1. Capability Spec

### Supported (exploratory)
- Likelihood wrapper around canonical_forward_solver
- dynesty nested sampling
- Grid emulator for fast approximate posterior
- Synthetic injection/recovery tests (component-level)

### NOT supported for headline claims
- Bayes factor B₀₁ for anisotropy (forward model maturity insufficient)
- NUTS/HMC with native AD (AD bridge is diagnostic only)
- Model comparison evidence (requires null FPR + injection power characterization)

### Required before evidence headlines
1. Null false-positive rate < 5% on isotropic synthetic
2. Injection recovery within 2σ
3. Prior sensitivity analysis
4. Posterior predictive check on real observables
5. Cross-sampler consistency

## 2. Permitted / Forbidden Claims
- ✅ "Exploratory Bayesian inference pipeline"
- ✅ "Parameter estimation with dynesty nested sampling"
- ❌ "Bayes factor B₀₁ = X conclusively favours anisotropy"
- ❌ "Publication-grade model comparison"
