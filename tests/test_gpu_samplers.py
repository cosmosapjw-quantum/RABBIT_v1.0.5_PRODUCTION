"""
Test: GPU-friendly parameter estimation pipeline.

§1. NUTS (BlackJAX) — Gaussian target recovery
§2. NSS (PolyChord-like) — Gaussian evidence computation
§3. API contracts — config, result schema, imports
§4. Integration — BBN wrappers importable
"""
import pytest
import numpy as np

pytest.importorskip("jax", reason="JAX required")
pytest.importorskip("blackjax", reason="BlackJAX required for NUTS")


@pytest.fixture(scope="module")
def jax_setup():
    import jax; jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════
# §1. NUTS on simple Gaussian
# ═══════════════════════════════════════════════════════════════

class TestNUTSGaussian:
    """NUTS should recover a 2D Gaussian posterior."""

    @pytest.fixture(scope="class")
    def nuts_result(self, jax_setup):
        import jax.numpy as jnp
        from rabbit.inference.jax_nuts import run_nuts, NUTSConfig

        # Target: N(μ=[3,5], Σ=diag(0.1², 0.2²))
        mu_true = jnp.array([3.0, 5.0])
        sigma = jnp.array([0.1, 0.2])

        def log_posterior(x):
            return -0.5 * jnp.sum(((x - mu_true) / sigma) ** 2)

        config = NUTSConfig(num_warmup=200, num_samples=500, num_chains=1)
        return run_nuts(log_posterior, jnp.array([2.5, 4.5]), config)

    def test_success(self, nuts_result):
        assert nuts_result.samples.shape == (500, 2)

    def test_mean_recovery(self, nuts_result):
        mean = np.mean(nuts_result.samples, axis=0)
        assert abs(mean[0] - 3.0) < 0.15, f"μ₁={mean[0]:.3f}"
        assert abs(mean[1] - 5.0) < 0.3, f"μ₂={mean[1]:.3f}"

    def test_std_recovery(self, nuts_result):
        std = np.std(nuts_result.samples, axis=0)
        assert abs(std[0] - 0.1) < 0.05, f"σ₁={std[0]:.3f}"
        assert abs(std[1] - 0.2) < 0.1, f"σ₂={std[1]:.3f}"

    def test_acceptance_rate(self, nuts_result):
        assert 0.5 < nuts_result.acceptance_rate < 1.0

    def test_no_divergences(self, nuts_result):
        assert nuts_result.num_divergent == 0

    def test_metadata(self, nuts_result):
        assert nuts_result.metadata['sampler'] == 'blackjax_nuts'
        assert nuts_result.metadata['n_dim'] == 2


# ═══════════════════════════════════════════════════════════════
# §2. NSS on simple Gaussian (evidence computation)
# ═══════════════════════════════════════════════════════════════

class TestNSSGaussian:
    """NSS should compute correct evidence for a Gaussian likelihood."""

    @pytest.fixture(scope="class")
    def nss_result(self, jax_setup):
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        from rabbit.inference.jax_nested import run_nss, NSSConfig

        # 1D Gaussian: L(x) = N(x; 0, 1), π(x) = U(-10, 10)
        # Exact ln(Z) = ln(1/20) ≈ -3.00 (Gaussian integral / prior volume)
        # More precisely: Z = ∫ N(x;0,1) × (1/20) dx = (1/20)×1 = 0.05
        # ln(Z) = ln(0.05) ≈ -2.996

        def loglike(x):
            return -0.5 * x[0]**2

        def logprior(x):
            return jnp.where((x[0] >= -10) & (x[0] <= 10), jnp.log(1.0/20.0), -jnp.inf)

        def prior_sample(key):
            return jr.uniform(key, shape=(1,), minval=-10.0, maxval=10.0)

        config = NSSConfig(
            n_live=25, max_iterations=200,
            n_repeats_factor=3.0, dlogz_threshold=0.5,
            slice_max_steps=20)

        return run_nss(loglike, logprior, prior_sample, 1, config,
                       jax.random.PRNGKey(123))

    def test_converged(self, nss_result):
        assert nss_result.n_dead > 0

    def test_evidence_order(self, nss_result):
        """ln(Z) should be near -3.0 (within ~2 nats for small n_live=25)."""
        exact = np.log(0.05)  # ≈ -2.996
        assert abs(nss_result.log_evidence - exact) < 2.5, (
            f"ln(Z)={nss_result.log_evidence:.2f} vs exact={exact:.2f}")

    def test_positive_information(self, nss_result):
        assert nss_result.H > 0

    def test_samples_finite(self, nss_result):
        assert np.all(np.isfinite(nss_result.samples))

    def test_metadata(self, nss_result):
        assert nss_result.metadata['sampler'] == 'jax_nss_polychord_like'
        assert nss_result.metadata['n_dim'] == 1


# ═══════════════════════════════════════════════════════════════
# §3. API contracts
# ═══════════════════════════════════════════════════════════════

class TestAPIContracts:

    def test_nuts_config_defaults(self):
        from rabbit.inference.jax_nuts import NUTSConfig
        c = NUTSConfig()
        assert c.num_warmup == 500
        assert c.num_samples == 2000
        assert c.target_acceptance_rate == 0.80

    def test_nss_config_defaults(self):
        from rabbit.inference.jax_nested import NSSConfig
        c = NSSConfig()
        assert c.n_live == 250
        assert c.n_delete == 1
        assert c.dlogz_threshold == 0.01

    def test_nuts_result_fields(self):
        from rabbit.inference.jax_nuts import NUTSResult
        r = NUTSResult(
            samples=np.zeros((10, 2)), log_posterior=np.zeros(10),
            acceptance_rate=0.8, num_divergent=0, step_size=0.1,
            wall_time_s=1.0, num_warmup=100, num_samples=10, num_chains=1)
        assert hasattr(r, 'metadata')

    def test_nss_result_fields(self):
        from rabbit.inference.jax_nested import NSSResult
        r = NSSResult(
            log_evidence=-5.0, log_evidence_error=0.5,
            samples=np.zeros((10, 2)), log_weights=np.zeros(10),
            log_likelihoods=np.zeros(10), n_dead=10, n_like_calls=1000,
            n_iterations=50, wall_time_s=1.0, H=2.0)
        assert hasattr(r, 'metadata')


# ═══════════════════════════════════════════════════════════════
# §4. BBN wrappers importable
# ═══════════════════════════════════════════════════════════════

class TestBBNWrappers:

    def test_nuts_bbn_importable(self):
        from rabbit.inference.jax_nuts import run_bbn_nuts
        assert callable(run_bbn_nuts)

    def test_nss_bbn_importable(self):
        from rabbit.inference.jax_nested import run_bbn_nss
        assert callable(run_bbn_nss)
