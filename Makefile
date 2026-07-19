.PHONY: help test test-fast check-registry release-smoke sync-counts profile-kernels profile-batch-inference

help:
	@echo "RABBIT v1.0 development targets"
	@echo ""
	@echo "  test             Run all pytest gates (slow + production)"
	@echo "  test-fast        Run pytest excluding @slow tests"
	@echo "  check-registry   v3.2 P3: registry-vs-test-list drift gate (fast)"
	@echo "  release-smoke    All @release_smoke gates including registry drift"
	@echo "  sync-counts      Refresh README/STATUS pytest collection counts"
	@echo "  profile-kernels  Profile staged JAX helper kernels"
	@echo "  profile-batch-inference  Profile batched JAX inference throughput"
	@echo ""

test:
	JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
		. venv/bin/activate && python -m pytest -v

test-fast:
	JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
		. venv/bin/activate && python -m pytest -v -m "not slow"

# v3.2 P3 (hostile re-audit fix): meta-test that fails LOUDLY on
# registry-vs-test-list drift. Run before any release tag to catch
# the class of silent drift that was missed across v3.0/v3.1/v3.2.
check-registry:
	JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
		. venv/bin/activate && python -m pytest -q \
			tests/test_registry_drift_gate.py \
			tests/test_inference_hierarchy_lock.py

# v3.2 P3: full release-readiness smoke gate. Anything tagged
# @release_smoke must pass before tagging vN.x.0(-rcN).
release-smoke:
	JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
		. venv/bin/activate && python -m pytest -v -m release_smoke

sync-counts:
	JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
		. venv/bin/activate && python scripts/sync_test_counts.py

profile-kernels:
	JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
		. venv/bin/activate && python scripts/profile_kernel_backends.py

profile-batch-inference:
	JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
		. venv/bin/activate && python scripts/profile_batch_inference.py --likelihood
