#!/usr/bin/env python3
"""Package a release zip with mandatory pre-flight checks.

Usage: python scripts/package_release.py [output_path]

Pre-flight pipeline:
  [0] require a clean tracked tree
  [1] render registries and sync test counts
  [2] require generated sources to remain unchanged
  [3] pytest exact-count and fast production gates
  [4] generate RELEASE_MANIFEST.json
  [5] create zip (only if all above pass)
"""
import subprocess, sys, re, os, json, glob, zipfile, datetime


FAST_PRODUCTION_TIMEOUT_SECONDS = 600


def build_env(extra=None):
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("RABBIT_JAX_CACHE_DIR", "/tmp/rabbit_jax_cache")
    if extra:
        env.update(extra)
    return env


def run(cmd, timeout=120, check=False):
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=build_env(),
        )
    except subprocess.TimeoutExpired:
        print(
            "RELEASE BLOCKED: command timed out after "
            f"{timeout} seconds: {' '.join(cmd)}"
        )
        raise SystemExit(124) from None
    if check and r.returncode != 0:
        print(f"FAIL: {' '.join(cmd)}")
        print(r.stdout[-500:] if r.stdout else "")
        print(r.stderr[-300:] if r.stderr else "")
        sys.exit(1)
    return r


def require_clean_tracked_tree(context):
    """Fail closed on staged or unstaged tracked changes."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if status.returncode != 0:
        print(f"RELEASE BLOCKED: cannot verify clean tracked tree ({context}).")
        raise SystemExit(1)
    changed = [line for line in status.stdout.splitlines() if line.strip()]
    if changed:
        print(f"RELEASE BLOCKED: tracked tree is dirty ({context}).")
        for line in changed:
            print(f"  {line}")
        raise SystemExit(1)


def release_path_allowed(path):
    """Exclude historical manuscript artifacts until the publication freeze."""
    normalized = str(path).replace(os.sep, "/")
    return not normalized.startswith("docs/RABBIT_report/")


def get_counts():
    def _run(marker=None):
        cmd = [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"]
        if marker:
            cmd.extend(["-m", marker])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=build_env())
        if marker:
            m = re.search(r"(\d+)/\d+ tests", r.stdout)
        else:
            m = re.search(r"(\d+) tests? collected", r.stdout)
        return int(m.group(1)) if m else 0
    return {
        "total": _run(),
        "production_total": _run("production"),
        "production_not_slow": _run("production and not slow"),
        "gold": _run("gold"), "smoke": _run("release_smoke"),
        "production_slow": _run("production and slow"),
        "files": len(glob.glob("tests/test_*.py")),
    }


def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "RABBIT_v9rc_release.zip"
    print("═" * 60)
    print("RABBIT Release Packaging — Mandatory Pre-flight")
    print("═" * 60)

    require_clean_tracked_tree("before pre-flight")

    # [0] Render capability tables from registries
    print("\n[0/5] Rendering capability tables from registries...")
    r = run([sys.executable, "scripts/render_capability_tables.py", "--apply"], check=True)
    print("  ✅ Tables rendered.")

    # [1] Sync counts
    print("\n[1/6] Syncing test counts...")
    run([sys.executable, "scripts/sync_test_counts.py"], check=True)
    print("  ✅ Counts synced.")

    # Rendering/count sync are verification operations for a release commit.
    # If either rewrites a tracked source, the generated update must be reviewed
    # and committed before packaging can continue.
    require_clean_tracked_tree("generated sources changed during pre-flight")

    # [2] Exact-count sync tests (HARD GATE)
    print("\n[2/6] Running exact-count sync tests (build-env only)...")
    r = subprocess.run([sys.executable, "-m", "pytest",
             "tests/test_registry_sync.py::TestExactCountSync",
             "-v", "--tb=short"],
             capture_output=True, text=True, timeout=60, env=build_env({"RABBIT_BUILD_ENV": "1"}))
    if r.returncode != 0:
        print("═" * 60)
        print("RELEASE BLOCKED: Exact-count sync tests FAILED (build-env).")
        print("README/STATUS counts do not match actual pytest collection.")
        print("Run 'python scripts/sync_test_counts.py' and retry.")
        print("═" * 60)
        print(r.stdout[-500:])
        sys.exit(1)
    print("  ✅ Exact-count sync: PASS")

    # [3] Fast production gates
    print("\n[3/6] Running fast production gates...")
    r = run([sys.executable, "-m", "pytest", "tests/",
             "-m", "production and not slow", "-q", "--tb=line"],
            timeout=FAST_PRODUCTION_TIMEOUT_SECONDS)
    last = r.stdout.strip().split('\n')[-1] if r.stdout else ""
    if r.returncode != 0:
        print(f"RELEASE BLOCKED: Fast gates failed.\n  {last}")
        sys.exit(1)
    print(f"  ✅ Fast gates: {last}")

    # [4] Generate RELEASE_MANIFEST.json
    print("\n[4/6] Generating RELEASE_MANIFEST.json...")
    counts = get_counts()
    # Build-env fingerprint
    import platform
    opt_deps = {}
    for dep in ["jax", "jaxlib", "blackjax", "diffrax", "equinox"]:
        try:
            mod = __import__(dep)
            opt_deps[dep] = getattr(mod, "__version__", "installed")
        except ImportError:
            opt_deps[dep] = None

    # Git provenance
    git_hash = "unknown"
    git_dirty = False
    try:
        gh = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if gh.returncode == 0:
            git_hash = gh.stdout.strip()[:12]
    except Exception:
        git_dirty = True

    manifest = {
        "package": "rabbit",
        "version": "9.0.0rc1",
        "build_timestamp": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_hash,
        "git_dirty": git_dirty,
        "optional_deps": opt_deps,
        "test_counts": counts,
        "exact_count_sync_build_env": "PASS",
        "counts_portable_across_envs": False,
        "fast_gates": "PASS",
        "render_tables_applied": True,
        "render_scope": [
            "README.md: README_HEADER, IDENTITY, CORE_SUMMARY, README_BACKENDS_HEADER, BACKEND_TABLE, README_QUICKSTART, TEST_COUNTS, README_FOOTER",
            "STATUS.md: STATUS_HEADER, TIER_SUMMARY, STATUS_BACKENDS, STATUS_DETAIL (incl. STATUS_TEST_COUNTS)",
            "SUPPORTED_CAPABILITIES.md: SC_HEADER, BACKEND_TABLE, SC_CANONICAL_CORE, FEATURE_TABLE, CLASSB_LAYERS, CLAIMS",
            "PROMOTION_GATES.md: PG_HEADER, PROMOTION_STATUS, PG_BODY, NEXT_QUEUE, PG_FOOTER",
            "docs/BACKEND_CAPABILITY_MATRIX.md: full overwrite",
        ],
        "render_targets_count": 23,
        "excluded_paths": [
            "docs/RABBIT_report/ (historical manuscript; excluded until M-08/B-06 freeze)"
        ],
        "pipeline": "scripts/package_release.py",
        "note": "Counts are build-environment metadata, not portable invariants. "
                "Optional JAX/BlackJAX deps affect collection. "
                "Run 'make sync-counts' to update for your environment."
    }
    with open("RELEASE_MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  ✅ RELEASE_MANIFEST.json (total={counts['total']}, gold={counts['gold']})")

    # [5] Package
    print(f"\n[5/6] Creating {output}...")
    includes = ["pyproject.toml", "README.md", "STATUS.md", "RELEASE_MANIFEST.json",
                "SUPPORTED_CAPABILITIES.md", "PROMOTION_GATES.md", "REBUTTAL.md", "Makefile"]
    dirs = ["src/rabbit", "tests", "scripts", "docs"]
    excludes = {"__pycache__", ".pyc", ".egg-info"}

    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in includes:
            if os.path.exists(f):
                zf.write(f)
        for d in dirs:
            for root, dirs_list, files in os.walk(d):
                dirs_list[:] = [x for x in dirs_list if not any(e in x for e in excludes)]
                for fn in files:
                    if not any(e in fn for e in excludes):
                        fp = os.path.join(root, fn)
                        if release_path_allowed(fp):
                            zf.write(fp)

    size = os.path.getsize(output)
    print(f"  ✅ {output} ({size / 1024:.0f} KB)")
    print(f"\n{'═' * 60}")
    print("RELEASE PACKAGING COMPLETE")
    print(f"  Manifest: RELEASE_MANIFEST.json")
    print(f"  Counts verified: total={counts['total']}, gold={counts['gold']}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
