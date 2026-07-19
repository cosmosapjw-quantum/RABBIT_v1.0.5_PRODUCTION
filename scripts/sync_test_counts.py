#!/usr/bin/env python3
"""Sync test counts in STATUS.md and README.md with actual pytest collection.

Uses generated block markers (BEGIN:TEST_COUNTS / END:TEST_COUNTS) for robust
replacement. Falls back to regex if markers not found.

Usage: python scripts/sync_test_counts.py [--dry-run]
"""
import subprocess, re, sys, glob
import os


def _pytest_env():
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("RABBIT_JAX_CACHE_DIR", "/tmp/rabbit_jax_cache")
    return env


def get_counts():
    def _run(marker=None):
        cmd = [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"]
        if marker:
            cmd.extend(["-m", marker])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=_pytest_env())
        if marker:
            m = re.search(r"(\d+)/\d+ tests? collected", r.stdout)
        else:
            m = re.search(r"(\d+) tests? collected", r.stdout)
        if not m:
            label = marker or "all"
            stdout_tail = r.stdout[-2000:]
            stderr_tail = r.stderr[-2000:]
            raise RuntimeError(
                "Could not parse pytest collection count "
                f"for marker {label!r}; returncode={r.returncode}; "
                f"stdout_tail={stdout_tail!r}; stderr_tail={stderr_tail!r}"
            )
        return int(m.group(1))
    return {
        "total": _run(),
        "production_total": _run("production"),
        "production_not_slow": _run("production and not slow"),
        "gold": _run("gold"),
        "smoke": _run("release_smoke"),
        "production_slow": _run("production and slow"),
        "files": len(glob.glob("tests/**/test_*.py", recursive=True)),
    }


def _replace_block(text, begin_tag, end_tag, new_content):
    """Replace content between BEGIN/END markers.

    Uses flexible matching: begin_tag is treated as a prefix.
    """
    # Flexible: match begin_tag possibly followed by extra text before -->
    begin_esc = re.escape(begin_tag.rstrip(" ->"))
    pattern = re.compile(
        rf'({begin_esc}[^\n]*?)\n(.*?)\n({re.escape(end_tag)})',
        re.DOTALL
    )
    m = pattern.search(text)
    if m:
        return text[:m.start()] + m.group(1) + "\n" + new_content + "\n" + m.group(3) + text[m.end():]
    return None  # markers not found


def update_readme(counts, dry_run=False):
    t, pt, pf, g, s, nf = (
        counts["total"],
        counts["production_total"],
        counts["production_not_slow"],
        counts["gold"],
        counts["smoke"],
        counts["files"],
    )
    text = open("README.md").read()
    original = text

    new_line = (
        f"Overlapping marker subsets: **{g} gold** BBN regression gates | "
        f"{s} release smoke | `@production`: {pt} total, "
        f"{pf} production-and-not-slow | "
        f"build-env total: {t} tests across {nf} files"
    )

    result = _replace_block(text, "<!-- BEGIN:TEST_COUNTS", "<!-- END:TEST_COUNTS -->", new_line)
    if result:
        text = result
    else:
        # Fallback regex
        text = re.sub(r"\d+ tests across \d+ files:.*?See `STATUS\.md`[^.]*\.",
                       new_line, text, flags=re.DOTALL)

    # Also update Makefile references in README
    text = re.sub(
        r'full production suite \(\d+ tests\)',
        f'`@production` marker family ({pt} total; {pf} production-and-not-slow)',
        text,
    )
    text = re.sub(r'BBN observable gold locks \(\d+ tests\)', f'BBN observable gold locks ({g} tests)', text)

    if text != original:
        if not dry_run:
            open("README.md", "w").write(text)
        print(f"  {'[DRY-RUN]' if dry_run else 'Updated'} README.md")
    else:
        print("  README.md: already correct")


def update_status(counts, dry_run=False):
    t, pt, pf, ps, g, s = (
        counts["total"],
        counts["production_total"],
        counts["production_not_slow"],
        counts["production_slow"],
        counts["gold"],
        counts["smoke"],
    )
    text = open("STATUS.md").read()
    original = text

    new_header = (
        f"## Tests ({t} collected; overlapping marker subsets: "
        f"**{g} gold**, {s} smoke; `@production` {pt} total, "
        f"{pf} production-and-not-slow)"
    )

    result = _replace_block(text, "<!-- BEGIN:TEST_COUNTS -->", "<!-- END:TEST_COUNTS -->", new_header)
    if result:
        text = result
    else:
        text = re.sub(r"## Tests \([^)]+\)", new_header, text)

    hierarchy = "\n".join(
        [
            "## Production gate hierarchy",
            "",
            f'1. Required fast release lane: `-m "production and not slow"` ({pf} collected)',
            f"2. `@production` is the total marker family ({pt} collected; {ps} also marked slow)",
            f"3. `@gold` ({g}) and `@release_smoke` ({s}) are overlapping subsets, not a partition or sum",
            "4. `@build_env_only`: exact count sync (packaging only)",
            "5. `@slow`: opt-in runtime classification; excluded from the required fast lane",
        ]
    )
    text = re.sub(
        r"## Production gate hierarchy\n\n.*?(?=\n<!-- END:STATUS_DETAIL -->)",
        hierarchy,
        text,
        flags=re.DOTALL,
    )

    if text != original:
        if not dry_run:
            open("STATUS.md", "w").write(text)
        print(f"  {'[DRY-RUN]' if dry_run else 'Updated'} STATUS.md")
    else:
        print("  STATUS.md: already correct")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    counts = get_counts()
    print(
        f"Actual: {counts['total']} total, "
        f"{counts['production_total']} production total, "
        f"{counts['production_not_slow']} production-and-not-slow, "
        f"{counts['gold']} gold, {counts['smoke']} smoke, "
        f"{counts['production_slow']} production-and-slow"
    )
    update_readme(counts, dry_run)
    update_status(counts, dry_run)
    print("Done.")
