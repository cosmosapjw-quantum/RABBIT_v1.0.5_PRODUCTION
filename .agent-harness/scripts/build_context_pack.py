#!/usr/bin/env python3
from __future__ import annotations

from _harness import dump_json, hash_files, load_json, root, utc_now


def main() -> None:
    repo = root()
    harness = repo / ".agent-harness"
    index_path = harness / "context" / "CONTEXT_INDEX.json"
    index = load_json(index_path)
    files = list(index.get("shared_files", []))
    if not files:
        raise SystemExit("CONTEXT_INDEX.json has no shared_files.")

    missing = [rel for rel in files if not (repo / rel).is_file()]
    if missing:
        raise SystemExit("Missing shared context files:\n" + "\n".join(missing))

    version, entries = hash_files(repo, files)
    index["context_version"] = version
    index["built_at"] = utc_now()
    index["file_hashes"] = {rel: sha for rel, sha in entries}
    dump_json(index_path, index)

    out = harness / "generated" / "CONTEXT_PACK.md"
    chunks = [
        "# Canonical Shared Context Pack",
        "",
        f"Context version: `{version}`",
        f"Built at: `{index['built_at']}`",
        "",
        "This pack contains only the shared Tier-0 context. Assignment-specific context and sibling results are intentionally excluded.",
    ]
    for rel, sha in entries:
        text = (repo / rel).read_text(encoding="utf-8")
        chunks.extend(["", f"---\n\n## Source: `{rel}`\n\nSHA-256: `{sha}`\n", text.rstrip()])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")
    print(f"Built {out.relative_to(repo)}")
    print(f"context_version={version}")


if __name__ == "__main__":
    main()
