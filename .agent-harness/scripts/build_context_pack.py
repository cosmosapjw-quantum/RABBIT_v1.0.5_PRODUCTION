#!/usr/bin/env python3
from __future__ import annotations

from _harness import (
    dump_json,
    hash_files,
    load_json,
    render_context_pack,
    root,
    utc_now,
    write_text_atomic,
)


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
    # Atomic, because a partially written pack used to validate: both checks
    # only looked at the first six lines, so a crash or a full disk mid-write
    # left a file that passed (BD623 R1).
    write_text_atomic(out, render_context_pack(repo, version, index["built_at"], entries))
    print(f"Built {out.relative_to(repo)}")
    print(f"context_version={version}")


if __name__ == "__main__":
    main()
