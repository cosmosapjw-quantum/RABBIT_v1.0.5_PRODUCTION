#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, importlib.util, inspect, json, re
from pathlib import Path

WATCH = [
    ("rabbit.transport.species_tagged_bridge", "apply_species_tagged_bridge"),
    ("rabbit.transport.teff_collision_bridge", "apply_gather_scatter_collision"),
    ("rabbit.drivers.full_coupled_typeI", "coupled_rhs"),
    ("rabbit.drivers.full_coupled_typeI", "run_full_coupled_typeI"),
]

def load_wrapper(path: Path):
    spec = importlib.util.spec_from_file_location("reduced_modal_wrapper_probe", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def snap():
    out = {}
    for modname, attr in WATCH:
        try:
            mod = importlib.import_module(modname)
            obj = getattr(mod, attr, None)
            if obj is None:
                out[f"{modname}:{attr}"] = {"exists": False}
                continue
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "<sig unavailable>"
            out[f"{modname}:{attr}"] = {
                "exists": True,
                "id": id(obj),
                "module": getattr(obj, "__module__", None),
                "qualname": getattr(obj, "__qualname__", None),
                "name": getattr(obj, "__name__", None),
                "signature": sig,
                "file": inspect.getsourcefile(obj),
            }
        except Exception as e:
            out[f"{modname}:{attr}"] = {"exists": False, "error": repr(e)}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapper", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wrapper_path = Path(args.wrapper)
    text = wrapper_path.read_text()
    env_keys = sorted(set(re.findall(r"RABBIT_[A-Z0-9_]+", text)))

    before = snap()
    wrapper = load_wrapper(wrapper_path)
    install_src = inspect.getsource(wrapper.install_patch)

    wrapper.install_patch()
    after = snap()

    changed = {}
    for k in sorted(set(before) | set(after)):
        b = before.get(k, {})
        a = after.get(k, {})
        changed[k] = {
            "before": b,
            "after": a,
            "changed": (b.get("id") != a.get("id")) or (b.get("signature") != a.get("signature")),
        }

    out = {
        "wrapper": str(wrapper_path.resolve()),
        "env_keys_in_wrapper": env_keys,
        "install_patch_source": install_src,
        "watch": changed,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()
