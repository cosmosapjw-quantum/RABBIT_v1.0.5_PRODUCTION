#!/usr/bin/env python3
"""Apply the bounded D-081R1F1 P0B module wiring.

The P0B production implementation lives in a focused sibling module. This
script registers that module and its RED-first test module, then applies two
lexical fixes to otherwise-unused primal shadow values. It performs no formula,
tolerance, quadrature, support, or physics mutation.
"""

from __future__ import annotations

from pathlib import Path


LIB = Path("native/rabbit_cpu/src/lib.rs")
KINEMATICS = Path("native/rabbit_cpu/src/f10_tgamma_kinematics.rs")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_lib() -> bool:
    text = LIB.read_text(encoding="utf-8")
    changed = False
    if "mod f10_tgamma_kinematics;" not in text:
        text = replace_once(
            text,
            "mod f10_tgamma_tangent;\n",
            "mod f10_tgamma_tangent;\nmod f10_tgamma_kinematics;\n",
            "production module anchor",
        )
        changed = True
    if "mod f10_tgamma_kinematic_tests;" not in text:
        anchor = "#[cfg(test)]\nmod f10_tgamma_tangent_tests;\n"
        if anchor in text:
            text = replace_once(
                text,
                anchor,
                anchor + "\n#[cfg(test)]\nmod f10_tgamma_kinematic_tests;\n",
                "test module anchor",
            )
        else:
            text += "\n#[cfg(test)]\nmod f10_tgamma_kinematic_tests;\n"
        changed = True
    if changed:
        LIB.write_text(text, encoding="utf-8")
    return changed


def patch_staged_source() -> bool:
    text = KINEMATICS.read_text(encoding="utf-8")
    changed = False
    for old, new, label in (
        (
            "let phase_space = if support { k_star / sqrt_s } else { 0.0 };",
            "let _phase_space = if support { k_star / sqrt_s } else { 0.0 };",
            "unused phase-space primal",
        ),
        (
            "let quadrature_weight = weight2 * weight12 * weight_star * weight_phi;",
            "let _quadrature_weight = weight2 * weight12 * weight_star * weight_phi;",
            "unused quadrature-weight primal",
        ),
    ):
        if old in text:
            text = replace_once(text, old, new, label)
            changed = True
    if changed:
        KINEMATICS.write_text(text, encoding="utf-8")
    return changed


if __name__ == "__main__":
    changed = patch_lib() | patch_staged_source()
    print("D-081R1F1 P0B wiring:", "CHANGED" if changed else "NOOP")
