#!/usr/bin/env python3
"""Generate structural D-078 plots and a machine-readable research receipt.

The probe uses a manufactured one-component relaxation law.  It never imports
or calls the private RABBIT comparator and therefore produces no trajectory,
checkpoint, endpoint, or gate evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from _d078_logit_linearization import push_forward_occupation_jacobian
from _d078_tangent_certificate import TangentSampleStatus, certify_directional_derivative

FLOOR = 1.0e-12
LAMBDA = 1.0
TARGET_OCCUPATION = 0.2
EPSILONS = np.array(
    [
        1.0e-1,
        3.0e-2,
        1.0e-2,
        3.0e-3,
        1.0e-3,
        3.0e-4,
        1.0e-4,
        3.0e-5,
        1.0e-5,
        3.0e-6,
        1.0e-6,
        3.0e-7,
        1.0e-7,
    ]
)


def sigmoid_scalar(logit: float) -> float:
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def raw_chain_scalar(logit: float) -> float:
    occupation = sigmoid_scalar(logit)
    return occupation * (1.0 - occupation)


def transformed_relaxation_rhs(state: np.ndarray) -> np.ndarray:
    logit = float(state[0])
    occupation = sigmoid_scalar(logit)
    chain = max(occupation * (1.0 - occupation), FLOOR)
    physical_rhs = LAMBDA * (TARGET_OCCUPATION - occupation)
    return np.array([physical_rhs / chain])


def floor_branch(state: np.ndarray) -> bool:
    return bool(raw_chain_scalar(float(state[0])) > FLOOR)


def exact_transformed_jacobian(logit: float) -> float:
    occupation = np.array([sigmoid_scalar(logit)])
    physical_rhs = LAMBDA * (TARGET_OCCUPATION - occupation)
    occupation_jacobian = np.array([[-LAMBDA]])
    return float(
        push_forward_occupation_jacobian(
            occupation,
            physical_rhs,
            occupation_jacobian,
            floor=FLOOR,
        ).jacobian[0, 0]
    )


def sample_to_dict(sample: Any) -> dict[str, Any]:
    return {
        "epsilon": sample.epsilon,
        "status": sample.status.value,
        "error_norm": sample.error_norm,
        "comparison_scale": sample.comparison_scale,
        "threshold": sample.threshold,
        "normalized_residual": sample.normalized_residual,
    }


def run_case(label: str, logit: float, jvp_multiplier: float = 1.0) -> dict[str, Any]:
    occupation = sigmoid_scalar(logit)
    raw_chain = occupation * (1.0 - occupation)
    analytic = exact_transformed_jacobian(logit) * jvp_multiplier
    certificate = certify_directional_derivative(
        transformed_relaxation_rhs,
        np.array([logit]),
        np.array([1.0]),
        np.array([analytic]),
        epsilons=EPSILONS,
        rtol=3.0e-5,
        atol=1.0e-10,
        min_valid_samples=5,
        required_consecutive_passes=2,
        state_validator=lambda state: bool(np.all(np.isfinite(state))),
        branch_signature=floor_branch,
    )
    return {
        "label": label,
        "logit": logit,
        "occupation": occupation,
        "raw_chain": raw_chain,
        "effective_chain": max(raw_chain, FLOOR),
        "chain_amplification": 1.0 / max(raw_chain, FLOOR),
        "analytic_jvp": analytic,
        "jvp_multiplier": jvp_multiplier,
        "certificate_status": certificate.status.value,
        "valid_samples": certificate.valid_samples,
        "passing_samples": certificate.passing_samples,
        "max_consecutive_passes": certificate.max_consecutive_passes,
        "best_normalized_residual": certificate.best_normalized_residual,
        "branch_crossing_samples": sum(
            sample.status is TangentSampleStatus.BRANCH_CROSSING
            for sample in certificate.samples
        ),
        "samples": [sample_to_dict(sample) for sample in certificate.samples],
    }


def plot_chain_amplification(output: Path) -> None:
    logits = np.linspace(-32.0, 32.0, 1601)
    occupations = np.array([sigmoid_scalar(float(value)) for value in logits])
    raw = occupations * (1.0 - occupations)
    raw_amplification = 1.0 / raw
    effective_amplification = 1.0 / np.maximum(raw, FLOOR)
    transition = math.log((1.0 + math.sqrt(1.0 - 4.0 * FLOOR)) / (2.0 * FLOOR))

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.semilogy(logits, raw_amplification, label=r"$1/[f(1-f)]$")
    axis.semilogy(logits, effective_amplification, linestyle="--", label=r"$1/D_{\rm eff}$")
    axis.axvline(transition, linewidth=1.0, linestyle=":", label="floor transition")
    axis.axvline(-transition, linewidth=1.0, linestyle=":")
    axis.set_xlabel(r"logit $z$")
    axis.set_ylabel("chain amplification")
    axis.set_title("D-078 manufactured logit-chain amplification")
    axis.grid(True, which="both", linewidth=0.4)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_residual_curves(cases: list[dict[str, Any]], output: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for case in cases:
        epsilon_values: list[float] = []
        residual_values: list[float] = []
        for sample in case["samples"]:
            residual = sample["normalized_residual"]
            if residual is not None and residual > 0.0:
                epsilon_values.append(sample["epsilon"])
                residual_values.append(residual)
        if epsilon_values:
            axis.loglog(
                epsilon_values,
                residual_values,
                marker="o",
                markersize=3.0,
                label=f"{case['label']} [{case['certificate_status']}]",
            )
    axis.axhline(3.0e-5, linewidth=1.0, linestyle=":", label="frozen rtol")
    axis.set_xlabel(r"centered-difference step $\epsilon$")
    axis.set_ylabel("normalized true tangent residual")
    axis.set_title(title)
    axis.grid(True, which="both", linewidth=0.4)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_markdown(receipt: dict[str, Any], output: Path) -> None:
    lines = [
        "# D-078 Structural Probe — Machine Summary",
        "",
        "This file is generated from a manufactured relaxation law. It is not a RABBIT trajectory or gate result.",
        "",
        "| Case | z | 1/D_eff | Certificate | Valid | Branch crossings | Best residual |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for case in receipt["cases"]:
        lines.append(
            "| {label} | {logit:.6g} | {chain_amplification:.6e} | {certificate_status} | "
            "{valid_samples} | {branch_crossing_samples} | {best_normalized_residual:.6e} |".format(
                **case
            )
        )
    lines.extend(
        [
            "",
            "## Mutation controls",
            "",
            "| Mutation | Certificate | Best residual |",
            "|---|---|---:|",
        ]
    )
    for case in receipt["mutations"]:
        lines.append(
            "| {label} | {certificate_status} | {best_normalized_residual:.6e} |".format(
                **case
            )
        )
    lines.extend(
        [
            "",
            "## Generated plots",
            "",
            "- `logit_chain_amplification.png`",
            "- `directional_residual_ladder.png`",
            "- `mutation_residual_ladder.png`",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/audit/artifacts/d078"),
    )
    arguments = parser.parse_args()
    output_directory = arguments.out_dir
    output_directory.mkdir(parents=True, exist_ok=True)

    transition = math.log((1.0 + math.sqrt(1.0 - 4.0 * FLOOR)) / (2.0 * FLOOR))
    cases = [
        run_case("interior", 0.0),
        run_case("moderate-tail", 20.0),
        run_case("near-floor", transition - 1.1e-2),
        run_case("clamped-tail", 30.0),
    ]
    mutations = [
        run_case("correct", 5.0, 1.0),
        run_case("sign-mutant", 5.0, -1.0),
        run_case("scale-1.01-mutant", 5.0, 1.01),
    ]

    receipt = {
        "schema_version": 1,
        "classification": "STRUCTURAL_MANUFACTURED_RESEARCH_ONLY",
        "canonical_parent": "ae3f6776bd6fc5bc84bca72d251dc0db1bba7da5",
        "vigilode_reference_commit": "8d0c79184e09efb5bdadc24a6315c60a71a44264",
        "floor": FLOOR,
        "positive_floor_transition_logit": transition,
        "lambda": LAMBDA,
        "target_occupation": TARGET_OCCUPATION,
        "epsilons": EPSILONS.tolist(),
        "cases": cases,
        "mutations": mutations,
        "explicit_non_authority": [
            "no private comparator import",
            "no physical trajectory",
            "no checkpoint or endpoint",
            "no wall-time projection",
            "no gate movement",
        ],
    }

    plot_chain_amplification(output_directory / "logit_chain_amplification.png")
    _plot_residual_curves(
        cases,
        output_directory / "directional_residual_ladder.png",
        "D-078 same-branch directional residual ladders",
    )
    _plot_residual_curves(
        mutations,
        output_directory / "mutation_residual_ladder.png",
        "D-078 adversarial JVP mutation controls",
    )
    (output_directory / "research_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(receipt, output_directory / "probe_summary.md")
    print(json.dumps({
        "classification": receipt["classification"],
        "cases": {case["label"]: case["certificate_status"] for case in cases},
        "mutations": {case["label"]: case["certificate_status"] for case in mutations},
        "out_dir": str(output_directory),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
