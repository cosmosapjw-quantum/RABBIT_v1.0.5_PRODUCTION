#!/usr/bin/env python3
"""Apply the bounded D-081R1F0 source-compile repair idempotently.

This script changes no physics coefficient, quadrature rule, support predicate,
state chart, or derivative formula. It only:
1. rewrites one Clippy-reported range loop without changing accumulation order;
2. extends the exact-zero JVP test so every diagnostic field is exercised.
"""

from __future__ import annotations

from pathlib import Path


PACKED = Path("native/rabbit_cpu/src/f10_packed_rhs_jvp.rs")
TESTS = Path("native/rabbit_cpu/src/f10_packed_rhs_jvp_tests.rs")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one repair anchor, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    replace_once(
        PACKED,
        """    let mut delta_rho_neutrino_by_flavour = [0.0_f64; PAIR_COUNT];
    let rho_prefactor = temperature_cm.powi(4) / (PI * PI);
    for flavour in 0..PAIR_COUNT {
        let mut integral = 0.0_f64;
        for node in 0..grid.order {
            integral += grid.weights[node]
                * grid.nodes[node].powi(3)
                * tangent.occupation_delta[flavour * grid.order + node];
        }
        delta_rho_neutrino_by_flavour[flavour] = rho_prefactor * integral;
    }
""",
        """    let mut delta_rho_neutrino_by_flavour = [0.0_f64; PAIR_COUNT];
    let rho_prefactor = temperature_cm.powi(4) / (PI * PI);
    for (flavour, delta_rho) in delta_rho_neutrino_by_flavour.iter_mut().enumerate() {
        let mut integral = 0.0_f64;
        for node in 0..grid.order {
            integral += grid.weights[node]
                * grid.nodes[node].powi(3)
                * tangent.occupation_delta[flavour * grid.order + node];
        }
        *delta_rho = rho_prefactor * integral;
    }
""",
    )

    replace_once(
        TESTS,
        """    assert_eq!(result.base.values.len(), 3 * grid.order + 2);
    assert_eq!(result.values.len(), 3 * grid.order + 2);
    assert!(result.values.iter().all(|value| value.to_bits() == 0));
    assert_eq!(result.delta_rho_neutrino.to_bits(), 0);
    assert_eq!(result.delta_hubble_over_hubble.to_bits(), 0);
    assert_eq!(result.delta_neutrino_energy_transfer.to_bits(), 0);
    assert_eq!(result.delta_electromagnetic_energy_transfer.to_bits(), 0);
""",
        """    assert_eq!(result.base.values.len(), 3 * grid.order + 2);
    assert_eq!(result.values.len(), 3 * grid.order + 2);
    assert!(result.values.iter().all(|value| value.to_bits() == 0));
    assert!(
        result
            .delta_rho_neutrino_by_flavour
            .iter()
            .all(|value| value.to_bits() == 0)
    );
    assert_eq!(result.delta_rho_neutrino.to_bits(), 0);
    assert_eq!(result.delta_hubble_over_hubble.to_bits(), 0);
    assert_eq!(result.delta_neutrino_energy_transfer.to_bits(), 0);
    assert_eq!(result.delta_electromagnetic_energy_transfer.to_bits(), 0);
    assert_eq!(result.first_law_tangent_residual.to_bits(), 0);

    let combined = &result.combined_action;
    assert!(combined.modal_total.iter().all(|value| value.to_bits() == 0));
    assert!(combined.native_total.iter().all(|value| value.to_bits() == 0));
    for value in [
        combined.moments.signed_number_rate,
        combined.moments.absolute_number_rate,
        combined.moments.signed_energy_rate,
        combined.moments.absolute_energy_rate,
        combined.neutrino_energy_transfer,
        combined.electromagnetic_energy_transfer,
        combined.first_law_residual,
        combined.self_event_energy_residual,
        combined.self_event_energy_relative_residual,
        combined.charge_conjugation_residual,
        combined.mu_tau_residual,
    ] {
        assert_eq!(value.to_bits(), 0);
    }

    let self_action = &combined.self_action;
    assert!(self_action.modal.iter().all(|value| value.to_bits() == 0));
    assert!(self_action.native.iter().all(|value| value.to_bits() == 0));
    for value in [
        self_action.moments.signed_number_rate,
        self_action.moments.absolute_number_rate,
        self_action.moments.signed_energy_rate,
        self_action.moments.absolute_energy_rate,
        self_action.event_energy_residual,
        self_action.event_energy_absolute,
    ] {
        assert_eq!(value.to_bits(), 0);
    }

    let electron_action = &combined.electron_action;
    assert!(
        electron_action
            .modal
            .iter()
            .all(|value| value.to_bits() == 0)
    );
    assert!(
        electron_action
            .native
            .iter()
            .all(|value| value.to_bits() == 0)
    );
    for value in [
        electron_action.moments.signed_number_rate,
        electron_action.moments.absolute_number_rate,
        electron_action.moments.signed_energy_rate,
        electron_action.moments.absolute_energy_rate,
        electron_action.neutrino_energy_transfer,
        electron_action.electromagnetic_energy_transfer,
        electron_action.first_law_residual,
    ] {
        assert_eq!(value.to_bits(), 0);
    }
""",
    )

    print("D-081R1F0 source compile repair R1: READY")


if __name__ == "__main__":
    main()
