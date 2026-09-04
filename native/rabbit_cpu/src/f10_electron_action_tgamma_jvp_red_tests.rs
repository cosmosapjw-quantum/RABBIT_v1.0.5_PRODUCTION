//! Pair-only first slice of the frozen P1 contract, before production output.
//! Full electron-action T_gamma API remains unimplemented in this slice.
//! Historical absent-full-API RED is preserved at b362a249f2bf781baa874f9cc5b76517e6478ebb.

use crate::f10_action_grid::F10ActionGrid;
use crate::f10_electron_action::{
    F10ElectronActionConfig, assemble_electron_action,
    tgamma_jvp::{F10PairActionTgammaJvp, assemble_pair_action_tgamma_jvp, stable_pauli_jvp},
};
use serde_json::Value;

fn bits(value: &Value) -> f64 {
    f64::from_bits(u64::from_str_radix(value.as_str().expect("hex bits"), 16).unwrap())
}

fn array(value: &Value) -> Vec<f64> {
    value["bits"]
        .as_array()
        .expect("array bits")
        .iter()
        .map(bits)
        .collect()
}

fn fixture() -> Value {
    let path = std::env::var("D081R1F1_P1_PAIR_ORACLE").expect("explicit pair oracle required");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn relative(actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    assert!(!actual.is_empty());
    assert!(actual.iter().chain(expected).all(|x| x.is_finite()));
    let scale = actual
        .iter()
        .chain(expected)
        .map(|x| x.abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max)
        / scale
}

#[test]
fn pair_channel_matches_d080b_and_preserves_energy_and_exact_zero_structure() {
    let data = fixture();
    assert_eq!(data["scope"], "P1_PAIR_ONLY_ORDER8_NO_RETAINED");
    assert_eq!(
        data["d080b_blob"],
        "78489c43f3046db09d8ba2d96070124ed7b0aa91"
    );
    let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let state = array(&case["state"]);
        let tg = bits(&case["tg"]);
        let result: F10PairActionTgammaJvp = assemble_pair_action_tgamma_jvp(
            &grid,
            &state,
            2.0,
            tg,
            F10ElectronActionConfig::default(),
        )
        .unwrap();
        let reference = array(&case["pair_native"]);
        let residual = relative(&result.native, &reference);
        assert!(residual <= 1.0e-7, "pair native parity: {residual:.17e}");
        // At equilibrium the primal is a numerical null, not a relative-shape observable.
        // T*dC/dT has the primal units and is a non-null, reference-only physical scale.
        let base_reference = array(&case["base_pair_native"]);
        let base_scale = reference
            .iter()
            .map(|x| tg * x.abs())
            .fold(f64::MIN_POSITIVE, f64::max);
        let base_error = result
            .base_native
            .iter()
            .zip(&base_reference)
            .map(|(a, b): (&f64, &f64)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(base_error / base_scale <= 1.0e-7);
        assert_eq!(result.modal.len(), 48);
        assert_eq!(result.base_modal.len(), 48);
        assert_eq!(result.family_names, vec!["e:pair", "mu:pair", "tau:pair"]);
        assert!(
            result
                .measure_modal
                .iter()
                .chain(&result.matrix_modal)
                .chain(&result.projection_modal)
                .all(|x: &f64| x.to_bits() == 0)
        );
        assert_eq!(result.measure_modal.len(), 48);
        assert_eq!(result.matrix_modal.len(), 48);
        assert_eq!(result.projection_modal.len(), 48);
        assert_eq!(
            result.support,
            serde_json::from_value::<Vec<bool>>(case["support"].clone()).unwrap()
        );
        assert_eq!(
            result.corrected,
            serde_json::from_value::<Vec<bool>>(case["corrected"].clone()).unwrap()
        );
        assert!(result.neutrino_energy_transfer > 0.0);
        assert!(result.electromagnetic_energy_transfer < 0.0);
        assert!(result.first_law_residual <= 2.0e-9);
        assert!(
            relative(
                &[result.electromagnetic_energy_transfer],
                &[bits(&case["qem"])]
            ) <= 1.0e-7
        );
        let mut reconstructed = vec![0.0_f64; 48];
        for family in 0..3 {
            let expected_family = array(&case["family_native"][family]);
            let observed_family = &result.family_native[family * 48..(family + 1) * 48];
            assert!(relative(observed_family, &expected_family) <= 1.0e-7);
            for (i, value) in reconstructed.iter_mut().enumerate() {
                *value += result.family_modal[family * 48 + i];
                if i / 8 != 2 * family && i / 8 != 2 * family + 1 {
                    assert_eq!(observed_family[i].to_bits(), 0);
                }
            }
            let [qnu, qem] = result.energy_by_family[family];
            assert!((qnu + qem).abs() / (qnu.abs() + qem.abs()).max(f64::MIN_POSITIVE) <= 2.0e-9);
        }
        assert!(relative(&reconstructed, &result.modal) <= 2.0e-12);
        let mutant: Vec<f64> = result.native.iter().map(|x: &f64| 1.01 * x).collect();
        assert!(relative(&mutant, &reference) > 1.0e-4);
        eprintln!(
            "P1_PAIR case={} native_relative={residual:.17e} first_law={:.17e} qnu={:.17e} qem={:.17e}",
            case["name"],
            result.first_law_residual,
            result.neutrino_energy_transfer,
            result.electromagnetic_energy_transfer
        );
    }
}

#[test]
fn pair_tangent_matches_centered_primal_and_has_temperature_independent_branch() {
    let data = fixture();
    let case = &data["cases"][1];
    let state = array(&case["state"]);
    let tg = bits(&case["tg"]);
    let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
    let config = F10ElectronActionConfig::default();
    let result: F10PairActionTgammaJvp =
        assemble_pair_action_tgamma_jvp(&grid, &state, 2.0, tg, config).unwrap();
    let base = assemble_electron_action(&grid, &state, 2.0, tg, config).unwrap();
    assert!(relative(&result.base_native, &base.pair_native) <= 2.0e-12);
    assert!(relative(&result.base_modal, &base.pair_modal) <= 2.0e-12);
    let mut best = f64::INFINITY;
    for epsilon in [1.0e-3, 3.0e-4, 1.0e-4] {
        let plus = assemble_electron_action(&grid, &state, 2.0, tg + epsilon, config).unwrap();
        let minus = assemble_electron_action(&grid, &state, 2.0, tg - epsilon, config).unwrap();
        let centered: Vec<f64> = plus
            .pair_native
            .iter()
            .zip(&minus.pair_native)
            .map(|(p, m)| (p - m) / (2.0 * epsilon))
            .collect();
        let residual = relative(&result.native, &centered);
        best = best.min(residual);
        eprintln!("P1_PAIR_CENTERED epsilon={epsilon:.17e} relative={residual:.17e}");
    }
    assert!(best <= 2.0e-6);
    let moved: F10PairActionTgammaJvp =
        assemble_pair_action_tgamma_jvp(&grid, &state, 2.0, 2.3, config).unwrap();
    assert_eq!(moved.support, result.support);
    assert_eq!(moved.corrected, result.corrected);
}

#[test]
fn affinity_derivative_matches_high_precision_and_rejects_invalid_input() {
    let data = fixture();
    for probe in data["pauli_probes"].as_array().unwrap() {
        let logits: [f64; 4] = array(&probe["logits"]).try_into().unwrap();
        let direction: [f64; 4] = array(&probe["direction"]).try_into().unwrap();
        let expected = bits(&probe["derivative"]);
        let observed: f64 = stable_pauli_jvp(logits, direction).unwrap();
        let tolerance = 1.0e-10 * expected.abs().max(1.0e-300);
        assert!(
            (observed - expected).abs() <= tolerance,
            "Pauli: {observed:.17e} vs {expected:.17e}"
        );
    }
    assert!(stable_pauli_jvp([0.0; 4], [f64::NAN, 0.0, 0.0, 0.0]).is_err());
    assert!(stable_pauli_jvp([f64::INFINITY, 0.0, 0.0, 0.0], [0.0; 4]).is_err());
    let grid = F10ActionGrid::affine_legendre(8, 8.0).unwrap();
    assert!(
        assemble_pair_action_tgamma_jvp(&grid, &[], 2.0, 2.0, F10ElectronActionConfig::default())
            .is_err()
    );
    assert!(
        assemble_pair_action_tgamma_jvp(
            &grid,
            &[0.0; 24],
            2.0,
            0.0,
            F10ElectronActionConfig::default()
        )
        .is_err()
    );
}
