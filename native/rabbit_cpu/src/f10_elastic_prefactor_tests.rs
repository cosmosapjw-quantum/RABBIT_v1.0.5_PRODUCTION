//! Prospective sample-level elastic prefactor tests; no collision assembly.
//! Base b40a78ac0e03a27f13761ff546267b127bf9278e; fixed p1, masses and outer weight.
//! MeV natural units. Frozen before output: two points, three relative steps,
//! global tangent cap 1e-7, exact primal values and support/correction masks.
//! No retained states, Pauli derivative, projection, solver or speed claim.

use crate::electron_hm::SIN2_THETA_W;
use crate::f10_action_kinematics::{
    F10CollisionConfig, F10KinematicBatch, F10KinematicInput, electron_half_line_rule,
    two_body_kinematics,
};
use crate::f10_elastic_prefactor_tangent::{
    F10MeasureTangent, elastic_matrix_tangent, event_measure_tangent,
};
use crate::f10_kernel_primitives::{
    F10ElectronCategory, F10EventMeasureInput, F10InvariantProducts, F10KernelError,
    F10MatrixValue, F10Species, f10_electron_matrix, f10_event_measure,
};
use crate::f10_tgamma_kinematics::{
    F10ElasticTgammaInput, evaluate_elastic_tgamma_kinematic_tangent,
};

const MASS: f64 = 0.510_998_95;
const CAP: f64 = 1.0e-7;
const ULP_BUDGET: f64 = 1024.0;

// Prospective scalar gate, before this replay's Rust output: <64 elementary
// operations per route and independent reference; 128 epsilon allows both
// evaluation paths. Normalize by absolute contribution scale, never primal M.
// References: scripts/audit/d081r1f1_elastic_signed_cas.py, exact residuals=0.
const SIGNED_SCALAR_CAP: f64 = 128.0 * f64::EPSILON;

fn signed_scalar_check(label: &str, actual: f64, expected: f64, scale: f64) {
    let absolute = (actual - expected).abs();
    let relative = absolute / scale;
    eprintln!(
        "SIGNED_SCALAR {label} actual={actual:.17e} expected={expected:.17e} absolute={absolute:.17e} contribution_scale={scale:.17e} relative={relative:.17e} cap={SIGNED_SCALAR_CAP:.17e}"
    );
    assert!(actual.is_finite() && expected.is_finite() && scale > 0.0);
    assert!(relative <= SIGNED_SCALAR_CAP, "{label}: {relative:.17e}");
}

#[test]
fn elastic_signed_scalar_cas_references() {
    use crate::electron_hm::G_F_MEV_MINUS_2;
    let pi4 = core::f64::consts::PI.powi(4);
    for head_on in [true, false] {
        let (a, b, da, db, phase, dphase, w, dw, parts) = if head_on {
            (
                9.0,
                6.0,
                18.0 / 5.0,
                2.0,
                1.0 / 3.0,
                2.0 / 45.0,
                1.0 / 240.0,
                197.0 / 36000.0,
                [1.0 / 480.0, 1.0 / 240.0, 1.0 / 1800.0, -1.0 / 750.0],
            )
        } else {
            (
                1.0,
                10.0 / 11.0,
                -2.0 / 5.0,
                -202.0 / 605.0,
                1.0 / 11.0,
                -18.0 / 605.0,
                1.0 / 880.0,
                469.0 / 484000.0,
                [1.0 / 1760.0, 1.0 / 880.0, -9.0 / 24200.0, -1.0 / 2750.0],
            )
        };
        let input = F10EventMeasureInput {
            p1: 1.0,
            p2: 4.0,
            e2: 5.0,
            phase_space: phase,
            quadrature_weight: 1.0,
            outer_weight: 1.0,
        };
        let dt = F10MeasureTangent {
            d_p2: 2.0,
            d_e2: 8.0 / 5.0,
            d_phase_space: dphase,
            d_quadrature_weight: 0.5,
        };
        let (primal_w, derivative_w) = event_measure_tangent(input, dt).unwrap();
        let scale = parts.into_iter().map(f64::abs).sum::<f64>() / pi4;
        signed_scalar_check("W", primal_w, w / pi4, w / pi4);
        signed_scalar_check("W_T", derivative_w, dw / pi4, scale);
        let zero = F10MeasureTangent {
            d_p2: 0.0,
            d_e2: 0.0,
            d_phase_space: 0.0,
            d_quadrature_weight: 0.0,
        };
        let directions = [
            F10MeasureTangent {
                d_quadrature_weight: dt.d_quadrature_weight,
                ..zero
            },
            F10MeasureTangent {
                d_p2: dt.d_p2,
                ..zero
            },
            F10MeasureTangent {
                d_phase_space: dt.d_phase_space,
                ..zero
            },
            F10MeasureTangent {
                d_e2: dt.d_e2,
                ..zero
            },
        ];
        for (i, direction) in directions.into_iter().enumerate() {
            let (_, actual) = event_measure_tangent(input, direction).unwrap();
            signed_scalar_check(&format!("W_T_part_{i}"), actual, parts[i] / pi4, scale);
            // Each nonzero contribution must independently kill its omission;
            // doubling the moving weight differs by the same nonzero amount.
            assert!(actual.abs() / scale > SIGNED_SCALAR_CAP);
        }
        let inv = F10InvariantProducts {
            d12: a,
            d13: a - b,
            d14: b,
            d23: b,
            d24: 9.0 + a - b,
            d34: a,
        };
        let dinv = F10InvariantProducts {
            d12: da,
            d13: da - db,
            d14: db,
            d23: db,
            d24: da - db,
            d34: da,
        };
        // Explicit twelve-family reference routing, independent of helper routing.
        for (target, electron, anti) in [
            (F10Species::NuE, true, false),
            (F10Species::NuMu, false, false),
            (F10Species::NuTau, false, false),
            (F10Species::AntiNuE, true, true),
            (F10Species::AntiNuMu, false, true),
            (F10Species::AntiNuTau, false, true),
        ] {
            for category in [
                F10ElectronCategory::ElasticMinus,
                F10ElectronCategory::ElasticPlus,
            ] {
                let l0 = if electron {
                    0.5 + SIN2_THETA_W
                } else {
                    -0.5 + SIN2_THETA_W
                };
                let r0 = SIN2_THETA_W;
                let swap = anti != (category == F10ElectronCategory::ElasticPlus);
                let (left, right) = if swap { (r0, l0) } else { (l0, r0) };
                let common = 64.0 * G_F_MEV_MINUS_2.powi(2);
                let terms = [
                    2.0 * left * left * a * da,
                    2.0 * right * right * b * db,
                    -9.0 * left * right * (da - db),
                ];
                let reduced = (2.0 * left * left * a - 9.0 * left * right) * da
                    + (2.0 * right * right * b + 9.0 * left * right) * db;
                let expected = if head_on {
                    (324.0 / 5.0) * left * left + 24.0 * right * right - (72.0 / 5.0) * left * right
                } else {
                    -(4.0 / 5.0) * left * left - (808.0 / 1331.0) * right * right
                        + (72.0 / 121.0) * left * right
                };
                let scale = common * terms.into_iter().map(f64::abs).sum::<f64>();
                let (matrix, derivative) =
                    elastic_matrix_tangent(target, category, inv, dinv, 3.0, true, ULP_BUDGET)
                        .unwrap();
                assert_eq!(
                    matrix,
                    f10_electron_matrix(target, category, inv, 3.0, true, ULP_BUDGET).unwrap()
                );
                assert!(matrix.value > 0.0 && !matrix.corrected);
                signed_scalar_check(
                    &format!("M_T_{head_on}_{target:?}_{category:?}"),
                    derivative,
                    common * expected,
                    scale,
                );
                signed_scalar_check("reduced_AB", derivative, common * reduced, scale);
                if !head_on {
                    assert!(derivative < 0.0, "signed tangent clipping");
                }
            }
        }
        let zero_weight = F10EventMeasureInput {
            quadrature_weight: 0.0,
            ..input
        };
        let (_, nonzero) = event_measure_tangent(zero_weight, directions[0]).unwrap();
        signed_scalar_check(
            "zero_weight_nonzero_direction",
            nonzero,
            parts[0] / pi4,
            scale,
        );
    }
}

#[test]
fn elastic_zero_raw_matrix_requires_nondifferentiable_discrete_event() {
    // Scalar-helper boundary input, not an on-shell finite-mass event claim.
    // The admitted primal accepts raw=0; varying d13 gives raw_T != 0.
    let zero = F10InvariantProducts {
        d12: 0.0,
        d13: 0.0,
        d14: 0.0,
        d23: 0.0,
        d24: 0.0,
        d34: 0.0,
    };
    let base = f10_electron_matrix(
        F10Species::NuE,
        F10ElectronCategory::ElasticMinus,
        zero,
        3.0,
        true,
        ULP_BUDGET,
    )
    .unwrap();
    assert_eq!(base.value, 0.0);
    assert!(!base.corrected);
    for d13 in [1.0, -1.0] {
        let tangent = F10InvariantProducts { d13, ..zero };
        let result = elastic_matrix_tangent(
            F10Species::NuE,
            F10ElectronCategory::ElasticMinus,
            zero,
            tangent,
            3.0,
            true,
            ULP_BUDGET,
        );
        eprintln!("RAW_ZERO_NONZERO_DIRECTION d13_T={d13} base={base:?} candidate={result:?}");
        let error =
            result.expect_err("NONDIFFERENTIABLE_DISCRETE_EVENT required at raw=0, raw_T!=0");
        assert_eq!(error, F10KernelError::NondifferentiableDiscreteEvent);
    }
    let (unchanged, derivative) = elastic_matrix_tangent(
        F10Species::NuE,
        F10ElectronCategory::ElasticMinus,
        zero,
        zero,
        3.0,
        true,
        ULP_BUDGET,
    )
    .unwrap();
    assert_eq!(unchanged, base);
    assert_eq!(derivative.to_bits(), 0);
    // Nonfinite directions and finite-direction overflow keep their original error.
    for d13 in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
        assert_eq!(
            elastic_matrix_tangent(
                F10Species::NuE,
                F10ElectronCategory::ElasticMinus,
                zero,
                F10InvariantProducts { d13, ..zero },
                3.0,
                true,
                ULP_BUDGET,
            ),
            Err(F10KernelError::NonFiniteInput)
        );
    }
}

fn invariants(batch: &F10KinematicBatch, i: usize) -> F10InvariantProducts {
    F10InvariantProducts {
        d12: batch.d12[i],
        d13: batch.d13[i],
        d14: batch.d14[i],
        d23: batch.d23[i],
        d24: batch.d24[i],
        d34: batch.d34[i],
    }
}

fn measure(batch: &F10KinematicBatch, i: usize, p1: f64) -> F10EventMeasureInput {
    F10EventMeasureInput {
        p1,
        p2: batch.p2[i],
        e2: batch.e2[i],
        phase_space: batch.phase_space[i],
        quadrature_weight: batch.quadrature_weight[i],
        outer_weight: 1.0,
    }
}

fn primal(p1: f64, temperature: f64) -> F10KinematicBatch {
    let config = F10CollisionConfig::default();
    let (nodes, weights) =
        electron_half_line_rule(config.electron_radial_order, temperature).unwrap();
    two_body_kinematics(F10KinematicInput {
        p1,
        p2_nodes: &nodes,
        p2_weights: &weights,
        mass2: MASS,
        mass3: 0.0,
        mass4: MASS,
        config,
    })
    .unwrap()
}

fn relative(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    assert!(!a.is_empty());
    assert!(a.iter().chain(b).all(|x| x.is_finite()));
    let scale = a
        .iter()
        .chain(b)
        .map(|x| x.abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
        / scale
}

#[test]
fn elastic_prefactors_match_original_primal_at_two_fixed_branch_points() {
    for (p1, temperature) in [(2.0, 2.05), (0.75, 0.4)] {
        let tangent = evaluate_elastic_tgamma_kinematic_tangent(F10ElasticTgammaInput {
            p1,
            temperature_gamma: temperature,
            electron_mass: MASS,
            config: F10CollisionConfig::default(),
        })
        .unwrap();
        let base = &tangent.base;
        let n = base.support.len();
        let mut dm = Vec::<f64>::with_capacity(n);
        let mut di = Vec::<F10InvariantProducts>::with_capacity(n);
        for i in 0..n {
            let (value, derivative): (f64, f64) = event_measure_tangent(
                measure(base, i, p1),
                F10MeasureTangent {
                    d_p2: tangent.d_p2[i],
                    d_e2: tangent.d_e2[i],
                    d_phase_space: tangent.d_phase_space[i],
                    d_quadrature_weight: tangent.d_quadrature_weight[i],
                },
            )
            .unwrap();
            assert_eq!(
                value.to_bits(),
                f10_event_measure(measure(base, i, p1)).unwrap().to_bits()
            );
            dm.push(derivative);
            di.push(F10InvariantProducts {
                d12: tangent.d_d12[i],
                d13: tangent.d_d13[i],
                d14: tangent.d_d14[i],
                d23: tangent.d_d23[i],
                d24: tangent.d_d24[i],
                d34: tangent.d_d34[i],
            });
        }
        let mut best_measure = f64::INFINITY;
        let mut best_matrix = [f64::INFINITY; 12];
        for factor in [1.0e-3, 3.0e-4, 1.0e-4] {
            let h = temperature * factor;
            let plus = primal(p1, temperature + h);
            let minus = primal(p1, temperature - h);
            assert_eq!(
                base.support, plus.support,
                "NONDIFFERENTIABLE_DISCRETE_EVENT"
            );
            assert_eq!(
                base.support, minus.support,
                "NONDIFFERENTIABLE_DISCRETE_EVENT"
            );
            let centered: Vec<f64> = (0..n)
                .map(|i| {
                    (f10_event_measure(measure(&plus, i, p1)).unwrap()
                        - f10_event_measure(measure(&minus, i, p1)).unwrap())
                        / (2.0 * h)
                })
                .collect();
            let residual = relative(&dm, &centered);
            best_measure = best_measure.min(residual);
            eprintln!(
                "ELASTIC_MEASURE p1={p1} T={temperature} h={h:.17e} residual={residual:.17e}"
            );
            for (s, target) in F10Species::ALL.into_iter().enumerate() {
                for (c, category) in [
                    F10ElectronCategory::ElasticMinus,
                    F10ElectronCategory::ElasticPlus,
                ]
                .into_iter()
                .enumerate()
                {
                    let mut actual = Vec::<f64>::with_capacity(n);
                    let mut expected = Vec::<f64>::with_capacity(n);
                    for (i, derivative_invariants) in di.iter().copied().enumerate() {
                        let (matrix, derivative): (F10MatrixValue, f64) = elastic_matrix_tangent(
                            target,
                            category,
                            invariants(base, i),
                            derivative_invariants,
                            MASS,
                            base.support[i],
                            ULP_BUDGET,
                        )
                        .unwrap();
                        let original = f10_electron_matrix(
                            target,
                            category,
                            invariants(base, i),
                            MASS,
                            base.support[i],
                            ULP_BUDGET,
                        )
                        .unwrap();
                        assert_eq!(matrix, original);
                        let mp = f10_electron_matrix(
                            target,
                            category,
                            invariants(&plus, i),
                            MASS,
                            plus.support[i],
                            ULP_BUDGET,
                        )
                        .unwrap();
                        let mm = f10_electron_matrix(
                            target,
                            category,
                            invariants(&minus, i),
                            MASS,
                            minus.support[i],
                            ULP_BUDGET,
                        )
                        .unwrap();
                        assert_eq!(
                            matrix.corrected, mp.corrected,
                            "MATRIX_CORRECTION_BRANCH_CHANGED"
                        );
                        assert_eq!(
                            matrix.corrected, mm.corrected,
                            "MATRIX_CORRECTION_BRANCH_CHANGED"
                        );
                        actual.push(derivative);
                        expected.push((mp.value - mm.value) / (2.0 * h));
                    }
                    let residual = relative(&actual, &expected);
                    best_matrix[2 * s + c] = best_matrix[2 * s + c].min(residual);
                    eprintln!(
                        "ELASTIC_MATRIX p1={p1} T={temperature} target={target:?} category={category:?} h={h:.17e} residual={residual:.17e}"
                    );
                    let mutant: Vec<f64> = actual.iter().map(|x| 1.01 * x).collect();
                    assert!(relative(&mutant, &expected) > 1.0e-4);
                }
            }
        }
        assert!(best_measure <= CAP, "measure: {best_measure:.17e}");
        assert!(
            best_matrix.iter().all(|x| *x <= CAP),
            "matrix: {best_matrix:?}"
        );
    }
}

#[test]
fn elastic_prefactors_preserve_zero_cp_mass_and_correction_contracts() {
    let inv = F10InvariantProducts {
        d12: 2.0,
        d13: 0.5,
        d14: 1.0,
        d23: 1.0,
        d24: 0.5,
        d34: 2.0,
    };
    let zero = F10InvariantProducts {
        d12: 0.0,
        d13: 0.0,
        d14: 0.0,
        d23: 0.0,
        d24: 0.0,
        d34: 0.0,
    };
    let (_, dz): (F10MatrixValue, f64) = elastic_matrix_tangent(
        F10Species::NuE,
        F10ElectronCategory::ElasticMinus,
        inv,
        zero,
        MASS,
        true,
        ULP_BUDGET,
    )
    .unwrap();
    assert_eq!(dz.to_bits(), 0);
    let d13 = F10InvariantProducts { d13: 1.0, ..zero };
    let a: (F10MatrixValue, f64) = elastic_matrix_tangent(
        F10Species::NuE,
        F10ElectronCategory::ElasticMinus,
        inv,
        d13,
        MASS,
        true,
        ULP_BUDGET,
    )
    .unwrap();
    let b: (F10MatrixValue, f64) = elastic_matrix_tangent(
        F10Species::AntiNuE,
        F10ElectronCategory::ElasticPlus,
        inv,
        d13,
        MASS,
        true,
        ULP_BUDGET,
    )
    .unwrap();
    assert_eq!(a, b);
    assert!(a.0.value > 0.0 && !a.0.corrected);
    assert!(
        a.1 < 0.0,
        "finite-mass interference tangent is load-bearing"
    );
    let (_, massless): (F10MatrixValue, f64) = elastic_matrix_tangent(
        F10Species::NuE,
        F10ElectronCategory::ElasticMinus,
        inv,
        d13,
        0.0,
        true,
        ULP_BUDGET,
    )
    .unwrap();
    assert_eq!(massless.to_bits(), 0);
    let left = 0.5 + SIN2_THETA_W;
    let right = SIN2_THETA_W;
    let corrected_inv = F10InvariantProducts {
        d13: (left * left * 4.0 + right * right) / (left * right * MASS * MASS)
            * (1.0 + 128.0 * f64::EPSILON),
        ..inv
    };
    let (corrected, derivative): (F10MatrixValue, f64) = elastic_matrix_tangent(
        F10Species::NuE,
        F10ElectronCategory::ElasticMinus,
        corrected_inv,
        d13,
        MASS,
        true,
        ULP_BUDGET,
    )
    .unwrap();
    assert!(corrected.corrected);
    assert_eq!(corrected.value.to_bits(), 0);
    assert_eq!(derivative.to_bits(), 0);
    let (unsupported, derivative): (F10MatrixValue, f64) = elastic_matrix_tangent(
        F10Species::NuE,
        F10ElectronCategory::ElasticMinus,
        inv,
        d13,
        MASS,
        false,
        ULP_BUDGET,
    )
    .unwrap();
    assert_eq!(unsupported.value.to_bits(), 0);
    assert_eq!(derivative.to_bits(), 0);
    assert!(
        elastic_matrix_tangent(
            F10Species::NuE,
            F10ElectronCategory::Pair,
            inv,
            zero,
            MASS,
            true,
            ULP_BUDGET
        )
        .is_err()
    );
    let input = F10EventMeasureInput {
        p1: 2.0,
        p2: 1.0,
        e2: 1.2,
        phase_space: 0.5,
        quadrature_weight: 0.25,
        outer_weight: 1.0,
    };
    let dt = F10MeasureTangent {
        d_p2: 0.0,
        d_e2: 0.0,
        d_phase_space: 0.0,
        d_quadrature_weight: 0.0,
    };
    let (_, derivative): (f64, f64) = event_measure_tangent(input, dt).unwrap();
    assert_eq!(derivative.to_bits(), 0);
    assert!(
        event_measure_tangent(
            input,
            F10MeasureTangent {
                d_p2: f64::NAN,
                ..dt
            }
        )
        .is_err()
    );
    assert!(event_measure_tangent(F10EventMeasureInput { p1: 0.0, ..input }, dt).is_err());
}
