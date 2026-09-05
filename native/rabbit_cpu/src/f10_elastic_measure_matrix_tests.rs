//! Prospective scalar-only tests: no retained states and no collision assembly.
//! Exact references use T=2 MeV, p1=1 MeV, p2(T)=2T, m=3 MeV,
//! fixed collinear incoming directions and a transverse outgoing COM direction.
//! The synthetic mass is a unit-test input, never a production configuration.
//! Caps fixed before Rust output: 256 epsilon for exact scalar references,
//! 1e-7 for the best unchanged-primal centered witness at [1e-3,3e-4,1e-4] MeV.
use crate::electron_hm::{G_F_MEV_MINUS_2, SIN2_THETA_W};
use crate::f10_elastic_measure_matrix::{
    F10ElasticScalarError, F10MeasureDirection, matrix_tgamma, measure_tgamma,
};
use crate::f10_kernel_primitives::{
    F10ElectronCategory, F10EventMeasureInput, F10InvariantProducts, F10Species,
    f10_electron_matrix, f10_event_measure,
};
use core::f64::consts::PI;

const EXACT_CAP: f64 = 256.0 * f64::EPSILON;

fn close(actual: f64, expected: f64, scale: f64) {
    assert!(actual.is_finite() && expected.is_finite() && scale.is_finite());
    assert!(
        (actual - expected).abs() <= EXACT_CAP * scale.abs().max(f64::MIN_POSITIVE),
        "actual={actual:.17e} expected={expected:.17e} scale={scale:.17e}"
    );
}

fn invariants(v: [f64; 6]) -> F10InvariantProducts {
    F10InvariantProducts {
        d12: v[0], d13: v[1], d14: v[2], d23: v[3], d24: v[4], d34: v[5],
    }
}

fn exact_case(head_on: bool) -> (F10InvariantProducts, F10InvariantProducts) {
    if head_on {
        (invariants([9.0, 3.0, 6.0, 6.0, 12.0, 9.0]),
         invariants([18.0/5.0, 8.0/5.0, 2.0, 2.0, 8.0/5.0, 18.0/5.0]))
    } else {
        (invariants([1.0, 1.0/11.0, 10.0/11.0, 10.0/11.0, 100.0/11.0, 1.0]),
         invariants([-2.0/5.0, -8.0/121.0, -202.0/605.0,
                     -202.0/605.0, -8.0/121.0, -2.0/5.0]))
    }
}

// Independent physical trajectory for validation-only centered differences.
fn primal_point(t: f64, head_on: bool) -> (F10EventMeasureInput, F10InvariantProducts) {
    let p2 = 2.0 * t;
    let e2 = (p2 * p2 + 9.0).sqrt();
    let s = 9.0 + 2.0 * (e2 + if head_on { p2 } else { -p2 });
    let a = (s - 9.0) / 2.0;
    let c = (s - 9.0).powi(2) / (4.0 * s);
    (F10EventMeasureInput {
        p1: 1.0, p2, e2, phase_space: (s - 9.0) / (2.0 * s),
        quadrature_weight: t / 2.0, outer_weight: 1.0,
    }, invariants([a, c, a-c, a-c, 9.0+c, a]))
}

fn measure_direction(head_on: bool) -> F10MeasureDirection {
    F10MeasureDirection {
        p2: 2.0, e2: 8.0/5.0, quadrature_weight: 0.5,
        phase_space: if head_on { 2.0/45.0 } else { -18.0/605.0 },
    }
}

fn channels() -> Vec<(F10Species, F10ElectronCategory, f64, f64)> {
    use F10ElectronCategory::{ElasticMinus, ElasticPlus};
    let r = SIN2_THETA_W;
    let mut out = Vec::new();
    for (nu, anti, l) in [
        (F10Species::NuE, F10Species::AntiNuE, 0.5+r),
        (F10Species::NuMu, F10Species::AntiNuMu, -0.5+r),
        (F10Species::NuTau, F10Species::AntiNuTau, -0.5+r),
    ] {
        out.extend([(nu, ElasticMinus, l, r), (anti, ElasticPlus, l, r),
                    (nu, ElasticPlus, r, l), (anti, ElasticMinus, r, l)]);
    }
    out
}

#[test]
fn exact_measure_references_keep_all_four_terms_and_zero_direction() {
    for head_on in [true, false] {
        let (input, _) = primal_point(2.0, head_on);
        let out = measure_tgamma(input, measure_direction(head_on)).unwrap();
        assert_eq!(out.base.to_bits(), f10_event_measure(input).unwrap().to_bits());
        let (base_ref, tangent_ref, phase_ratio) = if head_on {
            (1.0/(240.0*PI.powi(4)), 197.0/(36000.0*PI.powi(4)), 2.0/15.0)
        } else {
            (1.0/(880.0*PI.powi(4)), 469.0/(484000.0*PI.powi(4)), -18.0/55.0)
        };
        close(out.base, base_ref, base_ref);
        close(out.derivative, tangent_ref, tangent_ref);
        for (&observed, ratio) in out.components.iter().zip([0.5, 1.0, phase_ratio, -8.0/25.0]) {
            close(observed, base_ref * ratio, base_ref);
        }
        close(out.components.iter().sum(), out.derivative, tangent_ref);
        assert!(out.derivative > 0.0);
        let bound_value = 2.0 * out.derivative / out.base;
        assert!(bound_value > 1.0 && bound_value < 13.0/4.0);
        assert!((1.01*out.derivative-tangent_ref).abs() > EXACT_CAP*tangent_ref.abs());
        let zero = measure_tgamma(input, F10MeasureDirection {
            p2: 0.0, e2: 0.0, phase_space: 0.0, quadrature_weight: 0.0,
        }).unwrap();
        assert_eq!(zero.derivative.to_bits(), 0.0_f64.to_bits());
        assert!(zero.components.iter().all(|x| x.to_bits() == 0.0_f64.to_bits()));
    }
}

#[test]
fn exact_matrices_cover_all_twelve_channels_and_signed_negative_tangents() {
    let common = 64.0 * G_F_MEV_MINUS_2.powi(2);
    assert_eq!(channels().len(), 12);
    for head_on in [true, false] {
        let (base, direction) = exact_case(head_on);
        for (target, category, l, r) in channels() {
            let out = matrix_tgamma(target, category, base, direction, 3.0, true, 1024.0).unwrap();
            assert_eq!(out.base, f10_electron_matrix(target, category, base, 3.0, true, 1024.0).unwrap());
            let (expected_base, terms) = if head_on {
                (81.0*l*l + 36.0*r*r - 27.0*l*r,
                 [324.0/5.0*l*l, 24.0*r*r, -72.0/5.0*l*r])
            } else {
                (l*l + 100.0/121.0*r*r - 9.0/11.0*l*r,
                 [-4.0/5.0*l*l, -808.0/1331.0*r*r, 72.0/121.0*l*r])
            };
            let expected = common * terms.iter().sum::<f64>();
            let scale = common * terms.iter().map(|x| x.abs()).sum::<f64>();
            close(out.base.value, common*expected_base, common*expected_base);
            close(out.derivative, expected, scale);
            close(out.raw_derivative, expected, scale);
            for (&observed, term) in out.components.iter().zip(terms) {
                close(observed, common*term, scale);
            }
            assert!(out.base.value > 0.0 && !out.base.corrected);
            if !head_on {
                assert!(out.derivative < 0.0, "a nonnegative primal can have a negative tangent");
                let square = -common * (4.0/5.0*(l-45.0/121.0*r).powi(2)
                              + 7268.0/14641.0*r*r);
                close(out.derivative, square, scale);
                assert!((out.derivative.abs()-expected).abs() > EXACT_CAP*scale);
                assert!((out.derivative.max(0.0)-expected).abs() > EXACT_CAP*scale);
            }
        }
    }
}

#[test]
fn unchanged_primal_centered_ladders_match_both_scalar_helpers() {
    for head_on in [true, false] {
        let (input, _) = primal_point(2.0, head_on);
        let dm = measure_tgamma(input, measure_direction(head_on)).unwrap();
        let (base, direction) = exact_case(head_on);
        for (target, category, _, _) in channels() {
            let derivative = matrix_tgamma(target, category, base, direction, 3.0, true, 1024.0).unwrap();
            let mut best_measure = f64::INFINITY;
            let mut best_matrix = f64::INFINITY;
            for eps in [1e-3, 3e-4, 1e-4] {
                let (mp, ip) = primal_point(2.0+eps, head_on);
                let (mm, im) = primal_point(2.0-eps, head_on);
                let plus = f10_electron_matrix(target, category, ip, 3.0, true, 1024.0).unwrap();
                let minus = f10_electron_matrix(target, category, im, 3.0, true, 1024.0).unwrap();
                assert_eq!(plus.corrected, derivative.base.corrected);
                assert_eq!(minus.corrected, derivative.base.corrected);
                let fd_m = (f10_event_measure(mp).unwrap()-f10_event_measure(mm).unwrap())/(2.0*eps);
                let fd_k = (plus.value-minus.value)/(2.0*eps);
                best_measure = best_measure.min((fd_m-dm.derivative).abs()/dm.derivative.abs());
                best_matrix = best_matrix.min((fd_k-derivative.derivative).abs()/derivative.derivative.abs());
            }
            eprintln!("ELASTIC_SCALARS head_on={head_on} target={target:?} category={category:?} measure_fd={best_measure:.17e} matrix_fd={best_matrix:.17e}");
            assert!(best_measure <= 1e-7 && best_matrix <= 1e-7);
        }
    }
}

#[test]
fn invalid_directions_and_matrix_kinks_fail_closed() {
    let (input, _) = primal_point(2.0, true);
    let mut bad = measure_direction(true);
    bad.e2 = f64::NAN;
    assert!(measure_tgamma(input, bad).is_err());
    let mut invalid_base = input;
    invalid_base.p1 = 0.0;
    assert!(measure_tgamma(invalid_base, measure_direction(true)).is_err());
    let z = invariants([0.0; 6]);
    let d = invariants([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    assert!(matches!(matrix_tgamma(F10Species::NuE, F10ElectronCategory::Pair, z, d, 1.0, true, 1024.0), Err(F10ElasticScalarError::NonElasticCategory)));
    assert!(matches!(matrix_tgamma(F10Species::NuE, F10ElectronCategory::ElasticMinus, z, d, 1.0, true, 1024.0), Err(F10ElasticScalarError::NonDifferentiableDiscreteEvent)));
    let inactive = matrix_tgamma(F10Species::NuE, F10ElectronCategory::ElasticMinus, z, d, 1.0, false, 1024.0).unwrap();
    assert_eq!(inactive.derivative.to_bits(), 0.0_f64.to_bits());
    let zero = matrix_tgamma(F10Species::NuE, F10ElectronCategory::ElasticMinus, z, z, 1.0, true, 1024.0).unwrap();
    assert_eq!(zero.derivative.to_bits(), 0.0_f64.to_bits());
    let mut nonfinite = d;
    nonfinite.d24 = f64::INFINITY;
    assert!(matrix_tgamma(F10Species::NuE, F10ElectronCategory::ElasticMinus, z, nonfinite, 1.0, true, 1024.0).is_err());
}

#[test]
fn corrected_primal_branch_is_distinct_from_a_signed_tangent() {
    let l = 0.5 + SIN2_THETA_W;
    let r = SIN2_THETA_W;
    let interference = f64::from_bits((l/r).to_bits()+4);
    let base = invariants([1.0, interference, 0.0, 0.0, 0.0, 1.0]);
    let direction = invariants([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    let out = matrix_tgamma(F10Species::NuE, F10ElectronCategory::ElasticMinus, base, direction, 1.0, true, 1024.0).unwrap();
    assert!(out.base.corrected && out.base.correction > 0.0);
    assert!(out.raw_derivative < 0.0);
    assert_eq!(out.derivative.to_bits(), 0.0_f64.to_bits());
}
