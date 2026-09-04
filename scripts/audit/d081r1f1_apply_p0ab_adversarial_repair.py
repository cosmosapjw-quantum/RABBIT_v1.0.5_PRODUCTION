#!/usr/bin/env python3
"""Apply the bounded D-081R1F1 P0A/P0B adversarial primitive repair.

This script changes only the admitted thermal primitive implementation. It
adds the analytic derivative of the actual 256-panel finite Simpson EOS
algorithm, retains the previous 4096-panel result as a continuum reference,
and adds normalized branch-margin diagnostics matching the frozen D-080A
semantics. It does not alter collision coefficients, quadrature identities,
support predicates, solver tolerances, retained states, or downstream JVPs.
"""

from __future__ import annotations

from pathlib import Path


TGAMMA = Path("native/rabbit_cpu/src/f10_tgamma_tangent.rs")
KINEMATICS = Path("native/rabbit_cpu/src/f10_tgamma_kinematics.rs")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def patch_tgamma() -> bool:
    text = TGAMMA.read_text(encoding="utf-8")
    if "electromagnetic_eos_tgamma_discrete_d2rho_dt2" in text:
        return False

    text = replace_once(
        text,
        "const D2_RHO_SIMPSON_PANELS: usize = 4096;\nconst D2_RHO_TAIL_E_FOLDS: f64 = 48.0;\n",
        "const DISCRETE_D2_RHO_SIMPSON_PANELS: usize = 256;\n"
        "const CONTINUUM_D2_RHO_SIMPSON_PANELS: usize = 4096;\n"
        "const D2_RHO_TAIL_E_FOLDS: f64 = 48.0;\n",
        "EOS panel constants",
    )
    text = replace_once(
        text,
        "fn electron_pair_d2rho_dt2(temperature_gamma_mev: f64) -> Result<f64, F10TgammaTangentError> {",
        "pub(crate) fn electromagnetic_eos_tgamma_continuum_d2rho_dt2_reference(\n"
        "    temperature_gamma_mev: f64,\n"
        ") -> Result<f64, F10TgammaTangentError> {",
        "continuum reference rename",
    )
    text = replace_once(
        text,
        "let step = theta_max / D2_RHO_SIMPSON_PANELS as f64;",
        "let step = theta_max / CONTINUUM_D2_RHO_SIMPSON_PANELS as f64;",
        "continuum step panels",
    )
    text = replace_once(
        text,
        "for index in 0..=D2_RHO_SIMPSON_PANELS {",
        "for index in 0..=CONTINUUM_D2_RHO_SIMPSON_PANELS {",
        "continuum loop panels",
    )
    text = replace_once(
        text,
        "let weight = if index == 0 || index == D2_RHO_SIMPSON_PANELS {",
        "let weight = if index == 0 || index == CONTINUUM_D2_RHO_SIMPSON_PANELS {",
        "continuum endpoint panels",
    )

    anchor = "pub(crate) fn electromagnetic_eos_tgamma_tangent(\n"
    discrete = r'''pub(crate) fn electromagnetic_eos_tgamma_discrete_d2rho_dt2(
    temperature_gamma_mev: f64,
) -> Result<f64, F10TgammaTangentError> {
    if !temperature_gamma_mev.is_finite() || temperature_gamma_mev <= 0.0 {
        return Err(F10TgammaTangentError::InvalidInput);
    }
    let ratio = ELECTRON_MASS_MEV / temperature_gamma_mev;
    if !ratio.is_finite() || ratio <= 0.0 {
        return Err(F10TgammaTangentError::NonFiniteOutput);
    }
    let endpoint_argument = 1.0 + D2_RHO_TAIL_E_FOLDS / ratio;
    let theta_max = endpoint_argument.acosh();
    let endpoint_denominator = (endpoint_argument.powi(2) - 1.0).sqrt();
    let d_theta_max_dt =
        (D2_RHO_TAIL_E_FOLDS / ELECTRON_MASS_MEV) / endpoint_denominator;
    if !theta_max.is_finite()
        || theta_max <= 0.0
        || !d_theta_max_dt.is_finite()
        || d_theta_max_dt <= 0.0
    {
        return Err(F10TgammaTangentError::NonFiniteOutput);
    }

    let step = theta_max / DISCRETE_D2_RHO_SIMPSON_PANELS as f64;
    let d_step_dt = d_theta_max_dt / DISCRETE_D2_RHO_SIMPSON_PANELS as f64;
    let d_ratio_dt = -ratio / temperature_gamma_mev;
    let ratio_fourth = ratio.powi(4);
    let ratio_fifth = ratio_fourth * ratio;
    let mut weighted_sum = 0.0_f64;
    let mut weighted_tangent_sum = 0.0_f64;

    for index in 0..=DISCRETE_D2_RHO_SIMPSON_PANELS {
        let theta = index as f64 * step;
        let d_theta_dt = index as f64 * d_step_dt;
        let sinh = theta.sinh();
        let cosh = theta.cosh();
        let epsilon = ratio * cosh;
        let d_epsilon_dt = d_ratio_dt * cosh + ratio * sinh * d_theta_dt;
        let exponential_negative = (-epsilon).exp();
        let occupation = exponential_negative / (1.0 + exponential_negative);
        let blocking = occupation * (1.0 - occupation);
        let d_blocking_dt = -blocking * (1.0 - 2.0 * occupation) * d_epsilon_dt;
        let geometric = sinh.powi(2) * cosh.powi(3);
        let d_geometric_dtheta =
            2.0 * sinh * cosh.powi(4) + 3.0 * sinh.powi(3) * cosh.powi(2);
        let value = ratio_fifth * geometric * blocking;
        let tangent = 5.0 * ratio_fourth * d_ratio_dt * geometric * blocking
            + ratio_fifth * d_geometric_dtheta * d_theta_dt * blocking
            + ratio_fifth * geometric * d_blocking_dt;
        if !value.is_finite() || !tangent.is_finite() {
            return Err(F10TgammaTangentError::NonFiniteOutput);
        }
        let weight = if index == 0 || index == DISCRETE_D2_RHO_SIMPSON_PANELS {
            1.0
        } else if index.is_multiple_of(2) {
            2.0
        } else {
            4.0
        };
        weighted_sum += weight * value;
        weighted_tangent_sum += weight * tangent;
    }

    let dimensionless_sum = weighted_sum * step / 3.0;
    let d_dimensionless_sum_dt =
        d_step_dt * weighted_sum / 3.0 + step * weighted_tangent_sum / 3.0;
    let electron = 2.0
        * (3.0 * temperature_gamma_mev.powi(2) * dimensionless_sum
            + temperature_gamma_mev.powi(3) * d_dimensionless_sum_dt)
        / PI.powi(2);
    let photon = 4.0 * PI.powi(2) * temperature_gamma_mev.powi(2) / 5.0;
    let result = photon + electron;
    if !result.is_finite() || result <= 0.0 {
        return Err(F10TgammaTangentError::NonFiniteOutput);
    }
    Ok(result)
}

'''
    text = replace_once(text, anchor, discrete + anchor, "discrete EOS insertion")
    text = replace_once(
        text,
        "let d2_rho = electron_pair_d2rho_dt2(temperature_gamma_mev)?;",
        "let d2_rho = electromagnetic_eos_tgamma_discrete_d2rho_dt2(\n"
        "        temperature_gamma_mev,\n"
        "    )?;",
        "production discrete EOS selection",
    )
    TGAMMA.write_text(text, encoding="utf-8")
    return True


def patch_kinematics() -> bool:
    text = KINEMATICS.read_text(encoding="utf-8")
    if "minimum_supported_lambda_margin_relative" in text:
        return False

    text = replace_once(
        text,
        "    pub(crate) minimum_support_margin: f64,\n"
        "    pub(crate) minimum_lambda_margin: f64,\n",
        "    pub(crate) minimum_support_margin: f64,\n"
        "    pub(crate) minimum_lambda_margin: f64,\n"
        "    pub(crate) minimum_support_margin_relative: f64,\n"
        "    pub(crate) minimum_supported_lambda_margin_relative: f64,\n",
        "normalized margin result fields",
    )
    text = replace_once(
        text,
        "    let mut minimum_support_margin = f64::INFINITY;\n"
        "    let mut minimum_lambda_margin = f64::INFINITY;\n"
        "    let mut output_index = 0_usize;\n",
        "    let mut minimum_support_margin = f64::INFINITY;\n"
        "    let mut minimum_lambda_margin = f64::INFINITY;\n"
        "    let mut maximum_invariant_s_absolute = 0.0_f64;\n"
        "    let mut minimum_supported_lambda_margin_relative = f64::INFINITY;\n"
        "    let mut output_index = 0_usize;\n",
        "margin accumulator initialization",
    )
    text = replace_once(
        text,
        "            let invariant_s = invariant_s_raw.max(0.0);\n"
        "            let d_invariant_s = if invariant_s_raw > 0.0 {\n",
        "            let invariant_s = invariant_s_raw.max(0.0);\n"
        "            maximum_invariant_s_absolute =\n"
        "                maximum_invariant_s_absolute.max(invariant_s.abs());\n"
        "            let d_invariant_s = if invariant_s_raw > 0.0 {\n",
        "invariant-s scale accumulation",
    )
    text = replace_once(
        text,
        "            if support && lambda <= 0.0 {\n"
        "                return Err(F10TgammaKinematicError::NondifferentiableDiscreteEvent);\n"
        "            }\n"
        "            let d_lambda = 2.0 * (invariant_s - mass3_squared - mass4_squared) * d_invariant_s;\n",
        "            if support && lambda <= 0.0 {\n"
        "                return Err(F10TgammaKinematicError::NondifferentiableDiscreteEvent);\n"
        "            }\n"
        "            if support {\n"
        "                minimum_supported_lambda_margin_relative =\n"
        "                    minimum_supported_lambda_margin_relative.min(\n"
        "                        lambda / square(invariant_s).max(f64::MIN_POSITIVE),\n"
        "                    );\n"
        "            }\n"
        "            let d_lambda = 2.0 * (invariant_s - mass3_squared - mass4_squared) * d_invariant_s;\n",
        "supported lambda margin accumulation",
    )
    text = replace_once(
        text,
        "    if output_index != expected_size\n"
        "        || !minimum_support_margin.is_finite()\n"
        "        || minimum_support_margin <= 0.0\n"
        "        || !minimum_lambda_margin.is_finite()\n"
        "        || minimum_lambda_margin <= 0.0\n"
        "    {\n",
        "    let minimum_support_margin_relative = minimum_support_margin\n"
        "        / maximum_invariant_s_absolute\n"
        "            .max(threshold_squared)\n"
        "            .max(f64::MIN_POSITIVE);\n"
        "    if output_index != expected_size\n"
        "        || !minimum_support_margin.is_finite()\n"
        "        || minimum_support_margin <= 0.0\n"
        "        || !minimum_lambda_margin.is_finite()\n"
        "        || minimum_lambda_margin <= 0.0\n"
        "        || !minimum_support_margin_relative.is_finite()\n"
        "        || minimum_support_margin_relative <= 0.0\n"
        "        || !minimum_supported_lambda_margin_relative.is_finite()\n"
        "        || minimum_supported_lambda_margin_relative <= 0.0\n"
        "    {\n",
        "normalized margin validation",
    )
    text = replace_once(
        text,
        "        minimum_support_margin,\n"
        "        minimum_lambda_margin,\n"
        "    })\n",
        "        minimum_support_margin,\n"
        "        minimum_lambda_margin,\n"
        "        minimum_support_margin_relative,\n"
        "        minimum_supported_lambda_margin_relative,\n"
        "    })\n",
        "normalized margin result construction",
    )
    KINEMATICS.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    changed = patch_tgamma() | patch_kinematics()
    print("D-081R1F1 P0A/P0B adversarial repair:", "CHANGED" if changed else "NOOP")


if __name__ == "__main__":
    main()
