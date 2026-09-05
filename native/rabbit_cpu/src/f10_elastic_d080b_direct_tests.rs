//! Test-only array bridge; the frozen Python comparator gates its two outputs.
//! No helper formula is implemented here. Refusals remain typed and explicit.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::process::Command;

use serde_json::{Value, json};

use crate::f10_action_kinematics::F10CollisionConfig;
use crate::f10_elastic_prefactor_tangent::{
    F10MeasureTangent, elastic_matrix_tangent, event_measure_tangent,
};
use crate::f10_kernel_primitives::{
    F10ElectronCategory, F10EventMeasureInput, F10InvariantProducts, F10KernelError, F10Species,
    f10_electron_matrix,
};
use crate::f10_tgamma_kinematics::{
    F10ElasticTgammaInput, evaluate_elastic_tgamma_kinematic_tangent,
};

fn scalar(value: &Value) -> f64 {
    f64::from_bits(u64::from_str_radix(value.as_str().unwrap(), 16).unwrap())
}

fn array(values: impl Iterator<Item = Option<f64>>, shape: &[usize]) -> Value {
    let bits: Vec<Option<String>> = values
        .map(|value| value.map(|number| format!("{:016x}", number.to_bits())))
        .collect();
    assert_eq!(bits.len(), shape.iter().product::<usize>());
    json!({"shape": shape, "bits": bits})
}

fn invariants(
    input: &BTreeMap<String, Vec<f64>>,
    index: usize,
    prefix: &str,
) -> F10InvariantProducts {
    let get = |name| input[&format!("{prefix}{name}")][index];
    F10InvariantProducts {
        d12: get("d12"),
        d13: get("d13"),
        d14: get("d14"),
        d23: get("d23"),
        d24: get("d24"),
        d34: get("d34"),
    }
}

fn export(mode: &str) {
    let directory = PathBuf::from(std::env::var("D081R1F1_ELASTIC_DIRECT_DIR").unwrap());
    let oracle: Value = serde_json::from_reader(BufReader::new(
        File::open(directory.join("oracle.json")).unwrap(),
    ))
    .unwrap();
    let config = F10CollisionConfig::default();
    assert_eq!(
        oracle["config"]["incoming_polar_order"],
        config.incoming_polar_order
    );
    assert_eq!(
        oracle["config"]["final_polar_order"],
        config.final_polar_order
    );
    assert_eq!(
        oracle["config"]["final_azimuth_order"],
        config.final_azimuth_order
    );
    assert_eq!(
        oracle["config"]["electron_radial_order"],
        config.electron_radial_order
    );
    let roundoff_ulps = oracle["config"]["matrix_roundoff_ulps"].as_f64().unwrap();
    assert_eq!(roundoff_ulps, 1024.0);
    let cases = oracle["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 2);
    let mut outputs = Vec::new();
    let mut failures = Vec::new();
    for (case_index, case) in cases.iter().enumerate() {
        let p1 = scalar(&case["p1"]);
        let temperature_gamma = scalar(&case["temperature"]);
        let electron_mass = scalar(&case["electron_mass"]);
        let outer_weight = scalar(&case["outer_weight"]);
        let (shape, support, tangent_support, input): (
            [usize; 4],
            Vec<bool>,
            Vec<bool>,
            BTreeMap<String, Vec<f64>>,
        ) = if mode == "same_input" {
            (
                serde_json::from_value(case["shape"].clone()).unwrap(),
                serde_json::from_value(case["support"].clone()).unwrap(),
                serde_json::from_value(case["tangent_support"].clone()).unwrap(),
                case["input"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(name, values)| {
                        let values = values["bits"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(scalar)
                            .collect();
                        (name.clone(), values)
                    })
                    .collect(),
            )
        } else {
            let tangent = evaluate_elastic_tgamma_kinematic_tangent(F10ElasticTgammaInput {
                p1,
                temperature_gamma,
                electron_mass,
                config,
            })
            .unwrap();
            (
                tangent.base.shape,
                tangent.base.support,
                tangent.support,
                [
                    ("p2", tangent.base.p2),
                    ("e2", tangent.base.e2),
                    ("phase_space", tangent.base.phase_space),
                    ("quadrature_weight", tangent.base.quadrature_weight),
                    ("d12", tangent.base.d12),
                    ("d13", tangent.base.d13),
                    ("d14", tangent.base.d14),
                    ("d23", tangent.base.d23),
                    ("d24", tangent.base.d24),
                    ("d34", tangent.base.d34),
                    ("d_p2", tangent.d_p2),
                    ("d_e2", tangent.d_e2),
                    ("d_phase_space", tangent.d_phase_space),
                    ("d_quadrature_weight", tangent.d_quadrature_weight),
                    ("d_d12", tangent.d_d12),
                    ("d_d13", tangent.d_d13),
                    ("d_d14", tangent.d_d14),
                    ("d_d23", tangent.d_d23),
                    ("d_d24", tangent.d_d24),
                    ("d_d34", tangent.d_d34),
                ]
                .into_iter()
                .map(|(name, values)| (name.to_owned(), values))
                .collect(),
            )
        };
        let n: usize = shape.iter().product();
        assert_eq!(n, 27648);
        assert_eq!(support.len(), n);
        assert_eq!(tangent_support.len(), n);
        assert!(input.values().all(|values| values.len() == n));
        let mut measure = [Vec::new(), Vec::new()];
        for index in 0..n {
            let result = event_measure_tangent(
                F10EventMeasureInput {
                    p1,
                    p2: input["p2"][index],
                    e2: input["e2"][index],
                    phase_space: input["phase_space"][index],
                    quadrature_weight: input["quadrature_weight"][index],
                    outer_weight,
                },
                F10MeasureTangent {
                    d_p2: input["d_p2"][index],
                    d_e2: input["d_e2"][index],
                    d_phase_space: input["d_phase_space"][index],
                    d_quadrature_weight: input["d_quadrature_weight"][index],
                },
            );
            if let Err(error) = result {
                failures.push(json!([case_index, "measure", index, format!("{error:?}")]));
            }
            measure[0].push(result.ok().map(|value| value.0));
            measure[1].push(result.ok().map(|value| value.1));
        }
        let mut routes = Vec::new();
        let reference_routes = case["routes"].as_array().unwrap();
        assert_eq!(reference_routes.len(), 12);
        for route in reference_routes {
            let target = F10Species::from_name(route["target"].as_str().unwrap()).unwrap();
            let category =
                F10ElectronCategory::from_name(route["category"].as_str().unwrap()).unwrap();
            let mut columns = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            let mut corrected = Vec::new();
            let mut kink = Vec::new();
            let mut status = Vec::new();
            for (index, &supported) in support.iter().enumerate() {
                let base = invariants(&input, index, "");
                let direction = invariants(&input, index, "d_");
                let primal = f10_electron_matrix(
                    target,
                    category,
                    base,
                    electron_mass,
                    supported,
                    roundoff_ulps,
                );
                let result = elastic_matrix_tangent(
                    target,
                    category,
                    base,
                    direction,
                    electron_mass,
                    supported,
                    roundoff_ulps,
                );
                let (derivative, refusal, label) = match result {
                    Ok((_, derivative)) => (Some(derivative), false, "Ok".to_owned()),
                    Err(F10KernelError::NondifferentiableDiscreteEvent) => {
                        (None, true, "NondifferentiableDiscreteEvent".to_owned())
                    }
                    Err(error) => {
                        failures.push(json!([
                            case_index,
                            route["target"],
                            route["category"],
                            index,
                            format!("{error:?}")
                        ]));
                        (None, false, format!("{error:?}"))
                    }
                };
                columns[0].push(primal.ok().map(|value| value.value));
                columns[1].push(derivative);
                columns[2].push(primal.ok().map(|value| {
                    if value.corrected {
                        -value.correction
                    } else {
                        value.value
                    }
                }));
                columns[3].push(primal.ok().map(|value| value.scale));
                corrected.push(primal.ok().map(|value| value.corrected));
                kink.push(refusal);
                status.push(label);
            }
            routes.push(json!({
                "target": route["target"], "category": route["category"],
                "M": array(columns[0].iter().copied(), &shape),
                "M_T": array(columns[1].iter().copied(), &shape),
                "raw": array(columns[2].iter().copied(), &shape),
                "scale": array(columns[3].iter().copied(), &shape),
                "corrected": corrected, "kink": kink, "status": status,
            }));
        }
        let encoded_input: BTreeMap<_, _> = input
            .iter()
            .map(|(name, values)| (name, array(values.iter().copied().map(Some), &shape)))
            .collect();
        let indices: Vec<_> = support
            .iter()
            .enumerate()
            .filter_map(|(index, &supported)| supported.then_some(index))
            .collect();
        outputs.push(json!({
            "p1": case["p1"], "temperature": case["temperature"],
            "electron_mass": case["electron_mass"], "outer_weight": case["outer_weight"],
            "shape": shape, "sample_indices": (0..n).collect::<Vec<_>>(),
            "support": support, "tangent_support": tangent_support,
            "domain": support, "domain_indices": indices, "input": encoded_input,
            "measure": {"W": array(measure[0].iter().copied(), &shape),
                        "W_T": array(measure[1].iter().copied(), &shape)},
            "routes": routes,
        }));
    }
    serde_json::to_writer(
        BufWriter::new(File::create(directory.join(format!("{mode}.json"))).unwrap()),
        &outputs,
    )
    .unwrap();
    assert!(
        failures.is_empty(),
        "first helper failure: {:?}",
        failures.first()
    );
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let status = Command::new(std::env::var("D081R1F1_ELASTIC_DIRECT_PYTHON").unwrap())
        .current_dir(&root)
        .env(
            "PYTHONPATH",
            format!("{}:{}", root.join("src").display(), root.display()),
        )
        .arg("scripts/audit/d081r1f1_elastic_d080b_direct.py")
        .args(["compare", "--directory"])
        .arg(&directory)
        .args(["--mode", mode])
        .status()
        .unwrap();
    assert!(
        status.success(),
        "{mode} direct numerical comparison failed: {status}"
    );
}

#[test]
fn same_input_scalar_kernel_arrays() {
    export("same_input");
}

#[test]
fn end_to_end_prefactor_arrays() {
    export("end_to_end");
}
