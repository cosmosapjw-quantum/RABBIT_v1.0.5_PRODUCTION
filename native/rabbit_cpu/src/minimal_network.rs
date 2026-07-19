//! Checked nine-species, nested twelve/selected-31-reaction BBN substrate.
//!
//! The internal state order is fixed independently of either embedded JSON
//! file.  Every table row is selected by a checked physical reaction identity;
//! positional coincidence is never used.  This module evaluates only nuclear
//! fluxes.  Weak rates, FLRW cooling, QED, corrections and observables belong
//! to later foundation stages.

#![cfg_attr(not(test), allow(dead_code))]

use serde_json::Value;

pub(crate) const N_SPECIES: usize = 9;
pub(crate) const N_BACKBONE_REACTIONS: usize = 12;
pub(crate) const N_REACTIONS: usize = 31;
pub(crate) const MEV_TO_T9: f64 = 11.604_518_121_550_082;
const AVOGADRO_PER_MOL: f64 = 6.022_140_76e23;
const HBAR_C_MEV_CM: f64 = 197.326_980_4e-13;
pub(crate) const APERY_ZETA_THREE: f64 = 1.202_056_903_159_594_2;
const CANONICAL_TABLE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../src/rabbit/network/data/primat_ac2024_31rxn.json"
));

#[cfg(test)]
const STANDALONE_TABLE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../src/rabbit/network/data/primat_ac2024_12rxn.json"
));

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub(crate) enum Species {
    Neutron = 0,
    Proton = 1,
    Deuterium = 2,
    Tritium = 3,
    Helium3 = 4,
    Helium4 = 5,
    Lithium6 = 6,
    Lithium7 = 7,
    Beryllium7 = 8,
}

impl Species {
    pub(crate) const ALL: [Self; N_SPECIES] = [
        Self::Neutron,
        Self::Proton,
        Self::Deuterium,
        Self::Tritium,
        Self::Helium3,
        Self::Helium4,
        Self::Lithium6,
        Self::Lithium7,
        Self::Beryllium7,
    ];

    const fn index(self) -> usize {
        self as usize
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Neutron => "n",
            Self::Proton => "p",
            Self::Deuterium => "D",
            Self::Tritium => "T",
            Self::Helium3 => "He3",
            Self::Helium4 => "He4",
            Self::Lithium6 => "Li6",
            Self::Lithium7 => "Li7",
            Self::Beryllium7 => "Be7",
        }
    }
}

pub(crate) const MASS_NUMBERS: [f64; N_SPECIES] = [1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 6.0, 7.0, 7.0];
pub(crate) const CHARGE_NUMBERS: [f64; N_SPECIES] = [0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0];

#[derive(Clone, Copy, Debug)]
struct ReactionSpec {
    canonical_index: usize,
    canonical_name: &'static str,
    standalone_name: Option<&'static str>,
    reactants: [u8; N_SPECIES],
    products: [u8; N_SPECIES],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NetworkExtent {
    Backbone12,
    Selected31,
}

impl NetworkExtent {
    pub(crate) const fn reaction_count(self) -> usize {
        match self {
            Self::Backbone12 => N_BACKBONE_REACTIONS,
            Self::Selected31 => N_REACTIONS,
        }
    }

    pub(crate) const fn evolves_lithium6(self) -> bool {
        matches!(self, Self::Selected31)
    }
}

const fn counts(entries: &[(Species, u8)]) -> [u8; N_SPECIES] {
    let mut result = [0; N_SPECIES];
    let mut index = 0;
    while index < entries.len() {
        let (species, multiplicity) = entries[index];
        result[species.index()] = multiplicity;
        index += 1;
    }
    result
}

const REACTIONS: [ReactionSpec; N_REACTIONS] = [
    ReactionSpec {
        canonical_index: 0,
        canonical_name: "n + p > d + g",
        standalone_name: Some("n+p→D+γ"),
        reactants: counts(&[(Species::Neutron, 1), (Species::Proton, 1)]),
        products: counts(&[(Species::Deuterium, 1)]),
    },
    ReactionSpec {
        canonical_index: 1,
        canonical_name: "d + p > He3 + g",
        standalone_name: Some("D+p→³He+γ"),
        reactants: counts(&[(Species::Deuterium, 1), (Species::Proton, 1)]),
        products: counts(&[(Species::Helium3, 1)]),
    },
    ReactionSpec {
        canonical_index: 2,
        canonical_name: "d + d > He3 + n",
        standalone_name: Some("D+D→n+³He"),
        reactants: counts(&[(Species::Deuterium, 2)]),
        products: counts(&[(Species::Helium3, 1), (Species::Neutron, 1)]),
    },
    ReactionSpec {
        canonical_index: 3,
        canonical_name: "d + d > t + p",
        standalone_name: Some("D+D→p+T"),
        reactants: counts(&[(Species::Deuterium, 2)]),
        products: counts(&[(Species::Tritium, 1), (Species::Proton, 1)]),
    },
    ReactionSpec {
        canonical_index: 4,
        canonical_name: "t + p > a + g",
        standalone_name: Some("T+p→⁴He+γ"),
        reactants: counts(&[(Species::Tritium, 1), (Species::Proton, 1)]),
        products: counts(&[(Species::Helium4, 1)]),
    },
    ReactionSpec {
        canonical_index: 5,
        canonical_name: "t + d > a + n",
        standalone_name: Some("T+D→n+⁴He"),
        reactants: counts(&[(Species::Tritium, 1), (Species::Deuterium, 1)]),
        products: counts(&[(Species::Helium4, 1), (Species::Neutron, 1)]),
    },
    ReactionSpec {
        canonical_index: 6,
        canonical_name: "t + a > Li7 + g",
        standalone_name: Some("T+⁴He→⁷Li+γ"),
        reactants: counts(&[(Species::Tritium, 1), (Species::Helium4, 1)]),
        products: counts(&[(Species::Lithium7, 1)]),
    },
    ReactionSpec {
        canonical_index: 7,
        canonical_name: "He3 + n > t + p",
        standalone_name: Some("³He+n→p+T"),
        reactants: counts(&[(Species::Helium3, 1), (Species::Neutron, 1)]),
        products: counts(&[(Species::Tritium, 1), (Species::Proton, 1)]),
    },
    ReactionSpec {
        canonical_index: 8,
        canonical_name: "He3 + d > a + p",
        standalone_name: Some("³He+D→p+⁴He"),
        reactants: counts(&[(Species::Helium3, 1), (Species::Deuterium, 1)]),
        products: counts(&[(Species::Helium4, 1), (Species::Proton, 1)]),
    },
    ReactionSpec {
        canonical_index: 9,
        canonical_name: "He3 + a > Be7 + g",
        standalone_name: Some("³He+⁴He→⁷Be+γ"),
        reactants: counts(&[(Species::Helium3, 1), (Species::Helium4, 1)]),
        products: counts(&[(Species::Beryllium7, 1)]),
    },
    ReactionSpec {
        canonical_index: 10,
        canonical_name: "Be7 + n > Li7 + p",
        standalone_name: Some("⁷Be+n→⁷Li+p"),
        reactants: counts(&[(Species::Beryllium7, 1), (Species::Neutron, 1)]),
        products: counts(&[(Species::Lithium7, 1), (Species::Proton, 1)]),
    },
    ReactionSpec {
        canonical_index: 11,
        canonical_name: "Li7 + p > a + a",
        standalone_name: Some("⁷Li+p→⁴He+⁴He"),
        reactants: counts(&[(Species::Lithium7, 1), (Species::Proton, 1)]),
        products: counts(&[(Species::Helium4, 2)]),
    },
    ReactionSpec {
        canonical_index: 12,
        canonical_name: "Li7 + p > a + a + g",
        standalone_name: None,
        reactants: counts(&[(Species::Lithium7, 1), (Species::Proton, 1)]),
        products: counts(&[(Species::Helium4, 2)]),
    },
    ReactionSpec {
        canonical_index: 13,
        canonical_name: "Be7 + n > a + a",
        standalone_name: None,
        reactants: counts(&[(Species::Beryllium7, 1), (Species::Neutron, 1)]),
        products: counts(&[(Species::Helium4, 2)]),
    },
    ReactionSpec {
        canonical_index: 14,
        canonical_name: "Be7 + d > 2a + p",
        standalone_name: None,
        reactants: counts(&[(Species::Beryllium7, 1), (Species::Deuterium, 1)]),
        products: counts(&[(Species::Helium4, 2), (Species::Proton, 1)]),
    },
    ReactionSpec {
        canonical_index: 15,
        canonical_name: "d + a > Li6 + g",
        standalone_name: None,
        reactants: counts(&[(Species::Deuterium, 1), (Species::Helium4, 1)]),
        products: counts(&[(Species::Lithium6, 1)]),
    },
    ReactionSpec {
        canonical_index: 16,
        canonical_name: "Li6 + p > Be7 + g",
        standalone_name: None,
        reactants: counts(&[(Species::Lithium6, 1), (Species::Proton, 1)]),
        products: counts(&[(Species::Beryllium7, 1)]),
    },
    ReactionSpec {
        canonical_index: 17,
        canonical_name: "Li6 + p > He3 + a",
        standalone_name: None,
        reactants: counts(&[(Species::Lithium6, 1), (Species::Proton, 1)]),
        products: counts(&[(Species::Helium3, 1), (Species::Helium4, 1)]),
    },
    ReactionSpec {
        canonical_index: 18,
        canonical_name: "Li6 + He3 > a + a  + p",
        standalone_name: None,
        reactants: counts(&[(Species::Lithium6, 1), (Species::Helium3, 1)]),
        products: counts(&[(Species::Helium4, 2), (Species::Proton, 1)]),
    },
    ReactionSpec {
        canonical_index: 19,
        canonical_name: "Li6 + t > a + a  + n",
        standalone_name: None,
        reactants: counts(&[(Species::Lithium6, 1), (Species::Tritium, 1)]),
        products: counts(&[(Species::Helium4, 2), (Species::Neutron, 1)]),
    },
    ReactionSpec {
        canonical_index: 20,
        canonical_name: "Li7 + He3 > Li6 + a",
        standalone_name: None,
        reactants: counts(&[(Species::Lithium7, 1), (Species::Helium3, 1)]),
        products: counts(&[(Species::Lithium6, 1), (Species::Helium4, 1)]),
    },
    ReactionSpec {
        canonical_index: 21,
        canonical_name: "Be7 + t > Li6 + a",
        standalone_name: None,
        reactants: counts(&[(Species::Beryllium7, 1), (Species::Tritium, 1)]),
        products: counts(&[(Species::Lithium6, 1), (Species::Helium4, 1)]),
    },
    ReactionSpec {
        canonical_index: 22,
        canonical_name: "Li6 + t > Li7 + d",
        standalone_name: None,
        reactants: counts(&[(Species::Lithium6, 1), (Species::Tritium, 1)]),
        products: counts(&[(Species::Lithium7, 1), (Species::Deuterium, 1)]),
    },
    ReactionSpec {
        canonical_index: 23,
        canonical_name: "Li6 + He3 > Be7 + d",
        standalone_name: None,
        reactants: counts(&[(Species::Lithium6, 1), (Species::Helium3, 1)]),
        products: counts(&[(Species::Beryllium7, 1), (Species::Deuterium, 1)]),
    },
    ReactionSpec {
        canonical_index: 24,
        canonical_name: "Li7 + He3 > a + a + d",
        standalone_name: None,
        reactants: counts(&[(Species::Lithium7, 1), (Species::Helium3, 1)]),
        products: counts(&[(Species::Helium4, 2), (Species::Deuterium, 1)]),
    },
    ReactionSpec {
        canonical_index: 25,
        canonical_name: "Be7 + t > a + a + d",
        standalone_name: None,
        reactants: counts(&[(Species::Beryllium7, 1), (Species::Tritium, 1)]),
        products: counts(&[(Species::Helium4, 2), (Species::Deuterium, 1)]),
    },
    ReactionSpec {
        canonical_index: 26,
        canonical_name: "Be7 + t > Li7 + He3",
        standalone_name: None,
        reactants: counts(&[(Species::Beryllium7, 1), (Species::Tritium, 1)]),
        products: counts(&[(Species::Lithium7, 1), (Species::Helium3, 1)]),
    },
    ReactionSpec {
        canonical_index: 27,
        canonical_name: "Be7 + He3 > p + p + 2a",
        standalone_name: None,
        reactants: counts(&[(Species::Beryllium7, 1), (Species::Helium3, 1)]),
        products: counts(&[(Species::Proton, 2), (Species::Helium4, 2)]),
    },
    ReactionSpec {
        canonical_index: 28,
        canonical_name: "d + d > a + g",
        standalone_name: None,
        reactants: counts(&[(Species::Deuterium, 2)]),
        products: counts(&[(Species::Helium4, 1)]),
    },
    ReactionSpec {
        canonical_index: 29,
        canonical_name: "He3 + He3 > a + p + p",
        standalone_name: None,
        reactants: counts(&[(Species::Helium3, 2)]),
        products: counts(&[(Species::Helium4, 1), (Species::Proton, 2)]),
    },
    ReactionSpec {
        canonical_index: 30,
        canonical_name: "Li7 + d > a + a + n",
        standalone_name: None,
        reactants: counts(&[(Species::Lithium7, 1), (Species::Deuterium, 1)]),
        products: counts(&[(Species::Helium4, 2), (Species::Neutron, 1)]),
    },
];

const REACTION_REFERENCES: [&str; N_REACTIONS] = [
    "And06",
    "Moscoso2021",
    "Gom17",
    "Gom17",
    "Ser04",
    "deSouza19a",
    "DAACV04",
    "DAACV04",
    "deSouza19b",
    "Ili16",
    "deSouza2020",
    "DAACV04",
    "NACRE II",
    "Bar16",
    "Rij19",
    "Trezzi2017",
    "NACRE II",
    "NACRE II",
    "TALYS2",
    "TALYS2",
    "TALYS2",
    "TALYS2",
    "TALYS2",
    "TALYS2",
    "TALYS2",
    "TALYS2",
    "TALYS2",
    "TALYS2",
    "NACRE II",
    "NACRE II",
    "CGXSV12",
];

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum NetworkError {
    EmbeddedData(String),
    NonFiniteTemperature,
    TemperatureOutsideTable,
    InvalidBaryonToPhotonRatio,
    InvalidBaryonNumberDensity,
    InvalidAbundance { species: usize },
    InvalidReactionOrder,
    NonFiniteRate { reaction: usize },
    NonFiniteFlux { reaction: usize },
    NonFiniteDerivative { species: usize },
}

#[derive(Clone, Debug)]
struct RateRow {
    log_rates: Vec<f64>,
    q_mev: f64,
    reverse_factor: f64,
    t9_power: f64,
    gamma: f64,
}

#[derive(Clone, Debug)]
struct RateTable {
    t9: Vec<f64>,
    rows: Vec<RateRow>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct NetworkRates {
    pub(crate) forward: [f64; N_REACTIONS],
    pub(crate) reverse: [f64; N_REACTIONS],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FluxComponents {
    pub(crate) forward: [f64; N_REACTIONS],
    pub(crate) reverse: [f64; N_REACTIONS],
}

#[derive(Clone, Debug)]
pub(crate) struct MinimalNetwork {
    table: RateTable,
    extent: NetworkExtent,
}

impl MinimalNetwork {
    pub(crate) fn canonical_reaction_names() -> [&'static str; N_BACKBONE_REACTIONS] {
        core::array::from_fn(|index| REACTIONS[index].canonical_name)
    }

    pub(crate) fn selected_31_reaction_names() -> [&'static str; N_REACTIONS] {
        core::array::from_fn(|index| REACTIONS[index].canonical_name)
    }

    pub(crate) fn from_embedded_canonical_table() -> Result<Self, NetworkError> {
        Self::from_embedded_table_with_extent(NetworkExtent::Backbone12)
    }

    pub(crate) fn from_embedded_selected_31_table() -> Result<Self, NetworkError> {
        Self::from_embedded_table_with_extent(NetworkExtent::Selected31)
    }

    fn from_embedded_table_with_extent(extent: NetworkExtent) -> Result<Self, NetworkError> {
        let table = parse_canonical_table(CANONICAL_TABLE, extent.reaction_count())?;
        validate_reaction_conservation(extent.reaction_count())?;
        Ok(Self { table, extent })
    }

    pub(crate) const fn extent(&self) -> NetworkExtent {
        self.extent
    }

    pub(crate) const fn reaction_count(&self) -> usize {
        self.extent.reaction_count()
    }

    pub(crate) fn temperature_bounds_mev(&self) -> (f64, f64) {
        (
            self.table.t9[0] / MEV_TO_T9,
            self.table.t9[self.table.t9.len() - 1] / MEV_TO_T9,
        )
    }

    pub(crate) fn evaluate_rates(
        &self,
        temperature_mev: f64,
    ) -> Result<NetworkRates, NetworkError> {
        if !temperature_mev.is_finite() {
            return Err(NetworkError::NonFiniteTemperature);
        }
        let t9 = temperature_mev * MEV_TO_T9;
        let first = self.table.t9[0];
        let last = self.table.t9[self.table.t9.len() - 1];
        if t9 < first || t9 > last {
            return Err(NetworkError::TemperatureOutsideTable);
        }

        let upper = self.table.t9.partition_point(|value| *value <= t9);
        let lower = if upper == 0 {
            0
        } else if upper == self.table.t9.len() {
            upper - 2
        } else {
            upper - 1
        };
        let log_t9 = t9.ln();
        let log_low = self.table.t9[lower].ln();
        let log_high = self.table.t9[lower + 1].ln();
        let fraction = (log_t9 - log_low) / (log_high - log_low);
        let mut forward = [0.0; N_REACTIONS];
        let mut reverse = [0.0; N_REACTIONS];
        for reaction in 0..self.reaction_count() {
            let row = &self.table.rows[reaction];
            let log_forward =
                row.log_rates[lower] + fraction * (row.log_rates[lower + 1] - row.log_rates[lower]);
            let log_reverse =
                log_forward + row.reverse_factor.ln() + row.t9_power * log_t9 + row.gamma / t9;
            forward[reaction] = log_forward.exp();
            reverse[reaction] = log_reverse.exp();
            if !forward[reaction].is_finite()
                || forward[reaction] < 0.0
                || !reverse[reaction].is_finite()
                || reverse[reaction] < 0.0
            {
                return Err(NetworkError::NonFiniteRate { reaction });
            }
        }
        Ok(NetworkRates { forward, reverse })
    }

    pub(crate) fn flux_components(
        &self,
        mass_fractions: &[f64; N_SPECIES],
        temperature_mev: f64,
        baryon_to_photon_ratio: f64,
    ) -> Result<FluxComponents, NetworkError> {
        validate_mass_fractions(mass_fractions)?;
        let rates = self.evaluate_rates(temperature_mev)?;
        let rho = baryon_density_factor(temperature_mev, baryon_to_photon_ratio)?;
        directional_fluxes(mass_fractions, rho, &rates, self.reaction_count())
    }

    pub(crate) fn rhs(
        &self,
        mass_fractions: &[f64; N_SPECIES],
        temperature_mev: f64,
        baryon_to_photon_ratio: f64,
    ) -> Result<[f64; N_SPECIES], NetworkError> {
        let order: Vec<usize> = (0..self.reaction_count()).collect();
        self.rhs_in_order(
            mass_fractions,
            temperature_mev,
            baryon_to_photon_ratio,
            &order,
        )
    }

    pub(crate) fn rhs_with_baryon_number_density(
        &self,
        mass_fractions: &[f64; N_SPECIES],
        temperature_mev: f64,
        baryon_number_density_per_cm3: f64,
    ) -> Result<[f64; N_SPECIES], NetworkError> {
        validate_mass_fractions(mass_fractions)?;
        self.stage_rhs_with_baryon_number_density(
            mass_fractions,
            temperature_mev,
            baryon_number_density_per_cm3,
        )
    }

    /// Evaluate the finite signed polynomial extension used by implicit stages.
    ///
    /// This does not clamp or repair a stage. The strict public kernel above
    /// still rejects negative physical input states.
    pub(crate) fn stage_rhs_with_baryon_number_density(
        &self,
        mass_fractions: &[f64; N_SPECIES],
        temperature_mev: f64,
        baryon_number_density_per_cm3: f64,
    ) -> Result<[f64; N_SPECIES], NetworkError> {
        validate_finite_mass_fractions(mass_fractions)?;
        let rates = self.evaluate_rates(temperature_mev)?;
        let rho = density_factor_from_number_density(baryon_number_density_per_cm3)?;
        let fluxes = directional_fluxes(mass_fractions, rho, &rates, self.reaction_count())?;
        let order: Vec<usize> = (0..self.reaction_count()).collect();
        mass_fraction_derivative(&fluxes, &order)
    }

    fn rhs_in_order(
        &self,
        mass_fractions: &[f64; N_SPECIES],
        temperature_mev: f64,
        baryon_to_photon_ratio: f64,
        order: &[usize],
    ) -> Result<[f64; N_SPECIES], NetworkError> {
        validate_reaction_order(order, self.reaction_count())?;
        let fluxes =
            self.flux_components(mass_fractions, temperature_mev, baryon_to_photon_ratio)?;
        mass_fraction_derivative(&fluxes, order)
    }

    pub(crate) fn jacobian(
        &self,
        mass_fractions: &[f64; N_SPECIES],
        temperature_mev: f64,
        baryon_to_photon_ratio: f64,
    ) -> Result<[f64; N_SPECIES * N_SPECIES], NetworkError> {
        validate_mass_fractions(mass_fractions)?;
        let rates = self.evaluate_rates(temperature_mev)?;
        let rho = baryon_density_factor(temperature_mev, baryon_to_photon_ratio)?;
        abundance_jacobian_from_rates(mass_fractions, rho, &rates, self.reaction_count())
    }

    pub(crate) fn jacobian_with_baryon_number_density(
        &self,
        mass_fractions: &[f64; N_SPECIES],
        temperature_mev: f64,
        baryon_number_density_per_cm3: f64,
    ) -> Result<[f64; N_SPECIES * N_SPECIES], NetworkError> {
        validate_mass_fractions(mass_fractions)?;
        self.stage_jacobian_with_baryon_number_density(
            mass_fractions,
            temperature_mev,
            baryon_number_density_per_cm3,
        )
    }

    /// Jacobian of the finite signed polynomial stage extension.
    pub(crate) fn stage_jacobian_with_baryon_number_density(
        &self,
        mass_fractions: &[f64; N_SPECIES],
        temperature_mev: f64,
        baryon_number_density_per_cm3: f64,
    ) -> Result<[f64; N_SPECIES * N_SPECIES], NetworkError> {
        validate_finite_mass_fractions(mass_fractions)?;
        let rates = self.evaluate_rates(temperature_mev)?;
        let rho = density_factor_from_number_density(baryon_number_density_per_cm3)?;
        abundance_jacobian_polynomial_from_rates(mass_fractions, rho, &rates, self.reaction_count())
    }
}

fn embedded_data(message: impl Into<String>) -> NetworkError {
    NetworkError::EmbeddedData(message.into())
}

fn json_array<'a>(root: &'a Value, key: &str) -> Result<&'a Vec<Value>, NetworkError> {
    root.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| embedded_data(format!("missing array {key}")))
}

fn number_array(root: &Value, key: &str) -> Result<Vec<f64>, NetworkError> {
    json_array(root, key)?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .filter(|entry| entry.is_finite())
                .ok_or_else(|| embedded_data(format!("invalid number in {key}")))
        })
        .collect()
}

fn parse_canonical_table(raw: &str, reaction_count: usize) -> Result<RateTable, NetworkError> {
    if !matches!(reaction_count, N_BACKBONE_REACTIONS | N_REACTIONS) {
        return Err(embedded_data("unsupported canonical reaction extent"));
    }
    let root: Value = serde_json::from_str(raw)
        .map_err(|error| embedded_data(format!("canonical JSON: {error}")))?;
    if root.get("source").and_then(Value::as_str) != Some("BBNRatesAC2024.dat (Coc & Pitrou)") {
        return Err(embedded_data("canonical rate-table source mismatch"));
    }
    validate_canonical_species(&root)?;
    let t9 = number_array(&root, "T9")?;
    validate_temperature_grid(&t9)?;
    let reaction_values = json_array(&root, "reactions")?;
    if reaction_values.len() < reaction_count
        || (reaction_count == N_REACTIONS && reaction_values.len() != N_REACTIONS)
    {
        return Err(embedded_data("canonical reaction set has wrong length"));
    }
    let mut rows = Vec::with_capacity(reaction_count);
    for spec in REACTIONS.iter().copied().take(reaction_count) {
        let matches: Vec<&Value> = reaction_values
            .iter()
            .filter(|value| {
                value.get("reaction").and_then(Value::as_str) == Some(spec.canonical_name)
            })
            .collect();
        if matches.len() != 1 {
            return Err(embedded_data(format!(
                "canonical reaction identity {} occurs {} times",
                spec.canonical_name,
                matches.len()
            )));
        }
        let value = matches[0];
        let stored_index = value
            .get("index")
            .and_then(Value::as_u64)
            .ok_or_else(|| embedded_data("missing canonical reaction index"))?
            as usize;
        if stored_index != spec.canonical_index || stored_index >= reaction_count {
            return Err(embedded_data(format!(
                "canonical identity/index mismatch for {}",
                spec.canonical_name
            )));
        }
        if value.get("reference").and_then(Value::as_str) != Some(REACTION_REFERENCES[stored_index])
        {
            return Err(embedded_data(format!(
                "canonical reference mismatch for {}",
                spec.canonical_name
            )));
        }
        rows.push(parse_object_rate_row(value, t9.len())?);
    }
    Ok(RateTable { t9, rows })
}

fn parse_object_rate_row(value: &Value, expected_length: usize) -> Result<RateRow, NetworkError> {
    let log_rates = number_array(value, "log_rates")?;
    if log_rates.len() != expected_length {
        return Err(embedded_data("rate row length does not match T9 grid"));
    }
    let field = |name: &str| {
        value
            .get(name)
            .and_then(Value::as_f64)
            .filter(|entry| entry.is_finite())
            .ok_or_else(|| embedded_data(format!("invalid rate metadata {name}")))
    };
    let row = RateRow {
        log_rates,
        q_mev: field("Q_MeV")?,
        reverse_factor: field("rev_factor")?,
        t9_power: field("T9_power")?,
        gamma: field("gamma")?,
    };
    // AC2024 row 27 stores a negative Q metadata value while its tabulated
    // reverse-law exponent remains the authoritative signed `gamma`.  Q is a
    // provenance field here, not a value from which the reverse law is rebuilt.
    if row.q_mev == 0.0 || row.reverse_factor <= 0.0 || row.gamma >= 0.0 {
        return Err(embedded_data("invalid reverse-rate metadata signs"));
    }
    Ok(row)
}

fn validate_temperature_grid(t9: &[f64]) -> Result<(), NetworkError> {
    if t9.len() < 2
        || t9[0] <= 0.0
        || t9
            .windows(2)
            .any(|pair| !pair[0].is_finite() || pair[1] <= pair[0])
    {
        return Err(embedded_data(
            "T9 grid is not strictly increasing and positive",
        ));
    }
    Ok(())
}

fn validate_canonical_species(root: &Value) -> Result<(), NetworkError> {
    let names = json_array(root, "species")?;
    let masses = number_array(root, "atomic_masses")?;
    let charges = number_array(root, "charge_numbers")?;
    if names.len() != N_SPECIES || masses.len() != N_SPECIES || charges.len() != N_SPECIES {
        return Err(embedded_data("canonical species metadata has wrong length"));
    }
    let mut seen = [false; N_SPECIES];
    for index in 0..N_SPECIES {
        let name = names[index]
            .as_str()
            .ok_or_else(|| embedded_data("canonical species name is not a string"))?;
        let species = table_species(name)
            .ok_or_else(|| embedded_data(format!("unknown canonical species {name}")))?;
        let internal = species.index();
        if seen[internal]
            || masses[index] != MASS_NUMBERS[internal]
            || charges[index] != CHARGE_NUMBERS[internal]
        {
            return Err(embedded_data(format!(
                "invalid metadata for species {name}"
            )));
        }
        seen[internal] = true;
    }
    if seen.iter().any(|entry| !entry) {
        return Err(embedded_data("canonical species set is incomplete"));
    }
    Ok(())
}

fn table_species(name: &str) -> Option<Species> {
    match name {
        "n" => Some(Species::Neutron),
        "p" => Some(Species::Proton),
        "d" => Some(Species::Deuterium),
        "t" => Some(Species::Tritium),
        "He3" => Some(Species::Helium3),
        "a" => Some(Species::Helium4),
        "Li6" => Some(Species::Lithium6),
        "Li7" => Some(Species::Lithium7),
        "Be7" => Some(Species::Beryllium7),
        _ => None,
    }
}

fn validate_reaction_conservation(reaction_count: usize) -> Result<(), NetworkError> {
    for spec in REACTIONS.iter().copied().take(reaction_count) {
        let mut baryon = 0.0;
        let mut charge = 0.0;
        for species in 0..N_SPECIES {
            let delta = f64::from(stoichiometry(spec, species));
            baryon += MASS_NUMBERS[species] * delta;
            charge += CHARGE_NUMBERS[species] * delta;
        }
        if baryon != 0.0 || charge != 0.0 {
            return Err(embedded_data(format!(
                "nonconserving reaction {}",
                spec.canonical_name
            )));
        }
    }
    Ok(())
}

fn validate_mass_fractions(mass_fractions: &[f64; N_SPECIES]) -> Result<(), NetworkError> {
    for (species, value) in mass_fractions.iter().enumerate() {
        if !value.is_finite() || *value < 0.0 {
            return Err(NetworkError::InvalidAbundance { species });
        }
    }
    Ok(())
}

fn validate_finite_mass_fractions(mass_fractions: &[f64; N_SPECIES]) -> Result<(), NetworkError> {
    for (species, value) in mass_fractions.iter().enumerate() {
        if !value.is_finite() {
            return Err(NetworkError::InvalidAbundance { species });
        }
    }
    Ok(())
}

fn validate_reaction_order(order: &[usize], reaction_count: usize) -> Result<(), NetworkError> {
    if reaction_count > N_REACTIONS || order.len() != reaction_count {
        return Err(NetworkError::InvalidReactionOrder);
    }
    let mut seen = [false; N_REACTIONS];
    for &reaction in order {
        if reaction >= reaction_count || reaction >= N_REACTIONS || seen[reaction] {
            return Err(NetworkError::InvalidReactionOrder);
        }
        seen[reaction] = true;
    }
    Ok(())
}

fn baryon_density_factor(
    temperature_mev: f64,
    baryon_to_photon_ratio: f64,
) -> Result<f64, NetworkError> {
    if !baryon_to_photon_ratio.is_finite() || baryon_to_photon_ratio <= 0.0 {
        return Err(NetworkError::InvalidBaryonToPhotonRatio);
    }
    let photon_density_per_cm3 = photon_number_density_per_cm3(temperature_mev)?;
    let result = baryon_to_photon_ratio * photon_density_per_cm3 / AVOGADRO_PER_MOL;
    if !result.is_finite() || result <= 0.0 {
        return Err(NetworkError::InvalidBaryonToPhotonRatio);
    }
    Ok(result)
}

pub(crate) fn photon_number_density_per_cm3(temperature_mev: f64) -> Result<f64, NetworkError> {
    if !temperature_mev.is_finite() || temperature_mev <= 0.0 {
        return Err(NetworkError::NonFiniteTemperature);
    }
    let photon_density_mev_cubed =
        2.0 * APERY_ZETA_THREE / core::f64::consts::PI.powi(2) * temperature_mev.powi(3);
    let result = photon_density_mev_cubed / HBAR_C_MEV_CM.powi(3);
    if !result.is_finite() || result <= 0.0 {
        return Err(NetworkError::NonFiniteTemperature);
    }
    Ok(result)
}

fn density_factor_from_number_density(
    baryon_number_density_per_cm3: f64,
) -> Result<f64, NetworkError> {
    if !baryon_number_density_per_cm3.is_finite() || baryon_number_density_per_cm3 <= 0.0 {
        return Err(NetworkError::InvalidBaryonNumberDensity);
    }
    let result = baryon_number_density_per_cm3 / AVOGADRO_PER_MOL;
    if !result.is_finite() || result <= 0.0 {
        return Err(NetworkError::InvalidBaryonNumberDensity);
    }
    Ok(result)
}

fn molar_abundances(mass_fractions: &[f64; N_SPECIES]) -> [f64; N_SPECIES] {
    core::array::from_fn(|species| mass_fractions[species] / MASS_NUMBERS[species])
}

fn directional_fluxes(
    mass_fractions: &[f64; N_SPECIES],
    rho: f64,
    rates: &NetworkRates,
    reaction_count: usize,
) -> Result<FluxComponents, NetworkError> {
    let molar = molar_abundances(mass_fractions);
    let mut forward = [0.0; N_REACTIONS];
    let mut reverse = [0.0; N_REACTIONS];
    for (reaction, spec) in REACTIONS.iter().enumerate().take(reaction_count) {
        forward[reaction] = rho
            * symmetry_factor(&spec.reactants)
            * rates.forward[reaction]
            * monomial(&spec.reactants, &molar);
        let product_count = particle_count(&spec.products);
        reverse[reaction] = integer_power(rho, product_count.saturating_sub(1))
            * symmetry_factor(&spec.products)
            * rates.reverse[reaction]
            * monomial(&spec.products, &molar);
        if !forward[reaction].is_finite() || !reverse[reaction].is_finite() {
            return Err(NetworkError::NonFiniteFlux { reaction });
        }
    }
    Ok(FluxComponents { forward, reverse })
}

fn mass_fraction_derivative(
    fluxes: &FluxComponents,
    order: &[usize],
) -> Result<[f64; N_SPECIES], NetworkError> {
    validate_reaction_order(order, order.len())?;
    let mut derivative = [0.0; N_SPECIES];
    for &reaction in order {
        let net_flux = fluxes.forward[reaction] - fluxes.reverse[reaction];
        for species in 0..N_SPECIES {
            derivative[species] += MASS_NUMBERS[species]
                * f64::from(stoichiometry(REACTIONS[reaction], species))
                * net_flux;
        }
    }
    for (species, value) in derivative.iter().enumerate() {
        if !value.is_finite() {
            return Err(NetworkError::NonFiniteDerivative { species });
        }
    }
    Ok(derivative)
}

fn abundance_jacobian_from_rates(
    mass_fractions: &[f64; N_SPECIES],
    rho: f64,
    rates: &NetworkRates,
    reaction_count: usize,
) -> Result<[f64; N_SPECIES * N_SPECIES], NetworkError> {
    validate_mass_fractions(mass_fractions)?;
    abundance_jacobian_polynomial_from_rates(mass_fractions, rho, rates, reaction_count)
}

fn abundance_jacobian_polynomial_from_rates(
    mass_fractions: &[f64; N_SPECIES],
    rho: f64,
    rates: &NetworkRates,
    reaction_count: usize,
) -> Result<[f64; N_SPECIES * N_SPECIES], NetworkError> {
    validate_finite_mass_fractions(mass_fractions)?;
    let molar = molar_abundances(mass_fractions);
    let mut jacobian = [0.0; N_SPECIES * N_SPECIES];
    for (reaction, spec) in REACTIONS.iter().enumerate().take(reaction_count) {
        let forward_scale = rho * symmetry_factor(&spec.reactants) * rates.forward[reaction];
        let product_count = particle_count(&spec.products);
        let reverse_scale = integer_power(rho, product_count.saturating_sub(1))
            * symmetry_factor(&spec.products)
            * rates.reverse[reaction];
        for column in 0..N_SPECIES {
            let forward_derivative = forward_scale
                * monomial_derivative(&spec.reactants, &molar, column)
                / MASS_NUMBERS[column];
            let reverse_derivative = reverse_scale
                * monomial_derivative(&spec.products, &molar, column)
                / MASS_NUMBERS[column];
            let net_derivative = forward_derivative - reverse_derivative;
            for row in 0..N_SPECIES {
                jacobian[row * N_SPECIES + column] +=
                    MASS_NUMBERS[row] * f64::from(stoichiometry(*spec, row)) * net_derivative;
            }
        }
    }
    for (index, value) in jacobian.iter().enumerate() {
        if !value.is_finite() {
            return Err(NetworkError::NonFiniteDerivative {
                species: index / N_SPECIES,
            });
        }
    }
    Ok(jacobian)
}

fn particle_count(counts: &[u8; N_SPECIES]) -> usize {
    counts.iter().map(|value| usize::from(*value)).sum()
}

fn integer_power(value: f64, exponent: usize) -> f64 {
    let mut result = 1.0;
    for _ in 0..exponent {
        result *= value;
    }
    result
}

fn symmetry_factor(counts: &[u8; N_SPECIES]) -> f64 {
    let mut denominator = 1_u64;
    for &count in counts {
        for factor in 2..=u64::from(count) {
            denominator *= factor;
        }
    }
    1.0 / denominator as f64
}

fn monomial(counts: &[u8; N_SPECIES], values: &[f64; N_SPECIES]) -> f64 {
    let mut result = 1.0;
    for species in 0..N_SPECIES {
        for _ in 0..counts[species] {
            result *= values[species];
        }
    }
    result
}

fn monomial_derivative(
    counts: &[u8; N_SPECIES],
    values: &[f64; N_SPECIES],
    variable: usize,
) -> f64 {
    if counts[variable] == 0 {
        return 0.0;
    }
    let mut result = f64::from(counts[variable]);
    for species in 0..N_SPECIES {
        let power = counts[species] - u8::from(species == variable);
        for _ in 0..power {
            result *= values[species];
        }
    }
    result
}

const fn stoichiometry(spec: ReactionSpec, species: usize) -> i8 {
    spec.products[species] as i8 - spec.reactants[species] as i8
}

#[cfg(test)]
fn parse_standalone_table(
    raw: &str,
) -> Result<(RateTable, [usize; N_BACKBONE_REACTIONS]), NetworkError> {
    let root: Value = serde_json::from_str(raw)
        .map_err(|error| embedded_data(format!("standalone JSON: {error}")))?;
    let t9 = number_array(&root, "T9")?;
    validate_temperature_grid(&t9)?;
    let declared_reactions = root.get("N_reactions").and_then(Value::as_u64);
    let declared_temperatures = root.get("N_T9").and_then(Value::as_u64);
    if declared_reactions != Some(N_BACKBONE_REACTIONS as u64)
        || declared_temperatures != Some(t9.len() as u64)
    {
        return Err(embedded_data("standalone declared shape mismatch"));
    }
    let log_t9 = number_array(&root, "log_T9")?;
    if log_t9.len() != t9.len()
        || log_t9
            .iter()
            .zip(&t9)
            .any(|(stored, temperature)| (stored - temperature.ln()).abs() > 3.0e-15)
    {
        return Err(embedded_data("standalone log_T9 mismatch"));
    }
    let names = json_array(&root, "reaction_names")?;
    let shape = root
        .get("log_rates")
        .and_then(|value| value.get("shape"))
        .and_then(Value::as_array)
        .ok_or_else(|| embedded_data("missing standalone rate shape"))?;
    if shape.len() != 2
        || shape[0].as_u64() != Some(N_BACKBONE_REACTIONS as u64)
        || shape[1].as_u64() != Some(t9.len() as u64)
    {
        return Err(embedded_data("standalone rate shape mismatch"));
    }
    let flat = root
        .get("log_rates")
        .and_then(|value| value.get("data"))
        .and_then(Value::as_array)
        .ok_or_else(|| embedded_data("missing standalone flat rate data"))?;
    let flat: Vec<f64> = flat
        .iter()
        .map(|value| {
            value
                .as_f64()
                .filter(|entry| entry.is_finite())
                .ok_or_else(|| embedded_data("invalid standalone rate"))
        })
        .collect::<Result<_, _>>()?;
    let q = number_array(&root, "Q_MeV")?;
    let reverse_factor = number_array(&root, "rev_factor")?;
    let t9_power = number_array(&root, "T9_power")?;
    let gamma = number_array(&root, "gamma")?;
    if names.len() != N_BACKBONE_REACTIONS
        || flat.len() != N_BACKBONE_REACTIONS * t9.len()
        || q.len() != N_BACKBONE_REACTIONS
        || reverse_factor.len() != N_BACKBONE_REACTIONS
        || t9_power.len() != N_BACKBONE_REACTIONS
        || gamma.len() != N_BACKBONE_REACTIONS
    {
        return Err(embedded_data("standalone table shape mismatch"));
    }
    let mut rows = Vec::with_capacity(N_BACKBONE_REACTIONS);
    let mut canonical_to_standalone = [usize::MAX; N_BACKBONE_REACTIONS];
    for (canonical, spec) in REACTIONS.iter().take(N_BACKBONE_REACTIONS).enumerate() {
        let standalone_name = spec
            .standalone_name
            .ok_or_else(|| embedded_data("missing standalone reaction identity"))?;
        let matches: Vec<usize> = names
            .iter()
            .enumerate()
            .filter_map(|(index, value)| (value.as_str() == Some(standalone_name)).then_some(index))
            .collect();
        if matches.len() != 1 {
            return Err(embedded_data(format!(
                "standalone reaction identity {} occurs {} times",
                standalone_name,
                matches.len()
            )));
        }
        let source = matches[0];
        canonical_to_standalone[canonical] = source;
        rows.push(RateRow {
            log_rates: flat[source * t9.len()..(source + 1) * t9.len()].to_vec(),
            q_mev: q[source],
            reverse_factor: reverse_factor[source],
            t9_power: t9_power[source],
            gamma: gamma[source],
        });
    }
    Ok((RateTable { t9, rows }, canonical_to_standalone))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ode::{OdeConfig, OdeSystem, SolverKind, solve};

    const ETA: f64 = 6.1e-10;

    fn network() -> MinimalNetwork {
        MinimalNetwork::from_embedded_canonical_table().unwrap()
    }

    fn selected_31_network() -> MinimalNetwork {
        MinimalNetwork::from_embedded_selected_31_table().unwrap()
    }

    fn representative_mass_fractions() -> [f64; N_SPECIES] {
        [0.12, 0.50, 0.08, 0.05, 0.04, 0.16, 0.005, 0.02, 0.025]
    }

    fn assert_relative(actual: f64, expected: f64, tolerance: f64) {
        let scale = expected.abs().max(1.0e-300);
        assert!(
            (actual - expected).abs() <= tolerance * scale,
            "actual={actual:.17e}, expected={expected:.17e}, rel={:.3e}",
            (actual - expected).abs() / scale
        );
    }

    #[test]
    fn species_order_and_every_reaction_conserve_baryon_and_charge() {
        assert_eq!(
            Species::ALL.map(Species::name),
            ["n", "p", "D", "T", "He3", "He4", "Li6", "Li7", "Be7"]
        );
        validate_reaction_conservation(N_REACTIONS).unwrap();
        for (reaction, spec) in REACTIONS.iter().copied().enumerate() {
            assert_eq!(particle_count(&spec.reactants), 2);
            let baryon: f64 = (0..N_SPECIES)
                .map(|species| MASS_NUMBERS[species] * f64::from(stoichiometry(spec, species)))
                .sum();
            let charge: f64 = (0..N_SPECIES)
                .map(|species| CHARGE_NUMBERS[species] * f64::from(stoichiometry(spec, species)))
                .sum();
            assert_eq!(baryon, 0.0, "{}", spec.canonical_name);
            assert_eq!(charge, 0.0, "{}", spec.canonical_name);
            if reaction < N_BACKBONE_REACTIONS {
                assert_eq!(stoichiometry(spec, Species::Lithium6.index()), 0);
            }
        }
    }

    #[test]
    fn both_tables_match_only_after_the_named_nontrivial_permutation() {
        let canonical = parse_canonical_table(CANONICAL_TABLE, N_BACKBONE_REACTIONS).unwrap();
        let (standalone, canonical_to_standalone) =
            parse_standalone_table(STANDALONE_TABLE).unwrap();
        assert_eq!(
            canonical_to_standalone,
            [0, 1, 2, 3, 11, 4, 5, 6, 7, 8, 10, 9]
        );
        let mut standalone_to_canonical = [usize::MAX; N_BACKBONE_REACTIONS];
        for (canonical_index, standalone_index) in
            canonical_to_standalone.iter().copied().enumerate()
        {
            standalone_to_canonical[standalone_index] = canonical_index;
        }
        assert_eq!(
            standalone_to_canonical,
            [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 10, 4]
        );
        assert_eq!(canonical.t9, standalone.t9);
        for reaction in 0..N_BACKBONE_REACTIONS {
            let left = &canonical.rows[reaction];
            let right = &standalone.rows[reaction];
            assert_eq!(left.log_rates, right.log_rates);
            assert_eq!(left.q_mev, right.q_mev);
            assert_eq!(left.reverse_factor, right.reverse_factor);
            assert_eq!(left.t9_power, right.t9_power);
            assert_eq!(left.gamma, right.gamma);
        }
    }

    #[test]
    fn selected_31_identity_extent_and_reverse_metadata_are_explicit() {
        let selected = selected_31_network();
        assert_eq!(selected.extent(), NetworkExtent::Selected31);
        assert!(selected.extent().evolves_lithium6());
        assert_eq!(selected.reaction_count(), 31);
        assert_eq!(network().reaction_count(), 12);
        assert_eq!(
            MinimalNetwork::selected_31_reaction_names(),
            core::array::from_fn(|index| REACTIONS[index].canonical_name)
        );
        assert_eq!(selected.table.rows.len(), N_REACTIONS);
        assert_eq!(selected.table.rows[27].q_mev, -11.2724);
        assert_eq!(selected.table.rows[27].gamma, -130.8113);
        // AC2024 Q metadata is not a reverse-law reconstruction authority:
        // these additional rows also carry gamma-implied Q values distinct
        // from their raw Q fields. The tabulated gamma is consumed verbatim.
        assert_eq!(selected.table.rows[22].q_mev, 0.9927);
        assert_eq!(selected.table.rows[22].gamma, -11.5332);
        assert_eq!(selected.table.rows[23].q_mev, 0.1123);
        assert_eq!(selected.table.rows[23].gamma, -1.3157);
    }

    #[test]
    fn selected_31_extension_rates_match_independent_json_math_formulation() {
        // Generated without importing Rabbit's Python network module: the
        // JSON rows were identity-selected and log-interpolated with
        // `math.log/exp` at T_gamma=0.1 MeV. This checks implementation over
        // shared AC2024 data; it is not an independent rate measurement.
        let expected_forward = [
            3.135_648_078_206_429e4,
            6.410_319_386_332_5e6,
            7.439_004_903_582e6,
            1.621_922_114_672_967_6e-1,
            2.823_281_210_375_505e2,
            8.606_688_795_690_319e6,
            2.843_483_131_196_170_6e5,
            6.901_798_439_727_29e6,
            1.574_877_639_326_629e5,
            1.318_166_951_863_179_2e6,
            3.184_749_847_033_754e6,
            5.103_277_021_037_077e-14,
            1.051_499_087_853_804_1e5,
            9.216_250_081_308_536e5,
            1.041_995_551_614_641_6e5,
            5.387_017_737_192_292e4,
            1.342_102_416_690_603,
            3.561_565_428_315_896_4e5,
            1.444_578_523_854_950_4e7,
        ];
        let expected_reverse = [
            6.817_534_444_288_851e-71,
            9.995_724_387_662_577e-76,
            9.012_989_037_588_684e-76,
            1.234_083_814_219_149_2e3,
            1.871_491_336_344_529e-12,
            3.204_066_164_724_622e-11,
            8.153_809_430_278_936e-78,
            4.096_831_863_411_967e-73,
            4.549_500_061_890_961e-53,
            5.713_337_442_089_285e-56,
            1.118_300_456_945_764_4e2,
            1.194_267_302_836_127e-14,
            7.987_338_174_168_156e-57,
            1.050_369_219_008_824_3e-59,
            1.563_350_996_750_939_7e1,
            4.658_573_999_328_01e-64,
            2.074_446_545_938_938_5e-93,
            1.370_034_497_107_938_8e-60,
            2.417_160_227_738_494_4e-68,
        ];
        let rates = selected_31_network().evaluate_rates(0.1).unwrap();
        for (offset, (&forward, &reverse)) in
            expected_forward.iter().zip(&expected_reverse).enumerate()
        {
            let reaction = N_BACKBONE_REACTIONS + offset;
            assert_relative(rates.forward[reaction], forward, 2.0e-14);
            assert_relative(rates.reverse[reaction], reverse, 8.0e-14);
        }
    }

    #[test]
    fn interpolation_and_reverse_formula_match_independent_numeric_anchors() {
        let network = network();
        let rates = network.evaluate_rates(0.1 / MEV_TO_T9).unwrap();
        assert_relative(rates.forward[0], 38_471.999_999_999_97, 3.0e-15);
        assert_relative(rates.forward[4], 38.859_999_999_999_99, 3.0e-15);
        assert_relative(rates.reverse[0], 4.421_920_695_640_766e-100, 8.0e-14);
        assert_relative(rates.reverse[7], 1.911_895_177_920_374_4e-30, 8.0e-14);

        let t9 = (0.1_f64 * 0.11).sqrt();
        let interpolated = network.evaluate_rates(t9 / MEV_TO_T9).unwrap();
        let fraction = (t9.ln() - 0.1_f64.ln()) / (0.11_f64.ln() - 0.1_f64.ln());
        assert_relative(fraction, 0.5, 2.0e-15);
        let expected =
            ((1.0 - fraction) * 10.557_685_982_957_498 + fraction * 10.546_498_663_602_277).exp();
        assert_relative(interpolated.forward[0], expected, 3.0e-15);
    }

    #[test]
    fn all_rates_match_the_independent_python_vectorized_formulation_off_grid() {
        // Independently regenerated with Python's json/math modules at
        // T_gamma=0.1 MeV using the exact PRIMAT MeV-to-GK conversion.
        // This is implementation cross-check evidence over shared PRIMAT data,
        // not an independent nuclear-rate measurement or final BBN validation.
        let expected_forward = [
            2.582_281_934_120_794e4,
            4.474_813_794_700_311e2,
            1.701_864_298_923_042_8e7,
            1.449_811_804_718_935_5e7,
            3.238_703_416_877_004e3,
            4.764_821_984_561_948e8,
            2.427_998_025_380_187e2,
            4.849_864_014_277_108e8,
            1.019_300_851_397_224_8e8,
            1.974_722_022_508_199_6e1,
            1.557_291_642_908_954e9,
            3.117_927_012_848_921e5,
        ];
        let expected_reverse = [
            3.321_771_754_317_44e4,
            1.265_752_728_648_292_4e-11,
            1.873_889_245_477_242e-7,
            7.708_364_937_215_848e-11,
            9.411_514_847_493_511e-73,
            1.068_726_916_541_162_3e-67,
            6.487_525_191_594_244e1,
            2.341_851_586_325_714_6e5,
            1.102_988_743_364_280_2e-71,
            3.516_194_973_957_288e4,
            1.128_402_171_955_961_1e2,
            6.779_005_256_557_495e-70,
        ];
        let rates = network().evaluate_rates(0.1).unwrap();
        for reaction in 0..N_BACKBONE_REACTIONS {
            assert_relative(rates.forward[reaction], expected_forward[reaction], 2.0e-14);
            assert_relative(rates.reverse[reaction], expected_reverse[reaction], 8.0e-14);
        }
    }

    #[test]
    fn strict_table_domain_and_raw_inputs_fail_without_clamping() {
        let network = network();
        let (minimum, maximum) = network.temperature_bounds_mev();
        assert!(network.evaluate_rates(minimum).is_ok());
        assert!(network.evaluate_rates(maximum).is_ok());
        assert_eq!(
            network.evaluate_rates(minimum * 0.999).unwrap_err(),
            NetworkError::TemperatureOutsideTable
        );
        assert_eq!(
            network.evaluate_rates(maximum * 1.001).unwrap_err(),
            NetworkError::TemperatureOutsideTable
        );
        assert_eq!(
            network.evaluate_rates(f64::NAN).unwrap_err(),
            NetworkError::NonFiniteTemperature
        );
        let mut invalid = representative_mass_fractions();
        invalid[Species::Tritium.index()] = -1.0e-30;
        assert_eq!(
            network.flux_components(&invalid, 0.1, ETA).unwrap_err(),
            NetworkError::InvalidAbundance {
                species: Species::Tritium.index()
            }
        );
        invalid[Species::Tritium.index()] = f64::NAN;
        assert_eq!(
            network.flux_components(&invalid, 0.1, ETA).unwrap_err(),
            NetworkError::InvalidAbundance {
                species: Species::Tritium.index()
            }
        );
        invalid[Species::Tritium.index()] = f64::INFINITY;
        assert_eq!(
            network.flux_components(&invalid, 0.1, ETA).unwrap_err(),
            NetworkError::InvalidAbundance {
                species: Species::Tritium.index()
            }
        );
        let valid = representative_mass_fractions();
        for invalid_eta in [0.0, -ETA, f64::NAN, f64::INFINITY] {
            assert_eq!(
                network
                    .flux_components(&valid, 0.1, invalid_eta)
                    .unwrap_err(),
                NetworkError::InvalidBaryonToPhotonRatio
            );
        }
    }

    #[test]
    fn explicit_number_density_path_matches_eta_and_stage_extension_is_not_repair() {
        let network = network();
        let temperature = 0.1;
        let state = representative_mass_fractions();
        let number_density = ETA * photon_number_density_per_cm3(temperature).unwrap();
        let eta_rhs = network.rhs(&state, temperature, ETA).unwrap();
        let density_rhs = network
            .rhs_with_baryon_number_density(&state, temperature, number_density)
            .unwrap();
        let eta_jacobian = network.jacobian(&state, temperature, ETA).unwrap();
        let density_jacobian = network
            .jacobian_with_baryon_number_density(&state, temperature, number_density)
            .unwrap();
        let physical_stage_rhs = network
            .stage_rhs_with_baryon_number_density(&state, temperature, number_density)
            .unwrap();
        let physical_stage_jacobian = network
            .stage_jacobian_with_baryon_number_density(&state, temperature, number_density)
            .unwrap();
        assert_eq!(density_rhs, physical_stage_rhs);
        assert_eq!(density_jacobian, physical_stage_jacobian);
        for (left, right) in eta_rhs.iter().zip(density_rhs) {
            assert_relative(right, *left, 4.0e-15);
        }
        for (left, right) in eta_jacobian.iter().zip(density_jacobian) {
            assert_relative(right, *left, 4.0e-15);
        }

        let mut signed_stage = state;
        signed_stage[Species::Lithium7.index()] = -0.02;
        assert_eq!(
            network
                .rhs_with_baryon_number_density(&signed_stage, temperature, number_density)
                .unwrap_err(),
            NetworkError::InvalidAbundance {
                species: Species::Lithium7.index()
            }
        );
        let signed_rhs = network
            .stage_rhs_with_baryon_number_density(&signed_stage, temperature, number_density)
            .unwrap();
        let signed_jacobian = network
            .stage_jacobian_with_baryon_number_density(&signed_stage, temperature, number_density)
            .unwrap();
        let signed_scale: f64 = signed_rhs.iter().map(|value| value.abs()).sum();
        assert!(signed_rhs.iter().sum::<f64>().abs() < 3.0e-15 * signed_scale);
        let signed_charge_derivative: f64 = signed_rhs
            .iter()
            .enumerate()
            .map(|(species, value)| CHARGE_NUMBERS[species] / MASS_NUMBERS[species] * value)
            .sum();
        assert!(signed_charge_derivative.abs() < 3.0e-15 * signed_scale);
        for column in 0..N_SPECIES {
            let step = 1.0e-2 * signed_stage[column].abs().max(1.0e-3);
            let mut plus = signed_stage;
            let mut minus = signed_stage;
            plus[column] += step;
            minus[column] -= step;
            let rhs_plus = network
                .stage_rhs_with_baryon_number_density(&plus, temperature, number_density)
                .unwrap();
            let rhs_minus = network
                .stage_rhs_with_baryon_number_density(&minus, temperature, number_density)
                .unwrap();
            for row in 0..N_SPECIES {
                let finite_difference = (rhs_plus[row] - rhs_minus[row]) / (2.0 * step);
                let exact = signed_jacobian[row * N_SPECIES + column];
                let scale = exact.abs().max(finite_difference.abs()).max(1.0e-20);
                let cancellation_bound =
                    128.0 * f64::EPSILON * rhs_plus[row].abs().max(rhs_minus[row].abs()) / step;
                assert!(
                    (exact - finite_difference).abs() < 3.0e-10 * scale + cancellation_bound,
                    "signed row={row}, column={column}, exact={exact:.17e}, fd={finite_difference:.17e}"
                );
            }
        }
        assert_eq!(signed_stage[Species::Lithium7.index()], -0.02);
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut nonfinite = signed_stage;
            nonfinite[Species::Lithium7.index()] = invalid;
            assert_eq!(
                network
                    .stage_rhs_with_baryon_number_density(&nonfinite, temperature, number_density)
                    .unwrap_err(),
                NetworkError::InvalidAbundance {
                    species: Species::Lithium7.index()
                }
            );
        }
    }

    #[test]
    fn density_powers_and_identical_particle_factors_are_exact() {
        let network = network();
        let state = representative_mass_fractions();
        let temperature = 0.1;
        let rates = network.evaluate_rates(temperature).unwrap();
        let flux = network.flux_components(&state, temperature, ETA).unwrap();
        let rho = baryon_density_factor(temperature, ETA).unwrap();
        let y = molar_abundances(&state);
        let expected_forward_symmetry =
            [1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let expected_reverse_density_power = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1];
        let expected_reverse_symmetry =
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5];
        for reaction in 0..N_BACKBONE_REACTIONS {
            assert_eq!(particle_count(&REACTIONS[reaction].reactants), 2);
            assert_eq!(
                symmetry_factor(&REACTIONS[reaction].reactants),
                expected_forward_symmetry[reaction]
            );
            assert_eq!(
                particle_count(&REACTIONS[reaction].products) - 1,
                expected_reverse_density_power[reaction]
            );
            assert_eq!(
                symmetry_factor(&REACTIONS[reaction].products),
                expected_reverse_symmetry[reaction]
            );
        }
        assert_relative(
            flux.forward[0],
            rho * rates.forward[0] * y[0] * y[1],
            2.0e-15,
        );
        assert_relative(flux.reverse[0], rates.reverse[0] * y[2], 2.0e-15);
        assert_relative(
            flux.forward[2],
            0.5 * rho * rates.forward[2] * y[2].powi(2),
            2.0e-15,
        );
        assert_relative(
            flux.reverse[7],
            rho * rates.reverse[7] * y[1] * y[3],
            2.0e-15,
        );
        assert_relative(
            flux.reverse[11],
            0.5 * rho * rates.reverse[11] * y[5].powi(2),
            2.0e-15,
        );
    }

    #[test]
    fn selected_31_multi_particle_density_and_symmetry_contracts_are_exact() {
        let network = selected_31_network();
        for (reaction, spec) in REACTIONS.iter().enumerate() {
            assert_eq!(particle_count(&spec.reactants), 2, "reaction={reaction}");
            let expected_forward = if matches!(reaction, 2 | 3 | 28 | 29) {
                0.5
            } else {
                1.0
            };
            let expected_reverse = if reaction == 27 {
                0.25
            } else if matches!(reaction, 11 | 12 | 13 | 14 | 18 | 19 | 24 | 25 | 29 | 30) {
                0.5
            } else {
                1.0
            };
            assert_eq!(symmetry_factor(&spec.reactants), expected_forward);
            assert_eq!(symmetry_factor(&spec.products), expected_reverse);
        }
        assert_eq!(particle_count(&REACTIONS[14].products) - 1, 2);
        assert_eq!(particle_count(&REACTIONS[27].products) - 1, 3);
        assert_eq!(particle_count(&REACTIONS[29].products) - 1, 2);

        let fluxes = network
            .flux_components(&representative_mass_fractions(), 0.1, ETA)
            .unwrap();
        assert_relative(fluxes.forward[14], 3.412_648_859_165_967e-2, 3.0e-14);
        assert_relative(fluxes.reverse[15], 1.028_403_178_515_957_8, 8.0e-14);
        assert_relative(fluxes.forward[27], 8.237_660_893_902_587e-5, 3.0e-14);
        assert_relative(fluxes.forward[29], 1.016_630_661_681_259e-3, 3.0e-14);
        assert_relative(fluxes.forward[30], 5.301_611_508_857_005e-2, 3.0e-14);
    }

    #[test]
    fn selected_31_rhs_and_jacobian_match_independent_formulation() {
        let network = selected_31_network();
        let state = representative_mass_fractions();
        let expected = [
            1.292_976_508_983_059_4e3,
            1.376_969_722_871_377_1e3,
            -2.682_560_857_887_656e3,
            4.560_969_053_622_922e1,
            2.985_810_362_797_292_3e2,
            5.570_034_075_200_626e2,
            -6.872_723_140_535_299,
            1.482_560_927_287_277_4e2,
            -1.029_962_877_890_995e3,
        ];
        let derivative = network.rhs(&state, 0.1, ETA).unwrap();
        for species in 0..N_SPECIES {
            assert_relative(derivative[species], expected[species], 5.0e-13);
        }
        let scale: f64 = derivative.iter().map(|value| value.abs()).sum();
        assert!(derivative.iter().sum::<f64>().abs() < 3.0e-15 * scale);
        let charge: f64 = derivative
            .iter()
            .enumerate()
            .map(|(species, value)| CHARGE_NUMBERS[species] / MASS_NUMBERS[species] * value)
            .sum();
        assert!(charge.abs() < 3.0e-15 * scale);

        let analytic = network.jacobian(&state, 0.1, ETA).unwrap();
        for column in 0..N_SPECIES {
            // The largest selected reaction monomial is quartic only in the
            // reverse direction; a smaller centered step limits its cubic
            // truncation while the cancellation term remains explicit.
            let step = 1.0e-4 * state[column];
            let mut plus = state;
            let mut minus = state;
            plus[column] += step;
            minus[column] -= step;
            let rhs_plus = network.rhs(&plus, 0.1, ETA).unwrap();
            let rhs_minus = network.rhs(&minus, 0.1, ETA).unwrap();
            for row in 0..N_SPECIES {
                let finite = (rhs_plus[row] - rhs_minus[row]) / (2.0 * step);
                let exact = analytic[row * N_SPECIES + column];
                let local_scale = exact.abs().max(finite.abs()).max(1.0e-20);
                let cancellation =
                    256.0 * f64::EPSILON * rhs_plus[row].abs().max(rhs_minus[row].abs()) / step;
                assert!(
                    (exact - finite).abs() < 2.0e-8 * local_scale + cancellation,
                    "row={row}, column={column}, exact={exact:.17e}, fd={finite:.17e}"
                );
            }
        }
    }

    #[test]
    fn neutron_proton_only_handoff_has_finite_boundary_jacobian_and_expected_signs() {
        let network = network();
        let mut state = [0.0; N_SPECIES];
        state[Species::Neutron.index()] = 0.13;
        state[Species::Proton.index()] = 0.87;
        let flux = network.flux_components(&state, 0.08, ETA).unwrap();
        assert!(flux.forward[0] > 0.0);
        assert!(flux.forward[1..].iter().all(|value| *value == 0.0));
        assert!(flux.reverse.iter().all(|value| *value == 0.0));
        let derivative = network.rhs(&state, 0.08, ETA).unwrap();
        assert!(derivative[Species::Neutron.index()] < 0.0);
        assert!(derivative[Species::Proton.index()] < 0.0);
        assert!(derivative[Species::Deuterium.index()] > 0.0);
        assert!(
            derivative[Species::Tritium.index()..]
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            network
                .jacobian(&state, 0.08, ETA)
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn reaction_sum_is_permutation_invariant_and_rejects_bad_orders() {
        let network = network();
        let state = representative_mass_fractions();
        let canonical = network.rhs(&state, 0.1, ETA).unwrap();
        let reverse_order: Vec<usize> = (0..network.reaction_count()).rev().collect();
        let permuted = network
            .rhs_in_order(&state, 0.1, ETA, &reverse_order)
            .unwrap();
        for species in 0..N_SPECIES {
            assert_relative(permuted[species], canonical[species], 4.0e-15);
        }
        let mut duplicate = reverse_order;
        duplicate[0] = duplicate[1];
        assert_eq!(
            network
                .rhs_in_order(&state, 0.1, ETA, &duplicate)
                .unwrap_err(),
            NetworkError::InvalidReactionOrder
        );
    }

    #[test]
    fn analytic_abundance_jacobian_matches_centered_finite_difference() {
        let network = network();
        let state = representative_mass_fractions();
        let analytic = network.jacobian(&state, 0.1, ETA).unwrap();
        for column in 0..N_SPECIES {
            // Every abundance monomial has degree at most two, so a centered
            // difference has no truncation term here.  A deliberately broad
            // step suppresses cancellation between large opposing fluxes.
            let step = 1.0e-2 * state[column];
            let mut plus = state;
            let mut minus = state;
            plus[column] += step;
            minus[column] -= step;
            let rhs_plus = network.rhs(&plus, 0.1, ETA).unwrap();
            let rhs_minus = network.rhs(&minus, 0.1, ETA).unwrap();
            for row in 0..N_SPECIES {
                let finite_difference = (rhs_plus[row] - rhs_minus[row]) / (2.0 * step);
                let scale = analytic[row * N_SPECIES + column]
                    .abs()
                    .max(finite_difference.abs())
                    .max(1.0e-20);
                let cancellation_bound =
                    128.0 * f64::EPSILON * rhs_plus[row].abs().max(rhs_minus[row].abs()) / step;
                assert!(
                    (analytic[row * N_SPECIES + column] - finite_difference).abs()
                        < 3.0e-10 * scale + cancellation_bound,
                    "row={row}, column={column}, analytic={:.17e}, fd={finite_difference:.17e}",
                    analytic[row * N_SPECIES + column]
                );
            }
        }
    }

    #[test]
    fn radiative_reaction_can_be_constructed_at_exact_directional_balance() {
        let network = network();
        let temperature = 0.1;
        let rates = network.evaluate_rates(temperature).unwrap();
        let rho = baryon_density_factor(temperature, ETA).unwrap();
        let mut state = [0.0; N_SPECIES];
        state[Species::Neutron.index()] = 0.2;
        state[Species::Proton.index()] = 0.3;
        let y_d = rho * rates.forward[0] * 0.2 * 0.3 / rates.reverse[0];
        state[Species::Deuterium.index()] = MASS_NUMBERS[Species::Deuterium.index()] * y_d;
        let flux = network.flux_components(&state, temperature, ETA).unwrap();
        assert_relative(flux.forward[0], flux.reverse[0], 3.0e-15);
    }

    #[test]
    fn closed_network_rhs_preserves_baryon_and_charge_exactly() {
        let network = network();
        let state = representative_mass_fractions();
        let derivative = network.rhs(&state, 0.1, ETA).unwrap();
        let baryon_mass_fraction_derivative: f64 = derivative.iter().sum();
        let charge_per_baryon_derivative: f64 = derivative
            .iter()
            .enumerate()
            .map(|(species, value)| CHARGE_NUMBERS[species] / MASS_NUMBERS[species] * value)
            .sum();
        let scale: f64 = derivative.iter().map(|value| value.abs()).sum();
        assert!(baryon_mass_fraction_derivative.abs() < 3.0e-15 * scale);
        assert!(charge_per_baryon_derivative.abs() < 3.0e-15 * scale);
    }

    #[test]
    fn full_rhs_matches_independent_python_species_remap_anchor() {
        // A direct Python/NumPy formulation evaluated the same physical X
        // state with this module's explicit zeta(3), hbar*c, and N_A constants,
        // then remapped once from the legacy species order. Shared table data
        // means this is a formulation/remap check, not rate validation.
        let expected = [
            1.293_008_636_772_212_4e3,
            1.377_049_898_639_464_4e3,
            -2.684_449_999_475_787_5e3,
            4.563_662_662_799_805e1,
            2.982_425_755_740_411_6e2,
            5.509_585_962_250_975e2,
            0.0,
            1.486_281_897_909_179e2,
            -1.029_074_524_153_944_1e3,
        ];
        let derivative = network()
            .rhs(&representative_mass_fractions(), 0.1, ETA)
            .unwrap();
        for species in 0..N_SPECIES {
            if expected[species] == 0.0 {
                assert_eq!(derivative[species], 0.0);
            } else {
                assert_relative(derivative[species], expected[species], 3.0e-14);
            }
        }
    }

    #[derive(Clone, Copy)]
    struct FrozenReverseRadiativeSystem {
        first_rate: f64,
        second_rate: f64,
    }

    impl FrozenReverseRadiativeSystem {
        fn rates(self) -> NetworkRates {
            let mut rates = NetworkRates {
                forward: [0.0; N_REACTIONS],
                reverse: [0.0; N_REACTIONS],
            };
            rates.reverse[0] = self.first_rate;
            rates.reverse[1] = self.second_rate;
            rates
        }
    }

    impl OdeSystem for FrozenReverseRadiativeSystem {
        fn dimension(&self) -> usize {
            N_SPECIES
        }

        fn rhs(&self, _time: f64, state: &[f64], output: &mut [f64]) {
            let Ok(state) = <&[f64; N_SPECIES]>::try_from(state) else {
                output.fill(f64::NAN);
                return;
            };
            let order: Vec<usize> = (0..N_BACKBONE_REACTIONS).collect();
            let result = validate_mass_fractions(state)
                .and_then(|()| directional_fluxes(state, 1.0, &self.rates(), N_BACKBONE_REACTIONS))
                .and_then(|fluxes| mass_fraction_derivative(&fluxes, &order));
            match result {
                Ok(derivative) => output.copy_from_slice(&derivative),
                Err(_) => output.fill(f64::NAN),
            }
        }

        fn jacobian(&self, _time: f64, state: &[f64], output: &mut [f64]) {
            let Ok(state) = <&[f64; N_SPECIES]>::try_from(state) else {
                output.fill(f64::NAN);
                return;
            };
            match abundance_jacobian_from_rates(state, 1.0, &self.rates(), N_BACKBONE_REACTIONS) {
                Ok(jacobian) => output.copy_from_slice(&jacobian),
                Err(_) => output.fill(f64::NAN),
            }
        }

        fn dfdt(&self, _time: f64, _state: &[f64], output: &mut [f64]) {
            output.fill(0.0);
        }
    }

    fn manufactured_config() -> OdeConfig {
        OdeConfig {
            rtol: 1.0e-10,
            atol: vec![1.0e-12; N_SPECIES],
            h_init: 1.0e-3,
            h_min: 1.0e-12,
            h_max: 0.05,
            max_attempts: 100_000,
        }
    }

    #[test]
    fn manufactured_single_reverse_radiative_reaction_matches_exact_solution() {
        let rate = 3.25;
        let system = FrozenReverseRadiativeSystem {
            first_rate: rate,
            second_rate: 0.0,
        };
        let mut initial = [0.0; N_SPECIES];
        initial[Species::Deuterium.index()] = 0.4;
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let result = solve(
                kind,
                &system,
                (0.0, 1.2),
                &initial,
                &manufactured_config(),
                None,
            );
            assert_eq!(result.failure, None, "{kind:?}: {result:?}");
            let expected_d = initial[Species::Deuterium.index()] * (-rate * 1.2).exp();
            assert_relative(result.y[Species::Deuterium.index()], expected_d, 2.0e-8);
            assert_relative(
                result.y[Species::Neutron.index()],
                0.5 * (initial[Species::Deuterium.index()] - expected_d),
                2.0e-8,
            );
            assert_relative(
                result.y[Species::Proton.index()],
                0.5 * (initial[Species::Deuterium.index()] - expected_d),
                2.0e-8,
            );
        }
    }

    #[test]
    fn manufactured_two_reaction_linear_chain_matches_exact_solution() {
        let first_rate = 2.0;
        let second_rate = 0.7;
        let system = FrozenReverseRadiativeSystem {
            first_rate,
            second_rate,
        };
        let mut initial = [0.0; N_SPECIES];
        initial[Species::Deuterium.index()] = 0.15;
        initial[Species::Helium3.index()] = 0.2;
        let terminal_time = 1.4;
        let expected_he3 = initial[Species::Helium3.index()] * (-second_rate * terminal_time).exp();
        let expected_d = initial[Species::Deuterium.index()] * (-first_rate * terminal_time).exp()
            + (2.0 / 3.0)
                * second_rate
                * initial[Species::Helium3.index()]
                * ((-second_rate * terminal_time).exp() - (-first_rate * terminal_time).exp())
                / (first_rate - second_rate);
        for kind in [SolverKind::Bdf, SolverKind::Rodas5P] {
            let result = solve(
                kind,
                &system,
                (0.0, terminal_time),
                &initial,
                &manufactured_config(),
                None,
            );
            assert_eq!(result.failure, None, "{kind:?}: {result:?}");
            assert_relative(result.y[Species::Helium3.index()], expected_he3, 2.0e-8);
            assert_relative(result.y[Species::Deuterium.index()], expected_d, 2.0e-8);
            let initial_baryon: f64 = initial.iter().sum();
            let terminal_baryon: f64 = result.y.iter().sum();
            assert_relative(terminal_baryon, initial_baryon, 2.0e-11);
        }
    }
}
