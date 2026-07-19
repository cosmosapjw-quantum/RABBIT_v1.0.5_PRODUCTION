#![allow(dead_code)]

use crate::electron_catalog::RateMeV;
use crate::electron_event::pauli_gradient;
use crate::electron_supplied::SuppliedElectronEvents;

const DIM: &str = "explicit response dimension overflow";
const ALLOC: &str = "explicit response allocation failed";
const BAD_DIRECTION: &str = "explicit response direction has invalid dimension or value";
pub(crate) struct AppliedExplicitSixResponseMeV {
    pub(crate) channel_rows: Vec<RateMeV>,
    pub(crate) total_rows: Vec<RateMeV>,
}
pub(crate) struct GeneralizedExplicitSixResponseMeV {
    pub(crate) nq: usize,
    pub(crate) channel_row_major: Vec<RateMeV>,
    pub(crate) total_row_major: Vec<RateMeV>,
}
impl GeneralizedExplicitSixResponseMeV {
    pub(crate) fn from_leg_weights(
        stream: &SuppliedElectronEvents,
        f6: &[f64],
        leg_weights: &[[RateMeV; 2]],
    ) -> Result<Self, &'static str> {
        if leg_weights.len() != stream.events().len() {
            return Err("explicit generalized response leg weights have invalid length");
        }
        let (nq, channel_row_major, total_row_major) =
            assemble_response(stream, f6, Some(leg_weights))?;
        Ok(Self {
            nq,
            channel_row_major,
            total_row_major,
        })
    }
}
pub(crate) struct ExplicitSixJ0 {
    pub(crate) nq: usize,
    pub(crate) channel_row_major: Vec<RateMeV>,
    pub(crate) total_row_major: Vec<RateMeV>,
}
fn dimensions(nq: usize) -> Result<(usize, usize, usize, usize), &'static str> {
    let input = nq.checked_mul(6).filter(|value| *value > 0).ok_or(DIM)?;
    let channel_rows = nq.checked_mul(18).ok_or(DIM)?;
    let channel_len = channel_rows.checked_mul(input).ok_or(DIM)?;
    let total_len = input.checked_mul(input).ok_or(DIM)?;
    Ok((input, channel_rows, channel_len, total_len))
}
fn zero_rates(len: usize) -> Result<Vec<RateMeV>, &'static str> {
    let zero = RateMeV::new(0.0)?;
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| ALLOC)?;
    for _ in 0..len {
        values.push(zero);
    }
    Ok(values)
}
fn cell(row: usize, column: usize, width: usize, len: usize) -> Result<usize, &'static str> {
    let index = row
        .checked_mul(width)
        .and_then(|offset| offset.checked_add(column))
        .ok_or(DIM)?;
    (row < len / width && column < width && index < len)
        .then_some(index)
        .ok_or(DIM)
}
fn add(matrix: &mut [RateMeV], index: usize, value: RateMeV) -> Result<(), &'static str> {
    let slot = matrix.get_mut(index).ok_or(DIM)?;
    *slot = RateMeV::new(slot.value() + value.value())?;
    Ok(())
}
fn apply_rows(
    matrix: &[RateMeV],
    rows: usize,
    input: usize,
    direction: &[f64],
    output: &mut [RateMeV],
) -> Result<(), &'static str> {
    for row in 0..rows {
        let mut sum = RateMeV::new(0.0)?;
        for (column, delta) in direction.iter().enumerate() {
            let index = cell(row, column, input, matrix.len())?;
            let product = RateMeV::new(matrix[index].value() * delta)?;
            sum = RateMeV::new(sum.value() + product.value())?;
        }
        *output.get_mut(row).ok_or(DIM)? = sum;
    }
    Ok(())
}
impl ExplicitSixJ0 {
    pub(crate) fn apply(
        &self,
        direction6: &[f64],
    ) -> Result<AppliedExplicitSixResponseMeV, &'static str> {
        let (input, channel_rows, channel_len, total_len) = dimensions(self.nq)?;
        if self.channel_row_major.len() != channel_len || self.total_row_major.len() != total_len {
            return Err(DIM);
        }
        if direction6.len() != input || !direction6.iter().all(|value| value.is_finite()) {
            return Err(BAD_DIRECTION);
        }
        let mut channel_rows = zero_rates(channel_rows)?;
        let mut total_rows = zero_rates(input)?;
        apply_rows(
            &self.channel_row_major,
            channel_rows.len(),
            input,
            direction6,
            &mut channel_rows,
        )?;
        apply_rows(
            &self.total_row_major,
            input,
            input,
            direction6,
            &mut total_rows,
        )?;
        Ok(AppliedExplicitSixResponseMeV {
            channel_rows,
            total_rows,
        })
    }
}
pub(crate) fn explicit_six_action_j0(
    stream: &SuppliedElectronEvents,
    f6: &[f64],
) -> Result<ExplicitSixJ0, &'static str> {
    let (nq, channel_row_major, total_row_major) = assemble_response(stream, f6, None)?;
    Ok(ExplicitSixJ0 {
        nq,
        channel_row_major,
        total_row_major,
    })
}
fn assemble_response(
    stream: &SuppliedElectronEvents,
    f6: &[f64],
    leg_weights: Option<&[[RateMeV; 2]]>,
) -> Result<(usize, Vec<RateMeV>, Vec<RateMeV>), &'static str> {
    let nq = stream.nq();
    let (input, _, channel_len, total_len) = dimensions(nq)?;
    let mut channel_row_major = zero_rates(channel_len)?;
    let mut total_row_major = zero_rates(total_len)?;
    for (event_index, item) in stream.validated_contractions(f6)?.enumerate() {
        let item = item?;
        let gradient = pauli_gradient(item.occupancies)?;
        let row = item
            .process_slot
            .checked_mul(nq)
            .and_then(|offset| offset.checked_add(item.dynamic_legs.target.explicit_node.node))
            .ok_or(DIM)?;
        let weights = match leg_weights {
            None => [item.scalar_weight, item.scalar_weight],
            Some(values) => *values.get(event_index).ok_or(DIM)?,
        };
        for (leg, weight) in [
            (&item.dynamic_legs.target, weights[0]),
            (&item.dynamic_legs.coupled, weights[1]),
        ] {
            let column = leg.explicit_node.flat_index;
            let index = cell(row, column, input, channel_len)?;
            let coefficient = *gradient.get(leg.pauli_leg_zero_based).ok_or(DIM)?;
            let derivative = RateMeV::new(weight.value() * coefficient)?;
            add(&mut channel_row_major, index, derivative)?;
        }
    }
    for row in 0..input {
        let state = row.checked_div(nq).ok_or(DIM)?;
        let node = row.checked_rem(nq).ok_or(DIM)?;
        let first_row = state
            .checked_mul(3)
            .and_then(|process| process.checked_mul(nq))
            .and_then(|offset| offset.checked_add(node))
            .ok_or(DIM)?;
        let second_row = first_row.checked_add(nq).ok_or(DIM)?;
        let third_row = second_row.checked_add(nq).ok_or(DIM)?;
        for column in 0..input {
            let first = channel_row_major[cell(first_row, column, input, channel_len)?];
            let second = channel_row_major[cell(second_row, column, input, channel_len)?];
            let third = channel_row_major[cell(third_row, column, input, channel_len)?];
            let partial = RateMeV::new(first.value() + second.value())?;
            let index = cell(row, column, input, total_len)?;
            *total_row_major.get_mut(index).ok_or(DIM)? =
                RateMeV::new(partial.value() + third.value())?;
        }
    }
    Ok((nq, channel_row_major, total_row_major))
}
