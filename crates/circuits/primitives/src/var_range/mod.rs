//! A chip which uses preprocessed trace to provide a lookup table for range checking
//! a variable `x` has `b` bits where `b` can be any integer in `[0, range_max_bits]`.
//! In other words, the same chip can be used to range check for different bit sizes.
//! We define `0` to have `0` bits.

use core::mem::size_of;
use std::{
    borrow::{Borrow, BorrowMut},
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir, PairBuilder},
    p3_field::{Field, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::types::AirProofInput,
    rap::{get_air_name, BaseAirWithPublicValues, ColumnsAir, PartitionedBaseAir},
    AirRef, Chip, ChipUsageGetter, Stateful,
};
use struct_reflection::{StructReflection, StructReflectionHelper};
use tracing::instrument;

mod bus;
#[cfg(test)]
pub mod tests;

pub use bus::*;

#[derive(Default, AlignedBorrow, Copy, Clone, StructReflection)]
#[repr(C)]
pub struct VariableRangeCols<T> {
    pub mult: T,
}

#[derive(Default, AlignedBorrow, Copy, Clone, StructReflection)]
#[repr(C)]
pub struct VariableRangePreprocessedCols<T> {
    pub value: T,
    pub max_bits: T,
}

pub const NUM_VARIABLE_RANGE_COLS: usize = size_of::<VariableRangeCols<u8>>();
pub const NUM_VARIABLE_RANGE_PREPROCESSED_COLS: usize =
    size_of::<VariableRangePreprocessedCols<u8>>();

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct VariableRangeCheckerAir {
    pub bus: VariableRangeCheckerBus,
}

impl VariableRangeCheckerAir {
    pub fn range_max_bits(&self) -> usize {
        self.bus.range_max_bits
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for VariableRangeCheckerAir {}
impl<F: Field> PartitionedBaseAir<F> for VariableRangeCheckerAir {}
impl<F: Field> BaseAir<F> for VariableRangeCheckerAir {
    fn width(&self) -> usize {
        NUM_VARIABLE_RANGE_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let rows: Vec<F> = [F::ZERO; NUM_VARIABLE_RANGE_PREPROCESSED_COLS]
            .into_iter()
            .chain((0..=self.range_max_bits()).flat_map(|bits| {
                (0..(1 << bits)).flat_map(move |value| {
                    [F::from_canonical_u32(value), F::from_canonical_usize(bits)].into_iter()
                })
            }))
            .collect();
        Some(RowMajorMatrix::new(
            rows,
            NUM_VARIABLE_RANGE_PREPROCESSED_COLS,
        ))
    }
}

impl<F: Field> ColumnsAir<F> for VariableRangeCheckerAir {
    fn columns(&self) -> Option<Vec<String>> {
        VariableRangeCols::<F>::struct_reflection()
    }
}

impl<AB: InteractionBuilder + PairBuilder> Air<AB> for VariableRangeCheckerAir {
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_local = preprocessed.row_slice(0);
        let prep_local: &VariableRangePreprocessedCols<AB::Var> = (*prep_local).borrow();
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &VariableRangeCols<AB::Var> = (*local).borrow();
        // Omit creating separate bridge.rs file for brevity
        self.bus
            .receive(prep_local.value, prep_local.max_bits)
            .eval(builder, local.mult);
    }
}

pub struct VariableRangeCheckerChip {
    pub air: VariableRangeCheckerAir,
    pub count: Vec<AtomicU32>,
}

#[derive(Clone)]
pub struct SharedVariableRangeCheckerChip(pub Arc<VariableRangeCheckerChip>);

impl VariableRangeCheckerChip {
    pub fn new(bus: VariableRangeCheckerBus) -> Self {
        let num_rows = (1 << (bus.range_max_bits + 1)) as usize;
        let count = (0..num_rows).map(|_| AtomicU32::new(0)).collect();
        Self {
            air: VariableRangeCheckerAir::new(bus),
            count,
        }
    }

    pub fn bus(&self) -> VariableRangeCheckerBus {
        self.air.bus
    }

    pub fn range_max_bits(&self) -> usize {
        self.air.range_max_bits()
    }

    pub fn air_width(&self) -> usize {
        NUM_VARIABLE_RANGE_COLS
    }

    #[instrument(
        name = "VariableRangeCheckerChip::add_count",
        skip(self),
        level = "trace"
    )]
    pub fn add_count(&self, value: u32, max_bits: usize) {
        // index is 2^max_bits + value - 1 + 1 for the extra [0, 0] row
        // if each [value, max_bits] is valid, the sends multiset will be exactly the receives multiset
        let idx = (1 << max_bits) + (value as usize);
        assert!(
            idx < self.count.len(),
            "range exceeded: {} >= {}",
            idx,
            self.count.len()
        );
        let val_atomic = &self.count[idx];
        val_atomic.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        println!("multiplicity for {value} {max_bits} is {}", val_atomic.load(Ordering::Relaxed));
    }

    pub fn remove_count(&self, value: u32, max_bits: usize) {
        // index is 2^max_bits + value - 1 + 1 for the extra [0, 0] row
        // if each [value, max_bits] is valid, the sends multiset will be exactly the receives multiset
        let idx = (1 << max_bits) + (value as usize);
        assert!(
            idx < self.count.len(),
            "range exceeded: {} >= {}",
            idx,
            self.count.len()
        );
        let val_atomic = &self.count[idx];
        val_atomic.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        println!("multiplicity for {value} {max_bits} is {}", val_atomic.load(Ordering::Relaxed));
    }

    pub fn clear(&self) {
        for i in 0..self.count.len() {
            self.count[i].store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub fn generate_trace<F: Field>(&self) -> RowMajorMatrix<F> {
        let mut rows = F::zero_vec(self.count.len() * NUM_VARIABLE_RANGE_COLS);
        for (n, row) in rows.chunks_mut(NUM_VARIABLE_RANGE_COLS).enumerate() {
            let cols: &mut VariableRangeCols<F> = row.borrow_mut();
            cols.mult =
                F::from_canonical_u32(self.count[n].load(std::sync::atomic::Ordering::SeqCst));
        }
        RowMajorMatrix::new(rows, NUM_VARIABLE_RANGE_COLS)
    }

    /// Range checks that `value` is `bits` bits by decomposing into `limbs` where all but
    /// last limb is `range_max_bits` bits. Assumes there are enough limbs.
    pub(crate) fn decompose<F: Field>(&self, mut value: u32, bits: usize, limbs: &mut [F]) {
        println!("When decomposing, 251 14 has multiplicity {}", &self.count[(1 << 14) + (251 as usize)].load(Ordering::Relaxed));
        debug_assert!(
            limbs.len() <= bits.div_ceil(self.range_max_bits()),
            "Not enough limbs: len {}",
            limbs.len()
        );
        let mask = (1 << self.range_max_bits()) - 1;
        let mut bits_remaining = bits;
        for limb in limbs.iter_mut() {
            let limb_u32 = value & mask;
            *limb = F::from_canonical_u32(limb_u32);
            self.add_count(limb_u32, bits_remaining.min(self.range_max_bits()));

            value >>= self.range_max_bits();
            bits_remaining = bits_remaining.saturating_sub(self.range_max_bits());
        }
        debug_assert_eq!(value, 0);
        debug_assert_eq!(bits_remaining, 0);
    }
}

impl SharedVariableRangeCheckerChip {
    pub fn new(bus: VariableRangeCheckerBus) -> Self {
        Self(Arc::new(VariableRangeCheckerChip::new(bus)))
    }

    pub fn bus(&self) -> VariableRangeCheckerBus {
        self.0.bus()
    }

    pub fn range_max_bits(&self) -> usize {
        self.0.range_max_bits()
    }

    pub fn air_width(&self) -> usize {
        self.0.air_width()
    }

    pub fn add_count(&self, value: u32, max_bits: usize) {
        let id = Arc::as_ptr(&self.0) as usize;
        println!("add to {:#x}", id);
        self.0.add_count(value, max_bits)
    }

    pub fn remove_count(&self, value: u32, max_bits: usize) {
        let id = Arc::as_ptr(&self.0) as usize;
        println!("remove from {:#x}", id);
        self.0.remove_count(value, max_bits)
    }

    pub fn clear(&self) {
        let id = Arc::as_ptr(&self.0) as usize;
        println!("clear {:#x}", id);
        self.0.clear()
    }

    pub fn generate_trace<F: Field>(&self) -> RowMajorMatrix<F> {
        self.0.generate_trace()
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for VariableRangeCheckerChip
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let trace = self.generate_trace::<Val<SC>>();
        AirProofInput::simple_no_pis(trace)
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for SharedVariableRangeCheckerChip
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> AirRef<SC> {
        self.0.air()
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        self.0.generate_air_proof_input()
    }
}

impl ChipUsageGetter for VariableRangeCheckerChip {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn constant_trace_height(&self) -> Option<usize> {
        Some(self.count.len())
    }
    fn current_trace_height(&self) -> usize {
        self.count.len()
    }
    fn trace_width(&self) -> usize {
        NUM_VARIABLE_RANGE_COLS
    }
}

impl ChipUsageGetter for SharedVariableRangeCheckerChip {
    fn air_name(&self) -> String {
        self.0.air_name()
    }

    fn current_trace_height(&self) -> usize {
        self.0.current_trace_height()
    }

    fn trace_width(&self) -> usize {
        self.0.trace_width()
    }
}

impl Stateful<Vec<u8>> for SharedVariableRangeCheckerChip {
    fn load_state(&mut self, state: Vec<u8>) {
        let count_vals: Vec<u32> = bitcode::deserialize(&state).unwrap();
        for (x, val) in self.0.count.iter().zip(count_vals) {
            x.store(val, Ordering::Relaxed);
        }
    }

    fn store_state(&self) -> Vec<u8> {
        bitcode::serialize(&self.0.count).unwrap()
    }
}

impl AsRef<VariableRangeCheckerChip> for SharedVariableRangeCheckerChip {
    fn as_ref(&self) -> &VariableRangeCheckerChip {
        &self.0
    }
}
