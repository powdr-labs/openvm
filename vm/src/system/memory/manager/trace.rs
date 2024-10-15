use std::{borrow::BorrowMut, cmp::max};

use afs_primitives::{
    is_less_than::columns::IsLessThanAuxColsMut, utils::next_power_of_two_or_zero,
};
use p3_air::BaseAir;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;

use crate::system::memory::{
    adapter::AccessAdapterCols, manager::memory::AccessAdapterRecordKind, MemoryAddress, MemoryChip,
};

impl<F: PrimeField32> MemoryChip<F> {
    pub fn generate_access_adapter_trace<const N: usize>(&self) -> RowMajorMatrix<F> {
        let air = self.access_adapter_air::<N>();
        let width = BaseAir::<F>::width(&air);

        match self.adapter_records.get(&N) {
            None => RowMajorMatrix::new(vec![], width),
            Some(records) => {
                let height = next_power_of_two_or_zero(records.len());
                let mut values = vec![F::zero(); height * width];

                for (row, record) in values.chunks_mut(width).zip(records) {
                    let row: &mut AccessAdapterCols<F, N> = row.borrow_mut();

                    row.is_valid = F::one();
                    row.values = record.data.clone().try_into().unwrap();
                    row.address = MemoryAddress::new(record.address_space, record.start_index);

                    match record.kind {
                        AccessAdapterRecordKind::Split => {
                            row.left_timestamp = F::from_canonical_u32(record.timestamp);
                            row.right_timestamp = F::from_canonical_u32(record.timestamp);
                            row.is_split = F::one();
                        }
                        AccessAdapterRecordKind::Merge {
                            left_timestamp,
                            right_timestamp,
                        } => {
                            assert_eq!(max(left_timestamp, right_timestamp), record.timestamp);
                            row.left_timestamp = F::from_canonical_u32(left_timestamp);
                            row.right_timestamp = F::from_canonical_u32(right_timestamp);
                            row.is_split = F::zero();
                            row.is_right_larger = F::from_bool(left_timestamp < right_timestamp);
                        }
                    }
                    air.lt_air.generate_trace_row_aux(
                        row.left_timestamp.as_canonical_u32(),
                        row.right_timestamp.as_canonical_u32(),
                        &self.range_checker,
                        &mut IsLessThanAuxColsMut {
                            lower_decomp: &mut row.lt_aux,
                        },
                    );
                }
                RowMajorMatrix::new(values, width)
            }
        }
    }
}
