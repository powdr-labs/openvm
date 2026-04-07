use std::{
    borrow::{Borrow, BorrowMut},
    mem::{align_of, size_of},
};

use openvm_circuit::arch::{
    get_record_from_slice, CustomBorrow, MultiRowLayout, MultiRowMetadata, SizedRecord, TraceFiller,
    VmField,
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, AlignedBytesBorrow,
};
use openvm_stark_backend::p3_matrix::dense::RowMajorMatrix;

use crate::air::{COL_IS_FIRST, COL_IS_LAST, COL_IS_VALID, COL_PC, COL_SECTION_IDX, COL_TIMESTAMP};

// ---- Record types ----

/// Metadata type that does NOT implement Default, to avoid blanket CustomBorrow conflicts.
#[derive(Clone, Copy, Debug)]
pub struct BenchmarkMetadata {
    pub num_rows: usize,
}

impl MultiRowMetadata for BenchmarkMetadata {
    #[inline(always)]
    fn get_num_rows(&self) -> usize {
        self.num_rows
    }
}

pub type BenchmarkLayout = MultiRowLayout<BenchmarkMetadata>;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct BenchmarkRecordHeader {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub num_rows: u32,
}

/// Custom record wrapper (avoids blanket impl conflicts on `&mut T`).
pub struct BenchmarkRecordMut<'a> {
    pub header: &'a mut BenchmarkRecordHeader,
}

impl<'a> CustomBorrow<'a, BenchmarkRecordMut<'a>, BenchmarkLayout> for [u8] {
    fn custom_borrow(&'a mut self, _layout: BenchmarkLayout) -> BenchmarkRecordMut<'a> {
        let (header_buf, _rest) =
            unsafe { self.split_at_mut_unchecked(size_of::<BenchmarkRecordHeader>()) };
        BenchmarkRecordMut {
            header: header_buf.borrow_mut(),
        }
    }

    unsafe fn extract_layout(&self) -> BenchmarkLayout {
        let record: &BenchmarkRecordHeader =
            self[..size_of::<BenchmarkRecordHeader>()].borrow();
        BenchmarkLayout {
            metadata: BenchmarkMetadata {
                num_rows: record.num_rows as usize,
            },
        }
    }
}

impl SizedRecord<BenchmarkLayout> for BenchmarkRecordMut<'_> {
    fn size(_layout: &BenchmarkLayout) -> usize {
        size_of::<BenchmarkRecordHeader>()
    }

    fn alignment(_layout: &BenchmarkLayout) -> usize {
        align_of::<BenchmarkRecordHeader>()
    }
}

// ---- Filler ----

#[derive(Clone, derive_new::new)]
pub struct BenchmarkFiller {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub num_columns: usize,
    pub num_interactions: usize,
    pub rows_per_invocation: usize,
}

impl<F> TraceFiller<F> for BenchmarkFiller
where
    F: VmField,
{
    fn fill_trace(
        &self,
        _mem_helper: &openvm_circuit::system::memory::MemoryAuxColsFactory<F>,
        trace_matrix: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }

        let width = trace_matrix.width;
        let mut trace = &mut trace_matrix.values[..width * rows_used];

        while !trace.is_empty() {
            // Peek at the record header (uses blanket CustomBorrow<&mut T, ()>)
            let header: &BenchmarkRecordHeader =
                unsafe { get_record_from_slice(&mut trace, ()) };
            let num_rows = header.num_rows as usize;
            let from_pc = header.from_pc;
            let from_timestamp = header.from_timestamp;

            // Split off this section's rows
            let (section, rest) = trace.split_at_mut(width * num_rows);
            trace = rest;

            // Fill each row of the section
            for (row_idx, row) in section.chunks_exact_mut(width).enumerate() {
                row[COL_IS_VALID] = F::ONE;
                row[COL_IS_FIRST] = F::from_bool(row_idx == 0);
                row[COL_IS_LAST] = F::from_bool(row_idx + 1 == num_rows);
                row[COL_SECTION_IDX] = F::from_usize(row_idx);
                row[COL_PC] = F::from_u32(from_pc);
                row[COL_TIMESTAMP] = F::from_u32(from_timestamp);
                // All payload columns remain zero (default-initialized)

                // Track bitwise lookup multiplicities for range check interactions
                for _ in 0..self.num_interactions {
                    self.bitwise_lookup_chip.request_range(0, 0);
                }
            }
        }
    }
}
