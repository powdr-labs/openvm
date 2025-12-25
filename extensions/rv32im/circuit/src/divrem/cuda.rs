use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupChipGPU, range_tuple::RangeTupleCheckerChipGPU,
    var_range::VariableRangeCheckerChipGPU,
};
use openvm_cuda_backend::{
    base::DeviceMatrix,
    chip::{get_empty_air_proving_ctx, UInt2},
    prover_backend::GpuBackend,
    types::F,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_instructions::riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use openvm_stark_backend::{prover::types::AirProvingContext, ApcTracingContext, Chip};

use crate::{
    adapters::{Rv32MultAdapterCols, Rv32MultAdapterRecord},
    cuda_abi::divrem_cuda::tracegen,
    DivRemCoreCols, DivRemCoreRecord,
};

#[derive(new)]
pub struct Rv32DivRemChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub bitwise_lookup: Arc<BitwiseOperationLookupChipGPU<RV32_CELL_BITS>>,
    pub range_tuple_checker: Arc<RangeTupleCheckerChipGPU<2>>,
    pub pointer_max_bits: usize,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32DivRemChipGpu {
    fn generate_proving_ctx_direct(
        &self,
        arena: DenseRecordArena,
        ctx: Option<&ApcTracingContext>,
    ) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv32MultAdapterRecord,
            DivRemCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = DivRemCoreCols::<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>::width()
            + Rv32MultAdapterCols::<F>::width();
        let padded_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);
        let tuple_checker_sizes = UInt2::new(
            self.range_tuple_checker.sizes[0],
            self.range_tuple_checker.sizes[1],
        );
        let d_records = records.to_device().unwrap();
        let empty = DeviceBuffer::new();

        let owned_trace = ctx
            .is_none()
            .then(|| DeviceMatrix::<F>::with_capacity(padded_height, trace_width));

        unsafe {
            tracegen(
                ctx.map_or_else(|| owned_trace.as_ref().unwrap().buffer(), |c| c.d_trace),
                padded_height,
                trace_width,
                &d_records,
                &self.range_checker.count,
                &self.bitwise_lookup.count,
                RV32_CELL_BITS as u32,
                &self.range_tuple_checker.count,
                tuple_checker_sizes,
                self.timestamp_max_bits as u32,
                ctx.map_or(&empty, |c| c.d_subs),
                ctx.map_or(&empty, |c| c.d_opt_widths),
                ctx.map_or(&empty, |c| c.d_post_opt_offsets),
                ctx.map_or(0, |c| c.apc_height),
                ctx.map_or(0, |c| c.apc_width),
                ctx.map_or(1, |c| c.calls_per_apc_row),
            )
            .unwrap();
        }

        owned_trace.map_or_else(get_empty_air_proving_ctx::<GpuBackend>, AirProvingContext::simple_no_pis)
    }
}
