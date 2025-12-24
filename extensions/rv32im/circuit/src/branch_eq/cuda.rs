use std::mem::size_of;
use std::sync::Arc;

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::var_range::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::GpuBackend, types::F,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_stark_backend::{prover::types::AirProvingContext, ApcTracingContext, Chip};

use crate::{
    adapters::{Rv32BranchAdapterCols, Rv32BranchAdapterRecord, RV32_REGISTER_NUM_LIMBS},
    cuda_abi::beq_cuda::tracegen,
    BranchEqualCoreCols, BranchEqualCoreRecord,
};

#[derive(new)]
pub struct Rv32BranchEqualChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv32BranchEqualChipGpu {
    fn generate_proving_ctx_new(&self, arena: DenseRecordArena, ctx: &ApcTracingContext) {
        const RECORD_SIZE: usize = size_of::<(
            Rv32BranchAdapterRecord,
            BranchEqualCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return;
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = Rv32BranchAdapterCols::<F>::width()
            + BranchEqualCoreCols::<F, RV32_REGISTER_NUM_LIMBS>::width();
        let height = records.len() / RECORD_SIZE;
        let padded_height = next_power_of_two_or_zero(height);

        let d_records = records.to_device().unwrap();

        unsafe {
            tracegen(
                ctx.d_trace,
                padded_height,
                trace_width,
                &d_records,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                ctx.d_subs,
                ctx.d_opt_widths,
                ctx.d_post_opt_offsets,
                ctx.apc_height,
                ctx.apc_width,
                ctx.calls_per_apc_row,
            )
            .unwrap();
        }
    }

    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv32BranchAdapterRecord,
            BranchEqualCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = BranchEqualCoreCols::<F, RV32_REGISTER_NUM_LIMBS>::width()
            + Rv32BranchAdapterCols::<F>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);

        let d_records = records.to_device().unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity(trace_height, trace_width);

        unsafe {
            tracegen(
                d_trace.buffer(),
                trace_height,
                trace_width,
                &d_records,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                &DeviceBuffer::new(), // nullptr
                &DeviceBuffer::new(), // nullptr
                &DeviceBuffer::new(), // nullptr
                0,                    // apc_height: not used in this path so set to 0
                0,                    // apc_width: not used in this path so set to 0
                1,                    // calls_per_apc_row: 1 for non-apc
            )
            .unwrap();
        }
        AirProvingContext::simple_no_pis(d_trace)
    }
}
