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
    fn generate_proving_ctx_direct(
        &self,
        arena: DenseRecordArena,
        ctx: Option<&ApcTracingContext>,
    ) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(
            Rv32BranchAdapterRecord,
            BranchEqualCoreRecord<RV32_REGISTER_NUM_LIMBS>,
        )>();
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = Rv32BranchAdapterCols::<F>::width()
            + BranchEqualCoreCols::<F, RV32_REGISTER_NUM_LIMBS>::width();
        let trace_height = next_power_of_two_or_zero(records.len() / RECORD_SIZE);
        let d_records = records.to_device().unwrap();
        let empty = DeviceBuffer::new();

        let owned_trace = ctx
            .is_none()
            .then(|| DeviceMatrix::<F>::with_capacity(trace_height, trace_width));

        unsafe {
            tracegen(
                ctx.map_or_else(|| owned_trace.as_ref().unwrap().buffer(), |c| c.d_trace),
                trace_height,
                trace_width,
                &d_records,
                &self.range_checker.count,
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
