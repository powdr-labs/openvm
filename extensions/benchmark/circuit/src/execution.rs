use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
    slice::from_raw_parts,
};

use openvm_circuit::arch::{
    create_handler,
    execution_mode::{ExecutionCtxTrait, MeteredExecutionCtxTrait},
    E2PreCompute, InterpreterExecutor, InterpreterMeteredExecutor,
    PreflightExecutor, RecordArena, StaticProgramError, VmExecState, VmStateMut,
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::p3_field::PrimeField32;

#[cfg(not(feature = "tco"))]
use openvm_circuit::arch::ExecuteFunc;
#[cfg(feature = "tco")]
use openvm_circuit::arch::Handler;
#[cfg(feature = "aot")]
use openvm_circuit::arch::{AotExecutor, AotMeteredExecutor};

use crate::trace::{BenchmarkLayout, BenchmarkMetadata, BenchmarkRecordMut};

use openvm_circuit::system::memory::online::{GuestMemory, TracingMemory};

#[derive(Clone, Debug, derive_new::new)]
pub struct BenchmarkExecutor {
    pub rows_per_invocation: usize,
}

#[repr(C)]
#[derive(Clone, AlignedBytesBorrow)]
struct BenchmarkPreCompute {
    rows_per_invocation: u32,
}

impl<F: PrimeField32> InterpreterExecutor<F> for BenchmarkExecutor {
    fn pre_compute_size(&self) -> usize {
        size_of::<BenchmarkPreCompute>()
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        _pc: u32,
        _inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut BenchmarkPreCompute = data.borrow_mut();
        pre_compute.rows_per_invocation = self.rows_per_invocation as u32;
        Ok(execute_e1_handler::<_, _>)
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        _pc: u32,
        _inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        let pre_compute: &mut BenchmarkPreCompute = data.borrow_mut();
        pre_compute.rows_per_invocation = self.rows_per_invocation as u32;
        Ok(execute_e1_handler::<_, _>)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotExecutor<F> for BenchmarkExecutor {}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for BenchmarkExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<BenchmarkPreCompute>>()
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        air_idx: usize,
        _pc: u32,
        _inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<BenchmarkPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = air_idx as u32;
        pre_compute.data.rows_per_invocation = self.rows_per_invocation as u32;
        Ok(execute_e2_handler::<_, _>)
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        air_idx: usize,
        _pc: u32,
        _inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        let pre_compute: &mut E2PreCompute<BenchmarkPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = air_idx as u32;
        pre_compute.data.rows_per_invocation = self.rows_per_invocation as u32;
        Ok(execute_e2_handler::<_, _>)
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32> AotMeteredExecutor<F> for BenchmarkExecutor {}

// E3: Preflight executor (record generation)
impl<F, RA> PreflightExecutor<F, RA> for BenchmarkExecutor
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, BenchmarkLayout, BenchmarkRecordMut<'buf>>,
{
    fn get_opcode_name(&self, _opcode: usize) -> String {
        "BENCHMARK".to_string()
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        _instruction: &Instruction<F>,
    ) -> Result<(), openvm_circuit::arch::ExecutionError> {
        let num_rows = self.rows_per_invocation;
        let layout = BenchmarkLayout::new(BenchmarkMetadata { num_rows });
        let record: BenchmarkRecordMut = state.ctx.alloc(layout);
        record.header.from_pc = *state.pc;
        record.header.from_timestamp = state.memory.timestamp();
        record.header.num_rows = num_rows as u32;
        // Advance timestamp by 1 (must match timestamp_change in the AIR)
        state.memory.increment_timestamp();
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

// E1/E2 shared logic: just advance PC
#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: &BenchmarkPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> u32 {
    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));
    pre_compute.rows_per_invocation
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &BenchmarkPreCompute =
        from_raw_parts(pre_compute, size_of::<BenchmarkPreCompute>()).borrow();
    execute_e12_impl(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<BenchmarkPreCompute> = from_raw_parts(
        pre_compute,
        size_of::<E2PreCompute<BenchmarkPreCompute>>(),
    )
    .borrow();
    let height = execute_e12_impl(&pre_compute.data, exec_state);
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}
