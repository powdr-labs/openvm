use std::sync::Arc;

use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, InitFileGenerator, MatrixRecordArena,
        SystemConfig, VmBuilder, VmChipComplex, VmCircuitExtension, VmExecutionExtension, VmField,
        VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemChipInventory, SystemCpuBuilder, SystemExecutor},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor, VmConfig};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_cpu_backend::{CpuBackend, CpuDevice};
use openvm_instructions::VmOpcode;
use openvm_stark_backend::{StarkEngine, StarkProtocolConfig, Val};
use serde::{Deserialize, Serialize};

pub mod air;
pub mod execution;
pub mod trace;

pub use air::BenchmarkAir;
pub use execution::BenchmarkExecutor;
pub use trace::{
    BenchmarkFiller, BenchmarkLayout, BenchmarkMetadata, BenchmarkRecordHeader,
    BenchmarkRecordMut,
};

/// Base opcode offset for benchmark precompiles.
pub const BENCHMARK_OPCODE_BASE: usize = 0x900;

// ---- Extension ----

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkExtension {
    pub num_airs: usize,
    pub cols_per_air: usize,
    pub constraints_per_air: usize,
    pub interactions_per_air: usize,
    pub rows_per_invocation: usize,
}

#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
#[cfg_attr(
    feature = "aot",
    derive(
        openvm_circuit_derive::AotExecutor,
        openvm_circuit_derive::AotMeteredExecutor
    )
)]
pub enum BenchmarkExecutorEnum {
    Benchmark(BenchmarkExecutor),
}

impl<F: VmField> VmExecutionExtension<F> for BenchmarkExtension {
    type Executor = BenchmarkExecutorEnum;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, BenchmarkExecutorEnum>,
    ) -> Result<(), ExecutorInventoryError> {
        for i in 0..self.num_airs {
            let executor = BenchmarkExecutor::new(self.rows_per_invocation);
            let opcode = VmOpcode::from_usize(BENCHMARK_OPCODE_BASE + i);
            inventory.add_executor(executor, [opcode])?;
        }
        Ok(())
    }
}

impl<SC: StarkProtocolConfig> VmCircuitExtension<SC> for BenchmarkExtension
where
    Val<SC>: VmField,
{
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let execution_bridge = ExecutionBridge::new(
            inventory.system().port().execution_bus,
            inventory.system().port().program_bus,
        );

        // Get or create the bitwise operation lookup bus (same pattern as DeferralExtension)
        let bitwise_bus = {
            let existing_air = inventory.find_air::<BitwiseOperationLookupAir<8>>().next();
            if let Some(air) = existing_air {
                air.bus
            } else {
                let bus = BitwiseOperationLookupBus::new(inventory.new_bus_idx());
                let air = BitwiseOperationLookupAir::<8>::new(bus);
                inventory.add_air(air);
                air.bus
            }
        };

        for i in 0..self.num_airs {
            inventory.add_air(BenchmarkAir {
                execution_bridge,
                bitwise_bus,
                num_columns: self.cols_per_air,
                num_constraints: self.constraints_per_air,
                num_interactions: self.interactions_per_air,
                rows_per_invocation: self.rows_per_invocation,
                opcode: BENCHMARK_OPCODE_BASE + i,
            });
        }

        Ok(())
    }
}

pub struct BenchmarkCpuProverExt;

impl<SC, E, RA> VmProverExtension<E, RA, BenchmarkExtension> for BenchmarkCpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: openvm_circuit::arch::RowMajorMatrixArena<Val<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        extension: &BenchmarkExtension,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        // Get or create the bitwise lookup chip
        let bitwise_lu = {
            let existing_chip = inventory
                .find_chip::<SharedBitwiseOperationLookupChip<8>>()
                .next();
            if let Some(chip) = existing_chip {
                chip.clone()
            } else {
                let air: &BitwiseOperationLookupAir<8> = inventory.next_air()?;
                let chip = Arc::new(BitwiseOperationLookupChip::new(air.bus));
                inventory.add_periphery_chip(chip.clone());
                chip
            }
        };

        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker, timestamp_max_bits);

        for _ in 0..extension.num_airs {
            inventory.next_air::<BenchmarkAir>()?;
            let filler = BenchmarkFiller::new(
                bitwise_lu.clone(),
                extension.cols_per_air,
                extension.interactions_per_air,
                extension.rows_per_invocation,
            );
            inventory.add_executor_chip(openvm_circuit::arch::VmChipWrapper::new(
                filler,
                mem_helper.clone(),
            ));
        }

        Ok(())
    }
}

// ---- VmConfig ----

#[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
pub struct BenchmarkVmConfig {
    #[config(executor = "SystemExecutor<F>")]
    pub system: SystemConfig,
    #[extension(executor = "BenchmarkExecutorEnum")]
    pub benchmark: BenchmarkExtension,
}

impl InitFileGenerator for BenchmarkVmConfig {}

// ---- VmBuilder ----

#[derive(Clone)]
pub struct BenchmarkCpuBuilder;

impl<SC, E> VmBuilder<E> for BenchmarkCpuBuilder
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    type VmConfig = BenchmarkVmConfig;
    type SystemChipInventory = SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &BenchmarkVmConfig,
        circuit: AirInventory<SC>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&SystemCpuBuilder, &config.system, circuit)?;
        VmProverExtension::<E, _, _>::extend_prover(
            &BenchmarkCpuProverExt,
            &config.benchmark,
            &mut chip_complex.inventory,
        )?;
        Ok(chip_complex)
    }
}

#[cfg(test)]
mod tests {
    use openvm_circuit::{
        arch::{instructions::exe::VmExe, SystemConfig},
        utils::air_test,
    };
    use openvm_instructions::{
        instruction::Instruction, program::Program, LocalOpcode, SystemOpcode::TERMINATE,
    };
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_benchmark_air_small() {
        let num_airs = 1;
        let cols_per_air = 16;
        let rows_per_invocation = 4;
        let invocations = 4;

        let ext = BenchmarkExtension {
            num_airs,
            cols_per_air,
            constraints_per_air: 4,
            interactions_per_air: 2,
            rows_per_invocation,
        };
        let config = BenchmarkVmConfig {
            system: SystemConfig::default().with_max_segment_len(1 << 20),
            benchmark: ext,
        };

        let mut instructions: Vec<Instruction<F>> = Vec::new();
        for _ in 0..invocations {
            for air_idx in 0..num_airs {
                instructions.push(Instruction::from_isize(
                    VmOpcode::from_usize(BENCHMARK_OPCODE_BASE + air_idx),
                    0,
                    0,
                    0,
                    0,
                    0,
                ));
            }
        }
        instructions.push(Instruction::from_isize(
            TERMINATE.global_opcode(),
            0,
            0,
            0,
            0,
            0,
        ));

        let program = Program::from_instructions(&instructions);
        let exe = VmExe::new(program);
        air_test(BenchmarkCpuBuilder, config, exe);
    }

    #[test]
    fn test_benchmark_air_multi_air() {
        let num_airs = 3;
        let cols_per_air = 16;
        let rows_per_invocation = 2;
        let invocations_per_air = 4;

        let ext = BenchmarkExtension {
            num_airs,
            cols_per_air,
            constraints_per_air: 2,
            interactions_per_air: 1,
            rows_per_invocation,
        };
        let config = BenchmarkVmConfig {
            system: SystemConfig::default().with_max_segment_len(1 << 20),
            benchmark: ext,
        };

        let mut instructions: Vec<Instruction<F>> = Vec::new();
        for _ in 0..invocations_per_air {
            for air_idx in 0..num_airs {
                instructions.push(Instruction::from_isize(
                    VmOpcode::from_usize(BENCHMARK_OPCODE_BASE + air_idx),
                    0,
                    0,
                    0,
                    0,
                    0,
                ));
            }
        }
        instructions.push(Instruction::from_isize(
            TERMINATE.global_opcode(),
            0,
            0,
            0,
            0,
            0,
        ));

        let program = Program::from_instructions(&instructions);
        let exe = VmExe::new(program);
        air_test(BenchmarkCpuBuilder, config, exe);
    }
}
