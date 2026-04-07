use derive_more::derive::From;
use openvm_circuit::{
    arch::{
        AirInventory, AirInventoryError, ChipInventory, ChipInventoryError, ExecutionBridge,
        ExecutorInventoryBuilder, ExecutorInventoryError, InitFileGenerator, MatrixRecordArena,
        RowMajorMatrixArena, SystemConfig, VmBuilder, VmChipComplex, VmCircuitExtension,
        VmExecutionExtension, VmField, VmProverExtension,
    },
    system::{memory::SharedMemoryHelper, SystemPort},
};
use openvm_circuit_derive::{AnyEnum, Executor, MeteredExecutor, PreflightExecutor, VmConfig};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::VmOpcode;
use openvm_rv32im_circuit::{
    adapters::{Rv32BaseAluAdapterAir, Rv32BaseAluAdapterExecutor, Rv32BaseAluAdapterFiller},
    BaseAluCoreAir, BaseAluFiller, Rv32BaseAluAir, Rv32BaseAluChip, Rv32BaseAluExecutor,
    Rv32ImConfig, Rv32ImConfigExecutor, Rv32ImCpuBuilder,
};
use openvm_stark_backend::{
    p3_field::PrimeField32,
    prover::{CpuBackend, CpuDevice},
    StarkEngine, StarkProtocolConfig, Val,
};
use serde::{Deserialize, Serialize};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        mod cuda;
        pub use cuda::Rv32ImExtraAluGpuProverExt;
        use openvm_cuda_backend::{
            BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend,
        };
        use openvm_circuit::arch::DenseRecordArena;
        use openvm_circuit::system::cuda::SystemChipInventoryGPU;
        use openvm_rv32im_circuit::Rv32ImGpuBuilder;
        use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
        pub use self::{
            Rv32ImExtraAluCpuBuilder as Rv32ImExtraAluBuilder,
        };
    } else {
        pub use self::{
            Rv32ImExtraAluCpuBuilder as Rv32ImExtraAluBuilder,
        };
    }
}

/// Number of opcodes in a single BaseAlu group (ADD, SUB, XOR, OR, AND).
pub const BASE_ALU_COUNT: usize = 5;

/// Opcode offset for extra BaseAlu chips.
/// Extra chip k (0-indexed among the extra chips) uses
/// `BASE_EXTRA_ALU_OFFSET + k * BASE_ALU_COUNT`.
pub const BASE_EXTRA_ALU_OFFSET: usize = 0x900;

// ============ Extension Struct ============

/// Circuit extension that instantiates `num_extra` additional BaseAlu chips beyond
/// the one already present in `Rv32ImConfig`.
///
/// Total chip count = `num_extra + 1`, which should be a power of two.
/// Extra chip k (k = 0..num_extra-1) handles opcodes at
/// `BASE_EXTRA_ALU_OFFSET + k * BASE_ALU_COUNT`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rv32ExtraAlu {
    pub num_extra: usize,
}

impl Rv32ExtraAlu {
    /// `num_extra` is the number of BaseAlu chips to add beyond the original.
    /// Together with the one in `Rv32ImConfig`, total = `num_extra + 1`, which must be a
    /// power of two >= 2.
    pub fn new(num_extra: usize) -> Self {
        assert!(
            (num_extra + 1).is_power_of_two() && num_extra >= 1,
            "num_extra + 1 must be a power of two >= 2, got num_extra = {num_extra}",
        );
        Self { num_extra }
    }
}

// ============ Executor Enum ============

#[derive(Clone, From, AnyEnum, Executor, MeteredExecutor, PreflightExecutor)]
pub enum Rv32ExtraAluExecutor {
    BaseAlu(Rv32BaseAluExecutor),
}

// ============ VmExecutionExtension ============

impl<F: PrimeField32> VmExecutionExtension<F> for Rv32ExtraAlu {
    type Executor = Rv32ExtraAluExecutor;

    fn extend_execution(
        &self,
        inventory: &mut ExecutorInventoryBuilder<F, Rv32ExtraAluExecutor>,
    ) -> Result<(), ExecutorInventoryError> {
        for k in 0..self.num_extra {
            let offset = BASE_EXTRA_ALU_OFFSET + k * BASE_ALU_COUNT;
            let executor = Rv32BaseAluExecutor::new(Rv32BaseAluAdapterExecutor, offset);
            inventory.add_executor(
                executor,
                (0..BASE_ALU_COUNT).map(|i| VmOpcode::from_usize(offset + i)),
            )?;
        }
        Ok(())
    }
}

// ============ VmCircuitExtension ============

impl<SC: StarkProtocolConfig> VmCircuitExtension<SC> for Rv32ExtraAlu {
    fn extend_circuit(&self, inventory: &mut AirInventory<SC>) -> Result<(), AirInventoryError> {
        let SystemPort {
            execution_bus,
            program_bus,
            memory_bridge,
        } = inventory.system().port();
        let exec_bridge = ExecutionBridge::new(execution_bus, program_bus);

        // Reuse the BitwiseOperationLookupAir created by Rv32I (must already exist).
        let bitwise_lu_bus = inventory
            .find_air::<BitwiseOperationLookupAir<8>>()
            .next()
            .expect(
                "BitwiseOperationLookupAir<8> must exist; Rv32ExtraAlu requires Rv32ImConfig as base",
            )
            .bus;

        for k in 0..self.num_extra {
            let offset = BASE_EXTRA_ALU_OFFSET + k * BASE_ALU_COUNT;
            let air = Rv32BaseAluAir::new(
                Rv32BaseAluAdapterAir::new(exec_bridge, memory_bridge, bitwise_lu_bus),
                BaseAluCoreAir::new(bitwise_lu_bus, offset),
            );
            inventory.add_air(air);
        }
        Ok(())
    }
}

// ============ VM Config ============

/// Config wrapping `Rv32ImConfig` and adding `num_extra` extra BaseAlu chips.
#[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
pub struct Rv32ImExtraAluConfig {
    #[config]
    pub rv32im: Rv32ImConfig,
    #[extension]
    pub extra_alu: Rv32ExtraAlu,
}

impl InitFileGenerator for Rv32ImExtraAluConfig {}

impl Rv32ImExtraAluConfig {
    pub fn new(rv32im: Rv32ImConfig, num_extra: usize) -> Self {
        Self {
            rv32im,
            extra_alu: Rv32ExtraAlu::new(num_extra),
        }
    }
}

// ============ CPU Prover Extension ============

pub struct Rv32ImExtraAluCpuProverExt;

impl<SC, E, RA> VmProverExtension<E, RA, Rv32ExtraAlu> for Rv32ImExtraAluCpuProverExt
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    RA: RowMajorMatrixArena<Val<SC>>,
    Val<SC>: PrimeField32,
    SC::EF: Ord,
{
    fn extend_prover(
        &self,
        ext: &Rv32ExtraAlu,
        inventory: &mut ChipInventory<SC, RA, CpuBackend<SC>>,
    ) -> Result<(), ChipInventoryError> {
        let range_checker = inventory.range_checker()?.clone();
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let mem_helper = SharedMemoryHelper::new(range_checker.clone(), timestamp_max_bits);

        // Reuse the BitwiseOperationLookupChip created by Rv32I (must already exist).
        let bitwise_lu = inventory
            .find_chip::<SharedBitwiseOperationLookupChip<8>>()
            .next()
            .expect(
                "SharedBitwiseOperationLookupChip<8> must exist; \
                 Rv32ExtraAlu requires Rv32ImConfig as base",
            )
            .clone();

        for k in 0..ext.num_extra {
            let offset = BASE_EXTRA_ALU_OFFSET + k * BASE_ALU_COUNT;
            // Safeguard: verify AIR ordering matches circuit definition.
            inventory.next_air::<Rv32BaseAluAir>()?;
            let chip = Rv32BaseAluChip::new(
                BaseAluFiller::new(
                    Rv32BaseAluAdapterFiller::new(bitwise_lu.clone()),
                    bitwise_lu.clone(),
                    offset,
                ),
                mem_helper.clone(),
            );
            inventory.add_executor_chip(chip);
        }
        Ok(())
    }
}

// ============ CPU Builder ============

#[derive(Clone)]
pub struct Rv32ImExtraAluCpuBuilder;

impl<SC, E> VmBuilder<E> for Rv32ImExtraAluCpuBuilder
where
    SC: StarkProtocolConfig,
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Val<SC>: VmField,
    SC::EF: Ord,
{
    type VmConfig = Rv32ImExtraAluConfig;
    type SystemChipInventory = openvm_circuit::system::SystemChipInventory<SC>;
    type RecordArena = MatrixRecordArena<Val<SC>>;

    fn create_chip_complex(
        &self,
        config: &Self::VmConfig,
        circuit: AirInventory<SC>,
    ) -> Result<
        VmChipComplex<SC, Self::RecordArena, E::PB, Self::SystemChipInventory>,
        ChipInventoryError,
    > {
        let mut chip_complex =
            VmBuilder::<E>::create_chip_complex(&Rv32ImCpuBuilder, &config.rv32im, circuit)?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<E, _, _>::extend_prover(
            &Rv32ImExtraAluCpuProverExt,
            &config.extra_alu,
            inventory,
        )?;
        Ok(chip_complex)
    }
}

// ============ GPU Builder ============

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct Rv32ImExtraAluGpuBuilder;

#[cfg(feature = "cuda")]
impl VmBuilder<GpuBabyBearPoseidon2Engine> for Rv32ImExtraAluGpuBuilder {
    type VmConfig = Rv32ImExtraAluConfig;
    type SystemChipInventory = SystemChipInventoryGPU;
    type RecordArena = DenseRecordArena;

    fn create_chip_complex(
        &self,
        config: &Self::VmConfig,
        circuit: AirInventory<BabyBearPoseidon2Config>,
    ) -> Result<
        VmChipComplex<
            BabyBearPoseidon2Config,
            Self::RecordArena,
            GpuBackend,
            Self::SystemChipInventory,
        >,
        ChipInventoryError,
    > {
        let mut chip_complex = VmBuilder::<GpuBabyBearPoseidon2Engine>::create_chip_complex(
            &Rv32ImGpuBuilder,
            &config.rv32im,
            circuit,
        )?;
        let inventory = &mut chip_complex.inventory;
        VmProverExtension::<GpuBabyBearPoseidon2Engine, _, _>::extend_prover(
            &Rv32ImExtraAluGpuProverExt,
            &config.extra_alu,
            inventory,
        )?;
        Ok(chip_complex)
    }
}
