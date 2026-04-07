use openvm_circuit::{
    arch::{ChipInventory, ChipInventoryError, DenseRecordArena, VmProverExtension},
    system::cuda::extensions::{get_inventory_range_checker, get_or_create_bitwise_op_lookup},
};
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine as GpuBabyBearPoseidon2Engine, GpuBackend};
use openvm_rv32im_circuit::{Rv32BaseAluAir, Rv32BaseAluChipGpu};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use crate::Rv32ExtraAlu;

pub struct Rv32ImExtraAluGpuProverExt;

impl VmProverExtension<GpuBabyBearPoseidon2Engine, DenseRecordArena, Rv32ExtraAlu>
    for Rv32ImExtraAluGpuProverExt
{
    fn extend_prover(
        &self,
        ext: &Rv32ExtraAlu,
        inventory: &mut ChipInventory<BabyBearPoseidon2Config, DenseRecordArena, GpuBackend>,
    ) -> Result<(), ChipInventoryError> {
        let timestamp_max_bits = inventory.timestamp_max_bits();
        let range_checker = get_inventory_range_checker(inventory);
        let bitwise_lu = get_or_create_bitwise_op_lookup(inventory)?;

        for _ in 0..ext.num_extra {
            // Safeguard: verify AIR ordering matches circuit definition.
            inventory.next_air::<Rv32BaseAluAir>()?;
            // The GPU chip does not need an offset parameter: the opcode offset is encoded in
            // the AIR (BaseAluCoreAir::offset), while the chip trace generation works on raw
            // records that contain local opcodes (0–4).
            let chip = Rv32BaseAluChipGpu::new(
                range_checker.clone(),
                bitwise_lu.clone(),
                timestamp_max_bits,
            );
            inventory.add_executor_chip(chip);
        }
        Ok(())
    }
}
