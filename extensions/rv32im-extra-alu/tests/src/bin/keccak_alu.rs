use eyre::Result;
use openvm_circuit::utils::air_test;
use openvm_instructions::exe::VmExe;
use openvm_rv32im_circuit::{Rv32IConfig, Rv32ImConfig};
use openvm_rv32im_extra_alu_circuit::{Rv32ImExtraAluBuilder, Rv32ImExtraAluConfig};
use openvm_rv32im_extra_alu_transpiler::Rv32ExtraAluTranspilerExtension;
use openvm_rv32im_transpiler::{Rv32IoTranspilerExtension, Rv32MTranspilerExtension};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};
use openvm_transpiler::{transpiler::Transpiler, FromElf};

type F = BabyBear;

fn main() -> Result<()> {
    let num_total_alu: usize = std::env::args()
        .nth(1)
        .expect("Usage: keccak_alu <num_alus>")
        .parse()
        .expect("num_alus must be a positive integer");

    assert!(num_total_alu >= 1, "num_alus must be >= 1");

    println!("Running keccak with {num_total_alu} ALU chip(s)...");

    let config = Rv32ImExtraAluConfig::new(
        Rv32ImConfig {
            rv32i: Rv32IConfig {
                system: openvm_circuit::utils::test_system_config(),
                ..Default::default()
            },
            ..Default::default()
        },
        num_total_alu - 1,
    );

    let programs_dir = get_programs_dir!("programs");
    let elf = build_example_program_at_path(programs_dir, "keccak", &config)?;

    let exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ExtraAluTranspilerExtension::new(num_total_alu))
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;

    air_test(Rv32ImExtraAluBuilder, config, exe);

    println!("Done.");
    Ok(())
}
