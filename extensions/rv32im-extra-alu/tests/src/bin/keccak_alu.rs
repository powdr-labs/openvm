use eyre::Result;
use openvm_build::{build_guest_package, get_package, GuestOptions, TargetFilter};
use openvm_circuit::arch::InitFileGenerator;
use openvm_circuit::utils::air_test;
use openvm_instructions::exe::VmExe;
use openvm_rv32im_circuit::{Rv32IConfig, Rv32ImConfig};
use openvm_rv32im_extra_alu_circuit::{Rv32ImExtraAluBuilder, Rv32ImExtraAluConfig};
use openvm_rv32im_extra_alu_transpiler::Rv32ExtraAluTranspilerExtension;
use openvm_rv32im_transpiler::{Rv32IoTranspilerExtension, Rv32MTranspilerExtension};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use openvm_toolchain_tests::get_programs_dir;
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler, FromElf};

type F = BabyBear;

/// Like `build_example_program_at_path` but uses a stable target directory so cargo
/// can reuse the incremental build cache across runs.
fn build_guest(
    manifest_dir: std::path::PathBuf,
    example_name: &str,
    init_config: &impl InitFileGenerator,
) -> Result<Elf> {
    // Use a stable subdirectory inside the workspace target/ so cargo caches it.
    let target_dir = manifest_dir.join("../../..").join("target").join("guest-programs");
    let pkg = get_package(&manifest_dir);
    init_config.write_to_init_file(&manifest_dir, Some(&format!("openvm_init_{example_name}.rs")))?;
    let guest_opts = GuestOptions::default().with_target_dir(&target_dir);
    let examples_dir = build_guest_package(
        &pkg,
        &guest_opts,
        None,
        &Some(TargetFilter {
            name: example_name.to_string(),
            kind: "example".to_string(),
        }),
    ).map_err(|code| eyre::eyre!("Guest build failed with code {code:?}"))?;
    let elf_path = examples_dir.join(example_name);
    let data = std::fs::read(&elf_path)
        .map_err(|e| eyre::eyre!("Could not read ELF at {elf_path:?}: {e}"))?;
    Elf::decode(&data, MEM_SIZE as u32).map_err(Into::into)
}

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

    let elf = build_guest(get_programs_dir!("programs"), "keccak", &config)?;

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
