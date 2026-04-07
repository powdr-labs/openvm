#[cfg(test)]
mod tests {
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

    /// Point at the rv32im test programs which already contain `fibonacci` and `collatz`.
    fn programs_dir() -> std::path::PathBuf {
        get_programs_dir!("../../rv32im/tests/programs")
    }

    fn make_config(num_total_alu: usize) -> Rv32ImExtraAluConfig {
        Rv32ImExtraAluConfig::new(
            Rv32ImConfig {
                rv32i: Rv32IConfig {
                    system: openvm_circuit::utils::test_system_config(),
                    ..Default::default()
                },
                ..Default::default()
            },
            num_total_alu - 1, // num_extra = num_total_alu - 1
        )
    }

    fn make_transpiler(num_total_alu: usize) -> Transpiler<F> {
        Transpiler::<F>::default()
            .with_extension(Rv32ExtraAluTranspilerExtension::new(num_total_alu))
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
    }

    #[test]
    fn test_extra_alu_fibonacci_2chips() -> Result<()> {
        let num_total_alu = 2; // 1 original + 1 extra
        let config = make_config(num_total_alu);
        let elf = build_example_program_at_path(programs_dir(), "fibonacci", &config)?;
        let exe = VmExe::from_elf(elf, make_transpiler(num_total_alu))?;
        air_test(Rv32ImExtraAluBuilder, config, exe);
        Ok(())
    }

    #[test]
    fn test_extra_alu_fibonacci_4chips() -> Result<()> {
        let num_total_alu = 4; // 1 original + 3 extra
        let config = make_config(num_total_alu);
        let elf = build_example_program_at_path(programs_dir(), "fibonacci", &config)?;
        let exe = VmExe::from_elf(elf, make_transpiler(num_total_alu))?;
        air_test(Rv32ImExtraAluBuilder, config, exe);
        Ok(())
    }

    #[test]
    fn test_extra_alu_collatz_2chips() -> Result<()> {
        let num_total_alu = 2;
        let config = make_config(num_total_alu);
        let elf = build_example_program_at_path(programs_dir(), "collatz", &config)?;
        let exe = VmExe::from_elf(elf, make_transpiler(num_total_alu))?;
        air_test(Rv32ImExtraAluBuilder, config, exe);
        Ok(())
    }
}
