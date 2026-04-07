use std::time::Instant;

use clap::Parser;
use openvm_benchmark_circuit::{
    BenchmarkCpuBuilder, BenchmarkExtension, BenchmarkVmConfig, BENCHMARK_OPCODE_BASE,
};
use openvm_circuit::arch::{instructions::exe::VmExe, SystemConfig};
use openvm_circuit::utils::{air_test_impl, TestStarkEngine};
use openvm_instructions::{
    instruction::Instruction, program::Program, LocalOpcode, SystemOpcode::TERMINATE, VmOpcode,
};
use openvm_stark_backend::SystemParams;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

type F = BabyBear;

#[derive(Parser, Debug)]
#[command(about = "Benchmark proving time vs AIR complexity parameters")]
struct Cli {
    /// Number of distinct benchmark AIRs
    #[arg(long, default_value = "1")]
    num_airs: usize,

    /// Number of columns per AIR (including structural columns)
    #[arg(long, default_value = "64")]
    cols_per_air: usize,

    /// Number of boolean constraints per AIR
    #[arg(long, default_value = "32")]
    constraints_per_air: usize,

    /// Number of range check interactions per row per AIR
    #[arg(long, default_value = "4")]
    interactions_per_air: usize,

    /// Number of trace rows consumed per precompile invocation
    #[arg(long, default_value = "64")]
    rows_per_invocation: usize,

    /// Total trace cells across all benchmark AIRs (num_airs * trace_height * cols_per_air)
    #[arg(long, default_value = "4194304")]
    total_cells: usize,
}

fn main() -> eyre::Result<()> {
    let cli = Cli::parse();

    // Validate and compute derived parameters
    assert!(
        cli.cols_per_air >= 7,
        "cols_per_air must be >= 7 (6 structural + at least 1 payload)"
    );
    assert!(cli.num_airs > 0, "num_airs must be > 0");
    assert!(
        cli.rows_per_invocation > 0,
        "rows_per_invocation must be > 0"
    );

    let trace_height_per_air = cli.total_cells / (cli.num_airs * cli.cols_per_air);
    assert!(
        trace_height_per_air > 0,
        "total_cells too small for given num_airs and cols_per_air"
    );
    assert!(
        trace_height_per_air % cli.rows_per_invocation == 0,
        "trace_height_per_air ({trace_height_per_air}) must be divisible by rows_per_invocation ({})",
        cli.rows_per_invocation,
    );

    let invocations_per_air = trace_height_per_air / cli.rows_per_invocation;
    let total_invocations = invocations_per_air * cli.num_airs;
    let padded_height = trace_height_per_air.next_power_of_two();

    eprintln!("=== Benchmark AIR Complexity ===");
    eprintln!("  num_airs:             {}", cli.num_airs);
    eprintln!("  cols_per_air:         {}", cli.cols_per_air);
    eprintln!("  constraints_per_air:  {}", cli.constraints_per_air);
    eprintln!("  interactions_per_air: {}", cli.interactions_per_air);
    eprintln!("  rows_per_invocation:  {}", cli.rows_per_invocation);
    eprintln!("  total_cells:          {}", cli.total_cells);
    eprintln!("  ---");
    eprintln!(
        "  trace_height_per_air: {} (padded: {})",
        trace_height_per_air, padded_height
    );
    eprintln!("  invocations_per_air:  {}", invocations_per_air);
    eprintln!("  total_instructions:   {}", total_invocations + 1);
    eprintln!();

    // Build extension and config
    let ext = BenchmarkExtension {
        num_airs: cli.num_airs,
        cols_per_air: cli.cols_per_air,
        constraints_per_air: cli.constraints_per_air,
        interactions_per_air: cli.interactions_per_air,
        rows_per_invocation: cli.rows_per_invocation,
    };
    let config = BenchmarkVmConfig {
        system: SystemConfig::default().with_max_segment_len(padded_height),
        benchmark: ext,
    };

    // Construct flat instruction sequence
    let mut instructions: Vec<Instruction<F>> = Vec::with_capacity(total_invocations + 1);
    for _ in 0..invocations_per_air {
        for air_idx in 0..cli.num_airs {
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

    // Run proving pipeline with timing
    // Use at least 22 as the max log trace height (same as air_test), since system chips
    // like VariableRangeChecker may have large traces independent of our benchmark.
    let log_max_height = (padded_height.ilog2() as usize + 1).max(22);
    let params = SystemParams::new_for_testing(log_max_height);

    let total_start = Instant::now();
    let (_final_memory, proofs) = air_test_impl::<TestStarkEngine, BenchmarkCpuBuilder>(
        params,
        BenchmarkCpuBuilder,
        config,
        exe,
        openvm_circuit::arch::Streams::<F>::default(),
        1,
        false,
    )?;

    let total_time = total_start.elapsed();

    eprintln!("=== Results ===");
    eprintln!("  Segments proved:  {}", proofs.len());
    eprintln!("  Total time:       {:.3}s", total_time.as_secs_f64());

    Ok(())
}
