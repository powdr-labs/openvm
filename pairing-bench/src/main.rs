use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use eyre::Result;
use openvm_stark_sdk::{
    bench::run_with_metric_collection, openvm_stark_backend::codec::Encode,
};
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};
use sdk_v2::{
    config::{
        default_app_params, default_compression_params, default_internal_params,
        default_leaf_params, AggregationSystemParams, DEFAULT_APP_L_SKIP, DEFAULT_APP_LOG_BLOWUP,
        DEFAULT_COMPRESSION_LOG_BLOWUP, DEFAULT_INTERNAL_LOG_BLOWUP, DEFAULT_LEAF_LOG_BLOWUP,
    },
    Sdk, StdIn,
};

/// Max log of stacked trace height for app-level proofs.
/// Matches the value used in openvm-eth reth-benchmark.
const DEFAULT_LOG_STACKED_HEIGHT: usize = 24;

#[derive(Parser, Debug)]
#[command(name = "pairing-bench")]
struct Args {
    /// Path to a pre-compiled guest ELF. If not provided, builds the guest first.
    #[arg(long)]
    elf: Option<PathBuf>,

    /// Path to guest package directory (for building). Defaults to ./guest
    #[arg(long)]
    guest_dir: Option<PathBuf>,

    /// Execution mode
    #[arg(long, default_value = "prove-app")]
    mode: Mode,

    /// App-level log blowup
    #[arg(long, default_value_t = DEFAULT_APP_LOG_BLOWUP)]
    app_log_blowup: usize,

    /// App-level l_skip
    #[arg(long, default_value_t = DEFAULT_APP_L_SKIP)]
    app_l_skip: usize,

    /// Max trace height per segment
    #[arg(long)]
    max_segment_length: Option<u32>,
}

#[derive(Debug, Clone, ValueEnum)]
enum Mode {
    /// Just execute, no proving
    Execute,
    /// Execute with metering (segmentation info)
    ExecuteMetered,
    /// App-level proofs only (per-segment, no aggregation)
    ProveApp,
    /// Full STARK proof with recursive aggregation
    ProveStark,
}

fn load_or_build_elf(args: &Args) -> Result<Elf> {
    if let Some(elf_path) = &args.elf {
        let data = std::fs::read(elf_path)?;
        let elf = Elf::decode(&data, MEM_SIZE as u32)?;
        eprintln!("Loaded ELF from {}: {} bytes", elf_path.display(), data.len());
        return Ok(elf);
    }

    // Build guest
    let guest_dir = args
        .guest_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("guest"));
    eprintln!("Building guest from {}...", guest_dir.display());

    let pkg = openvm_build::get_package(&guest_dir);
    let temp_dir = tempfile::tempdir()?;
    let guest_opts = openvm_build::GuestOptions::default()
        .with_target_dir(temp_dir.path())
        .with_profile("release".to_string());

    if let Err(Some(code)) = openvm_build::build_guest_package(&pkg, &guest_opts, None, &None) {
        std::process::exit(code);
    }

    let elf_path = openvm_build::guest_methods(
        &pkg,
        temp_dir.path(),
        &guest_opts.features,
        &guest_opts.profile,
    )
    .pop()
    .expect("no ELF built");
    let data = std::fs::read(&elf_path)?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    eprintln!("Guest ELF built: {} bytes", data.len());
    Ok(elf)
}

fn main() -> Result<()> {
    let args = Args::parse();

    let elf = load_or_build_elf(&args)?;

    // Configure SDK: rv32im only (no precompiles)
    // max_log_height = l_skip + n_stack; use 24 like reth-benchmark
    let app_n_stack = DEFAULT_LOG_STACKED_HEIGHT - args.app_l_skip;
    let app_params = default_app_params(args.app_log_blowup, args.app_l_skip, app_n_stack);

    let agg_params = AggregationSystemParams {
        leaf: default_leaf_params(DEFAULT_LEAF_LOG_BLOWUP),
        internal: default_internal_params(DEFAULT_INTERNAL_LOG_BLOWUP),
        compression: Some(default_compression_params(DEFAULT_COMPRESSION_LOG_BLOWUP)),
    };

    let mut sdk: Sdk = Sdk::riscv32(app_params, agg_params);
    if let Some(max_height) = args.max_segment_length {
        sdk.app_config_mut()
            .app_vm_config
            .as_mut()
            .segmentation_config
            .limits
            .set_max_trace_height(max_height);
    }

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        match args.mode {
            Mode::Execute => {
                eprintln!("Executing...");
                let _pvs = sdk.execute(elf.clone(), StdIn::default())?;
                eprintln!("Execution complete.");
            }
            Mode::ExecuteMetered => {
                eprintln!("Executing (metered)...");
                let (_pvs, segments) = sdk.execute_metered(elf.clone(), StdIn::default())?;
                eprintln!("Metered execution complete: {} segments", segments.len());
                for (i, seg) in segments.iter().enumerate() {
                    eprintln!(
                        "  segment {}: {} insns, {} trace heights",
                        i,
                        seg.num_insns,
                        seg.trace_heights.len()
                    );
                }
            }
            Mode::ProveApp => {
                eprintln!("Generating app proofs (per-segment, no aggregation)...");
                let mut prover = sdk.app_prover(elf.clone())?;
                prover.set_program_name("pairing".to_string());
                let app_proof = prover.prove(StdIn::default())?;
                eprintln!(
                    "App proving complete: {} segment proofs",
                    app_proof.per_segment.len()
                );
            }
            Mode::ProveStark => {
                eprintln!("Generating full STARK proof with recursive aggregation...");
                let (proof, baseline) = sdk.prove(elf.clone(), StdIn::default())?;

                let encoded = proof.encode_to_vec()?;
                eprintln!("Proof size: {} bytes", encoded.len());

                // Verify
                Sdk::verify_proof((*sdk.agg_vk()).clone(), baseline, &proof)?;
                eprintln!("Proof verified successfully!");
            }
        }
        Ok(())
    })
}
