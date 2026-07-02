use openvm_circuit_primitives::{AlignedBytesBorrow, StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, PhantomDiscriminant, VmOpcode,
};
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, PermutationCheckBus},
    p3_field::PrimeCharacteristicRing,
};
use rand::rngs::StdRng;
use thiserror::Error;

use super::{execution_mode::ExecutionCtxTrait, Streams, VmExecState};
#[cfg(feature = "tco")]
use crate::arch::interpreter::InterpretedInstance;
#[cfg(feature = "aot")]
use crate::arch::SystemConfig;
#[cfg(feature = "metrics")]
use crate::metrics::VmMetrics;
use crate::{
    arch::{execution_mode::MeteredExecutionCtxTrait, ExecutorInventoryError, MatrixRecordArena},
    system::{
        memory::online::{GuestMemory, TracingMemory},
        program::ProgramBus,
    },
};

/// Number of extra ISA-specific registers carried in the *runtime* VM execution state alongside
/// `pc` (e.g. a frame pointer). This governs the array widths of the runtime state
/// ([`VmState`](super::VmState)/[`VmStateMut`]) and of the system connector's execution-bus fields.
///
/// It is `0` by default, so upstream ISAs pay nothing — `[u32; 0]` is a zero-sized array, the
/// connector gets no extra columns, and everything is byte-identical to plain `pc`/`timestamp`
/// state. Enabling the `fp` feature makes it `1`; crush turns that feature on and uses the single
/// slot (`extra_regs[0]`) as its frame pointer.
///
/// Note this is distinct from the per-AIR const-generic [`ExecutionState`] `EXTRA_LEN`: individual
/// instruction chips choose their own AIR width (RISC-V chips stay at `0`), whereas this constant
/// pins the runtime/connector width for the whole build.
#[cfg(not(feature = "fp"))]
pub const EXTRA_EXEC_REGS: usize = 0;
#[cfg(feature = "fp")]
pub const EXTRA_EXEC_REGS: usize = 1;

#[derive(Error, Debug)]
pub enum ExecutionError {
    #[error("execution failed at pc {pc}, err: {msg}")]
    Fail { pc: u32, msg: &'static str },
    #[error("pc {0} out of bounds")]
    PcOutOfBounds(u32),
    #[error("unreachable instruction at pc {0}")]
    Unreachable(u32),
    #[error("at pc {pc}, opcode {opcode} was not enabled")]
    DisabledOperation { pc: u32, opcode: VmOpcode },
    #[error("at pc = {pc}")]
    HintOutOfBounds { pc: u32 },
    #[error("at pc {pc}, hint buffer num_words is zero")]
    HintBufferZeroWords { pc: u32 },
    #[error("at pc {pc}, hint buffer num_words {num_words} exceeds MAX_HINT_BUFFER_WORDS {max_hint_buffer_words}")]
    HintBufferTooLarge {
        pc: u32,
        num_words: u32,
        max_hint_buffer_words: u32,
    },
    #[error("at pc {pc}, tried to publish into index {public_value_index} when num_public_values = {num_public_values}")]
    PublicValueIndexOutOfBounds {
        pc: u32,
        num_public_values: usize,
        public_value_index: usize,
    },
    #[error("at pc {pc}, tried to publish {new_value} into index {public_value_index} but already had {existing_value}")]
    PublicValueNotEqual {
        pc: u32,
        public_value_index: usize,
        existing_value: usize,
        new_value: usize,
    },
    #[error("at pc {pc}, phantom sub-instruction not found for discriminant {}", .discriminant.0)]
    PhantomNotFound {
        pc: u32,
        discriminant: PhantomDiscriminant,
    },
    #[error("at pc {pc}, discriminant {}, phantom error: {inner}", .discriminant.0)]
    Phantom {
        pc: u32,
        discriminant: PhantomDiscriminant,
        inner: eyre::Error,
    },
    #[error("program must terminate")]
    DidNotTerminate,
    #[error("program exit code {0}")]
    FailedWithExitCode(u32),
    #[error("trace buffer out of bounds: requested {requested} but capacity is {capacity}")]
    TraceBufferOutOfBounds { requested: usize, capacity: usize },
    #[error("instruction counter overflow: {instret} + {num_insns} > u64::MAX")]
    InstretOverflow { instret: u64, num_insns: u64 },
    #[error("inventory error: {0}")]
    Inventory(#[from] ExecutorInventoryError),
    #[error("static program error: {0}")]
    Static(#[from] StaticProgramError),
}

/// Errors in the program that can be statically analyzed before runtime.
#[derive(Error, Debug)]
pub enum StaticProgramError {
    #[error("invalid instruction at pc {0}")]
    InvalidInstruction(u32),
    #[error("Too many executors")]
    TooManyExecutors,
    #[error("at pc {pc}, opcode {opcode} was not enabled")]
    DisabledOperation { pc: u32, opcode: VmOpcode },
    #[error("Executor not found for opcode {opcode}")]
    ExecutorNotFound { opcode: VmOpcode },
    #[error("Failed to create temporary file: {err}")]
    FailToCreateTemporaryFile { err: String },
    #[error("Failed to write into temporary file: {err}")]
    FailToWriteTemporaryFile { err: String },
    #[error("Failed to generate dynamic library: {err}")]
    FailToGenerateDynamicLibrary { err: String },
}

#[cfg(feature = "aot")]
#[derive(Error, Debug)]
pub enum AotError {
    #[error("AOT compilation not supported for this opcode")]
    NotSupported,

    #[error("No executor found for opcode {0}")]
    NoExecutorFound(VmOpcode),

    #[error("Invalid instruction format")]
    InvalidInstruction,

    #[error("Other AOT error: {0}")]
    Other(String),
}

/// Function pointer for interpreter execution with function signature `(pre_compute,
/// arg, exec_state)`. The `pre_compute: *const u8` is a pre-computed buffer of data
/// corresponding to a single instruction. The contents of `pre_compute` are determined from the
/// program code as specified by the [Executor] and [MeteredExecutor] traits.
pub type ExecuteFunc<F, CTX> =
    unsafe fn(pre_compute: *const u8, exec_state: &mut VmExecState<F, GuestMemory, CTX>);

/// Handler for tail call elimination. The `CTX` is assumed to contain pointers to the pre-computed
/// buffer and the function handler table.
///
/// - `pre_compute_buf` is the starting pointer of the pre-computed buffer.
/// - `handlers` is the starting pointer of the table of function pointers of `Handler` type. The
///   pointer is typeless to avoid self-referential types.
#[cfg(feature = "tco")]
pub type Handler<F, CTX> = unsafe fn(
    interpreter: &InterpretedInstance<F, CTX>,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
);

/// Trait for pure execution via a host interpreter. The trait methods provide the methods to
/// pre-process the program code into function pointers which operate on `pre_compute` instruction
/// data.
// @dev: In the codebase this is sometimes referred to as (E1).
pub trait InterpreterExecutor<F> {
    fn pre_compute_size(&self) -> usize;

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait;

    /// Returns a function pointer with tail call optimization. The handler function assumes that
    /// the pre-compute buffer it receives is the populated `data`.
    // NOTE: we could have used `pre_compute` above to populate `data`, but the implementations were
    // simpler to keep `handler` entirely separate from `pre_compute`.
    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait;
}

#[cfg(feature = "aot")]
pub trait AotExecutor<F> {
    fn is_aot_supported(&self, _inst: &Instruction<F>) -> bool {
        false
    }

    /*
    Function: Generate x86 assembly for the given RV32 instruction, and transfer control to the next RV32 instruction

    Preconditions:
    x86 Registers: rbx = vm_exec_state_ptr, rbp = pre_compute_insns_ptr,
    - instruction: the instruction to be executed

    Postcondition:
    - x86's PC should be set to the label of the next RV32 instruction, and transfers control to the next instruction
    */
    fn generate_x86_asm(&self, _inst: &Instruction<F>, _pc: u32) -> Result<String, AotError> {
        unimplemented!()
    }
    // TODO: add air_idx:usize parameter to the function, for AotMeteredExecutor::generate_x86_asm
}
#[cfg(feature = "aot")]
pub trait Executor<F>: InterpreterExecutor<F> + AotExecutor<F> {}
#[cfg(feature = "aot")]
impl<F, T> Executor<F> for T where T: InterpreterExecutor<F> + AotExecutor<F> {}

#[cfg(not(feature = "aot"))]
pub trait Executor<F>: InterpreterExecutor<F> {}
#[cfg(not(feature = "aot"))]
impl<F, T> Executor<F> for T where T: InterpreterExecutor<F> {}

/// Trait for metered execution via a host interpreter. The trait methods provide the methods to
/// pre-process the program code into function pointers which operate on `pre_compute` instruction
/// data which contains auxiliary data (e.g., corresponding AIR ID) for metering purposes.
// @dev: In the codebase this is sometimes referred to as (E2).
pub trait InterpreterMeteredExecutor<F> {
    fn metered_pre_compute_size(&self) -> usize;

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        air_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait;

    /// Returns a function pointer with tail call optimization. The handler function assumes that
    /// the pre-compute buffer it receives is the populated `data`.
    // NOTE: we could have used `metered_pre_compute` above to populate `data`, but the
    // implementations were simpler to keep `metered_handler` entirely separate from
    // `metered_pre_compute`.
    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        air_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait;
}

#[cfg(feature = "aot")]
pub trait AotMeteredExecutor<F> {
    fn is_aot_metered_supported(&self, _inst: &Instruction<F>) -> bool {
        false
    }

    fn generate_x86_metered_asm(
        &self,
        _inst: &Instruction<F>,
        _pc: u32,
        _chip_idx: usize,
        _config: &SystemConfig,
    ) -> Result<String, AotError> {
        unimplemented!()
    }
}

#[cfg(feature = "aot")]
pub trait MeteredExecutor<F>: InterpreterMeteredExecutor<F> + AotMeteredExecutor<F> {}
#[cfg(feature = "aot")]
impl<F, T> MeteredExecutor<F> for T where T: InterpreterMeteredExecutor<F> + AotMeteredExecutor<F> {}

#[cfg(not(feature = "aot"))]
pub trait MeteredExecutor<F>: InterpreterMeteredExecutor<F> {}
#[cfg(not(feature = "aot"))]
impl<F, T> MeteredExecutor<F> for T where T: InterpreterMeteredExecutor<F> {}

/// Trait for preflight execution via a host interpreter. The trait methods allow execution of
/// instructions via enum dispatch within an interpreter. This execution is specialized to record
/// "records" of execution which will be ingested later for trace matrix generation. The records are
/// stored in a record arena, which is provided in the [VmStateMut] argument.
// NOTE: In the codebase this is sometimes referred to as (E3).
pub trait PreflightExecutor<F, RA = MatrixRecordArena<F>> {
    /// Runtime execution of the instruction, if the instruction is owned by the
    /// current instance. May internally store records of this call for later trace generation.
    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError>;

    /// For display purposes. From absolute opcode as `usize`, return the string name of the opcode
    /// if it is a supported opcode by the present executor.
    fn get_opcode_name(&self, opcode: usize) -> String;
}

/// Global VM state accessible during instruction execution.
/// The state is generic in guest memory `MEM` and additional record arena `RA`.
/// The host state is execution context specific.
#[derive(derive_new::new)]
pub struct VmStateMut<'a, F, MEM, RA> {
    pub pc: &'a mut u32,
    /// Extra ISA-specific registers carried in the VM state alongside `pc` (no longer stored in
    /// memory). Empty (`[u32; 0]`) unless the `fp` feature is on; see [`EXTRA_EXEC_REGS`].
    pub extra_regs: &'a mut [u32; EXTRA_EXEC_REGS],
    pub memory: &'a mut MEM,
    pub streams: &'a mut Streams<F>,
    pub rng: &'a mut StdRng,
    pub ctx: &'a mut RA,
    #[cfg(feature = "metrics")]
    pub metrics: &'a mut VmMetrics,
}

/// Wrapper type for metered pre-computed data, which is always an AIR index together with the
/// pre-computed data for pure execution.
#[derive(Clone, AlignedBytesBorrow)]
#[repr(C)]
pub struct E2PreCompute<DATA> {
    pub chip_idx: u32,
    pub data: DATA,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, AlignedBorrow, StructReflection)]
// Note: `Serialize`/`Deserialize`/`Default` are not derived because `serde` and `std` only provide
// those impls for concrete-length arrays, not a generic `const EXTRA_LEN`. `Default` is provided
// manually below; no code (de)serializes an `ExecutionState` directly.
pub struct ExecutionState<T, const EXTRA_LEN: usize = 0> {
    pub pc: T,
    pub timestamp: T,
    /// Extra ISA-specific state registers carried on the execution bus alongside `pc`/`timestamp`.
    /// `EXTRA_LEN == 0` (the default) is a zero-sized array: ISAs that don't use it pay nothing —
    /// no columns, no bus fields, no constraints, byte-identical to upstream. Crush sets
    /// `EXTRA_LEN == 1` and uses `extra_regs[0]` as the frame pointer.
    pub extra_regs: [T; EXTRA_LEN],
}

impl<T: Default, const EXTRA_LEN: usize> Default for ExecutionState<T, EXTRA_LEN> {
    fn default() -> Self {
        Self {
            pc: T::default(),
            timestamp: T::default(),
            extra_regs: std::array::from_fn(|_| T::default()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExecutionBus {
    pub inner: PermutationCheckBus,
}

impl ExecutionBus {
    pub const fn new(index: BusIndex) -> Self {
        Self {
            inner: PermutationCheckBus::new(index),
        }
    }

    #[inline(always)]
    pub fn index(&self) -> BusIndex {
        self.inner.index
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ExecutionBridge {
    execution_bus: ExecutionBus,
    program_bus: ProgramBus,
}

pub struct ExecutionBridgeInteractor<AB: InteractionBuilder, const EXTRA_LEN: usize = 0> {
    execution_bus: ExecutionBus,
    program_bus: ProgramBus,
    opcode: AB::Expr,
    operands: Vec<AB::Expr>,
    from_state: ExecutionState<AB::Expr, EXTRA_LEN>,
    to_state: ExecutionState<AB::Expr, EXTRA_LEN>,
}

pub enum PcIncOrSet<T> {
    Inc(T),
    Set(T),
}

impl<T> ExecutionState<T, 0> {
    /// Construct the default (no extra registers) execution state. Kept 2-arg so existing
    /// call sites (`ExecutionState::new(pc, timestamp)`) are unchanged.
    pub fn new(pc: impl Into<T>, timestamp: impl Into<T>) -> Self {
        Self {
            pc: pc.into(),
            timestamp: timestamp.into(),
            extra_regs: [],
        }
    }
}

impl<T, const EXTRA_LEN: usize> ExecutionState<T, EXTRA_LEN> {
    pub fn from_parts(
        pc: impl Into<T>,
        timestamp: impl Into<T>,
        extra_regs: [T; EXTRA_LEN],
    ) -> Self {
        Self {
            pc: pc.into(),
            timestamp: timestamp.into(),
            extra_regs,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_iter<I: Iterator<Item = T>>(iter: &mut I) -> Self {
        let mut next = || iter.next().unwrap();
        let pc = next();
        let timestamp = next();
        let extra_regs = std::array::from_fn(|_| next());
        Self {
            pc,
            timestamp,
            extra_regs,
        }
    }

    /// Field order on the execution bus: `[pc, timestamp, extra_regs..]`.
    pub fn flatten(self) -> impl Iterator<Item = T> {
        [self.pc, self.timestamp].into_iter().chain(self.extra_regs)
    }

    pub fn get_width() -> usize {
        2 + EXTRA_LEN
    }

    pub fn map<U, F: Fn(T) -> U>(self, function: F) -> ExecutionState<U, EXTRA_LEN> {
        ExecutionState {
            pc: function(self.pc),
            timestamp: function(self.timestamp),
            extra_regs: self.extra_regs.map(function),
        }
    }
}

impl ExecutionBus {
    /// Caller must constrain that `enabled` is boolean.
    pub fn execute_and_increment_pc<AB: InteractionBuilder, const EXTRA_LEN: usize>(
        &self,
        builder: &mut AB,
        enabled: impl Into<AB::Expr>,
        prev_state: ExecutionState<AB::Expr, EXTRA_LEN>,
        timestamp_change: impl Into<AB::Expr>,
    ) {
        let next_state = ExecutionState {
            pc: prev_state.pc.clone() + AB::F::ONE,
            timestamp: prev_state.timestamp.clone() + timestamp_change.into(),
            // extra registers are unchanged by the default pc increment.
            extra_regs: prev_state.extra_regs.clone(),
        };
        self.execute(builder, enabled, prev_state, next_state);
    }

    /// Caller must constrain that `enabled` is boolean.
    pub fn execute<AB: InteractionBuilder, const EXTRA_LEN: usize>(
        &self,
        builder: &mut AB,
        enabled: impl Into<AB::Expr>,
        prev_state: ExecutionState<impl Into<AB::Expr>, EXTRA_LEN>,
        next_state: ExecutionState<impl Into<AB::Expr>, EXTRA_LEN>,
    ) {
        let enabled = enabled.into();
        self.inner.receive(
            builder,
            prev_state.flatten().map(Into::into),
            enabled.clone(),
        );
        self.inner
            .send(builder, next_state.flatten().map(Into::into), enabled);
    }
}

impl ExecutionBridge {
    pub fn new(execution_bus: ExecutionBus, program_bus: ProgramBus) -> Self {
        Self {
            execution_bus,
            program_bus,
        }
    }

    /// If `to_pc` is `Some`, then `pc_inc` is ignored and the `to_state` uses `to_pc`. Otherwise
    /// `to_pc = from_pc + pc_inc`.
    pub fn execute_and_increment_or_set_pc<AB: InteractionBuilder, const EXTRA_LEN: usize>(
        &self,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = impl Into<AB::Expr>>,
        from_state: ExecutionState<impl Into<AB::Expr> + Clone, EXTRA_LEN>,
        timestamp_change: impl Into<AB::Expr>,
        pc_kind: impl Into<PcIncOrSet<AB::Expr>>,
    ) -> ExecutionBridgeInteractor<AB, EXTRA_LEN> {
        let from_state = from_state.map(Into::into);
        let to_state = ExecutionState {
            pc: match pc_kind.into() {
                PcIncOrSet::Set(to_pc) => to_pc,
                PcIncOrSet::Inc(pc_inc) => from_state.pc.clone() + pc_inc,
            },
            timestamp: from_state.timestamp.clone() + timestamp_change.into(),
            // extra registers are kept by default; instructions that change them (e.g. CALL
            // updating fp) use `execute` with an explicit `to_state`.
            extra_regs: from_state.extra_regs.clone(),
        };
        self.execute(opcode, operands, from_state, to_state)
    }

    pub fn execute_and_increment_pc<AB: InteractionBuilder, const EXTRA_LEN: usize>(
        &self,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = impl Into<AB::Expr>>,
        from_state: ExecutionState<impl Into<AB::Expr> + Clone, EXTRA_LEN>,
        timestamp_change: impl Into<AB::Expr>,
    ) -> ExecutionBridgeInteractor<AB, EXTRA_LEN> {
        let from_state = from_state.map(Into::into);
        let to_state = ExecutionState {
            pc: from_state.pc.clone() + AB::Expr::from_u32(DEFAULT_PC_STEP),
            timestamp: from_state.timestamp.clone() + timestamp_change.into(),
            extra_regs: from_state.extra_regs.clone(),
        };
        self.execute(opcode, operands, from_state, to_state)
    }

    pub fn execute<AB: InteractionBuilder, const EXTRA_LEN: usize>(
        &self,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = impl Into<AB::Expr>>,
        from_state: ExecutionState<impl Into<AB::Expr>, EXTRA_LEN>,
        to_state: ExecutionState<impl Into<AB::Expr>, EXTRA_LEN>,
    ) -> ExecutionBridgeInteractor<AB, EXTRA_LEN> {
        ExecutionBridgeInteractor {
            execution_bus: self.execution_bus,
            program_bus: self.program_bus,
            opcode: opcode.into(),
            operands: operands.into_iter().map(Into::into).collect(),
            from_state: from_state.map(Into::into),
            to_state: to_state.map(Into::into),
        }
    }
}

impl<AB: InteractionBuilder, const EXTRA_LEN: usize> ExecutionBridgeInteractor<AB, EXTRA_LEN> {
    /// Caller must constrain that `enabled` is boolean.
    pub fn eval(self, builder: &mut AB, enabled: impl Into<AB::Expr>) {
        let enabled = enabled.into();

        // Interaction with program
        self.program_bus.lookup_instruction(
            builder,
            self.from_state.pc.clone(),
            self.opcode,
            self.operands,
            enabled.clone(),
        );

        self.execution_bus
            .execute(builder, enabled, self.from_state, self.to_state);
    }
}

impl<T: PrimeCharacteristicRing> From<(u32, Option<T>)> for PcIncOrSet<T> {
    fn from((pc_inc, to_pc): (u32, Option<T>)) -> Self {
        match to_pc {
            None => PcIncOrSet::Inc(T::from_u32(pc_inc)),
            Some(to_pc) => PcIncOrSet::Set(to_pc),
        }
    }
}

/// Phantom sub-instructions affect the runtime of the VM and the trace matrix values.
/// However they all have no AIR constraints besides advancing the pc by
/// [DEFAULT_PC_STEP].
///
/// They should not mutate memory, but they can mutate the input & hint streams.
///
/// Phantom sub-instructions are only allowed to use operands
/// `a,b` and `c_upper = c.as_canonical_u32() >> 16`.
#[allow(clippy::too_many_arguments)]
pub trait PhantomSubExecutor<F>: Send + Sync {
    fn phantom_execute(
        &self,
        memory: &GuestMemory,
        streams: &mut Streams<F>,
        rng: &mut StdRng,
        extra_regs: [u32; EXTRA_EXEC_REGS],
        discriminant: PhantomDiscriminant,
        a: u32,
        b: u32,
        c_upper: u16,
    ) -> eyre::Result<()>;
}
