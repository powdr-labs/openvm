# RISC-V ELF Transpilation to OpenVM Executable

The OpenVM framework supports transpilation of a RISC-V ELF consisting of the RV32IM instruction set as well as [custom RISC-V instructions](./RISCV.md) specified by VM extensions into an OpenVM executable.

## Transpiler Framework

The transpiler is a function that converts a [RISC-V ELF](https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-elf.adoc) into an OpenVM executable, where an **OpenVM executable** is defined as the following pieces of data:

- Program ROM
- Starting program counter `pc_0`
- Initial data memory

The OpenVM executable forms a part of the [initial VM state](./ISA.md#virtual-machine-state).

We define a RISC-V **machine code block** to be a 32-bit aligned contiguous sequence of bits in the RISC-V program memory, where the bit length is variable and a multiple of 32. The code block _may_ contain instructions from standard or non-standard RISC-V ISA extensions, but it may also contain arbitrary bits.

The transpiler is configured upon construction with the set of VM extensions to support. In order to be supported by the transpiler, a VM extension must specify a set of RISC-V machine code blocks and rules for mapping each code block to a sequences of _potentially multiple_ [OpenVM instructions](./ISA.md#openvm-instruction-set).

The transpilation rules must satisfy:

- A read or write to the RISC-V program counter corresponds to a read or write to the program counter of the same value in OpenVM. This includes the implicit read of the program counter to fetch the instruction from program code, as well as any implicit `pc += 4` advancement in some RISC-V instructions. In transpilations where a single RISC-V code block is mapped to multiple OpenVM instructions (e.g., [Kernel Code](#openvm-kernel-code-transpilation)), the intermediate OpenVM instructions **may** change the value of the program counter to a program address that is not the start of a RISC-V instruction. It is required that at the end of the RISC-V code block, the program counter is set to the start of a valid RISC-V instruction in the RISC-V machine code.
- A RISC-V 32-bit register `x{i}` read or write access corresponds to an OpenVM memory access at `[4 * i: 4]_1` **except** for writes to `x0`, see [below](#register-x0-handling). The 32-bits of `x{i}` are represented as 4 little-endian bytes in OpenVM memory.
  - A RISC-V code block must **never** map to any OpenVM instruction that changes the value of `[0:4]_1` in OpenVM memory.
- A RISC-V 32-bit user memory access of the `j`th byte in word `i` corresponds to an OpenVM memory access at `[4 * i + j]_2`.
- If the RISC-V code block is a standard instruction from the [RISC-V Instruction Set Manual Volume I: Unprivileged ISA](https://lf-riscv.atlassian.net/wiki/spaces/HOME/pages/16154769/RISC-V+Technical+Specifications) ([pdf](https://drive.google.com/file/d/1uviu1nH-tScFfgrovvFCrj7Omv8tFtkp/view)), then the transpilation rule must map the RISC-V instruction to an OpenVM instruction that follows the RISC-V specification after applying the above correspondences to register and memory accesses.

The above requirements, together with the invariants of the OpenVM ISA, imply that transpilation will only be valid for programs where:

- The program code does not have program address greater than or equal to `2^PC_BITS`.
- The program does not access memory outside the range `[0, 2^addr_max_bits)`: programs that attempt such accesses will fail to execute.

A transpiler configuration is only considered valid if there are no two transpilation rules that may map the same RISC-V code block to different OpenVM instructions.

- When defining a new VM extension with transpiler support, the associated RISC-V code blocks should be chosen to avoid conflicts with RISC-V code blocks from other pre-existing VM extensions that the new VM extension expects to be compatible with.

### Register `x0` Handling

As specified in Section 2.1 of [RISC-V Instruction Set Manual Volume I: Unprivileged ISA](https://lf-riscv.atlassian.net/wiki/spaces/HOME/pages/16154769/RISC-V+Technical+Specifications) ([pdf](https://drive.google.com/file/d/1uviu1nH-tScFfgrovvFCrj7Omv8tFtkp/view)), register `x0` is hardwired to zero and must **never** be written to.

The OpenVM ISA treats `[0:4]_1` as normal read/write memory and makes no guarantees on memory accesses to this location. The transpiler must **never** transpile a RISC-V code block to any OpenVM instruction that changes the value of `[0:4]_1` in OpenVM memory. For compatibility with the RISC-V ISA, the transpiler must always transpile a RISC-V instruction to an OpenVM instruction that matches the RISC-V specification. In particular, any RISC-V instruction that has `rd=x0` must be transpiled to either the `NOP` OpenVM instruction if it has no side effects or to an OpenVM instruction that executes the expected side effect and does not change the value of `[0:4]_1`.

## Transpiler Specification for Default VM Extensions

This section specifies the behavior of the transpiler for the default VM extensions with the custom RISC-V instructions specified [here](./RISCV.md). We use the following notation:

- Let `ind(rd)` denote `4 * (register index)`, which is in `0..128`. In particular, it fits in one field element.
- We use `itof` for the function that takes 12-bits (or 21-bits in case of J-type) to a signed integer and then mapping to the corresponding field element. So `0b11…11` goes to `-1` in `F`.
- We use `sign_extend_24` to convert a 12-bit integer into a 24-bit integer via sign extension. We use this in conjunction with `utof`, which converts 24 bits into an unsigned integer and then maps it to the corresponding field element. Note that each 24-bit unsigned integer fits in one field element.
- We use `sign_extend_16` for the analogous conversion into a 16-bit integer via sign extension.
- We use `zero_extend_24` to convert an unsigned integer with at most 24 bits into a 24-bit unsigned integer by zero extension. This is used in conjunction with `utof` to convert unsigned integers to field elements.
- We use `sign_of(imm)` to get the sign bit of the immediate `imm`.
- The notation `imm[0:4]` means the lowest 5 bits of the immediate.
- For a phantom instruction `ins`, `disc(ins)` is the discriminant specified in the [ISA specification](./ISA.md#system-instructions).
- For a phantom instruction `ins` and a 16-bit `c_upper`, `phantom_c(c_upper, ins) = c_upper << 16 | disc(ins)` is the corresponding 32-bit operand `c` for PHANTOM.

We now specify the transpilation for system instructions and the default set of VM extensions.

## System Instructions

| RISC-V Inst | OpenVM Instruction          |
| ----------- | --------------------------- |
| terminate   | TERMINATE `_, _, utof(imm)` |

## RV32IM Extension

Transpilation from RV32IM to OpenVM assembly follows the mapping below, which is generally
a 1-1 translation between RV32IM instructions and OpenVM instructions. The main exception relates
to handling of the `x0` register, which discards writes and has value `0` in all reads.
We handle writes to `x0` in transpilation as follows:

- Instructions that write to `x0` with no side effects are transpiled to the PHANTOM instruction with `c = 0x00` (`Nop`).
- Instructions that write to a register which might be `x0` with side effects (JAL, JALR) are transpiled to the corresponding custom instruction whose write behavior is controlled by a flag specifying whether the target register is `x0`.

Because `[0:4]_1` is initialized to `0` and never written to, this guarantees that reads from `x0` yield `0` and enforces that any OpenVM program transpiled from RV32IM conforms to the RV32IM specification for `x0`.

### System Level Extensions to RV32IM

| RISC-V Inst | OpenVM Instruction                                               |
| ----------- | ---------------------------------------------------------------- |
| hintstorew  | HINT_STOREW_RV32 `0, ind(rd), _, 1, 2`                           |
| hintbuffer  | HINT_BUFFER_RV32 `ind(rs1), ind(rd), _, 1, 2`                    |
| reveal      | STOREW_RV32 `ind(rs1), ind(rd), utof(sign_extend_16(imm)), 1, 3, 1, sign_of(imm)` |
| hintinput   | PHANTOM `_, _, disc(Rv32HintInput)`                              |
| printstr    | PHANTOM `ind(rd), ind(rs1), disc(Rv32PrintStr)`                  |
| hintrandom  | PHANTOM `ind(rd), _, disc(Rv32HintRandom)`                       |

### Standard RV32IM Instructions

| RISC-V Inst | OpenVM Instruction                                                                                                          |
| ----------- | --------------------------------------------------------------------------------------------------------------------------- |
| add         | ADD_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                             |
| sub         | SUB_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                             |
| xor         | XOR_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                             |
| or          | OR_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                              |
| and         | AND_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                             |
| sll         | SLL_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                             |
| srl         | SRL_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                             |
| sra         | SRA_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                             |
| slt         | SLT_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                             |
| sltu        | SLTU_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                            |
| addi        | ADD_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`            |
| xori        | XOR_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`            |
| ori         | OR_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`             |
| andi        | AND_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`            |
| slli        | SLL_RV32 `ind(rd), ind(rs1), utof(zero_extend_24(imm[0:4])), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`       |
| srli        | SRL_RV32 `ind(rd), ind(rs1), utof(zero_extend_24(imm[0:4])), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`       |
| srai        | SRA_RV32 `ind(rd), ind(rs1), utof(zero_extend_24(imm[0:4])), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`       |
| slti        | SLT_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`            |
| sltiu       | SLTU_RV32 `ind(rd), ind(rs1), utof(sign_extend_24(imm)), 1, 0` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`           |
| lb          | LOADB_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2, (rd != x0), sign_of(imm)`                                   |
| lh          | LOADH_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2, (rd != x0), sign_of(imm)`                                   |
| lw          | LOADW_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2, (rd != x0), sign_of(imm)`                                   |
| lbu         | LOADBU_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2, (rd != x0), sign_of(imm)`                                  |
| lhu         | LOADHU_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 2, (rd != x0), sign_of(imm)`                                  |
| sb          | STOREB_RV32 `ind(rs2), ind(rs1), utof(sign_extend_16(imm)), 1, 2, 1, sign_of(imm)`                                          |
| sh          | STOREH_RV32 `ind(rs2), ind(rs1), utof(sign_extend_16(imm)), 1, 2, 1, sign_of(imm)`                                          |
| sw          | STOREW_RV32 `ind(rs2), ind(rs1), utof(sign_extend_16(imm)), 1, 2, 1, sign_of(imm)`                                          |
| beq         | BEQ_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                                                                              |
| bne         | BNE_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                                                                              |
| blt         | BLT_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                                                                              |
| bge         | BGE_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                                                                              |
| bltu        | BLTU_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                                                                             |
| bgeu        | BGEU_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 1`                                                                             |
| jal         | JAL_RV32 `ind(rd), 0, itof(imm), 1, 0, (rd != x0)`                                                                          |
| jalr        | JALR_RV32 `ind(rd), ind(rs1), utof(sign_extend_16(imm)), 1, 0, (rd != x0), sign_of(imm)`                                    |
| lui         | LUI_RV32 `ind(rd), 0, utof(zero_extend_24(imm[12:31])), 1, 0, 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`         |
| auipc       | AUIPC_RV32 `ind(rd), 0, utof(zero_extend_24(imm[12:31]) << 4), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`        |
| mul         | MUL_RV32 `ind(rd), ind(rs1), ind(rs2), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                |
| mulh        | MULH_RV32 `ind(rd), ind(rs1), ind(rs2), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                               |
| mulhsu      | MULHSU_RV32 `ind(rd), ind(rs1), ind(rs2), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                             |
| mulhu       | MULHU_RV32 `ind(rd), ind(rs1), ind(rs2), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                              |
| div         | DIV_RV32 `ind(rd), ind(rs1), ind(rs2), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                |
| divu        | DIVU_RV32 `ind(rd), ind(rs1), ind(rs2), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                               |
| rem         | REM_RV32 `ind(rd), ind(rs1), ind(rs2), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                |
| remu        | REMU_RV32 `ind(rd), ind(rs1), ind(rs2), 1` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                               |

## OpenVM Intrinsic VM Extensions

The following sections specify the transpilation of the default set of intrinsic extensions
to OpenVM. In order to preserve correctness of handling of `x0`, the transpilation must respect
the constraint that any instruction that writes to a register must:

- Transpile to `Nop` if the register is `x0` and there are no side effects.
- Transpile to an OpenVM assembly instruction that does not write to `[0:4]_1` and processes side effects if the register is `x0` and there are side effects.

Each VM extension's behavior is specified below.

### Keccak Extension

| RISC-V Inst | OpenVM Instruction                                 |
| ----------- | -------------------------------------------------- |
| keccak256   | KECCAK256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2` |

### SHA2-256 Extension

| RISC-V Inst | OpenVM Instruction                              |
| ----------- | ----------------------------------------------- |
| sha256      | SHA256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2` |

### BigInt Extension

| RISC-V Inst | OpenVM Instruction                                |
| ----------- | ------------------------------------------------- |
| add256      | ADD256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| sub256      | SUB256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| xor256      | XOR256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| or256       | OR256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`    |
| and256      | AND256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| sll256      | SLL256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| srl256      | SRL256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| sra256      | SRA256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| slt256      | SLT256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| sltu256     | SLTU256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`  |
| mul256      | MUL256_RV32 `ind(rd), ind(rs1), ind(rs2), 1, 2`   |
| beq256      | BEQ256_RV32 `ind(rs1), ind(rs2), itof(imm), 1, 2` |

### Algebra Extension

#### Modular Arithmetic

| RISC-V Inst  | OpenVM Instruction                                                                                                                                                                                                            |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| addmod\<N\>  | ADDMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                                                                                          |
| submod\<N\>  | SUBMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                                                                                          |
| mulmod\<N\>  | MULMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                                                                                          |
| divmod\<N\>  | DIVMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                                                                                          |
| iseqmod\<N\> | ISEQMOD_RV32\<N\> `ind(rd), ind(rs1), ind(rs2), 1, 2` if `rd != x0`, otherwise PHANTOM `_, _, disc(Nop)`                                                                                                                      |
| setup\<N\>   | SETUP_ADDSUBMOD_RV32\<N\> `ind(rd), ind(rs1), x0, 1, 2` if `ind(rs2) = 0`, SETUP_MULDIVMOD_RV32\<N\> `ind(rd), ind(rs1), x0, 1, 2` if `ind(rs2) = 1`, SETUP_ISEQMOD_RV32\<N\> `ind(rd), ind(rs1), x0, 1, 2` if `ind(rs2) = 2` |
| hint_non_qr  | PHANTOM `0, 0, phantom_c(curve_idx, HintNonQr)`                                                                                                |
| hint_sqrt    | PHANTOM `ind(rs1), 0, phantom_c(curve_idx, HintSqrt)`                                                                                                |

#### Complex Extension Field Arithmetic

| RISC-V Inst  | OpenVM Instruction                                                                                                                                 |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| addcomplex   | ADD\<Fp2\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                     |
| subcomplex   | SUB\<Fp2\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                     |
| mulcomplex   | MUL\<Fp2\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                     |
| divcomplex   | DIV\<Fp2\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                     |
| setupcomplex | SETUP_ADDSUB_RV32\<Fp2\> `ind(rd), ind(rs1), x0, 1, 2` if `ind(rs2) = 0`, SETUP_MULDIV_RV32\<Fp2\> `ind(rd), ind(rs1), x0, 1, 2` if `ind(rs2) = 1` |

### Elliptic Curve Extension

| RISC-V Inst     | OpenVM Instruction                                                                                                                                                |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| sw_add_ne\<C\>  | EC_ADD_NE_RV32\<C\> `ind(rd), ind(rs1), ind(rs2), 1, 2`                                                                                                           |
| sw_double\<C\>  | EC_DOUBLE_RV32\<C\> `ind(rd), ind(rs1), 0, 1, 2`                                                                                                                  |
| setup\<C\>      | SETUP_EC_ADD_NE_RV32\<C\> `ind(rd), ind(rs1), ind(rs2), 1, 2` if `ind(rs2) != 0`, SETUP_EC_DOUBLE_RV32\<C\> `ind(rd), ind(rs1), ind(rs2), 1, 2` if `ind(rs2) = 0` |

### Pairing Extension

| RISC-V Inst                | OpenVM Instruction                                                       |
| -------------------------- | ------------------------------------------------------------------------ |
| hint_final_exp             | PHANTOM `ind(rs1), ind(rs2), phantom_c(pairing_idx, HintFinalExp)`       |

## OpenVM Kernel Code Transpilation

This section specifies the transpilation of custom RISC-V [kernel code](./RISCV.md#classification-of-custom-risc-v-machine-code) to OpenVM instructions.
This transpilation differs from the ones described above in that a custom RISC-V code block of more than 32-bits is used to specify a single OpenVM instruction,
and a single 32-bit RISC-V instruction is also used to specify multiple (nonexistent) instructions.

We use the following 32-bit RISC-V code blocks in conjunction with other arbitrary code blocks to transpile custom RISC-V kernel code to OpenVM instructions.

We have 3 special 32-bit code blocks:

| Abbr.       | 32-bit Code | Name                                                   |
| ----------- | ----------- | ------------------------------------------------------ |
| lfii        | 0b00000000000000000111000000001011 | Long Form Instruction Indicator |
| gi          | 0b00000010000000000111000000001011 | Gap Indicator                   |
| vri         | 0b10000000000000000000000001110100 | Variable Register Indicator     |

Note that the `vri` code block does not conform to RISC-V instruction naming conventions and is only used after `lfii` as described below. The `vri` code is the 32-bit big-endian encoding of `2^31 + 116`.

### Overview

We specify a format in which an arbitrary sequence of OpenVM instructions can be serialized into a 32-bit aligned code block
which can be inserted into the RISC-V ELF. The transpiler is then able to recognize this code block and transpile it (effectively deserializing) back into the original OpenVM instructions.

To do this, suppose we have a sequence of OpenVM instructions `[i_1, ..., i_l]`. These will be serialized into the
concatenation of the code blocks:

```
lfii [i_1 encoding]
lfii [i_2 encoding]
...
lfii [i_l encoding]
gi [gap encoding]
```

This will be an overall code block of `32 * m` bits, where `m > l`, which will be transpiled to only `l` OpenVM instructions. The instructions are encoded as described [below](#openvm-instruction-encoding).
If the starting program address of the RISC-V code block is `a` (in bytes), then the OpenVM instructions `i_1, ..., i_l` will be at addresses `[a, a + 4, ..., a + 4 * (l - 1)]` in the OpenVM program ROM. The addresses `[a + 4 * l, ..., a + 4 * (m - 1)]` will be left empty in the OpenVM program ROM to maintain compatibility with RISC-V program addresses. The _gap_ between `l` and `m` is encoded by the [gap encoding](#gap-encoding).

### OpenVM Instruction Encoding

An OpenVM instruction is encoded to a RISC-V code block as follows.
We identify the 31-bit field `F` with `{0, ..., p - 1}` where `p` is the prime modulus.
We encode `u32` as 32-bits in little-endian format.

Let the instruction be `opcode operand_1 operand_2 ... operand_n` where each opcode and operand is a field element.
Then to encode it into a 32-bit aligned code block, we first write `lfii`, followed by the number of operands `n` (as `u32`), followed by `opcode` (as `u32`).
We then encode each operand simply by its canonical 32-bit representation.

### Gap Encoding

The transpiler also allows for the transpilation of gaps, i.e., addresses in the RISC-V program memory that do not map to OpenVM instructions.
The purpose of this is to maintain the validity of `pc` offsets when using the above encoding of OpenVM instructions.

A gap is encoded by first writing `gi`, then the number of instructions to be skipped (i.e. the length of the gap) as `u32`.
Note that the number of instructions to be skipped is not the same as the number of bytes to be skipped --
on a 32-bit architecture, these will differ by a factor of 4.

The gap code block is used to signify the end of a block of kernel code.

### Kernel Code Assumptions

The above transpilation procedure allows the insertion of arbitrary serialized OpenVM instructions into the RISC-V ELF as kernel code. To maintain the guarantees of the transpiler framework, the kernel code must satisfy the following safety assumptions:

- All code exiting the code block must jump to a valid RISC-V instruction in the machine code
- Code from outside of the kernel code block must not jump into the middle of a kernel code block
- Kernel code must not write to `[0:4]_1` in OpenVM memory
- Kernel code must only write bytes to address space `2` in OpenVM memory
