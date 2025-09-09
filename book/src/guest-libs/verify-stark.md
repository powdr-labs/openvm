# Verify STARK
A library to facilitate the recursive verification of OpenVM STARK proofs within a guest program. The library includes helpers for both the guest to verify the proofs and for the host to supply them.

You can find an example of using the library for recursive verification [here](https://github.com/openvm-org/openvm/blob/main/guest-libs/verify_stark/tests/integration_test.rs).

## Guest 

For guest programs, the library provides the `define_verify_stark_proof!` macro which can generate a function to verify an OpenVM STARK proof, directly usable within the guest. Users will need to specify:
- A function name.
- An ASM file containing the instructions to verify OpenVM STARK proofs. More specifically, the ASM must verify proofs outputted by the CLI's `cargo openvm prove stark` (or `prove_e2e_stark` from the SDK). The verification logic itself is tied to an aggregation config  (so the ASM is reusable across OpenVM STARKs with different app VM configs) and therefore, can be generated during aggregation keygen.
  - The SDK provides the helper `generate_root_verifier_asm` to generate the ASM for a given aggregation config.
  - Since OpenVM maintains verifier compatibility across patch releases, the same ASM is also reusable with STARKs generated across all `1.x.*` for some `x`.

The macro will output a function with the following interface:

```rust
fn verify_stark(app_exe_commit: &[u32; 8], app_vm_commit: &[u32; 8], user_pvs: &[u8])
```

where 

- `verify_stark` is the user-supplied name of the function.
- `app_exe_commit` is the commitment to the OpenVM application executable whose execution is being verified.
- `app_vm_commit` is the commitment to the app VM configuration.
- `user_pvs` are the public values revealed by the app.

Proofs are expected to be passed via the [hintable key-value store](https://github.com/openvm-org/openvm/blob/main/docs/specs/ISA.md#inputs-and-hints). Guests will query for proofs at the key `asm_filename || exe_commit_in_u32 || vm_commit_in_u32 || user_pvs`.

The function will panic on failure. Successful STARK verification implies that the app execution was successful and terminated with exit code 0.

> ⚠️ For Advanced Users
>
> Note that if your guest program directly writes data to the native address space (address space 4), the `verify_stark` function will likely overwrite it. Any data the guest placed in the native address space should be persisted (or treated as corrupt) before invocations to `verify_stark`.
> 
> This is not a concern if you are writing vanilla rust with the default RV32 I and M extensions.

## Host

Hosts are expected to properly hint the [hintable key-value store](https://github.com/openvm-org/openvm/blob/main/docs/specs/ISA.md#inputs-and-hints) with the OpenVM STARK proofs. To that end, the library provides the following utilities:
- `compute_hint_key_for_verify_openvm_stark` will compute the exact key at which the guest will look for the proof.
- `encode_proof_to_kv_store_value` will serialize the proof into the structure expected by the `verify_stark` function.

