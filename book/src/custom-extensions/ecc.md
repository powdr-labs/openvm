# Elliptic Curve Cryptography

The OpenVM Elliptic Curve Cryptography Extension provides support for elliptic curve operations through the `openvm-ecc-guest` crate.

Developers can enable arbitrary Weierstrass curves by configuring this extension with the modulus for the coordinate field and the coefficients in the curve equation. Preset configurations for the secp256k1 and secp256r1 curves are provided through the [K256](../guest-libs/k256.md) and [P256](../guest-libs/p256.md) guest libraries.

## Available traits and methods

- `Group` trait:
  This represents an element of a [group](<https://en.wikipedia.org/wiki/Group_(mathematics)>) where the operation is addition. Therefore the trait includes functions for `add`, `sub`, and `double`.

  - `IDENTITY` is the identity element of the group.

- `CyclicGroup` trait:
  It's a group that has a generator, so it defines `GENERATOR` and `NEG_GENERATOR`.

- `WeierstrassPoint` trait:
  It represents an affine point on a Weierstrass elliptic curve and it extends `Group`.

  - `Coordinate` type is the type of the coordinates of the point, and it implements `IntMod`.
  - `x()`, `y()` are used to get the affine coordinates
  - `from_xy` is a constructor for the point, which checks if the point is either identity or on the affine curve.
  - The point supports elliptic curve operations through intrinsic functions `add_ne_nonidentity` and `double_nonidentity`.
  - `decompress`: Sometimes an elliptic curve point is compressed and represented by its `x` coordinate and the odd/even parity of the `y` coordinate. `decompress` is used to decompress the point back to `(x, y)`.

- `msm`: for multi-scalar multiplication.

- `ecdsa`: for doing ECDSA signature verification and public key recovery from signature.

## Macros

For elliptic curve cryptography, the `openvm-ecc-guest` crate provides macros similar to those in [`openvm-algebra-guest`](./algebra.md):

1. **Declare**: Use `sw_declare!` to define elliptic curves over the previously declared moduli. For example:

```rust
sw_declare! {
    Bls12_381G1Affine { mod_type = Bls12_381Fp, b = BLS12_381_B },
    P256Affine { mod_type = P256Coord, a = P256_A, b = P256_B },
}
```

Each declared curve must specify the `mod_type` (implementing `IntMod`) and a constant `b` for the Weierstrass curve equation \\(y^2 = x^3 + ax + b\\). `a` is optional and defaults to 0 for short Weierstrass curves.
This creates `Bls12_381G1Affine` and `P256Affine` structs which implement the `Group` and `WeierstrassPoint` traits. The underlying memory layout of the structs uses the memory layout of the `Bls12_381Fp` and `P256Coord` structs, respectively.

2. **Init**: Called once, the [`openvm::init!` macro](./overview.md#automating-the-init-step) produces a call to `sw_init!` that enumerates these curves and allows the compiler to produce optimized instructions:

```rust
openvm::init!();
/* This expands to
sw_init! {
    Bls12_381G1Affine, P256Affine,
}
*/
```

**Summary**:

- `sw_declare!`: Declares elliptic curve structures.
- `init!`: Initializes them once, linking them to the underlying moduli.

To use elliptic curve operations on a struct defined with `sw_declare!`, it is expected that the struct for the curve's coordinate field was defined using `moduli_declare!`. In particular, the coordinate field needs to be initialized and set up as described in the [algebra extension](./algebra.md) chapter.

For the basic operations provided by the `WeierstrassPoint` trait, the scalar field is not needed. For the ECDSA functions in the `ecdsa` module, the scalar field must also be declared, initialized, and set up.

## ECDSA

The ECC extension supports ECDSA signature verification on any elliptic curve, and pre-defined implementations are provided for the secp256k1 and secp256r1 curves.
To verify an ECDSA signature, first call the `VerifyingKey::recover_from_prehash_noverify` associated function to recover the verifying key, then call the `VerifyingKey::verify_prehashed` method on the recovered verifying key.
