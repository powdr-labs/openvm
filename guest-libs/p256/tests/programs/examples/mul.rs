#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use elliptic_curve::{CurveArithmetic, Group, PrimeField};
use openvm_p256::NistP256;
// clippy thinks this is unused, but it's used in the init! macro
#[allow(unused)]
use openvm_p256::P256Point;

openvm::init!("openvm_init_mul.rs");

openvm::entry!(main);

mod test_vectors;
use test_vectors::{ADD_TEST_VECTORS, MUL_TEST_VECTORS};

// Taken from https://github.com/RustCrypto/elliptic-curves/blob/master/primeorder/src/dev.rs
pub fn main() {
    let generator = <NistP256 as CurveArithmetic>::ProjectivePoint::generator();

    for (k, coords) in ADD_TEST_VECTORS
        .iter()
        .enumerate()
        .map(|(k, coords)| {
            (
                <NistP256 as CurveArithmetic>::Scalar::from(k as u64 + 1),
                *coords,
            )
        })
        .chain(MUL_TEST_VECTORS.iter().cloned().map(|(k, x, y)| {
            (
                <NistP256 as CurveArithmetic>::Scalar::from_repr(k.into()).unwrap(),
                (x, y),
            )
        }))
    {
        let p = generator * k;
        assert_eq!(p.x_be_bytes(), coords.0);
        assert_eq!(p.y_be_bytes(), coords.1);
    }
}
