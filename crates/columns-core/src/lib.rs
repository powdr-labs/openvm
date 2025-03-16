use p3_keccak_air::KeccakCols;
use p3_poseidon2_air::Poseidon2Cols;

pub trait FlattenFieldsHelper {
    fn flatten_fields() -> Option<Vec<String>>;
}

//Implement FlattenFieldsHelper for arrays of any size
impl<T: FlattenFieldsHelper, const N: usize> FlattenFieldsHelper for [T; N] {
    fn flatten_fields() -> Option<Vec<String>> {
        let mut fields = Vec::new();
        for i in 0..N {
            for field in T::flatten_fields()? {
                fields.push(format!("{i}__{field}"));
            }
        }
        Some(fields)
    }
}

impl<T> FlattenFieldsHelper for KeccakCols<T> {
    fn flatten_fields() -> Option<Vec<String>> {
        let mut fields = Vec::new();

        // Add fields based on the known structure
        // step_flags array
        for i in 0..24 {
            // NUM_ROUNDS is 24
            fields.push(format!("step_flags__{}", i));
        }

        // export field
        fields.push("export".to_string());

        // preimage (3D array)
        for y in 0..5 {
            for x in 0..5 {
                for limb in 0..8 {
                    // U64_LIMBS is 8
                    fields.push(format!("preimage__{}_{}__{}", y, x, limb));
                }
            }
        }

        // 'a' field (3D array)
        for y in 0..5 {
            for x in 0..5 {
                for limb in 0..8 {
                    fields.push(format!("a__{}_{}__{}", y, x, limb));
                }
            }
        }

        // 'c' field (2D array)
        for x in 0..5 {
            for bit in 0..64 {
                fields.push(format!("c__{}__{}", x, bit));
            }
        }

        // 'c_prime' field (2D array)
        for x in 0..5 {
            for bit in 0..64 {
                fields.push(format!("c_prime__{}__{}", x, bit));
            }
        }

        // 'a_prime' field (3D array)
        for y in 0..5 {
            for x in 0..5 {
                for bit in 0..64 {
                    fields.push(format!("a_prime__{}_{}__{}", y, x, bit));
                }
            }
        }

        // 'a_prime_prime' field (3D array)
        for y in 0..5 {
            for x in 0..5 {
                for limb in 0..8 {
                    fields.push(format!("a_prime_prime__{}_{}__{}", y, x, limb));
                }
            }
        }

        // 'a_prime_prime_0_0_bits' field
        for bit in 0..64 {
            fields.push(format!("a_prime_prime_0_0_bits__{}", bit));
        }

        // 'a_prime_prime_prime_0_0_limbs' field
        for limb in 0..8 {
            fields.push(format!("a_prime_prime_prime_0_0_limbs__{}", limb));
        }

        Some(fields)
    }
}

impl<
        F,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    > FlattenFieldsHelper
    for Poseidon2Cols<F, WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>
{
    fn flatten_fields() -> Option<Vec<String>> {
        let mut fields = Vec::new();

        // Generate field names using exact parameter values
        fields.push("export".to_string());

        // Use actual parameters for array sizes
        for i in 0..WIDTH {
            fields.push(format!("inputs__{}", i));
        }

        // Other fields with their array sizes
        for i in 0..HALF_FULL_ROUNDS {
            fields.push(format!("beginning_full_rounds__{}", i));
        }

        for i in 0..PARTIAL_ROUNDS {
            fields.push(format!("partial_rounds__{}", i));
        }

        for i in 0..HALF_FULL_ROUNDS {
            fields.push(format!("ending_full_rounds__{}", i));
        }

        Some(fields)
    }
}
