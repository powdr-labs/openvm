// ANCHOR: imports
use core::hint::black_box;

use hex::FromHex;
use openvm_keccak256::keccak256;
use openvm::io::reveal_u32;
// ANCHOR_END: imports

// ANCHOR: main
openvm::entry!(main);

pub fn main() {
    /*
    let test_vectors = [
        (
            "",
            "C5D2460186F7233C927E7DB2DCC703C0E500B653CA82273B7BFAD8045D85A470",
        ),
        (
            "CC",
            "EEAD6DBFC7340A56CAEDC044696A168870549A6A7F6F56961E84A54BD9970B8A",
        ),
    ];
    */
    reveal_u32(black_box(222), 0);
    /*
    for (input, expected_output) in test_vectors.iter() {
        let input = Vec::from_hex(input).unwrap();
        let expected_output = Vec::from_hex(expected_output).unwrap();
        let output = keccak256(&black_box(input));
        if output != *expected_output {
            panic!();
        }
    }
    */
    reveal_u32(666, 1);
}
// ANCHOR_END: main
