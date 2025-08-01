use alloc::vec::Vec;

use halo2curves_axiom::bn256::{Fq, Fq12, Fq2, FROBENIUS_COEFF_FQ6_C1, XI_TO_Q_MINUS_1_OVER_2};
use itertools::izip;
use openvm_ecc_guest::{
    algebra::{field::FieldExtension, DivUnsafe, Field},
    AffinePoint,
};

use super::Bn254;
use crate::pairing::{
    Evaluatable, EvaluatedLine, LineMulDType, MillerStep, MultiMillerLoop, UnevaluatedLine,
};

pub const BN254_SEED: u64 = 0x44e992b44a6909f1;
// Encodes 6x+2 where x is the BN254 seed.
// 6*x+2 = sum_i BN254_PSEUDO_BINARY_ENCODING[i] * 2^i
// where BN254_PSEUDO_BINARY_ENCODING[i] is in {-1, 0, 1}
pub const BN254_PSEUDO_BINARY_ENCODING: [i8; 66] = [
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0,
    -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, -1, 0, -1, 0,
    0, 0, 1, 0, -1, 0, 1,
];

#[test]
fn test_bn254_pseudo_binary_encoding() {
    let mut x: i128 = 0;
    let mut power_of_2 = 1;
    for b in BN254_PSEUDO_BINARY_ENCODING.iter() {
        x += (*b as i128) * power_of_2;
        power_of_2 *= 2;
    }
    assert_eq!(x.unsigned_abs(), 6 * (BN254_SEED as u128) + 2);
}

impl MillerStep for Bn254 {
    type Fp2 = Fq2;

    /// Miller double step
    fn miller_double_step(
        s: &AffinePoint<Self::Fp2>,
    ) -> (AffinePoint<Self::Fp2>, UnevaluatedLine<Self::Fp2>) {
        let one = &Self::Fp2::ONE;
        let two = &(one + one);
        let three = &(one + two);

        let x = &s.x;
        let y = &s.y;
        // λ = (3x^2) / (2y)
        let lambda = &((three * x * x).div_unsafe(&(two * y)));
        // x_2s = λ^2 - 2x
        let x_2s = lambda * lambda - two * x;
        // y_2s = λ(x - x_2s) - y
        let y_2s = lambda * (x - x_2s) - y;
        let two_s = AffinePoint { x: x_2s, y: y_2s };

        // Tangent line
        //   1 + b' (x_P / y_P) w^-1 + c' (1 / y_P) w^-3
        // where
        //   l_{\Psi(S),\Psi(S)}(P) = (λ * x_S - y_S) (1 / y_P)  - λ (x_P / y_P) w^2 + w^3
        // x0 = λ * x_S - y_S
        // x2 = - λ
        let b = lambda.neg();
        let c = lambda * x - y;

        (two_s, UnevaluatedLine { b, c })
    }

    fn miller_add_step(
        s: &AffinePoint<Self::Fp2>,
        q: &AffinePoint<Self::Fp2>,
    ) -> (AffinePoint<Self::Fp2>, UnevaluatedLine<Self::Fp2>) {
        let x_s = &s.x;
        let y_s = &s.y;
        let x_q = &q.x;
        let y_q = &q.y;

        // λ1 = (y_s - y_q) / (x_s - x_q)
        let x_delta = x_s - x_q;
        let lambda = (y_s - y_q).div_unsafe(&x_delta);
        let x_s_plus_q = lambda * lambda - x_s - x_q;
        let y_s_plus_q = lambda * (x_q - x_s_plus_q) - y_q;

        let s_plus_q = AffinePoint {
            x: x_s_plus_q,
            y: y_s_plus_q,
        };

        // l_{\Psi(S),\Psi(Q)}(P) = (λ_1 * x_S - y_S) (1 / y_P) - λ_1 (x_P / y_P) w^2 + w^3
        let b = lambda.neg();
        let c = lambda * x_s - y_s;

        (s_plus_q, UnevaluatedLine { b, c })
    }

    /// Miller double and add step (2S + Q implemented as S + Q + S for efficiency)
    #[allow(clippy::type_complexity)]
    fn miller_double_and_add_step(
        s: &AffinePoint<Self::Fp2>,
        q: &AffinePoint<Self::Fp2>,
    ) -> (
        AffinePoint<Self::Fp2>,
        UnevaluatedLine<Self::Fp2>,
        UnevaluatedLine<Self::Fp2>,
    ) {
        let one = &Self::Fp2::ONE;
        let two = &(one + one);

        let x_s = &s.x;
        let y_s = &s.y;
        let x_q = &q.x;
        let y_q = &q.y;

        // λ1 = (y_s - y_q) / (x_s - x_q)
        let lambda1 = &((y_s - y_q).div_unsafe(&(x_s - x_q)));
        let x_s_plus_q = lambda1 * lambda1 - x_s - x_q;

        // λ2 = -λ1 - 2y_s / (x_{s+q} - x_s)
        let lambda2 = &(lambda1.neg() - (two * y_s).div_unsafe(&(x_s_plus_q - x_s)));
        let x_s_plus_q_plus_s = lambda2 * lambda2 - x_s - x_s_plus_q;
        let y_s_plus_q_plus_s = lambda2 * (x_s - x_s_plus_q_plus_s) - y_s;

        let s_plus_q_plus_s = AffinePoint {
            x: x_s_plus_q_plus_s,
            y: y_s_plus_q_plus_s,
        };

        // l_{\Psi(S),\Psi(Q)}(P) = (λ_1 * x_S - y_S) (1 / y_P) - λ_1 (x_P / y_P) w^2 + w^3
        let b0 = lambda1.neg();
        let c0 = lambda1 * x_s - y_s;

        // l_{\Psi(S+Q),\Psi(S)}(P) = (λ_2 * x_S - y_S) (1 / y_P) - λ_2 (x_P / y_P) w^2 + w^3
        let b1 = lambda2.neg();
        let c1 = lambda2 * x_s - y_s;

        (
            s_plus_q_plus_s,
            UnevaluatedLine { b: b0, c: c0 },
            UnevaluatedLine { b: b1, c: c1 },
        )
    }
}

#[allow(non_snake_case)]
impl MultiMillerLoop for Bn254 {
    type Fp = Fq;
    type Fp12 = Fq12;

    const SEED_ABS: u64 = BN254_SEED;
    const PSEUDO_BINARY_ENCODING: &[i8] = &BN254_PSEUDO_BINARY_ENCODING;

    fn evaluate_lines_vec(f: Fq12, lines: Vec<EvaluatedLine<Fq2>>) -> Fq12 {
        let mut f = f;
        let mut lines = lines;
        if lines.len() % 2 == 1 {
            f = Self::mul_by_013(&f, &lines.pop().unwrap());
        }
        for chunk in lines.chunks(2) {
            if let [line0, line1] = chunk {
                let prod = Self::mul_013_by_013(line0, line1);
                f = Self::mul_by_01234(&f, &prod);
            } else {
                panic!("lines.len() % 2 should be 0 at this point");
            }
        }
        f
    }

    fn pre_loop(
        Q_acc: Vec<AffinePoint<Fq2>>,
        _Q: &[AffinePoint<Fq2>],
        c: Option<Fq12>,
        xy_fracs: &[(Fq, Fq)],
    ) -> (Fq12, Vec<AffinePoint<Fq2>>) {
        let mut f = if let Some(mut c) = c {
            c.square_assign();
            c
        } else {
            Self::Fp12::ONE
        };

        let mut Q_acc = Q_acc;
        let mut initial_lines = Vec::<EvaluatedLine<Fq2>>::new();

        let (Q_out_double, lines_2S) = Q_acc
            .into_iter()
            .map(|Q| Self::miller_double_step(&Q))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_double;

        let lines_iter = izip!(lines_2S.iter(), xy_fracs.iter());
        for (line_2S, xy_frac) in lines_iter {
            let line = line_2S.evaluate(xy_frac);
            initial_lines.push(line);
        }

        f = Self::evaluate_lines_vec(f, initial_lines);

        (f, Q_acc)
    }

    fn post_loop(
        f: &Fq12,
        Q_acc: Vec<AffinePoint<Fq2>>,
        Q: &[AffinePoint<Fq2>],
        _c: Option<Fq12>,
        xy_fracs: &[(Fq, Fq)],
    ) -> (Fq12, Vec<AffinePoint<Fq2>>) {
        let mut Q_acc = Q_acc;
        let mut lines = Vec::<EvaluatedLine<Fq2>>::new();

        let x_to_q_minus_1_over_3 = FROBENIUS_COEFF_FQ6_C1[1];
        let x_to_q_sq_minus_1_over_3 = FROBENIUS_COEFF_FQ6_C1[2];
        let q1_vec = Q
            .iter()
            .map(|Q| {
                let x = Q.x.frobenius_map(1);
                let x = x * x_to_q_minus_1_over_3;
                let y = Q.y.frobenius_map(1);
                let y = y * XI_TO_Q_MINUS_1_OVER_2;
                AffinePoint { x, y }
            })
            .collect::<Vec<_>>();

        let (Q_out_add, lines_S_plus_Q) = Q_acc
            .iter()
            .zip(q1_vec.iter())
            .map(|(Q_acc, q1)| Self::miller_add_step(Q_acc, q1))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_add;

        let lines_iter = izip!(lines_S_plus_Q.iter(), xy_fracs.iter());
        for (lines_S_plus_Q, xy_frac) in lines_iter {
            let line = lines_S_plus_Q.evaluate(xy_frac);
            lines.push(line);
        }

        let q2_vec = Q
            .iter()
            .map(|Q| {
                // There is a frobenius mapping π²(Q) that we skip here since it is equivalent to
                // the identity mapping
                let x = Q.x * x_to_q_sq_minus_1_over_3;
                AffinePoint { x, y: Q.y }
            })
            .collect::<Vec<_>>();

        let (Q_out_add, lines_S_plus_Q) = Q_acc
            .iter()
            .zip(q2_vec.iter())
            .map(|(Q_acc, q2)| Self::miller_add_step(Q_acc, q2))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_add;

        let lines_iter = izip!(lines_S_plus_Q.iter(), xy_fracs.iter());
        for (lines_S_plus_Q, xy_frac) in lines_iter {
            let line = lines_S_plus_Q.evaluate(xy_frac);
            lines.push(line);
        }

        let mut f = *f;
        f = Self::evaluate_lines_vec(f, lines);

        (f, Q_acc)
    }
}
