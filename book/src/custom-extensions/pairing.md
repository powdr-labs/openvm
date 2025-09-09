# Elliptic Curve Pairing

The pairing extension enables usage of the optimal Ate pairing check on the BN254 and BLS12-381 elliptic curves. The following field extension tower for \\(\mathbb{F}\_{p^{12}}\\) is used for pairings in this crate:

$$
\mathbb{F_{p^2}} = \mathbb{F_{p}}[u]/(u^2 - \beta)\\\\
\mathbb{F_{p^6}} = \mathbb{F_{p^2}}[v]/(v^3 - \xi)\\\\
\mathbb{F_{p^{12}}} = \mathbb{F_{p^6}}[w]/(w^2 - v)
$$

The main feature of the pairing extension is the `pairing_check` function, which asserts that a product of pairings evaluates to 1.
For example, for the BLS12-381 curve,

```rust,no_run,noplayground
{{ #include ../../../examples/pairing/src/main.rs:pairing_check }}
```

This asserts that \\(e(p_0, q_0) e(p_1, q_1) = 1\\).
Naturally, this can be extended to more points by adding more elements to the arrays.

The pairing extension additionally provides field operations in \\(\mathbb{F_{p^{12}}}\\) for both BN254 and BLS12-381 curves where \\(\mathbb{F}\\) is the coordinate field.

See the [pairing guest library](../guest-libs/pairing.md) for usage details.
