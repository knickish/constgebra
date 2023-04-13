# Const Linear Algebra 

[![Crates.io][crates_img]][crates_lnk]

[crates_img]: https://img.shields.io/crates/v/constgebra.svg
[crates_lnk]: https://crates.io/crates/constgebra

Do your math ahead of time and embed the result in the binary. Floating-point code is from `compiler_builtins` and `libm` via the [const_soft_float](https://crates.io/crates/const_soft_float) crate. Uses const generics to check shape of inputs, and is `no_std`.

Please file an issue or make a test PR if your use case is not supported.


```rust
const START: CMatrix<2, 2> = CMatrix::new([
    [4.0, 1.0], 
    [2.0, 3.0]
]);

const ADD: CMatrix<2, 2> = CMatrix::new([
    [0.0, 6.0], 
    [0.0, 3.0]]
);

const EXPECTED: [[f64; 2]; 2] = [
    [0.6, -0.7], 
    [-0.2, 0.4]
];

const RESULT: [[f64; 2]; 2] = START
    .add(ADD)
    .pinv(f64::EPSILON)
    .finish();

for i in 0..2 {
    for j in 0..2 {
        assert!(float_equal(RESULT[i][j], EXPECTED[i][j], 1e-5));
    }
}
```
