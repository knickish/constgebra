# Const Linear Algebra 

Float-point code is from `compiler_builtins` via the [const_soft_float](https://crates.io/crates/const_soft_float) crate. Uses const generics to check shape of inputs. File an issue or PR a test if your use case is not supported.

```rust
const START: SMatrix<2, 2> = SMatrix::new([[4.0, 1.0], [2.0, 3.0]]);
const ADD: SMatrix<2, 2> = SMatrix::new([[0.0, 6.0], [0.0, 3.0]]);
const EXPECTED: [[f64; 2]; 2] = [[0.6, -0.7], [-0.2, 0.4]];

const INVERSE: [[f64;2];2] = START.add(ADD).pinv(f64::EPSILON).finish();
for i in 0..2 {
    for j in 0..2 {
        assert!(float_equal(INVERSE[i][j], EXPECTED[i][j], 1e-5));
    }
}
```
