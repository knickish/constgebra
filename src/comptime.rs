#![allow(clippy::needless_range_loop)]
#![allow(clippy::erasing_op)]

use const_soft_float::soft_f64::SoftF64 as Sf64;

/// Used with `CMatrix::apply_each` to specify an operation, as `const` function
/// pointers and closures are not yet stable
pub enum Operation {
    Mul(f64),
    Div(f64),
    Sqrt,
}

/// A helper type to construct row vectors. `CMatrix` is used
/// internally, but makes working with 1-dimensional data less verbose.
/// ```
/// # use constgebra::CVector;
/// const ARRAY: [f64; 2] = [4.0, 7.0];
///
/// const ROW_VECTOR: CVector::<2> = CVector::new_vector(ARRAY);
/// const RESULT: [f64; 2] = ROW_VECTOR.finish_vector();
///
/// assert_eq!(ARRAY, RESULT)
/// ```
pub type CVector<const N: usize> = CMatrix<1, N>;

/// A `const` matrix type, with dimensions checked at compile time
/// for all operations.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct CMatrix<const R: usize, const C: usize>([[Sf64; C]; R]);

impl<const R: usize, const C: usize> Default for CMatrix<R, C> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const R: usize, const C: usize> CMatrix<R, C> {
    /// Create a `CMatrix` from a 2D array of `f64`.
    /// ```rust
    /// # use constgebra::CMatrix;
    /// const ARRAY: [[f64; 2]; 2] = [
    ///     [4.0, 7.0],
    ///     [2.0, 6.0]
    /// ];
    ///
    /// const CMATRIX: CMatrix::<2, 2> = CMatrix::new(ARRAY);
    /// ```
    pub const fn new(vals: [[f64; C]; R]) -> Self {
        let mut ret = [[Sf64(0.0); C]; R];

        let mut i = 0;
        while i < R {
            let mut j = 0;
            while j < C {
                ret[i][j] = Sf64(vals[i][j]);
                j += 1;
            }
            i += 1
        }
        CMatrix(ret)
    }

    /// Equivalent to `CMatrix::new` using `const_soft_float::SoftF64`
    /// instead of `f64.`
    pub const fn new_from_soft(vals: [[Sf64; C]; R]) -> Self {
        CMatrix(vals)
    }

    /// Create a `CMatrix` filled with zeroes.
    pub const fn zero() -> Self {
        // TODO: Replace this with `::default()` once this PR is merged and released:
        // https://github.com/823984418/const_soft_float/pull/7
        Self([[Sf64::from_f64(0.0); C]; R])
    }

    /// Create an identity `CMatrix`.
    ///```rust
    /// # use constgebra::CMatrix;
    /// const LEFT: CMatrix<4, 3> = CMatrix::new([
    ///     [1.0, 0.0, 1.0],
    ///     [2.0, 1.0, 1.0],
    ///     [0.0, 1.0, 1.0],
    ///     [1.0, 1.0, 2.0],
    /// ]);
    ///
    /// const RIGHT: CMatrix<3, 3> = CMatrix::identity();
    ///
    /// const EXPECTED: [[f64; 3]; 4] = LEFT.finish();
    ///
    /// const RESULT: [[f64; 3]; 4] = LEFT.mul(RIGHT).finish();
    ///
    /// assert_eq!(EXPECTED, RESULT);
    /// ```
    pub const fn identity() -> Self {
        let mut ret = Self::zero();
        let diag_max = match R > C {
            true => C,
            false => R,
        };

        let mut idx = 0;
        while idx < diag_max {
            ret.0[idx][idx] = Sf64::from_f64(1.0);
            idx += 1;
        }

        ret
    }

    /// Converts a `CMatrix` back into a two-dimensional array.
    /// ```rust
    /// # use constgebra::CMatrix;
    /// const ARRAY: [[f64; 2]; 2] = [
    ///     [4.0, 7.0],
    ///     [2.0, 6.0]
    /// ];
    ///
    /// const CMATRIX: CMatrix::<2, 2> = CMatrix::new(ARRAY);
    ///
    /// const RESULT: [[f64; 2]; 2] = CMATRIX.finish();
    ///
    /// assert_eq!(ARRAY, RESULT)
    /// ```
    pub const fn finish(self) -> [[f64; C]; R] {
        let mut ret = [[0.0; C]; R];

        let mut i = 0;
        while i < R {
            let mut j = 0;
            while j < C {
                ret[i][j] = self.0[i][j].0;
                j += 1;
            }
            i += 1
        }
        ret
    }

    /// `CMatrix::finish`, but returns `const_soft_float::SoftF64`
    pub const fn finish_soft(self) -> [[Sf64; C]; R] {
        self.0
    }

    const fn rows(&self) -> usize {
        R
    }

    const fn columns(&self) -> usize {
        C
    }

    const fn get_dims(&self) -> [usize; 2] {
        if R == C {
            return [R, C];
        }

        if R > C {
            [R, C]
        } else {
            [C, R]
        }
    }

    /// Multiply two `CMatrix` and return the result. Columns
    /// of self and rows of multiplier must agree in number.
    ///```rust
    /// # use constgebra::CMatrix;
    /// const LEFT: CMatrix<4, 3> = CMatrix::new([
    ///     [1.0, 0.0, 1.0],
    ///     [2.0, 1.0, 1.0],
    ///     [0.0, 1.0, 1.0],
    ///     [1.0, 1.0, 2.0],
    /// ]);
    ///
    /// const RIGHT: CMatrix<3, 3> = CMatrix::new([
    ///     [1.0, 2.0, 1.0],
    ///     [2.0, 3.0, 1.0],
    ///     [4.0, 2.0, 2.0]
    /// ]);
    ///
    /// const EXPECTED: [[f64; 3]; 4] = [
    ///     [5.0, 4.0, 3.0],
    ///     [8.0, 9.0, 5.0],
    ///     [6.0, 5.0, 3.0],
    ///     [11.0, 9.0, 6.0],
    /// ];
    ///
    /// const RESULT: [[f64; 3]; 4] = LEFT.mul(RIGHT).finish();
    ///
    /// assert_eq!(EXPECTED, RESULT);
    /// ```
    pub const fn mul<const OC: usize>(self, rhs: CMatrix<C, OC>) -> CMatrix<R, OC> {
        let mut ret = [[Sf64(0.0); OC]; R];

        let mut i = 0;
        while i < R {
            let mut j = 0;
            while j < OC {
                let mut k = 0;
                let mut acc: Sf64 = Sf64(0.0_f64);
                while k < C {
                    acc = acc.add(self.0[i][k].mul(rhs.0[k][j]));
                    k += 1
                }
                ret[i][j] = acc;
                j += 1;
            }
            i += 1;
        }
        CMatrix(ret)
    }

    /// Add two `CMatrix` and return the result.
    /// ```rust
    /// # use constgebra::CMatrix;
    /// const LEFT: CMatrix<3, 3> = CMatrix::new([
    ///     [1.0, 0.0, 1.0],
    ///     [2.0, 1.0, 1.0],
    ///     [0.0, 1.0, 1.0]]
    /// );
    ///
    /// const RIGHT: CMatrix<3, 3> = CMatrix::new([
    ///     [1.0, 2.0, 1.0],
    ///     [2.0, 3.0, 1.0],
    ///     [4.0, 2.0, 2.0]]
    /// );
    ///
    /// const EXPECTED: [[f64; 3]; 3] = [
    ///     [2.0, 2.0, 2.0],
    ///     [4.0, 4.0, 2.0],
    ///     [4.0, 3.0, 3.0]
    /// ];
    ///
    /// const RESULT: [[f64; 3]; 3] = LEFT.add(RIGHT).finish();
    ///
    /// assert_eq!(EXPECTED, RESULT);
    ///```
    pub const fn add(self, rhs: Self) -> Self {
        let mut ret = [[Sf64(0.0); C]; R];

        let mut i = 0;
        while i < R {
            let mut j = 0;
            while j < C {
                ret[i][j] = self.0[i][j].add(rhs.0[i][j]);
                j += 1;
            }
            i += 1;
        }
        CMatrix(ret)
    }

    /// Subtract two `CMatrix` and return the result.
    /// ```rust
    /// # use constgebra::CMatrix;
    /// const LEFT: CMatrix<3, 3> = CMatrix::new([
    ///     [1.0, 2.0, 1.0],
    ///     [2.0, 3.0, 1.0],
    ///     [4.0, 2.0, 2.0]]
    /// );
    ///
    /// const RIGHT: CMatrix<3, 3> = CMatrix::new([
    ///     [1.0, 0.0, 1.0],
    ///     [2.0, 1.0, 1.0],
    ///     [0.0, 1.0, 1.0]]
    /// );
    ///
    /// const EXPECTED: [[f64; 3]; 3] = [
    ///     [0.0, 2.0, 0.0],
    ///     [0.0, 2.0, 0.0],
    ///     [4.0, 1.0, 1.0]
    /// ];
    ///
    /// const RESULT: [[f64; 3]; 3] = LEFT.sub(RIGHT).finish();
    ///
    /// assert_eq!(EXPECTED, RESULT);
    ///```
    pub const fn sub(self, rhs: Self) -> Self {
        let mut ret = [[Sf64(0.0); C]; R];

        let mut i = 0;
        while i < R {
            let mut j = 0;
            while j < C {
                ret[i][j] = self.0[i][j].sub(rhs.0[i][j]);
                j += 1;
            }
            i += 1;
        }
        CMatrix(ret)
    }

    /// Apply an operation to each member of the matrix separately. Especially
    /// useful for scaling vectors
    /// ```
    /// # use constgebra::{CMatrix, Operation};
    /// const BASE: CMatrix<1, 3> = CMatrix::new([[1.0, 2.0, 3.0]]);
    /// const MUL: CMatrix<1, 3> = BASE.apply_each(Operation::Mul(3.0));
    /// assert_eq!([[3.0, 6.0, 9.0]], MUL.finish());
    pub const fn apply_each(mut self, op: Operation) -> Self {
        let mut i = 0;
        while i < R {
            let mut j = 0;
            while j < C {
                self.0[i][j] = match op {
                    Operation::Mul(val) => self.0[i][j].mul(Sf64(val)),
                    Operation::Div(val) => self.0[i][j].div(Sf64(val)),
                    Operation::Sqrt => self.0[i][j].sqrt(),
                };
                j += 1;
            }
            i += 1;
        }
        self
    }

    const fn get(&self, row: usize, column: usize) -> Sf64 {
        self.0[row][column]
    }

    #[must_use]
    // TODO Replace once const_mut_refs are stabilized
    const fn set(self, row: usize, column: usize, value: Sf64) -> Self {
        let mut ret = self.0;
        ret[row][column] = value;
        Self(ret)
    }

    /// Return the transpose of a `CMatrix`.
    /// ```rust
    /// # use constgebra::CMatrix;
    /// const START: [[f64; 2]; 2] = [
    ///     [4.0, 7.0],
    ///     [2.0, 6.0]
    /// ];
    ///
    /// const EXPECTED: [[f64; 2]; 2] = [
    ///     [4.0, 2.0],
    ///     [7.0, 6.0]
    /// ];
    ///
    /// const RESULT: [[f64; 2]; 2] =
    ///     CMatrix::new(START).transpose().finish();
    ///
    /// assert_eq!(EXPECTED, RESULT)
    /// ```
    pub const fn transpose(self) -> CMatrix<C, R> {
        let mut i = 0;
        let mut ret = [[Sf64(0.0); R]; C];
        while i < R {
            let mut j = 0;
            while j < C {
                ret[j][i] = self.0[i][j];
                j += 1;
            }
            i += 1;
        }
        CMatrix(ret)
    }

    #[must_use]
    pub const fn givens_l(self, m: usize, a: Sf64, b: Sf64) -> Self {
        let r = (a.powi(2).add(b.powi(2))).sqrt();
        if eq(r, Sf64(0.0)) {
            return self;
        }

        let mut mut_self = self;
        let c = a.div(r);
        let s = b.neg().div(r);
        let mut i = 0;
        while i < C {
            let s0 = mut_self.get(m, i);
            let s1 = mut_self.get(m + 1, i);

            let val = mut_self.get(m, i).add(s0.mul(c.sub(Sf64(1.0))));
            mut_self = mut_self.set(m, i, val);

            let val = mut_self.get(m, i).add(s1.mul(s.neg()));
            mut_self = mut_self.set(m, i, val);

            let val = mut_self.get(m + 1, i).add(s0.mul(s));
            mut_self = mut_self.set(m + 1, i, val);

            let val = mut_self.get(m + 1, i).add(s1.mul(c.sub(Sf64(1.0))));
            mut_self = mut_self.set(m + 1, i, val);
            i += 1;
        }
        mut_self
    }

    #[must_use]
    pub const fn givens_r(self, m: usize, a: Sf64, b: Sf64) -> Self {
        let r = (a.powi(2).add(b.powi(2))).sqrt();
        if eq(r, Sf64(0.0)) {
            return self;
        }

        let mut mut_self = self;
        let c = a.div(r);
        let s = b.neg().div(r);
        let mut i = 0;
        while i < R {
            let s0 = mut_self.get(i, m);
            let s1 = mut_self.get(i, m + 1);

            let val = mut_self.get(i, m).add(s0.mul(c.sub(Sf64(1.0))));
            mut_self = mut_self.set(i, m, val);

            let val = mut_self.get(i, m).add(s1.mul(s.neg()));
            mut_self = mut_self.set(i, m, val);

            let val = mut_self.get(i, m + 1).add(s0.mul(s));
            mut_self = mut_self.set(i, m + 1, val);

            let val = mut_self.get(i, m + 1).add(s1.mul(c.sub(Sf64(1.0))));
            mut_self = mut_self.set(i, m + 1, val);
            i += 1;
        }
        mut_self
    }

    const fn gemm<const M: usize>(
        mut self,
        a: &CMatrix<R, M>,
        b: &CMatrix<M, C>,
        alpha: f64,
        beta: f64,
    ) -> Self {
        let alpha = Sf64(alpha);
        let beta = Sf64(beta);
        const fn beta_term(x: Sf64, beta: Sf64) -> Sf64 {
            match is_zero(beta) {
                true => Sf64(0.0),
                false => beta.mul(x),
            }
        }
        let mut n = 0;

        while n < C {
            let mut m = 0;
            while m < R {
                let mut axb = Sf64(0.0);
                let mut k = 0;
                while k < M {
                    axb = axb.add(a.0[m][k].mul(b.0[k][n]));
                    k += 1;
                }
                let b_term = beta_term(self.0[m][n], beta);
                self.0[m][n] = alpha.mul(axb.add(b_term));
                m += 1;
            }
            n += 1;
        }
        self
    }

    /// Singular Value Decomposition
    pub const fn svd(self, epsilon: f64) -> CMatrix<C, R> {
        const fn less_zero_sign(x: Sf64) -> Sf64 {
            let Some(cmp) = x.cmp(Sf64(0.0)) else {
                panic!("failed to get sign of value")
            };
            match cmp {
                core::cmp::Ordering::Less => Sf64(-1.0),
                _ => Sf64(1.0),
            }
        }

        let dim = self.get_dims();

        if self.rows() == self.columns() {
            let mut s_working = CMatrix([[Sf64(0.0); R]; R]);
            let mut u_working = CMatrix([[Sf64(0.0); R]; R]);
            let mut v_working = CMatrix([[Sf64(0.0); R]; R]);

            let mut u_out = CMatrix::new([[0.0; C]; C]);
            let mut s_out = CMatrix::new([[0.0; R]; 1]);
            let mut vt_out = CMatrix::new([[0.0; C]; R]);
            {
                let mut i = 0;
                while i < R {
                    let mut j = 0;
                    while j < R {
                        s_working.0[i][j] = self.0[i][j];
                        j += 1;
                    }
                    i += 1;
                }
            }

            {
                let mut i = 0;
                while i < R {
                    u_working.0[i][i] = Sf64(1.0);
                    i += 1;
                }
            }

            {
                let mut i = 0;
                while i < R {
                    v_working.0[i][i] = Sf64(1.0);
                    i += 1;
                }
            }

            (u_working, s_working, v_working) = Self::svd_inner(
                self.get_dims(),
                u_working,
                s_working,
                v_working,
                Sf64(epsilon),
            );

            {
                let mut i = 0;
                while i < R {
                    // Set S
                    s_out.0[0][i] = s_working.get(i, i);
                    i += 1;
                }
            }

            let mut i = 0;
            while i < C {
                let mut j = 0;
                while j < C {
                    u_out.0[i][j] = v_working.get(i, j).mul(less_zero_sign(s_out.get(0, i)));
                    j += 1;
                }
                i += 1;
            }

            // Set V
            let mut i = 0;
            while i < R {
                let mut j = 0;
                while j < dim[1] {
                    vt_out.0[i][j] = u_working.0[i][j];
                    j += 1
                }
                i += 1;
            }
            {
                let mut i = 0;
                while i < R {
                    s_out.0[0][i] = s_out.get(0, i).mul(less_zero_sign(s_out.get(0, i)));
                    i += 1;
                }
            }

            // set all below epsilon to zero
            let eps_ = Sf64(epsilon); //s_out[0] * epsilon;
            let mut i = 0;
            while i < R {
                if lt(s_out.0[0][i], eps_) {
                    s_out.0[0][i] = Sf64(0.0);
                } else {
                    s_out.0[0][i] = Sf64(1.0).div(s_out.0[0][i]);
                }
                i += 1;
            }

            let mut i = 0;
            while i < C {
                let mut j = 0;
                while j < C {
                    u_out.0[j][i] = u_out.0[j][i].mul(s_out.0[0][j]);
                    j += 1;
                }
                i += 1;
            }

            self.gemm(&vt_out, &u_out, 1.0, 0.0).transpose()
        } else {
            panic!("Non-square matrices not yet supported");
        }
    }

    #[must_use]
    const fn svd_inner<const L: usize, const S: usize>(
        dim: [usize; 2],
        mut u_working: CMatrix<L, L>,
        mut s_working: CMatrix<L, S>,
        mut v_working: CMatrix<S, S>,
        eps: Sf64,
    ) -> (CMatrix<L, L>, CMatrix<L, S>, CMatrix<S, S>) {
        let n = S;

        let mut house_vec = [Sf64(0.0); L];
        let mut i = 0;
        while i < S {
            // Column Householder
            {
                let x1 = abs(s_working.get(i, i));

                let x_inv_norm = {
                    let mut x_inv_norm = Sf64(0.0);
                    let mut j = i;
                    while j < dim[0] {
                        x_inv_norm = x_inv_norm.add(s_working.get(j, i).powi(2));
                        j += 1;
                    }
                    if gt(x_inv_norm, Sf64(0.0)) {
                        x_inv_norm = Sf64(1.0).div(x_inv_norm.sqrt());
                    }
                    x_inv_norm
                };

                let (alpha, beta) = {
                    let mut alpha = (Sf64(1.0).add(x1.mul(x_inv_norm))).sqrt();
                    let beta = x_inv_norm.div(alpha);
                    if eq(x_inv_norm, Sf64(0.0)) {
                        alpha = Sf64(0.0);
                    } // nothing to do
                    (alpha, beta)
                };

                house_vec[i] = alpha.neg();
                let mut j = i + 1;
                while j < L {
                    house_vec[j] = beta.neg().mul(s_working.get(j, i));
                    j += 1;
                }

                if lt(s_working.get(i, i), Sf64(0.0)) {
                    let mut j = i + 1;
                    while j < dim[0] {
                        house_vec[j] = house_vec[j].neg();
                        j += 1;
                    }
                }
            }

            let mut k = i;
            while k < dim[1] {
                let mut dot_prod = Sf64(0.0);
                let mut j = i;
                while j < dim[0] {
                    dot_prod = dot_prod.add(s_working.get(j, k).mul(house_vec[j]));
                    j += 1;
                }
                let mut j = i;
                while j < dim[0] {
                    let val = s_working.get(j, k).sub(dot_prod.mul(house_vec[j]));
                    s_working = s_working.set(j, k, val);
                    j += 1;
                }
                k += 1;
            }
            let mut k = 0;
            while k < dim[0] {
                let mut dot_prod = Sf64(0.0);
                let mut j = i;
                while j < dim[0] {
                    dot_prod = dot_prod.add(u_working.get(k, j).mul(house_vec[j]));
                    j += 1;
                }

                let mut j = i;
                while j < dim[0] {
                    let val = u_working.get(k, j).sub(dot_prod.mul(house_vec[j]));
                    u_working = u_working.set(k, j, val);
                    j += 1;
                }
                k += 1;
            }

            if i >= n - 1 {
                i += 1;
                continue;
            }

            {
                let x1 = abs(s_working.get(i, i + 1));

                let x_inv_norm = {
                    let mut x_inv_norm = Sf64(0.0);
                    let mut j = i + 1;
                    while j < dim[1] {
                        x_inv_norm = x_inv_norm.add(s_working.get(i, j).powi(2));
                        j += 1;
                    }
                    if gt(x_inv_norm, Sf64(0.0)) {
                        x_inv_norm = Sf64(1.0).div(x_inv_norm.sqrt());
                    }
                    x_inv_norm
                };

                let (alpha, beta) = {
                    let mut alpha = Sf64(1.0).add(x1.mul(x_inv_norm)).sqrt();
                    let beta = x_inv_norm.div(alpha);
                    if eq(x_inv_norm, Sf64(0.0)) {
                        alpha = Sf64(0.0); // nothing to do
                    }
                    (alpha, beta)
                };

                house_vec[i + 1] = alpha.neg();
                let mut j = i + 2;
                while j < dim[1] {
                    house_vec[j] = beta.neg().mul(s_working.get(i, j));
                    j += 1;
                }
                if lt(s_working.get(i, i + 1), Sf64(0.0)) {
                    let mut j = i + 2;
                    while j < dim[1] {
                        house_vec[j] = house_vec[j].neg();
                        j += 1;
                    }
                }
            }

            let mut k = i;
            while k < dim[0] {
                let mut dot_prod = Sf64(0.0);
                let mut j = i + 1;
                while j < dim[1] {
                    dot_prod = dot_prod.add(s_working.get(k, j).mul(house_vec[j]));
                    j += 1;
                }
                let mut j = i + 1;
                while j < dim[1] {
                    let val = s_working.get(k, j).sub(dot_prod.mul(house_vec[j]));
                    s_working = s_working.set(k, j, val);
                    j += 1;
                }
                k += 1;
            }

            let mut k = 0;
            while k < dim[1] {
                let mut dot_prod = Sf64(0.0);
                let mut j = i + 1;
                while j < dim[1] {
                    dot_prod = dot_prod.add(v_working.get(j, k).mul(house_vec[j]));
                    j += 1;
                }
                let mut j = i + 1;
                while j < dim[1] {
                    let val = v_working.get(j, k).sub(dot_prod.mul(house_vec[j]));
                    v_working = v_working.set(j, k, val);
                    j += 1;
                }
                k += 1;
            }

            i += 1;
        }

        let eps = if lt(eps, Sf64(0.0)) {
            let mut eps = Sf64(1.0);
            while gt(eps.add(Sf64(1.0)), Sf64(1.0)) {
                eps = eps.mul(Sf64(0.5));
            }

            eps = eps.mul(Sf64(64.0));
            eps
        } else {
            eps
        };

        let mut k0 = 0;
        while k0 < dim[1] - 1 {
            // Diagonalization
            let s_max = {
                let mut s_max = Sf64(0.0);
                let mut i = 0;
                while i < dim[1] {
                    let tmp = abs(s_working.get(i, i));

                    if gt(tmp, s_max) {
                        s_max = tmp;
                    }
                    i += 1;
                }

                let mut i = 0;
                while i < (dim[1] - 1) {
                    let tmp = abs(s_working.get(i, i + 1));
                    if gt(tmp, s_max) {
                        s_max = tmp
                    }
                    i += 1;
                }
                s_max
            };

            while (k0 < dim[1] - 1) && le(abs(s_working.get(k0, k0 + 1)), eps.mul(s_max)) {
                k0 += 1;
            }
            if k0 == dim[1] - 1 {
                continue;
            }

            let n = {
                let mut n = k0 + 2;
                while n < dim[1] && gt(abs(s_working.get(n - 1, n)), eps.mul(s_max)) {
                    n += 1;
                }
                n
            };

            let (alpha, beta) = {
                if n - k0 == 2
                    && lt(abs(s_working.get(k0, k0)), eps.mul(s_max))
                    && lt(abs(s_working.get(k0 + 1, k0 + 1)), eps.mul(s_max))
                {
                    // Compute mu
                    (Sf64(0.0), Sf64(1.0))
                } else {
                    let mut c_vec = [Sf64(0.0); 4];
                    c_vec[0 * 2] = s_working.get(n - 2, n - 2).mul(s_working.get(n - 2, n - 2));
                    if n - k0 > 2 {
                        c_vec[0 * 2] = c_vec[0 * 2]
                            .add(s_working.get(n - 3, n - 2).mul(s_working.get(n - 3, n - 2)));
                    }
                    c_vec[1] = s_working.get(n - 2, n - 2).mul(s_working.get(n - 2, n - 1));
                    c_vec[2] = s_working.get(n - 2, n - 2).mul(s_working.get(n - 2, n - 1));
                    c_vec[2 + 1] = s_working
                        .get(n - 1, n - 1)
                        .mul(s_working.get(n - 1, n - 1))
                        .add(s_working.get(n - 2, n - 1).mul(s_working.get(n - 2, n - 1)));

                    let (b, d) = {
                        let mut b = (c_vec[0 * 2].add(c_vec[2 + 1])).neg().div(Sf64(2.0));
                        let mut c = c_vec[0 * 2].mul(c_vec[2 + 1]).sub(c_vec[1].mul(c_vec[2]));
                        let mut d = Sf64(0.0);
                        if gt(abs(b.powi(2).sub(c)), eps.mul(b.powi(2))) {
                            d = (b.powi(2).sub(c)).sqrt();
                        } else {
                            b = c_vec[0 * 2].sub((c_vec[2 + 1]).div(Sf64(2.0)));
                            c = c_vec[1].neg().mul(c_vec[2]);
                            if gt(b.mul(b).sub(c), Sf64(0.0)) {
                                d = (b.mul(b).sub(c)).sqrt();
                            }
                        }

                        (b, d)
                    };

                    let lambda1 = b.neg().add(d);
                    let lambda2 = b.neg().sub(d);

                    let d1 = abs(lambda1.sub(c_vec[2 + 1]));
                    let d2 = abs(lambda2.sub(c_vec[2 + 1]));
                    let mu = if lt(d1, d2) { lambda1 } else { lambda2 };

                    let alpha = s_working.get(k0, k0).powi(2).sub(mu);
                    let beta = s_working.get(k0, k0).mul(s_working.get(k0, k0 + 1));

                    (alpha, beta)
                }
            };
            {
                let mut alpha = alpha;
                let mut beta = beta;
                let mut k = k0;
                while k < (n - 1) {
                    s_working = s_working.givens_r(k, alpha, beta);
                    v_working = v_working.givens_l(k, alpha, beta);

                    alpha = s_working.get(k, k);
                    beta = s_working.get(k + 1, k);
                    s_working = s_working.givens_l(k, alpha, beta);
                    u_working = u_working.givens_r(k, alpha, beta);

                    alpha = s_working.get(k, k + 1);

                    if k != n - 2 {
                        beta = s_working.get(k, k + 2);
                    }
                    k += 1;
                }
            }

            {
                // Make S bi-diagonal again
                let mut i0 = k0;
                while i0 < (n - 1) {
                    let mut i1 = 0;
                    while i1 < dim[1] {
                        if i0 > i1 || i0 + 1 < i1 {
                            s_working = s_working.set(i0, i1, Sf64(0.0));
                        }
                        i1 += 1;
                    }
                    i0 += 1;
                }
                let mut i0 = 0;
                while i0 < dim[0] {
                    let mut i1 = k0;
                    while i1 < (n - 1) {
                        if i0 > i1 || i0 + 1 < i1 {
                            s_working = s_working.set(i0, i1, Sf64(0.0));
                        }
                        i1 += 1;
                    }
                    i0 += 1;
                }
                let mut i = 0;
                while i < (dim[1] - 1) {
                    if le(abs(s_working.get(i, i + 1)), eps.mul(s_max)) {
                        s_working = s_working.set(i, i + 1, Sf64(0.0));
                    }
                    i += 1;
                }
            }
        }
        (u_working, s_working, v_working)
    }

    pub const fn pinv(self, epsilon: f64) -> CMatrix<C, R> {
        if self.rows() * self.columns() == 0 {
            return self.transpose();
        }

        self.svd(epsilon)
    }
}

impl<const N: usize> CMatrix<1, N> {
    /// Special case of `CMatrix::new` for constructing a CVector
    /// Always returns a row vector, follow with `transpose` to build
    /// a column vector
    /// ```
    /// # use constgebra::CVector;
    /// const ARRAY: [f64; 2] = [4.0, 7.0];
    ///
    /// const ROWVECTOR: CVector::<2> = CVector::new_vector(ARRAY);
    pub const fn new_vector(vals: [f64; N]) -> CVector<N> {
        CMatrix::new([vals])
    }

    pub const fn new_vector_from_soft(vals: [Sf64; N]) -> Self {
        CMatrix::new_from_soft([vals])
    }

    /// Dot product of two `CVector` of the same size.
    /// ```rust
    /// # use constgebra::{CVector, const_soft_float::soft_f64::SoftF64 as Sf64};
    /// const LEFT: CVector<3> = CVector::new_vector([1.0, 3.0, -5.0]);
    /// const RIGHT: CVector<3> = CVector::new_vector([4.0, -2.0, -1.0]);
    ///
    /// const EXPECTED: f64 = 3.0;
    /// const RESULT: f64 = LEFT.dot(RIGHT);
    ///
    /// assert_eq!(EXPECTED, RESULT)
    /// ```
    pub const fn dot(self, other: Self) -> f64 {
        self.mul(other.transpose()).get(0, 0).to_f64()
    }

    /// Special case of `CMatrix::finish` for use with a CVector,
    /// returns `[f64 ; N]` instead of `[[f64 ; N]; 1]`
    /// ```rust
    /// # use constgebra::CVector;
    /// const ARRAY: [f64; 2] = [4.0, 7.0];
    ///
    /// const CVECTOR: CVector::<2> = CVector::new_vector(ARRAY);
    ///
    /// const RESULT: [f64; 2] = CVECTOR.finish_vector();
    ///
    /// assert_eq!(ARRAY, RESULT)
    /// ```
    pub const fn finish_vector(self) -> [f64; N] {
        self.finish()[0]
    }

    /// `CVector::finish_vector`, but returns soft floats
    pub const fn finish_vector_soft(self) -> [Sf64; N] {
        self.finish_soft()[0]
    }
}

#[cfg(unused)]
const fn panic_if_ne(val: Sf64, test: f64) {
    if gt(abs(val.sub(Sf64(test))), Sf64(0.00001)) {
        panic!();
    }
}

const fn is_zero(mut arg: Sf64) -> bool {
    let mut i = 0;
    while i < 64 {
        if arg.0 as u64 != 0 {
            return false;
        }
        arg = arg.mul(Sf64(2.0));
        i += 1;
    }
    true
}

const fn abs(x: Sf64) -> Sf64 {
    let Some(cmp) = x.cmp(Sf64(0.0)) else {
        panic!("failed to get sign of value")
    };
    match cmp {
        core::cmp::Ordering::Less => x.neg(),
        _ => x,
    }
}

const fn gt(x: Sf64, y: Sf64) -> bool {
    let Some(cmp) = x.cmp(y) else {
        panic!("failed to compare values")
    };
    matches!(cmp, core::cmp::Ordering::Greater)
}

const fn lt(x: Sf64, y: Sf64) -> bool {
    let Some(cmp) = x.cmp(y) else {
        panic!("failed to compare values")
    };
    matches!(cmp, core::cmp::Ordering::Less)
}

const fn le(x: Sf64, y: Sf64) -> bool {
    let Some(cmp) = x.cmp(y) else {
        panic!("failed to compare values")
    };
    matches!(cmp, core::cmp::Ordering::Less | core::cmp::Ordering::Equal)
}

const fn eq(x: Sf64, y: Sf64) -> bool {
    let Some(cmp) = x.cmp(y) else {
        panic!("failed to compare values")
    };
    matches!(cmp, core::cmp::Ordering::Equal)
}

#[cfg(test)]
mod const_tests {
    use super::*;

    fn float_equal(one: f64, two: f64, eps: f64) -> bool {
        match (one, two) {
            (a, b) if (a - b).abs() < eps => true,
            (a, b) if a == -0.0 || b == -0.0 => a + b == 0.0 || -(a + b) == 0.0,
            _ => false,
        }
    }

    #[test]
    fn test_cvec() {
        const VEC_BASE: CVector<3> = CVector::new_vector([1.0, 2.0, 3.0]);
        const MAT_BASE: CMatrix<1, 3> = CMatrix::new([[1.0, 2.0, 3.0]]);

        assert_eq!(VEC_BASE.finish(), MAT_BASE.finish());

        const ADDED: CVector<3> = VEC_BASE.add(MAT_BASE);
        const DIVIDED: CVector<3> = ADDED.apply_each(Operation::Div(3.0));
        assert_eq!(DIVIDED.finish_vector()[2], 2.0)
    }

    #[test]
    fn test_apply_each() {
        const VEC_BASE: CVector<3> = CVector::new_vector([1.0, 2.0, 3.0]);
        const MAT_BASE: CMatrix<1, 3> = CMatrix::new([[1.0, 2.0, 3.0]]);

        assert_eq!(VEC_BASE.finish(), MAT_BASE.finish());

        const ADDED: CVector<3> = VEC_BASE.add(MAT_BASE);
        const DIVIDED: CVector<3> = ADDED.apply_each(Operation::Div(3.0));
        assert_eq!(DIVIDED.finish_vector()[2], 2.0)
    }

    #[test]
    fn test_dot_product_normal() {
        const VEC1: CVector<3> = CVector::new_vector([2.0, 3.0, 4.0]);
        const VEC2: CVector<3> = CVector::new_vector([1.0, 5.0, 7.0]);
        const EXPECTED: f64 = 2.0 * 1.0 + 3.0 * 5.0 + 4.0 * 7.0;
        const RESULT: f64 = VEC1.dot(VEC2);
        assert_eq!(EXPECTED, RESULT);
    }

    #[test]
    fn test_dot_product_zero_vector() {
        const VEC1: CVector<3> = CVector::new_vector([0.0, 0.0, 0.0]);
        const VEC2: CVector<3> = CVector::new_vector([1.0, 5.0, 7.0]);
        const EXPECTED: f64 = 0.0;
        const RESULT: f64 = VEC1.dot(VEC2);
        assert_eq!(EXPECTED, RESULT);
    }

    #[test]
    fn test_dot_product_with_negatives() {
        const VEC1: CVector<3> = CVector::new_vector([-1.0, -3.0, 5.0]);
        const VEC2: CVector<3> = CVector::new_vector([4.0, -2.0, -1.0]);
        const EXPECTED: f64 = (-1.0) * 4.0 + (-3.0) * (-2.0) + 5.0 * (-1.0);
        const RESULT: f64 = VEC1.dot(VEC2);
        assert_eq!(EXPECTED, RESULT);
    }

    #[test]
    fn test_dot_product_large_values() {
        const VEC1: CVector<3> = CVector::new_vector([1000000.0, 3000000.0, -5000000.0]);
        const VEC2: CVector<3> = CVector::new_vector([4000000.0, -2000000.0, -1000000.0]);
        const EXPECTED: f64 =
            1000000.0 * 4000000.0 + 3000000.0 * (-2000000.0) + (-5000000.0) * (-1000000.0);
        const RESULT: f64 = VEC1.dot(VEC2);
        assert_eq!(EXPECTED, RESULT);
    }

    #[test]
    fn test_dot_product_identity() {
        const VEC1: CVector<3> = CVector::new_vector([1.0, 0.0, 0.0]);
        const VEC2: CVector<3> = CVector::new_vector([0.0, 1.0, 0.0]);
        const EXPECTED: f64 = 0.0;
        const RESULT: f64 = VEC1.dot(VEC2);
        assert_eq!(EXPECTED, RESULT);
    }

    #[test]
    fn test_2_x_2_example() {
        const START: CMatrix<2, 2> = CMatrix::new([[4.0, 1.0], [2.0, 3.0]]);
        const ADD: CMatrix<2, 2> = CMatrix::new([[0.0, 6.0], [0.0, 3.0]]);
        const EXPECTED: [[f64; 2]; 2] = [[0.6, -0.7], [-0.2, 0.4]];

        const RESULT: [[f64; 2]; 2] = START.add(ADD).pinv(f64::EPSILON).finish();
        for i in 0..2 {
            for j in 0..2 {
                assert!(float_equal(RESULT[i][j], EXPECTED[i][j], 1e-5));
            }
        }
    }

    #[test]
    fn test_2_x_2_invert() {
        const START: [[f64; 2]; 2] = [[4.0, 7.0], [2.0, 6.0]];
        const EXPECTED: [[f64; 2]; 2] = [[0.6, -0.7], [-0.2, 0.4]];

        const RESULT: [[f64; 2]; 2] = CMatrix::new(START).pinv(f64::EPSILON).finish();
        for i in 0..2 {
            for j in 0..2 {
                assert!(float_equal(RESULT[i][j], EXPECTED[i][j], 1e-5));
            }
        }
    }

    #[test]
    fn check_4_x_4() {
        const START: [[f64; 4]; 4] = [
            [13.0, 17.0, 25.0, 12.0],
            [19.0, 24.0, 16.0, 21.0],
            [29.0, 9.0, 3.0, 14.0],
            [23.0, 27.0, 20.0, 15.0],
        ];
        const EXPECTED: [[f64; 4]; 4] = [
            [
                0.005_304_652_520_926_611,
                -0.053_014_080_851_339_955,
                0.043_653_883_589_643_76,
                0.029_232_366_491_467_134,
            ],
            [
                -0.072_318_473_817_403_15,
                0.004_087_989_098_695_737,
                -0.044_675_880_864_317_695,
                0.093_829_083_122_445,
            ],
            [
                0.077_331_127_116_994_36,
                -0.029_719_031_860_359_485,
                0.003_357_991_045_357_212,
                -0.023_392_382_064_758_938,
            ],
            [
                0.018_931_282_849_912_4,
                0.113_555_252_741_548_25,
                0.009_003_309_324_508_468,
                -0.115_858_802_154_305_37,
            ],
        ];
        const INVERSE: [[f64; 4]; 4] = CMatrix::new(START).pinv(f64::EPSILON).finish();
        for i in 0..4 {
            for j in 0..4 {
                assert!(float_equal(INVERSE[i][j], EXPECTED[i][j], 1e-5))
            }
        }
    }

    #[test]
    fn check_8_x_8() {
        #[rustfmt::skip]
        const START: [[f64;8];8] = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 400000000.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 400000000.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 400000000.0],
        ];

        #[rustfmt::skip]
        const EXPECTED: [[f64;8];8] = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0000000025, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0000000025, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0000000025],
        ];

        const INVERSE: [[f64; 8]; 8] = CMatrix::new(START).pinv(f64::EPSILON).finish();
        for i in 0..8 {
            for j in 0..8 {
                assert!(float_equal(INVERSE[i][j], EXPECTED[i][j], 1e-5))
            }
        }
    }
}
