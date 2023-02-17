#![allow(clippy::needless_range_loop)]
#![allow(clippy::erasing_op)]

use std::{
    cmp::{max, min},
    fmt::Display,
    ops::{Add, Index, IndexMut, Mul, Sub},
};

/// invert a row-major square matrix
pub fn pseudo_invert_square(matrix: Vec<f64>) -> Vec<f64> {
    let tmp = (matrix.len() as f32).sqrt() as usize;
    debug_assert!(tmp.pow(2) == matrix.len());
    let mut uninverted = Matrix {
        vals: matrix,
        rows: tmp,
        columns: tmp,
    };
    uninverted.pinv(f64::EPSILON);
    uninverted.vals
}

/// invert a row-major matrix
pub fn pseudo_invert(matrix: Vec<f64>, row_len: u16) -> Vec<f64> {
    let mut uninverted = Matrix::new(matrix, row_len);
    if let Matrix { rows: 1, .. } = uninverted {
        let mag = uninverted.vals.iter().fold(0.0, |acc, &x| acc + x.powi(2));
        return uninverted.vals.into_iter().map(|x| x / mag).collect();
    }
    // println!("Uninverted matrix:\n{}", &uninverted);
    uninverted.pinv(f64::EPSILON);
    uninverted.vals
}

pub fn mul(left: Vec<f64>, right: &Vec<f64>, shared_dim: usize) -> Vec<f64> {
    let leftlen = left.len();
    let rightlen = right.len();
    let left = Matrix {
        rows: leftlen / shared_dim,
        vals: left,
        columns: shared_dim,
    };
    let right = Matrix {
        rows: shared_dim,
        vals: right.to_vec(),
        columns: rightlen / shared_dim,
    };
    debug_assert_eq!(left.rows * left.columns, leftlen);
    debug_assert_eq!(right.rows * right.columns, rightlen);
    left.mul(right).vals
}

#[derive(Default, Debug, Clone, PartialEq, PartialOrd)]
struct Matrix {
    vals: Vec<f64>,
    rows: usize,
    columns: usize,
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "[")?;
        for row in 0..self.rows {
            write!(f, "\t")?;
            for column in 0..self.columns {
                write!(f, "{:10.5e}, ", self.vals[row * self.columns + column])?;
            }
            writeln!(f)?;
        }
        writeln!(f, "]")?;
        Ok(())
    }
}

impl Mul for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.columns, rhs.rows);
        let mut ret = Vec::new();
        for i in 0..self.rows {
            for j in 0..rhs.columns {
                let mut acc: f64 = 0.0;
                for (k, l) in (0..self.columns).zip(0..rhs.rows) {
                    acc += self.vals[(i * self.columns) + k] * rhs.vals[(l * rhs.columns) + j];
                }
                if acc.is_nan() {
                    println!("{self}\n{rhs}");
                    debug_assert!(false)
                }
                ret.push(acc);
            }
        }
        debug_assert_eq!(ret.len(), (self.rows * rhs.columns));
        debug_assert!(!ret[0].is_nan());
        Matrix {
            vals: ret,
            rows: self.rows,
            columns: rhs.columns,
        }
    }
}

impl Add for Matrix {
    type Output = Matrix;
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert_eq!((self.columns, self.rows), (rhs.columns, rhs.rows));

        let mut ret: Vec<f64> = vec![0.0; self.vals.len()];
        for i in 0..self.rows {
            for j in 0..self.columns {
                ret[i * self.columns + j] =
                    self.vals[i * self.columns + j] + rhs.vals[i * self.columns + j];
            }
        }
        Matrix { vals: ret, ..self }
    }
}

impl Sub for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Self) -> Self::Output {
        debug_assert_eq!((self.columns, self.rows), (rhs.columns, rhs.rows));

        let mut ret: Vec<f64> = vec![0.0; self.vals.len()];
        for i in 0..self.rows {
            for j in 0..self.columns {
                ret[i * self.columns + j] =
                    self.vals[i * self.columns + j] - rhs.vals[i * self.columns + j];
            }
        }
        Matrix { vals: ret, ..self }
    }
}

impl Index<usize> for Matrix {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.vals[index]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.vals[index]
    }
}

impl Matrix {
    #[cfg(test)]
    const EPSILON: f64 = 1e-12;

    fn new(vals: Vec<f64>, row_len: u16) -> Matrix {
        let len = vals.len();
        Matrix {
            vals,
            rows: len / row_len as usize,
            columns: row_len as usize,
        }
    }

    fn get(&self, row: usize, column: usize) -> f64 {
        debug_assert!(row < self.rows && column < self.columns);
        self.vals[self.columns * row + column]
    }

    fn set(&mut self, row: usize, column: usize, value: f64) {
        debug_assert!(row < self.rows && column < self.columns);
        self.vals[self.columns * row + column] = value;
    }

    #[cfg(test)]
    fn transpose(&mut self) {
        let mut ret = Vec::new();
        for i in 0..self.columns {
            for j in 0..self.rows {
                ret.push(self.vals[j * self.columns + i])
            }
        }

        std::mem::swap(&mut self.rows, &mut self.columns);
        self.vals = ret;
    }

    #[cfg(test)]
    fn retain_non_zero_rows(&mut self) {
        self.vals = {
            let chained: &mut Vec<&f64> = &mut Vec::new();
            {
                let mut starts = self
                    .vals
                    .iter()
                    .enumerate()
                    .step_by(self.columns)
                    .map(|(idx, _)| idx);
                let (val_ref, columns) = (&self.vals, self.columns);
                let _: Vec<()> = std::iter::from_fn(|| match starts.next() {
                    Some(idx) => {
                        match val_ref[idx..idx + columns].iter() {
                            x if x.clone().sum::<f64>().abs() > f32::EPSILON as f64 => {
                                chained.extend(x);
                            }
                            _ => (),
                        }
                        Some(())
                    }
                    None => None,
                })
                .collect();
            }

            chained.iter_mut().map(|x| **x).collect()
        };
        self.rows = self.vals.len() / self.columns;
    }

    #[cfg(test)]
    fn swap_rows(&mut self, first: usize, second: usize) {
        let l = first * self.columns..((first + 1) * self.columns);
        let r = second * self.columns..((second + 1) * self.columns);
        let pairator = l.zip(r);
        for (l, r) in pairator {
            // dbg!(l, r);
            self.vals.swap(l, r);
        }
    }

    #[cfg(test)]
    fn multiply_row(&mut self, row: usize, value: f64) {
        for column in 0..self.columns {
            let val_ref = self.vals.get_mut(row * self.columns + column).unwrap();
            if *val_ref == 0.0 {
                continue;
            }
            *val_ref *= value;
        }
    }

    #[cfg(test)]
    fn subtract_mul_row(&mut self, subtract_row: usize, from_row: usize, mul: f64) {
        for column in 0..self.columns {
            let sub = mul * self.vals[subtract_row * self.columns + column];
            self.vals[from_row * self.columns + column] -= sub;
        }
    }

    #[cfg(test)]
    fn row_echelon(&mut self) {
        fn max_i<'a, T>(slice: T) -> usize
        where
            T: Iterator<Item = &'a f64>,
        {
            let mut first: Option<(f64, usize)> = None;
            for (index, val) in slice.enumerate() {
                first = match first {
                    None => Some((*val, index)),
                    Some((f_val, _)) if val.abs() > f_val => Some((val.abs(), index)),
                    _ => first,
                };
            }
            match first {
                Some(tup) => tup.1,
                None => {
                    debug_assert!(false);
                    0
                }
            }
        }
        let m = self.rows;
        let n = self.columns;
        let mut h = 0;
        let mut k = 0;
        while h < m && k < n {
            let column_iter = self
                .vals
                .iter()
                .skip(h * self.columns + k)
                .step_by(self.columns);
            let i_max = h + max_i(column_iter);
            if self.vals[i_max * self.columns + k] == 0.0 {
                k += 1;
            } else {
                self.swap_rows(h, i_max);
                for i in (h + 1)..m {
                    let f = self.vals[i * self.columns + k] / self.vals[h * self.columns + k];
                    self.vals[i * self.columns + k] = 0.0;
                    for j in (k + 1)..n {
                        self.vals[i * self.columns + j] -= self.vals[h * self.columns + j] * f;
                    }
                }
                h += 1;
                k += 1;
            }
        }
    }

    #[cfg(test)]
    fn reduced_row_echelon(&mut self) {
        self.row_echelon();

        for i in 0..self.rows {
            for j in 0..self.columns {
                let val = self.vals[i * self.columns + j];
                if val != 0.0 {
                    self.multiply_row(i, 1.0 / val);
                    break;
                }
            }
        }

        for i in (1..self.rows).rev() {
            'j: for j in 0..self.columns {
                let val = self.vals[i * self.columns + j];
                if val != 0.0 {
                    (0..i).rev().next();
                    for mul_row in (0..i).rev() {
                        let to_zero = self.vals[(mul_row) * self.columns + j];
                        if to_zero != 0.0 {
                            self.subtract_mul_row(i, mul_row, to_zero);
                            // self.vals[mul_row*self.columns + j] = 0.0; // hard-set it to zero to remove precision weirdness
                        }
                    }
                    break 'j;
                }
            }
        }
    }

    #[cfg(test)]
    fn fill_identity(&mut self) {
        let mut base = 1;
        while base < self.rows * self.columns {
            base *= 4;
        }
        let new_square_size = (dbg!(base) as f64).sqrt() as usize;
        let old_square_size = ((self.rows * self.columns) as f64).sqrt() as usize;

        if new_square_size == old_square_size {
            return;
        }

        let size_diff = dbg!(new_square_size - old_square_size);
        let mut ret: Vec<f64> = vec![0.0; base];
        for i in 0..new_square_size {
            for j in 0..new_square_size {
                if i >= size_diff && j >= size_diff {
                    ret[i * new_square_size + j] =
                        self.vals[(i - (size_diff)) * old_square_size + j - (size_diff)];
                } else {
                    ret[i * new_square_size + j] = if i == j { 1.0 } else { 0.0 }
                }
            }
        }
        self.vals = ret;
        self.rows = new_square_size;
        self.columns = new_square_size;
    }

    #[cfg(test)]
    fn trim_identity(&mut self, new_square_size: usize) {
        let mut base = 1;
        while base < self.vals.len() {
            base *= 4;
        }
        let old_square_size = (base as f64).sqrt() as usize;

        if new_square_size == old_square_size {
            return;
        }

        let size_diff = old_square_size - new_square_size;
        let mut ret: Vec<f64> = vec![0.0; base * 4];
        for i in 0..new_square_size {
            for j in 0..new_square_size {
                ret[i * new_square_size + j] =
                    self.vals[(i + size_diff) * old_square_size + j + (size_diff)];
            }
        }
        self.vals = ret;
        self.rows = new_square_size;
        self.columns = new_square_size;
    }

    // Look at using an enum flag for this instead
    fn givens_l(&mut self, m: usize, a: f64, b: f64) {
        let r = (a.powi(2) + b.powi(2)).sqrt();
        if r == 0.0 {
            return;
        }
        let c = a / r;
        let s = -b / r;

        for i in 0..self.columns {
            let s0 = self.get(m, i);
            let s1 = self.get(m + 1, i);
            self.set(m, i, self.get(m, i) + s0 * (c - 1.0));
            self.set(m, i, self.get(m, i) + s1 * (-s));

            self.set(m + 1, i, self.get(m + 1, i) + s0 * (s));
            self.set(m + 1, i, self.get(m + 1, i) + s1 * (c - 1.0));
        }
    }

    fn givens_r(&mut self, m: usize, a: f64, b: f64) {
        let r = (a.powi(2) + b.powi(2)).sqrt();
        if r == 0.0 {
            return;
        }
        let c = a / r;
        let s = -b / r;

        for i in 0..self.rows {
            let s0 = self.get(i, m);
            let s1 = self.get(i, m + 1);
            self.set(i, m, self.get(i, m) + s0 * (c - 1.0));
            self.set(i, m, self.get(i, m) + s1 * (-s));

            self.set(i, m + 1, self.get(i, m + 1) + s0 * (s));
            self.set(i, m + 1, self.get(i, m + 1) + s1 * (c - 1.0));
        }
    }

    fn gemm(&mut self, _k: usize, a: &Matrix, b: &Matrix, alpha: f64, beta: f64) {
        let beta_term = |x| -> f64 {
            if beta == 0.0 || beta == -0.0 {
                0.0_f64
            } else {
                beta * x
            }
        };
        // dbg!(&self, a, b, _k);
        for n in 0..self.rows {
            for m in 0..self.columns {
                let mut axb = 0.0;
                for k in 0.._k {
                    axb += a.vals[k + a.columns * m] * b.vals[n + b.columns * k]
                }
                // dbg!(axb);
                self.vals[self.columns * n + m] =
                    alpha * axb + beta_term(self.vals[n * self.columns + m]);
            }
        }
    }

    fn svd_inner(
        dim: [usize; 2],
        u_working: &mut Matrix,
        s_working: &mut Matrix,
        v_working: &mut Matrix,
        eps: f64,
    ) {
        let n = min(dim[0], dim[1]);
        debug_assert!(dim[0] >= dim[1]);
        let mut house_vec = vec![0.0; max(dim[0], dim[1])];
        for i in 0..n {
            // Column Householder
            {
                let x1 = match s_working.get(i, i) {
                    val if val < 0.0 => -val,
                    val => val,
                };

                let x_inv_norm = {
                    let mut x_inv_norm = 0.0;
                    for j in i..dim[0] {
                        x_inv_norm += s_working.get(j, i).powi(2);
                    }
                    if x_inv_norm > 0.0 {
                        x_inv_norm = 1.0 / x_inv_norm.sqrt();
                    }
                    x_inv_norm
                };

                dbg!(x_inv_norm);

                let (alpha, beta) = {
                    let mut alpha = (1.0 + x1 * x_inv_norm).sqrt();
                    let beta = x_inv_norm / alpha;
                    if x_inv_norm == 0.0 {
                        alpha = 0.0;
                    } // nothing to do
                    (alpha, beta)
                };

                house_vec[i] = -alpha;
                for j in (i + 1)..dim[0] {
                    house_vec[j] = -beta * s_working.get(j, i);
                }

                if s_working.get(i, i) < 0.0 {
                    for j in (i + 1)..dim[0] {
                        house_vec[j] = -house_vec[j];
                    }
                }
            }

            dbg!(&house_vec);

            for k in i..dim[1] {
                let mut dot_prod = 0.0;
                for j in i..dim[0] {
                    dot_prod += s_working.get(j, k) * house_vec[j];
                }
                for j in i..dim[0] {
                    s_working.set(j, k, s_working.get(j, k) - (dot_prod * house_vec[j]));
                }
            }
            for k in 0..dim[0] {
                let mut dot_prod = 0.0;
                for j in i..dim[0] {
                    dot_prod += u_working.get(k, j) * house_vec[j];
                }
                dbg!(k, dot_prod);
                for j in i..dim[0] {
                    let val = u_working.get(k, j) - (dot_prod * house_vec[j]);
                    dbg!((val, i, j, k));
                    u_working.set(k, j, val);
                }
            }

            if i >= n - 1 {
                continue;
            }

            {
                let x1 = s_working.get(i, i + 1).abs();

                let x_inv_norm = {
                    let mut x_inv_norm = 0.0;
                    for j in (i + 1)..dim[1] {
                        x_inv_norm += s_working.get(i, j).powi(2);
                    }
                    if x_inv_norm > 0.0 {
                        x_inv_norm = 1.0 / x_inv_norm.sqrt();
                    }
                    x_inv_norm
                };

                let (alpha, beta) = {
                    let mut alpha = (1.0 + x1 * x_inv_norm).sqrt();
                    let beta = x_inv_norm / alpha;
                    if x_inv_norm == 0.0 {
                        alpha = 0.0; // nothing to do
                    }
                    (alpha, beta)
                };

                house_vec[i + 1] = -alpha;
                for j in (i + 2)..dim[1] {
                    house_vec[j] = -beta * s_working.get(i, j);
                }
                if s_working.get(i, i + 1) < 0.0 {
                    for j in (i + 2)..dim[1] {
                        house_vec[j] = -house_vec[j];
                    }
                }
            }
            for k in i..dim[0] {
                let mut dot_prod = 0.0;
                for j in (i + 1)..dim[1] {
                    dot_prod += s_working.get(k, j) * house_vec[j];
                }
                for j in (i + 1)..dim[1] {
                    s_working.set(k, j, s_working.get(k, j) - (dot_prod * house_vec[j]));
                }
            }
            for k in 0..dim[1] {
                let mut dot_prod = 0.0;
                for j in (i + 1)..dim[1] {
                    dot_prod += v_working.get(j, k) * house_vec[j];
                }
                for j in (i + 1)..dim[1] {
                    v_working.set(j, k, v_working.get(j, k) - (dot_prod * house_vec[j]));
                }
            }
        }

        dbg!(&v_working);

        let mut k0 = 0;
        let eps = if eps < 0.0 {
            let mut eps = 1.0;
            while eps + 1.0 > 1.0 {
                eps *= 0.5;
            }
            eps *= 64.0;
            eps
        } else {
            eps
        };

        while k0 < dim[1] - 1 {
            // Diagonalization
            let s_max = {
                let mut s_max = 0.0;
                for i in 0..dim[1] {
                    let tmp = s_working.get(i, i).abs();
                    if tmp > s_max {
                        s_max = tmp
                    }
                }
                for i in 0..(dim[1] - 1) {
                    let tmp = s_working.get(i, i + 1).abs();
                    if tmp > s_max {
                        s_max = tmp
                    }
                }
                s_max
            };

            while k0 < dim[1] - 1 && s_working.get(k0, k0 + 1).abs() <= eps * s_max {
                k0 += 1;
            }
            if k0 == dim[1] - 1 {
                continue;
            }

            let n = {
                let mut n = k0 + 2;
                while n < dim[1] && s_working.get(n - 1, n).abs() > eps * s_max {
                    n += 1;
                }
                dbg!(n)
            };

            let (alpha, beta) = {
                if n - k0 == 2
                    && s_working.get(k0, k0).abs() < eps * s_max
                    && s_working.get(k0 + 1, k0 + 1).abs() < eps * s_max
                {
                    // Compute mu
                    (0.0, 1.0)
                } else {
                    let mut c_vec = [0.0; 4];
                    c_vec[0 * 2] = s_working.get(n - 2, n - 2) * s_working.get(n - 2, n - 2);
                    if n - k0 > 2 {
                        c_vec[0 * 2] += s_working.get(n - 3, n - 2) * s_working.get(n - 3, n - 2);
                    }
                    c_vec[1] = s_working.get(n - 2, n - 2) * s_working.get(n - 2, n - 1);
                    c_vec[2] = s_working.get(n - 2, n - 2) * s_working.get(n - 2, n - 1);
                    c_vec[2 + 1] = s_working.get(n - 1, n - 1) * s_working.get(n - 1, n - 1)
                        + s_working.get(n - 2, n - 1) * s_working.get(n - 2, n - 1);

                    let (b, d) = {
                        let mut b = -(c_vec[0 * 2] + c_vec[2 + 1]) / 2.0;
                        let mut c = c_vec[0 * 2] * c_vec[2 + 1] - c_vec[1] * c_vec[2];
                        let mut d = 0.0;
                        if (b.powi(2) - c).abs() > eps * b.powi(2) {
                            d = (b.powi(2) - c).sqrt();
                        } else {
                            b = (c_vec[0 * 2] - c_vec[2 + 1]) / 2.0;
                            c = -c_vec[1] * c_vec[2];
                            if b * b - c > 0.0 {
                                d = (b * b - c).sqrt();
                            }
                        }
                        (b, d)
                    };

                    let lambda1 = -b + d;
                    let lambda2 = -b - d;

                    let d1 = (lambda1 - c_vec[2 + 1]).abs();
                    let d2 = (lambda2 - c_vec[2 + 1]).abs();
                    let mu = if d1 < d2 { lambda1 } else { lambda2 };

                    let alpha = s_working.get(k0, k0).powi(2) - dbg!(mu);
                    let beta = s_working.get(k0, k0) * s_working.get(k0, k0 + 1);
                    (alpha, beta)
                }
            };
            {
                let mut alpha = alpha;
                let mut beta = beta;
                for k in k0..(n - 1) {
                    s_working.givens_r(k, alpha, beta);
                    v_working.givens_l(k, alpha, beta);

                    alpha = s_working.get(k, k);
                    beta = s_working.get(k + 1, k);
                    s_working.givens_l(k, alpha, beta);
                    u_working.givens_r(k, alpha, beta);

                    alpha = s_working.get(k, k + 1);

                    if k != n - 2 {
                        beta = s_working.get(k, k + 2);
                    }
                }
            }

            {
                // Make S bi-diagonal again
                for i0 in k0..(n - 1) {
                    for i1 in 0..dim[1] {
                        if i0 > i1 || i0 + 1 < i1 {
                            s_working.set(i0, i1, 0.0);
                        }
                    }
                }
                for i0 in 0..dim[0] {
                    for i1 in k0..(n - 1) {
                        if i0 > i1 || i0 + 1 < i1 {
                            s_working.set(i0, i1, 0.0);
                        }
                    }
                }
                for i in 0..(dim[1] - 1) {
                    if s_working.get(i, i + 1).abs() <= eps * s_max {
                        s_working.set(i, i + 1, 0.0);
                    }
                }
            }
        }
    }

    fn svd(&mut self, m: usize, n: usize, k: usize, epsilon: f64) {
        let dim = [max(m, n), min(m, n)];
        let mut s_working = Matrix::new(vec![0.0; dim[0] * dim[1]], dim[1] as u16);
        let mut u_working = Matrix::new(vec![0.0; dim[0] * dim[0]], dim[0] as u16);
        let mut v_working = Matrix::new(vec![0.0; dim[1] * dim[1]], dim[1] as u16);

        let ldu = m;
        let ldv = k;
        let mut u_out = Matrix::new(vec![0.0; m * k], ldu as u16);
        let mut s_out = Matrix::new(vec![0.0; k], k as u16);
        let mut vt_out = Matrix::new(vec![0.0; k * n], ldv as u16);

        if dim[1] == m {
            for i in 0..dim[0] {
                for j in 0..dim[1] {
                    s_working[i * dim[1] + j] = self.vals[i * m + j];
                }
            }
        } else {
            for i in 0..dim[0] {
                for j in 0..dim[1] {
                    s_working[i * dim[1] + j] = self.vals[j * self.columns + i];
                }
            }
        }
        for i in 0..dim[0] {
            u_working[i * dim[0] + i] = 1.0;
        }
        for i in 0..dim[1] {
            v_working[i * dim[1] + i] = 1.0;
        }

        println!("u: {}", u_working);

        // println!("u: {}\ns: {}\nv: {}", u_working, s_working, v_working);
        Self::svd_inner(dim, &mut u_working, &mut s_working, &mut v_working, epsilon);

        dbg!(&v_working);

        let less_zero_sign = |x: f64| -> f64 {
            if x < 0.0 {
                return -1.0;
            }
            1.0
        };

        for i in 0..dim[1] {
            // Set S
            s_out[i] = s_working[i * dim[1] + i];
        }
        if dim[1] == m {
            // Set U
            for i in 0..dim[1] {
                for j in 0..m {
                    u_out[j + ldu * i] = v_working[j + i * dim[1]] * (less_zero_sign(s_out[i]));
                }
            }
        } else {
            for i in 0..dim[1] {
                for j in 0..m {
                    u_out[j + ldu * i] = u_working[i + j * dim[0]] * (less_zero_sign(s_out[i]));
                }
            }
        }
        dbg!(&u_out);

        if dim[0] == n {
            // Set V
            for i in 0..n {
                for j in 0..dim[1] {
                    vt_out[j + ldv * i] = u_working[j + i * dim[0]];
                }
            }
        } else {
            for i in 0..n {
                for j in 0..dim[1] {
                    vt_out[j + ldv * i] = v_working[i + j * dim[1]];
                }
            }
        }
        for i in 0..dim[1] {
            s_out[i] = s_out[i] * (less_zero_sign(s_out[i]));
        }

        //set all below epsilon to zero
        let eps_ = epsilon; //s_out[0] * epsilon;
        for i in 0..k {
            if s_out[i] < eps_ {
                s_out[i] = 0.0;
            } else {
                s_out[i] = 1.0 / s_out[i];
            }
        }

        for i in 0..m {
            for j in 0..k {
                u_out[i + j * m] *= s_out[j];
            }
        }

        let mut ret_matrix = Matrix::new(vec![0.0; n * m], n as u16);

        dbg!(&u_out);
        dbg!(&vt_out);

        // fn gemm(&mut self, _k: usize, a: &Matrix, b: &Matrix, alpha: f64, beta: f64) {
        ret_matrix.gemm(k, &vt_out, &u_out, 1.0, 0.0);

        *self = ret_matrix;
    }

    pub fn pinv(&mut self, epsilon: f64) {
        if self.rows * self.columns == 0 {
            return;
        }

        let m = self.columns;
        let n = self.rows;

        let k = n.min(m);

        self.svd(m, n, k, epsilon)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float_equal(one: f64, two: f64) -> bool {
        match (one, two) {
            (a, b) if (a - b).abs() < Matrix::EPSILON => true,
            (a, b) if a == -0.0 || b == -0.0 => a + b == 0.0 || -(a + b) == 0.0,
            _ => false,
        }
    }

    fn generate_identity(size: usize) -> Vec<f64> {
        let new_square_size = (size as f64).sqrt() as usize;
        let mut ret: Vec<f64> = vec![0.0; size];
        for i in 0..new_square_size {
            for j in 0..new_square_size {
                if i == j {
                    ret[i * new_square_size + j] = 1.0
                }
            }
        }
        ret
    }

    fn is_identity(check: &Matrix) -> bool {
        for i in 0..check.columns {
            for j in 0..check.rows {
                if i == j {
                    if let false = float_equal(check.get(j, i), 1.0) {
                        dbg!(j, i, check.get(j, i));
                        return false;
                    }
                } else if let false = float_equal(check.get(j, i), 0.0) {
                    dbg!(j, i, check.get(j, i));
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_2_x_2_invert() {
        let vals = vec![4.0, 7.0, 2.0, 6.0];
        let inv_given = vec![0.6, -0.7, -0.2, 0.4];
        let inv = pseudo_invert(vals, 2);

        // let mut tmp = Matrix::new(vals, 2);
        // tmp.svd(2, 2, 2, f64::EPSILON);
        // dbg!(tmp);
        for i in 0..inv_given.len() {
            assert!(float_equal(inv[i], inv_given[i]))
        }
    }

    // #[test]
    // fn test_2_x_2_invert_zeros() {
    //     let vals = vec![0.0, 0.0, 0.0, 0.0];
    //     let inv_given = vec![0.0, 0.0, 0.0, 0.0];
    //     let inv = dbg!(pseudo_invert_square(vals));
    //     for i in 0..inv_given.len() {
    //         assert!(float_equal(inv[i], inv_given[i]))
    //     }
    // }

    #[test]
    fn check_invert_identity() {
        for i in 1..6 {
            let ident = generate_identity(4_usize.pow(i));
            let inv = pseudo_invert_square(generate_identity(4_usize.pow(i)));
            for j in 0..ident.len() {
                assert!(float_equal(ident[j], inv[j]))
            }
        }
    }

    #[test]
    fn check_4_x_4() {
        let vals = vec![
            13.0, 17.0, 25.0, 12.0, 19.0, 24.0, 16.0, 21.0, 29.0, 9.0, 3.0, 14.0, 23.0, 27.0, 20.0,
            15.0,
        ];
        let inv_given = vec![
            0.005_304_652_520_926_611,
            -0.053_014_080_851_339_955,
            0.043_653_883_589_643_76,
            0.029_232_366_491_467_134,
            -0.072_318_473_817_403_15,
            0.004_087_989_098_695_737,
            -0.044_675_880_864_317_695,
            0.093_829_083_122_445,
            0.077_331_127_116_994_36,
            -0.029_719_031_860_359_485,
            0.003_357_991_045_357_212,
            -0.023_392_382_064_758_938,
            0.018_931_282_849_912_4,
            0.113_555_252_741_548_25,
            0.009_003_309_324_508_468,
            -0.115_858_802_154_305_37,
        ];
        let inv = pseudo_invert_square(vals);
        for i in 0..inv_given.len() {
            assert!(float_equal(inv[i], inv_given[i]))
        }
    }

    #[test]
    fn check_8_x_8() {
        #[rustfmt::skip]
        let vals = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 400000000.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 400000000.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 400000000.0,
        ];

        #[rustfmt::skip]
        let inv_given = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0000000025, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0000000025, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0000000025,
        ];

        let inv = pseudo_invert_square(vals.clone());
        for i in 0..inv_given.len() {
            assert!(float_equal(inv[i], inv_given[i]))
        }

        //self * inverse = identity
        let mut l = Matrix::new(vals, 8);
        let mut r = Matrix::new(inv, 8);
        l.fill_identity();
        r.fill_identity();
        let mut mul = l * r;
        mul.trim_identity(8);
        let mul = mul.vals;
        let ident = generate_identity(64);
        for i in 0..ident.len() {
            assert!(float_equal(mul[i], ident[i]))
        }
    }

    #[test]
    fn non_square_mul() {
        let l = Matrix::new(
            vec![1.0, 0.0, 1.0, 2.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0],
            3,
        );
        let r = Matrix::new(vec![1.0, 2.0, 1.0, 2.0, 3.0, 1.0, 4.0, 2.0, 2.0], 3);
        let expected = vec![5.0, 4.0, 3.0, 8.0, 9.0, 5.0, 6.0, 5.0, 3.0, 11.0, 9.0, 6.0];

        let result = dbg!(l * r);

        for i in 0..expected.len() {
            assert!(float_equal(result.vals[i], expected[i]))
        }
    }

    #[test]
    fn non_square_transpose() {
        let mut l = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3);
        let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];

        //transpose
        let original = l.vals.clone();
        l.transpose();
        for i in 0..expected.len() {
            assert!(float_equal(l.vals[i], expected[i]))
        }

        //and go back
        l.transpose();
        for i in 0..original.len() {
            assert!(float_equal(l.vals[i], original[i]))
        }
    }

    #[test]
    fn square_row_echelon() {
        let mut l = Matrix::new(vec![2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0], 3);

        l.row_echelon();
        assert!(float_equal(l.vals[l.rows * l.columns - l.columns], 0.0));
    }

    #[test]
    fn non_square_row_echelon() {
        let mut l = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3);

        //transpose
        l.row_echelon();
        assert!(float_equal(l.vals[l.rows * l.columns - l.columns], 0.0))
    }

    #[test]
    fn non_square_reduced_row_echelon() {
        let mut l = Matrix::new(vec![7.0, 3.0, -1.0, 0.0, 1.0, 7.0], 3);
        let expected = Matrix::new(vec![1.0, 0.0, -3.142857142857143, 0.0, 1.0, 7.0], 3);

        l.reduced_row_echelon();
        for i in 0..expected.vals.len() {
            assert!(float_equal(l.vals[i], expected.vals[i]))
        }
    }

    #[test]
    fn check_8_x_8_pseudo_inverse_is_inverse() {
        #[rustfmt::skip]
        let vals = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 400000000.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 400000000.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 400000000.0,
        ];

        #[rustfmt::skip]
        let inv_given = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0000000025, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0000000025, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0000000025,
        ];

        let inv = pseudo_invert(vals.clone(), 8);
        for i in 0..inv_given.len() {
            assert!(float_equal(inv[i], inv_given[i]))
        }

        //self * inverse = identity
        let mut l = Matrix::new(vals, 8);
        let mut r = Matrix::new(inv, 8);
        l.fill_identity();
        r.fill_identity();
        let mut mul = l * r;
        mul.trim_identity(8);
        let mul = mul.vals;
        let ident = generate_identity(64);
        for i in 0..ident.len() {
            assert!(float_equal(mul[i], ident[i]))
        }
    }

    #[test]
    fn check_8_x_8_pseudo_inverse_is_inverse_2() {
        #[rustfmt::skip]
        let vals = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 400000000.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 400000000.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 400000000.0,
        ];

        #[rustfmt::skip]
        let inv_given = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0000000025, 0.0, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0000000025, 0.0, 
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0000000025,
        ];

        let mut inv = Matrix::new(vals.clone(), 8);
        inv.pinv(f64::EPSILON);
        println!("{}", &inv);
        for i in 0..inv_given.len() {
            assert!(float_equal(inv.vals[i], inv_given[i]))
        }

        //self * inverse = identity
        let mut l = Matrix::new(vals, 8);
        let mut r = Matrix::new(inv.vals, 8);
        l.fill_identity();
        r.fill_identity();
        let mut mul = l * r;
        mul.trim_identity(8);
        let mul = mul.vals;
        let ident = generate_identity(64);
        for i in 0..ident.len() {
            assert!(float_equal(mul[i], ident[i]))
        }
    }

    #[test]
    fn check_retain_non_zer0_rows() {
        #[rustfmt::skip]
        let vals = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 75.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        #[rustfmt::skip]
        let given = vec![
            100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 75.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        let mut l = Matrix::new(vals, 12);

        l.retain_non_zero_rows();
        dbg!(&l);

        for i in 0..given.len() {
            assert!(float_equal(l.vals[i], given[i]))
        }
    }

    #[test]
    fn check_pseudo_inverse_non_square() {
        #[rustfmt::skip]
        let vals = vec![
            100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 75.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];

        #[rustfmt::skip]
        let inv_given = vec![
            0.01, 0.0, 
            0.0, 0.013333333333333334, 
            0.0, 0.0, 
            0.0, 0.0, 
            0.0, 0.0, 
            0.0, 0.0,
            0.0, 0.0, 
            0.0, 0.0, 
            0.0, 0.0, 
            0.0, 0.0, 
            0.0, 0.0, 
            0.0, 0.0,
        ];

        let inv = dbg!(pseudo_invert(vals, 12));
        for i in 0..inv_given.len() {
            assert!(float_equal(inv[i], inv_given[i]))
        }
    }

    #[test]
    fn test_pseudo_invert_12x12() {
        let vals = vec![
            1.53000e2, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0,
            0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 1.58000e2, 0.00000e0, 0.00000e0,
            0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0,
            0.00000e0, 0.00000e0, 1.64000e2, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0,
            0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 8.70000e1,
            0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0,
            0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 1.78000e2, 0.00000e0, 0.00000e0, 0.00000e0,
            0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0,
            0.00000e0, 1.83000e2, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0,
            0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 1.51000e2, 6.16695e3, 0.00000e0,
            0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0,
            7.80000e1, 0.00000e0, 0.00000e0, 6.73872e3, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0,
            0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 1.03000e2, 0.00000e0, 0.00000e0, 0.00000e0,
            1.40759e4, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0,
            0.00000e0, 1.67000e2, 0.00000e0, 0.00000e0, 0.00000e0, 1.86249e4, 0.00000e0, 0.00000e0,
            0.00000e0, 0.00000e0, 0.00000e0, 9.80000e1, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0,
            0.00000e0, 0.00000e0, 3.07876e3, 0.00000e0, 0.00000e0, 0.00000e0, 1.90000e2, 0.00000e0,
            0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 0.00000e0, 7.75973e3,
        ];

        let inv = dbg!(pseudo_invert(vals.clone(), 12));
        let val_mat = Matrix::new(vals, 12);
        let inv_mat = Matrix::new(inv, 12);
        let ident = val_mat * inv_mat;
        println!("{}", ident);
        dbg!(f32::EPSILON, f64::EPSILON);

        is_identity(&ident);
    }
}
