//! Dense steps of the block Krylov SVD, on faer 0.24 with explicit `Par::Seq` inside each
//! task so no result depends on faer's global parallelism or the rayon pool size.
//!
//! - `orthonormalize`: thin Q of a tall matrix by TSQR. Rows are split into a fixed number
//!   of chunks (from the shape only), each chunk is QR-factored in parallel, the stacked R
//!   factors are QR-factored once more, and each chunk's Householder reflectors are applied
//!   to its slice of the second Q. Works in place: the only extra memory is one chunk-sized
//!   buffer per task and the stacked R (chunks x l x l).
//! - `thin_svd`: faer's SVD of the small genes x l matrix.
//! - `gaussian`: the test matrix, from ChaCha8 seeded with the PCA seed.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::householder::{
    apply_block_householder_sequence_on_the_left_in_place_scratch,
    apply_block_householder_sequence_on_the_left_in_place_with_conj,
};
use faer::linalg::qr::no_pivoting::factor::{qr_in_place, qr_in_place_scratch, recommended_block_size};
use faer::linalg::svd::{svd, svd_scratch, ComputeSvdVectors};
use faer::reborrow::{Reborrow, ReborrowMut};
use faer::{Conj, Mat, MatMut, MatRef, Par};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::error::{Error, Result};

/// Target rows per TSQR chunk. Matrices with at most this many rows take one plain QR.
pub(super) const TSQR_ROWS: usize = 8192;

/// One factored chunk: Householder block coefficients and the R factor (s x l).
struct Factored {
    coeff: Mat<f64>,
    r: Mat<f64>,
}

/// Factor `a` (m x l, m >= 1) in place: Householder basis below the diagonal, R in the
/// upper triangle. Returns the block coefficients and a copy of R (min(m, l) x l).
fn factor_in_place(mut a: MatMut<'_, f64>) -> Factored {
    let (m, l) = a.shape();
    let s = m.min(l);
    let bs = recommended_block_size::<f64>(m, l);
    let mut coeff = Mat::<f64>::zeros(bs, s);
    let mut mem = MemBuffer::new(qr_in_place_scratch::<f64>(m, l, bs, Par::Seq, Default::default()));
    qr_in_place(a.rb_mut(), coeff.as_mut(), Par::Seq, MemStack::new(&mut mem), Default::default());
    let r = Mat::from_fn(s, l, |i, j| if i <= j { a[(i, j)] } else { 0.0 });
    Factored { coeff, r }
}

/// `target <- H target`, with H the Householder sequence stored in `basis` / `coeff`.
fn apply_q(basis: MatRef<'_, f64>, coeff: MatRef<'_, f64>, target: MatMut<'_, f64>) {
    let s = coeff.ncols();
    let basis = basis.subcols(0, s);
    let mut mem = MemBuffer::new(apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(
        basis.nrows(),
        coeff.nrows(),
        target.ncols(),
    ));
    apply_block_householder_sequence_on_the_left_in_place_with_conj(
        basis,
        coeff,
        Conj::No,
        target,
        Par::Seq,
        MemStack::new(&mut mem),
    );
}

/// Replace `a` (m x l, m >= l) by an orthonormal basis Q (m x l) with range(Q) ⊇ range(a).
/// Sequential: used for one chunk, the stacked R, and small genes x l matrices.
fn orthonormalize_seq(mut a: MatMut<'_, f64>) {
    let (m, l) = a.shape();
    debug_assert!(m >= l);
    let f = factor_in_place(a.rb_mut());
    let mut q = Mat::<f64>::from_fn(m, l, |i, j| if i == j { 1.0 } else { 0.0 });
    apply_q(a.rb(), f.coeff.as_ref(), q.as_mut());
    a.copy_from(&q);
}

/// In-place thin Q of a tall matrix (m x l, m >= l) by TSQR with a fixed row partition, so
/// the result is bitwise identical for any number of threads.
pub(super) fn orthonormalize(a: MatMut<'_, f64>) {
    orthonormalize_chunked(a, TSQR_ROWS);
}

pub(super) fn orthonormalize_chunked(mut a: MatMut<'_, f64>, chunk_rows: usize) {
    let (m, l) = a.shape();
    assert!(m >= l, "orthonormalize needs a tall matrix");
    let chunk_rows = chunk_rows.max(2 * l).max(1);
    let count = m.div_ceil(chunk_rows);
    if count <= 1 {
        orthonormalize_seq(a);
        return;
    }
    // Phase 1: factor every chunk in place (equal-sized chunks, each with at least l rows).
    let factored: Vec<Factored> = a.rb_mut().par_row_partition_mut(count).map(factor_in_place).collect();
    // Phase 2: QR of the stacked R factors.
    let total: usize = factored.iter().map(|f| f.r.nrows()).sum();
    let mut stacked = Mat::<f64>::zeros(total, l);
    let mut offsets = Vec::with_capacity(count);
    let mut off = 0;
    for f in &factored {
        stacked.as_mut().subrows_mut(off, f.r.nrows()).copy_from(&f.r);
        offsets.push(off);
        off += f.r.nrows();
    }
    orthonormalize_seq(stacked.as_mut());
    // Phase 3: Q_chunk = H_chunk [Q2_chunk; 0].
    a.par_row_partition_mut(count).zip(factored.par_iter().zip(offsets.par_iter())).for_each(|(mut ac, (f, &off))| {
        let (mc, s) = (ac.nrows(), f.r.nrows());
        let mut t = Mat::<f64>::zeros(mc, l);
        t.as_mut().subrows_mut(0, s).copy_from(stacked.as_ref().subrows(off, s));
        apply_q(ac.rb(), f.coeff.as_ref(), t.as_mut());
        ac.copy_from(&t);
    });
}

/// Thin SVD `a = U diag(s) V^T` (a: m x n), singular values in non-increasing order.
pub(super) fn thin_svd(a: MatRef<'_, f64>) -> Result<(Mat<f64>, Vec<f64>, Mat<f64>)> {
    let (m, n) = a.shape();
    let k = m.min(n);
    let mut u = Mat::<f64>::zeros(m, k);
    let mut v = Mat::<f64>::zeros(n, k);
    let mut s = faer::diag::Diag::<f64>::zeros(k);
    let mut mem = MemBuffer::new(svd_scratch::<f64>(
        m,
        n,
        ComputeSvdVectors::Thin,
        ComputeSvdVectors::Thin,
        Par::Seq,
        Default::default(),
    ));
    svd(a, s.as_mut(), Some(u.as_mut()), Some(v.as_mut()), Par::Seq, MemStack::new(&mut mem), Default::default())
        .map_err(|e| Error::Backend { context: "faer thin SVD", message: format!("{e:?}") })?;
    let s = (0..k).map(|i| s[i]).collect();
    Ok((u, s, v))
}

/// Standard normal m x l test matrix from ChaCha8 seeded with `seed` (Box-Muller), filled
/// column by column, so adding oversampling columns leaves the first ones unchanged.
pub(super) fn gaussian(m: usize, l: usize, seed: u64) -> Mat<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut values = Vec::with_capacity(m * l + 1);
    while values.len() < m * l {
        let u1: f64 = 1.0 - rng.random::<f64>(); // (0, 1]
        let u2: f64 = rng.random::<f64>();
        let r = (-2.0 * u1.ln()).sqrt();
        let (sin, cos) = (std::f64::consts::TAU * u2).sin_cos();
        values.push(r * cos);
        values.push(r * sin);
    }
    Mat::from_fn(m, l, |i, j| values[j * m + i])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(m: usize, l: usize, seed: u64) -> Mat<f64> {
        gaussian(m, l, seed)
    }

    fn check_q(q: MatRef<'_, f64>, a: MatRef<'_, f64>) {
        let l = q.ncols();
        let qtq = q.transpose() * q;
        for i in 0..l {
            for j in 0..l {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((qtq[(i, j)] - want).abs() < 1e-12, "Q^T Q ({i},{j}) = {}", qtq[(i, j)]);
            }
        }
        // a lies in range(Q): a - Q Q^T a = 0.
        let resid = a - q * (q.transpose() * a);
        let scale = a.norm_l2();
        assert!(resid.norm_l2() <= 1e-12 * scale, "residual {}", resid.norm_l2());
    }

    #[test]
    fn tsqr_matches_range_and_is_orthonormal() {
        for &(m, l, chunk) in &[(50, 7, 10_000), (1000, 13, 100), (1001, 40, 81), (333, 33, 66)] {
            let a = random(m, l, 7);
            let mut q = a.clone();
            orthonormalize_chunked(q.as_mut(), chunk);
            check_q(q.as_ref(), a.as_ref());
        }
    }

    #[test]
    fn tsqr_rank_deficient() {
        // 3 independent columns repeated: Q is still orthonormal and spans a.
        let base = random(500, 3, 1);
        let a = Mat::from_fn(500, 9, |i, j| base[(i, j % 3)] * (1.0 + j as f64));
        let mut q = a.clone();
        orthonormalize_chunked(q.as_mut(), 64);
        check_q(q.as_ref(), a.as_ref());
    }

    #[test]
    fn tsqr_is_thread_count_independent() {
        let a = random(5000, 20, 3);
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut q = a.clone();
                orthonormalize_chunked(q.as_mut(), 300);
                q
            })
        };
        let q1 = run(1);
        for t in [2, 3, 4] {
            assert!(q1 == run(t), "TSQR differs between 1 and {t} threads");
        }
    }

    #[test]
    fn gaussian_is_seeded_and_standard() {
        let a = gaussian(400, 50, 0);
        assert!(a == gaussian(400, 50, 0));
        assert!(a != gaussian(400, 50, 1));
        // First columns do not depend on the number of columns.
        let b = gaussian(400, 60, 0);
        assert!(a.as_ref() == b.as_ref().subcols(0, 50));
        let n = (400 * 50) as f64;
        let mean: f64 = (0..50).map(|j| a.col(j).iter().sum::<f64>()).sum::<f64>() / n;
        let var: f64 = (0..50).map(|j| a.col(j).iter().map(|x| x * x).sum::<f64>()).sum::<f64>() / n;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
    }
}
