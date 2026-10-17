//! The implicitly scaled operator `Z = (X - 1 mu^T) diag(1/sigma)` and the one-pass gene
//! statistics it needs.
//!
//! `X` is the borrowed CSR (cells x genes). Neither `Z` nor a transposed copy of `X` is ever
//! built:
//!
//! - `Z V   = X (S V) - 1 (mu^T S V)` with `S = diag(1/sigma)`: one row-parallel pass over
//!   the CSR. Each output row depends on one input row only, so it is deterministic.
//! - `Z^T U = S (X^T U - mu (1^T U))`: also a row-parallel pass over the CSR, scattering
//!   `x_ij * U[i, :]` into a small genes x l accumulator. Row blocks are fixed by the input
//!   shape (never by the thread count) and their partial accumulators are summed by a
//!   fixed binary tree, so the floating-point summation order, and therefore the result, is
//!   identical for any number of threads.
//!
//! Why not a CSC copy for `X^T U`: a gene-parallel gather over a CSC copy needs no
//! reduction, but it costs a second copy of the HVG matrix (8 bytes per non-zero as
//! u32 + f32, 12 as i32 + f64) and reads `U[i, :]` once per non-zero from all over an
//! n x l matrix that does not fit in cache at large n. The scatter reads `U` once,
//! sequentially, and keeps the genes x l accumulator (640 KB at 2,000 HVGs and l = 40) in
//! cache. The price is one accumulator per live tree node, bounded by roughly
//! threads x log2(blocks) of them and by at most `MAX_BLOCKS` leaves.

use faer::reborrow::ReborrowMut;
use faer::{Mat, MatMut, MatRef};
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::types::{CsrView, Idx, Real};

/// Upper bound on the number of row blocks for the reductions (stats and `Z^T U`).
const MAX_BLOCKS: usize = 256;
/// Lower bound on rows per reduction block.
const MIN_BLOCK_ROWS: usize = 64;
/// A leaf should do at least this many times more scatter work than it spends zeroing and
/// merging its accumulator.
const ACC_WORK_RATIO: usize = 64;
/// Rows per task for the row-parallel `Z V` product (no reduction, so any size is exact).
const ZV_ROWS: usize = 256;
/// Rows of `U` transposed into a local row-major buffer at a time inside a `Z^T U` leaf.
const ZTU_SUB_ROWS: usize = 256;

/// Fixed partition of the rows for the reductions. Depends on the input shape only.
#[derive(Clone, Copy, Debug)]
pub(super) struct Blocks {
    pub rows: usize,
    pub count: usize,
}

impl Blocks {
    pub fn new(n: usize, p: usize, nnz: usize) -> Self {
        let avg_row_nnz = (nnz / n.max(1)).max(1);
        let for_acc = (ACC_WORK_RATIO * (p + 1)).div_ceil(avg_row_nnz);
        let rows = n.div_ceil(MAX_BLOCKS).max(for_acc).max(MIN_BLOCK_ROWS).min(n.max(1));
        Self { rows, count: n.div_ceil(rows).max(1) }
    }

    #[inline]
    fn range(&self, b: usize, n: usize) -> (usize, usize) {
        (b * self.rows, ((b + 1) * self.rows).min(n))
    }
}

/// Deterministic parallel reduction: leaves `lo..hi` combined by a binary tree whose shape
/// depends only on the leaf count, never on the scheduling.
fn tree_reduce<A, L, M>(lo: usize, hi: usize, leaf: &L, merge: &M) -> Result<A>
where
    A: Send,
    L: Fn(usize) -> Result<A> + Sync,
    M: Fn(&mut A, A) + Sync,
{
    if hi - lo == 1 {
        return leaf(lo);
    }
    let mid = lo + (hi - lo) / 2;
    let (a, b) = rayon::join(|| tree_reduce(lo, mid, leaf, merge), || tree_reduce(mid, hi, leaf, merge));
    let mut a = a?;
    merge(&mut a, b?);
    Ok(a)
}

/// Per-gene statistics for the implicit scaling.
#[derive(Clone, Debug)]
pub(super) struct GeneStats {
    pub mean: Vec<f64>,
    /// Sample standard deviation (n - 1 denominator); 0 for constant genes.
    pub sd: Vec<f64>,
    /// Exactly constant genes (all zero, or the same stored value in every cell).
    pub constant: Vec<bool>,
}

struct StatsAcc {
    sum: Vec<f64>,
    sumsq: Vec<f64>,
    min: Vec<f64>,
    max: Vec<f64>,
    nnz: Vec<u64>,
}

/// One parallel pass over the CSR: per-gene sum, sum of squares, min/max of stored values and
/// stored-entry counts, reduced by the fixed tree. Also validates the sparse structure, so a
/// `CsrView` built without `CsrView::new` cannot cause out-of-bounds access later.
pub(super) fn gene_stats<I: Idx, T: Real>(x: &CsrView<'_, I, T>, blocks: Blocks) -> Result<GeneStats> {
    let (n, p, nnz) = (x.n_rows, x.n_cols, x.data.len());
    let leaf = |b: usize| -> Result<StatsAcc> {
        let (r0, r1) = blocks.range(b, n);
        let mut acc = StatsAcc {
            sum: vec![0.0; p],
            sumsq: vec![0.0; p],
            min: vec![f64::INFINITY; p],
            max: vec![f64::NEG_INFINITY; p],
            nnz: vec![0; p],
        };
        // Row stamp per gene to detect duplicate column indices within a row.
        let mut seen = vec![u32::MAX; p];
        for i in r0..r1 {
            let (s, e) = (x.indptr[i].to_usize(), x.indptr[i + 1].to_usize());
            if s > e || e > nnz {
                return Err(Error::InvalidInput(format!("indptr is invalid at row {i}")));
            }
            let stamp = (i - r0) as u32;
            for (j, v) in x.indices[s..e].iter().zip(&x.data[s..e]) {
                let j = j.to_usize();
                if j >= p {
                    return Err(Error::InvalidInput(format!("column index {j} out of range for {p} columns")));
                }
                if seen[j] == stamp {
                    return Err(Error::InvalidInput(format!(
                        "duplicate column index {j} in row {i}; sum duplicates first"
                    )));
                }
                seen[j] = stamp;
                let v = v.to_f64();
                acc.sum[j] += v;
                acc.sumsq[j] += v * v;
                acc.min[j] = acc.min[j].min(v);
                acc.max[j] = acc.max[j].max(v);
                acc.nnz[j] += 1;
            }
        }
        Ok(acc)
    };
    let merge = |a: &mut StatsAcc, b: StatsAcc| {
        for j in 0..p {
            a.sum[j] += b.sum[j];
            a.sumsq[j] += b.sumsq[j];
            a.min[j] = a.min[j].min(b.min[j]);
            a.max[j] = a.max[j].max(b.max[j]);
            a.nnz[j] += b.nnz[j];
        }
    };
    let acc = tree_reduce(0, blocks.count, &leaf, &merge)?;

    let nf = n as f64;
    let mut mean = vec![0.0; p];
    let mut sd = vec![0.0; p];
    let mut constant = vec![false; p];
    for j in 0..p {
        if !acc.sum[j].is_finite() || !acc.sumsq[j].is_finite() {
            return Err(Error::InvalidInput(format!("non-finite values in column {j}")));
        }
        let all_stored = acc.nnz[j] as usize == n;
        let is_const = acc.nnz[j] == 0 || (acc.min[j] == acc.max[j] && (all_stored || acc.max[j] == 0.0));
        if is_const {
            constant[j] = true;
            // The exact constant, so the centred column is exactly zero.
            mean[j] = if all_stored { acc.min[j] } else { 0.0 };
            continue;
        }
        let m = acc.sum[j] / nf;
        let m2 = (acc.sumsq[j] - acc.sum[j] * m).max(0.0);
        mean[j] = m;
        sd[j] = (m2 / (nf - 1.0)).sqrt();
    }
    Ok(GeneStats { mean, sd, constant })
}

/// `Z = (X - 1 mu^T) diag(inv_sd)`, applied implicitly.
pub(super) struct ScaledOp<'a, I: Idx, T: Real> {
    x: CsrView<'a, I, T>,
    mean: &'a [f64],
    /// 1/sigma for scaled genes, 1 when scaling is off, and 0 for constant genes (their
    /// centred column is zero, so this is the same operator, with exact zeros).
    inv_sd: &'a [f64],
    blocks: Blocks,
}

impl<'a, I: Idx, T: Real> ScaledOp<'a, I, T> {
    pub fn new(x: CsrView<'a, I, T>, mean: &'a [f64], inv_sd: &'a [f64], blocks: Blocks) -> Self {
        Self { x, mean, inv_sd, blocks }
    }

    pub fn nrows(&self) -> usize {
        self.x.n_rows
    }
    pub fn ncols(&self) -> usize {
        self.x.n_cols
    }

    /// `out = Z v`, with `v` genes x l and `out` cells x l (column-major).
    pub fn apply(&self, v: MatRef<'_, f64>, out: MatMut<'_, f64>) {
        let (n, p, l) = (self.nrows(), self.ncols(), v.ncols());
        assert!(v.nrows() == p && out.nrows() == n && out.ncols() == l);
        // vs = diag(inv_sd) v, row-major so one gene's l values are contiguous.
        let mut vs = vec![0.0f64; p * l];
        for c in 0..l {
            let col = v.col(c);
            for j in 0..p {
                vs[j * l + c] = col[j] * self.inv_sd[j];
            }
        }
        // shift = mu^T diag(inv_sd) v, summed in gene order.
        let mut shift = vec![0.0f64; l];
        for j in 0..p {
            let m = self.mean[j];
            if m != 0.0 {
                for (s, w) in shift.iter_mut().zip(&vs[j * l..(j + 1) * l]) {
                    *s += m * w;
                }
            }
        }
        let x = &self.x;
        out.par_row_chunks_mut(ZV_ROWS).enumerate().for_each(|(b, mut oc)| {
            let r0 = b * ZV_ROWS;
            let m = oc.nrows();
            let mut buf = vec![0.0f64; m * l];
            for r in 0..m {
                let acc = &mut buf[r * l..(r + 1) * l];
                let (idx, val) = x.row(r0 + r);
                for (j, xv) in idx.iter().zip(val) {
                    let j = j.to_usize();
                    let xv = xv.to_f64();
                    for (a, w) in acc.iter_mut().zip(&vs[j * l..(j + 1) * l]) {
                        *a += xv * w;
                    }
                }
                for (a, s) in acc.iter_mut().zip(&shift) {
                    *a -= s;
                }
            }
            for c in 0..l {
                let col = oc.rb_mut().col_mut(c).try_as_col_major_mut().expect("column-major").as_slice_mut();
                for (r, y) in col.iter_mut().enumerate() {
                    *y = buf[r * l + c];
                }
            }
        });
    }

    /// `Z^T u`, with `u` cells x l (column-major); returns genes x l.
    pub fn apply_t(&self, u: MatRef<'_, f64>) -> Mat<f64> {
        let (n, p, l) = (self.nrows(), self.ncols(), u.ncols());
        assert!(u.nrows() == n);
        let x = &self.x;
        let blocks = self.blocks;
        // Accumulator rows 0..p hold X^T u, row p holds the column sums 1^T u.
        let leaf = |b: usize| -> Result<Vec<f64>> {
            let (r0, r1) = blocks.range(b, n);
            let mut acc = vec![0.0f64; (p + 1) * l];
            let mut ub = vec![0.0f64; ZTU_SUB_ROWS.min(r1 - r0) * l];
            let mut s0 = r0;
            while s0 < r1 {
                let s1 = (s0 + ZTU_SUB_ROWS).min(r1);
                let m = s1 - s0;
                for c in 0..l {
                    let col = u.col(c).subrows(s0, m);
                    for r in 0..m {
                        ub[r * l + c] = col[r];
                    }
                }
                for r in 0..m {
                    let ur = &ub[r * l..(r + 1) * l];
                    for (a, w) in acc[p * l..].iter_mut().zip(ur) {
                        *a += w;
                    }
                    let (idx, val) = x.row(s0 + r);
                    for (j, xv) in idx.iter().zip(val) {
                        let j = j.to_usize();
                        let xv = xv.to_f64();
                        for (a, w) in acc[j * l..(j + 1) * l].iter_mut().zip(ur) {
                            *a += xv * w;
                        }
                    }
                }
                s0 = s1;
            }
            Ok(acc)
        };
        let merge = |a: &mut Vec<f64>, b: Vec<f64>| {
            for (x, y) in a.iter_mut().zip(&b) {
                *x += y;
            }
        };
        let acc = tree_reduce(0, blocks.count, &leaf, &merge).expect("leaves are infallible");
        let colsum = &acc[p * l..];
        Mat::from_fn(p, l, |j, c| (acc[j * l + c] - self.mean[j] * colsum[c]) * self.inv_sd[j])
    }
}
