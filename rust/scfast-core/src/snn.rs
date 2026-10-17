//! Shared-nearest-neighbour graph: replaces the SNN part of Seurat's `FindNeighbors`
//! (`ComputeSNN`), which is the graph `FindClusters` clusters on.
//!
//! Seurat's definition: with each cell's neighbour set N(i) of size k INCLUDING itself,
//! the edge weight is the Jaccard index |N(i) & N(j)| / |N(i) | N(j)| =
//! s / (2k - s) with s = |N(i) & N(j)|, and weights below `prune` (default 1/15) are set to
//! zero. The result is symmetric and includes the diagonal (weight 1).
//!
//! # How it is computed
//!
//! Seurat forms the binary n x n kNN matrix A and takes A A^T. Here the same counts come
//! from a reverse kNN index (for each cell l, the cells that list l as a neighbour): the cells
//! sharing at least one neighbour with i are exactly the cells in `rev(l)` for `l` in N(i),
//! and the number of times j is reached that way is s = |N(i) & N(j)|. Each row is
//! therefore found from about k^2 lookups with a per-thread dense counter, rows are
//! independent (parallel with rayon), and nothing n x n is ever built.
//!
//! The output is written in two passes over the rows (count, then fill), so peak memory is
//! the final CSR plus the reverse index (n * k `u32`) and one O(n) counter per worker, never
//! a second copy of the graph. Every weight is computed with Seurat's own floating-point
//! expression, `s / (k + (k - s))` in f64, and the prune test is done in f64 before storing as
//! f32, so the nonzero pattern is identical to Seurat's for the same kNN indices.

use std::ops::Range;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::types::{Knn, SparseGraph};

#[derive(Clone, Debug)]
pub struct SnnParams {
    /// Seurat `prune.SNN` (default 1/15).
    pub prune: f64,
    /// Keep the diagonal (self weight 1), as Seurat's `RNA_snn` does.
    pub keep_diagonal: bool,
}

impl Default for SnnParams {
    fn default() -> Self {
        Self { prune: 1.0 / 15.0, keep_diagonal: true }
    }
}

/// Build the SNN graph from a kNN result (self included, as returned by `knn::knn`).
/// Output: full symmetric CSR (both directions stored), rows with sorted column indices.
///
/// Each row of `knn.indices` is taken as the neighbour set exactly as given, which is what
/// Seurat does with its Annoy indices (so passing Seurat's own `Indices()` reproduces
/// `RNA_snn`). `knn.distances` is not read and may be empty.
///
/// Errors: `indices.len() != n * k`, `k == 0` with cells present, an index `>= n`, or the same
/// neighbour listed twice in one row (`InvalidInput`); `prune` outside `[0, 1]`
/// (`InvalidParameter`).
pub fn snn_from_knn(knn: &Knn, params: &SnnParams) -> Result<SparseGraph> {
    if !(0.0..=1.0).contains(&params.prune) {
        return Err(Error::InvalidParameter(format!("prune must be in [0, 1], got {}", params.prune)));
    }
    validate_knn(knn)?;
    let n = knn.n;
    if n == 0 {
        return Ok(SparseGraph { n: 0, indptr: vec![0], indices: Vec::new(), weights: Vec::new() });
    }

    let rev = ReverseIndex::build(knn);
    let table = WeightTable::new(knn.k, params.prune);
    let keep_diagonal = params.keep_diagonal;
    let blocks = row_blocks(n);

    // Pass 1: number of surviving entries per row, written to indptr[i + 1].
    let mut indptr = vec![0usize; n + 1];
    {
        let mut parts = Vec::with_capacity(blocks.len());
        let mut rest: &mut [usize] = &mut indptr[1..];
        for rows in &blocks {
            let (head, tail) = rest.split_at_mut(rows.len());
            parts.push((rows.clone(), head));
            rest = tail;
        }
        parts.into_par_iter().for_each(|(rows, out)| {
            let mut acc = Accumulator::new(n);
            for (slot, i) in out.iter_mut().zip(rows) {
                acc.accumulate(knn, &rev, i);
                *slot = acc.touched.iter().filter(|&&j| table.keeps(acc.count[j as usize], i, j, keep_diagonal)).count();
                acc.reset();
            }
        });
    }
    for i in 0..n {
        indptr[i + 1] += indptr[i];
    }

    // Pass 2: fill each row's columns (sorted) and weights into its slice of the output.
    let nnz = indptr[n];
    let mut indices = vec![0u32; nnz];
    let mut weights = vec![0.0f32; nnz];
    {
        let mut parts = Vec::with_capacity(blocks.len());
        let mut rest_i: &mut [u32] = &mut indices;
        let mut rest_w: &mut [f32] = &mut weights;
        for rows in &blocks {
            let len = indptr[rows.end] - indptr[rows.start];
            let (hi, ti) = rest_i.split_at_mut(len);
            let (hw, tw) = rest_w.split_at_mut(len);
            parts.push((rows.clone(), hi, hw));
            rest_i = ti;
            rest_w = tw;
        }
        let indptr = &indptr;
        parts.into_par_iter().for_each(|(rows, out_i, out_w)| {
            let base = indptr[rows.start];
            let mut acc = Accumulator::new(n);
            for i in rows {
                let (lo, hi) = (indptr[i] - base, indptr[i + 1] - base);
                acc.accumulate(knn, &rev, i);
                let cols = &mut out_i[lo..hi];
                let mut pos = 0;
                for &j in &acc.touched {
                    if table.keeps(acc.count[j as usize], i, j, keep_diagonal) {
                        cols[pos] = j;
                        pos += 1;
                    }
                }
                debug_assert_eq!(pos, cols.len(), "pass 1 and pass 2 disagree on row {i}");
                cols.sort_unstable();
                for (w, &j) in out_w[lo..hi].iter_mut().zip(cols.iter()) {
                    *w = table.weight[acc.count[j as usize] as usize];
                }
                acc.reset();
            }
        });
    }

    Ok(SparseGraph { n, indptr, indices, weights })
}

/// Shape, bounds and duplicate checks, parallel over rows.
fn validate_knn(knn: &Knn) -> Result<()> {
    let (n, k) = (knn.n, knn.k);
    if n > u32::MAX as usize {
        return Err(Error::InvalidInput(format!("{n} cells exceed the u32 index range")));
    }
    let expected = n.checked_mul(k).ok_or_else(|| Error::InvalidInput(format!("n * k overflows ({n} x {k})")))?;
    if knn.indices.len() != expected {
        return Err(Error::InvalidInput(format!(
            "knn.indices has {} entries, expected n * k = {n} x {k} = {expected}",
            knn.indices.len()
        )));
    }
    if n > 0 && k == 0 {
        return Err(Error::InvalidInput("k must be at least 1 (each row is the cell's neighbour set)".into()));
    }
    (0..n).into_par_iter().try_for_each_init(
        || Vec::with_capacity(k),
        |buf: &mut Vec<u32>, i| {
            let row = &knn.indices[i * k..(i + 1) * k];
            if let Some(&bad) = row.iter().find(|&&j| j as usize >= n) {
                return Err(Error::InvalidInput(format!("knn row {i}: neighbour index {bad} out of range for {n} cells")));
            }
            buf.clear();
            buf.extend_from_slice(row);
            buf.sort_unstable();
            if let Some(w) = buf.windows(2).find(|w| w[0] == w[1]) {
                return Err(Error::InvalidInput(format!("knn row {i}: neighbour {} is listed more than once", w[0])));
            }
            Ok(())
        },
    )
}

/// For each cell l, the cells whose neighbour set contains l (CSR, rows in increasing order).
struct ReverseIndex {
    ptr: Vec<usize>,
    idx: Vec<u32>,
}

impl ReverseIndex {
    fn build(knn: &Knn) -> Self {
        let (n, k) = (knn.n, knn.k);
        let mut ptr = vec![0usize; n + 1];
        for &l in &knn.indices {
            ptr[l as usize + 1] += 1;
        }
        for l in 0..n {
            ptr[l + 1] += ptr[l];
        }
        let mut cursor = ptr.clone();
        let mut idx = vec![0u32; knn.indices.len()];
        for i in 0..n {
            for &l in &knn.indices[i * k..(i + 1) * k] {
                idx[cursor[l as usize]] = i as u32;
                cursor[l as usize] += 1;
            }
        }
        Self { ptr, idx }
    }

    #[inline]
    fn row(&self, l: usize) -> &[u32] {
        &self.idx[self.ptr[l]..self.ptr[l + 1]]
    }
}

/// Weight and prune decision for every possible overlap s = 0..=k, computed once with
/// Seurat's exact expression so the comparison against `prune` matches it bit for bit.
struct WeightTable {
    keep: Vec<bool>,
    weight: Vec<f32>,
}

impl WeightTable {
    fn new(k: usize, prune: f64) -> Self {
        let kf = k as f64;
        let (mut keep, mut weight) = (Vec::with_capacity(k + 1), Vec::with_capacity(k + 1));
        for s in 0..=k {
            let sf = s as f64;
            // ComputeSNN: value / (k + (k - value)); `value < prune` -> 0 -> removed, so
            // equality survives (w and prune are finite here, so `>=` is exactly `!(<)`)
            let w = sf / (kf + (kf - sf));
            keep.push(s > 0 && w >= prune);
            weight.push(w as f32);
        }
        Self { keep, weight }
    }

    #[inline]
    fn keeps(&self, s: u32, i: usize, j: u32, keep_diagonal: bool) -> bool {
        self.keep[s as usize] && (keep_diagonal || j as usize != i)
    }
}

/// Per-worker dense overlap counter plus the list of cells it touched, so resetting costs
/// only the row's own candidates.
struct Accumulator {
    count: Vec<u32>,
    touched: Vec<u32>,
}

impl Accumulator {
    fn new(n: usize) -> Self {
        // vec![0; n] is a zeroed allocation: pages are only materialised where touched
        Self { count: vec![0u32; n], touched: Vec::new() }
    }

    /// count[j] = |N(i) & N(j)| for every j sharing a neighbour with i.
    #[inline]
    fn accumulate(&mut self, knn: &Knn, rev: &ReverseIndex, i: usize) {
        let k = knn.k;
        for &l in &knn.indices[i * k..(i + 1) * k] {
            for &j in rev.row(l as usize) {
                let c = &mut self.count[j as usize];
                if *c == 0 {
                    self.touched.push(j);
                }
                *c += 1;
            }
        }
    }

    #[inline]
    fn reset(&mut self) {
        for &j in &self.touched {
            self.count[j as usize] = 0;
        }
        self.touched.clear();
    }
}

/// Contiguous row ranges, about 8 per worker thread so work-stealing can balance them, and
/// few enough that the O(n) counter allocated per block stays negligible.
fn row_blocks(n: usize) -> Vec<Range<usize>> {
    let threads = rayon::current_num_threads().max(1);
    let size = n.div_ceil(threads * 8).max(256);
    (0..n).step_by(size).map(|s| s..(s + size).min(n)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn knn(n: usize, k: usize, indices: Vec<u32>) -> Knn {
        Knn { n, k, indices, distances: Vec::new() }
    }

    #[test]
    fn weight_table_matches_seurat_prune_rule() {
        // k = 20, prune = 1/15: s / (40 - s) >= 1/15  <=>  s >= 2.5
        let t = WeightTable::new(20, 1.0 / 15.0);
        assert!(!t.keep[0] && !t.keep[1] && !t.keep[2]);
        assert!(t.keep[3]);
        assert_eq!(t.weight[20], 1.0);
        // boundary: k = 8, s = 1 gives exactly 1/15 in f64, which Seurat keeps (`<` prune)
        let t = WeightTable::new(8, 1.0 / 15.0);
        assert!(t.keep[1]);
    }

    #[test]
    fn empty_and_invalid_inputs() {
        let g = snn_from_knn(&knn(0, 5, vec![]), &SnnParams::default()).unwrap();
        assert_eq!((g.n, g.indptr.as_slice(), g.nnz()), (0, &[0usize][..], 0));
        assert!(snn_from_knn(&knn(2, 2, vec![0, 1, 1]), &SnnParams::default()).is_err());
        assert!(snn_from_knn(&knn(2, 2, vec![0, 1, 1, 2]), &SnnParams::default()).is_err());
        assert!(snn_from_knn(&knn(2, 2, vec![0, 0, 1, 0]), &SnnParams::default()).is_err());
        assert!(snn_from_knn(&knn(2, 0, vec![]), &SnnParams::default()).is_err());
        for prune in [-0.1, 1.5, f64::NAN] {
            let p = SnnParams { prune, keep_diagonal: true };
            assert!(snn_from_knn(&knn(2, 2, vec![0, 1, 1, 0]), &p).is_err(), "prune {prune}");
        }
    }

    #[test]
    fn blocks_cover_all_rows() {
        for n in [1, 255, 256, 257, 10_000, 123_457] {
            let b = row_blocks(n);
            assert_eq!(b.first().unwrap().start, 0);
            assert_eq!(b.last().unwrap().end, n);
            assert!(b.windows(2).all(|w| w[0].end == w[1].start));
        }
    }
}
