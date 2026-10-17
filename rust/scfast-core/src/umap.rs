//! UMAP layout from a kNN graph, replacing `RunUMAP` (uwot) and `sc.tl.umap` (umap-learn).
//!
//! Same algorithm as umap-learn: fuzzy simplicial set from the kNN distances
//! (`smooth_knn_dist` + `compute_membership_strengths`), fuzzy union, then the SGD layout
//! (`optimize_layout_euclidean`) with edge sampling by weight and negative sampling. The SGD
//! runs edges in parallel without locks (Hogwild, as umap-learn's `parallel=True`), so the
//! layout is not bit-reproducible across thread counts; the embedding coordinates live in
//! relaxed atomics so concurrent updates are defined behaviour.

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};

use rayon::prelude::*;

use crate::error::{Error, Result};

#[derive(Clone, Debug)]
pub struct UmapParams {
    /// Curve parameters of the low-dimensional similarity `1 / (1 + a d^(2b))`.
    pub a: f32,
    pub b: f32,
    pub n_epochs: usize,
    pub negative_sample_rate: usize,
    pub learning_rate: f32,
    pub seed: u64,
}

/// Fuzzy union of the kNN membership strengths, as a symmetric CSR graph without self loops.
pub struct FuzzyGraph {
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
    pub weights: Vec<f32>,
}

const SMOOTH_K_TOLERANCE: f32 = 1e-5;
const MIN_K_DIST_SCALE: f32 = 1e-3;
/// Edges per SGD task at least: every epoch ends in a barrier, so small graphs stay on few threads.
const MIN_TASK_EDGES: usize = 16_384;

/// `indices` / `distances` are (n, k) row-major with the cell itself in column 0.
pub fn fuzzy_simplicial_set(indices: &[u32], distances: &[f32], n: usize, k: usize) -> Result<FuzzyGraph> {
    if indices.len() != n * k || distances.len() != n * k || k < 2 {
        return Err(Error::InvalidInput(format!("kNN arrays must be {n} x {k} with k >= 2")));
    }
    let mean_all = distances.iter().map(|&d| d as f64).sum::<f64>() / distances.len() as f64;
    let target = (k as f32).log2();
    let mut w = vec![0f32; n * k];
    w.par_chunks_mut(k).enumerate().for_each(|(i, wi)| {
        let d = &distances[i * k..(i + 1) * k];
        let rho = d.iter().copied().find(|&x| x > 0.0).unwrap_or(0.0);
        let (mut lo, mut hi, mut mid) = (0f32, f32::INFINITY, 1f32);
        for _ in 0..64 {
            let psum: f32 = d[1..].iter().map(|&x| {
                let e = x - rho;
                if e > 0.0 { (-e / mid).exp() } else { 1.0 }
            }).sum();
            if (psum - target).abs() < SMOOTH_K_TOLERANCE {
                break;
            }
            if psum > target {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                mid = if hi.is_infinite() { mid * 2.0 } else { (lo + hi) / 2.0 };
            }
        }
        let mean_i = d.iter().sum::<f32>() / k as f32;
        let floor = MIN_K_DIST_SCALE * if rho > 0.0 { mean_i } else { mean_all as f32 };
        let sigma = mid.max(floor);
        for j in 0..k {
            let nb = indices[i * k + j] as usize;
            wi[j] = if nb == i {
                0.0
            } else if d[j] - rho <= 0.0 {
                1.0
            } else {
                (-(d[j] - rho) / sigma).exp()
            };
        }
    });

    let mut deg = vec![0usize; n + 1];
    for (p, &j) in indices.iter().enumerate() {
        if w[p] > 0.0 {
            deg[p / k + 1] += 1;
            deg[j as usize + 1] += 1;
        }
    }
    for i in 0..n {
        deg[i + 1] += deg[i];
    }
    let total = deg[n];
    let mut cols = vec![0u32; total];
    let mut vals = vec![0f32; total];
    let mut fill = deg.clone();
    for (p, &j) in indices.iter().enumerate() {
        if w[p] > 0.0 {
            let (i, j) = (p / k, j as usize);
            cols[fill[i]] = j as u32;
            vals[fill[i]] = w[p];
            fill[i] += 1;
            cols[fill[j]] = i as u32;
            vals[fill[j]] = w[p];
            fill[j] += 1;
        }
    }

    let rows: Vec<Vec<(u32, f32)>> = (0..n).into_par_iter().map(|i| {
        let mut r: Vec<(u32, f32)> = cols[deg[i]..deg[i + 1]].iter().copied().zip(vals[deg[i]..deg[i + 1]].iter().copied()).collect();
        r.sort_unstable_by_key(|e| e.0);
        let mut out: Vec<(u32, f32)> = Vec::with_capacity(r.len());
        for (c, v) in r {
            match out.last_mut() {
                Some(last) if last.0 == c => last.1 = last.1 + v - last.1 * v,
                _ => out.push((c, v)),
            }
        }
        out
    }).collect();

    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0);
    for r in &rows {
        indptr.push(indptr.last().unwrap() + r.len());
    }
    let mut indices_out = Vec::with_capacity(indptr[n]);
    let mut weights = Vec::with_capacity(indptr[n]);
    for r in rows {
        for (c, v) in r {
            indices_out.push(c);
            weights.push(v);
        }
    }
    Ok(FuzzyGraph { indptr, indices: indices_out, weights })
}

#[inline(always)]
fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

#[inline(always)]
fn clip(g: f32) -> f32 {
    g.clamp(-4.0, 4.0)
}

/// SGD layout of `graph` starting from `init` (n x 2, row-major). Returns the n x 2 embedding.
pub fn optimize_layout(graph: &FuzzyGraph, init: &[f32], p: &UmapParams) -> Result<Vec<f32>> {
    const DIM: usize = 2;
    let n = graph.indptr.len() - 1;
    if init.len() != n * DIM {
        return Err(Error::InvalidInput(format!("init must be {n} x {DIM}")));
    }
    if p.n_epochs == 0 {
        return Err(Error::InvalidParameter("n_epochs must be at least 1".into()));
    }
    let wmax = graph.weights.iter().copied().fold(0f32, f32::max);
    let cutoff = wmax / p.n_epochs as f32;
    let mut heads = Vec::new();
    let mut tails = Vec::new();
    let mut eps = Vec::new();
    for i in 0..n {
        for q in graph.indptr[i]..graph.indptr[i + 1] {
            let wq = graph.weights[q];
            if wq >= cutoff && wq > 0.0 {
                heads.push(i as u32);
                tails.push(graph.indices[q]);
                eps.push(wmax / wq);
            }
        }
    }

    let (mut lo, mut hi) = ([f32::INFINITY; DIM], [f32::NEG_INFINITY; DIM]);
    for r in init.chunks_exact(DIM) {
        for d in 0..DIM {
            lo[d] = lo[d].min(r[d]);
            hi[d] = hi[d].max(r[d]);
        }
    }
    let emb: Vec<AtomicU32> = init.chunks_exact(DIM).flat_map(|r| {
        (0..DIM).map(move |d| {
            let span = if hi[d] > lo[d] { hi[d] - lo[d] } else { 1.0 };
            AtomicU32::new((10.0 * (r[d] - lo[d]) / span).to_bits())
        }).collect::<Vec<_>>()
    }).collect();

    let (a, b) = (p.a, p.b);
    let neg_rate = p.negative_sample_rate as f32;
    let mut next = eps.clone();
    let mut next_neg: Vec<f32> = eps.iter().map(|e| e / neg_rate).collect();
    let chunk = 4096;
    let get = |i: usize| [f32::from_bits(emb[i * DIM].load(Relaxed)), f32::from_bits(emb[i * DIM + 1].load(Relaxed))];
    let put = |i: usize, v: [f32; DIM]| {
        emb[i * DIM].store(v[0].to_bits(), Relaxed);
        emb[i * DIM + 1].store(v[1].to_bits(), Relaxed);
    };

    for epoch in 0..p.n_epochs {
        let ep = epoch as f32;
        let alpha = p.learning_rate * (1.0 - ep / p.n_epochs as f32);
        next.par_chunks_mut(chunk).zip(next_neg.par_chunks_mut(chunk)).enumerate().with_min_len(MIN_TASK_EDGES / chunk).for_each(|(c, (nx, nn))| {
            let base = c * chunk;
            let mut rng = (p.seed ^ 0x9E37_79B9_7F4A_7C15).wrapping_mul(epoch as u64 + 1).wrapping_add(base as u64 * 0x2545_F491_4F6C_DD1D) | 1;
            let mut row = (usize::MAX, [0f32; DIM], [0f32; DIM]);
            for t in 0..nx.len() {
                if nx[t] > ep {
                    continue;
                }
                let e = base + t;
                let (j, kk) = (heads[e] as usize, tails[e] as usize);
                if j != row.0 {
                    if row.0 != usize::MAX {
                        put(row.0, [row.1[0] + row.2[0], row.1[1] + row.2[1]]);
                    }
                    row = (j, get(j), [0.0; DIM]);
                }
                let (cur0, delta) = (row.1, &mut row.2);
                let mut oth = get(kk);
                let dx = [cur0[0] - oth[0], cur0[1] - oth[1]];
                let d2 = dx[0] * dx[0] + dx[1] * dx[1];
                if d2 > 0.0 {
                    let pb = d2.powf(b);
                    let gc = -2.0 * a * b * pb / (d2 * (a * pb + 1.0));
                    for d in 0..DIM {
                        let g = clip(gc * dx[d]) * alpha;
                        delta[d] += g;
                        oth[d] -= g;
                    }
                    put(kk, oth);
                }
                nx[t] += eps[e];
                let eneg = eps[e] / neg_rate;
                let n_neg = ((ep - nn[t]) / eneg) as usize;
                for _ in 0..n_neg {
                    let q = (((xorshift(&mut rng) >> 32) * n as u64) >> 32) as usize;
                    if q == j {
                        continue;
                    }
                    let o = get(q);
                    let dx = [cur0[0] - o[0], cur0[1] - o[1]];
                    let d2 = dx[0] * dx[0] + dx[1] * dx[1];
                    let gc = if d2 > 0.0 { 2.0 * b / ((0.001 + d2) * (a * d2.powf(b) + 1.0)) } else { 0.0 };
                    for d in 0..DIM {
                        delta[d] += if gc > 0.0 { clip(gc * dx[d]) } else { 4.0 } * alpha;
                    }
                }
                nn[t] += n_neg as f32 * eneg;
            }
            if row.0 != usize::MAX {
                put(row.0, [row.1[0] + row.2[0], row.1[1] + row.2[1]]);
            }
        });
    }
    Ok(emb.into_iter().map(|v| f32::from_bits(v.into_inner())).collect())
}

/// Fuzzy simplicial set + SGD layout in one call.
pub fn umap(indices: &[u32], distances: &[f32], n: usize, k: usize, init: &[f32], p: &UmapParams) -> Result<Vec<f32>> {
    let g = fuzzy_simplicial_set(indices, distances, n, k)?;
    optimize_layout(&g, init, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_blobs_separate() {
        let (n, k) = (400usize, 10usize);
        let mut s = 7u64;
        let pts: Vec<[f32; 2]> = (0..n).map(|i| {
            let c = if i < n / 2 { 0.0 } else { 50.0 };
            [c + (xorshift(&mut s) % 1000) as f32 / 100.0, (xorshift(&mut s) % 1000) as f32 / 100.0]
        }).collect();
        let mut idx = vec![0u32; n * k];
        let mut dist = vec![0f32; n * k];
        for i in 0..n {
            let mut o: Vec<(f32, u32)> = (0..n).map(|j| {
                let d = ((pts[i][0] - pts[j][0]).powi(2) + (pts[i][1] - pts[j][1]).powi(2)).sqrt();
                (if i == j { -1.0 } else { d }, j as u32)
            }).collect();
            o.sort_by(|x, y| x.partial_cmp(y).unwrap());
            for q in 0..k {
                idx[i * k + q] = o[q].1;
                dist[i * k + q] = o[q].0.max(0.0);
            }
        }
        let g = fuzzy_simplicial_set(&idx, &dist, n, k).unwrap();
        for i in 0..n {
            for q in g.indptr[i]..g.indptr[i + 1] {
                let j = g.indices[q] as usize;
                let back = (g.indptr[j]..g.indptr[j + 1]).find(|&r| g.indices[r] as usize == i).unwrap();
                assert!((g.weights[q] - g.weights[back]).abs() < 1e-6);
            }
        }
        let init: Vec<f32> = pts.iter().flat_map(|p| [p[1], p[0]]).collect();
        let p = UmapParams { a: 0.583, b: 1.334, n_epochs: 200, negative_sample_rate: 5, learning_rate: 1.0, seed: 0 };
        let e = optimize_layout(&g, &init, &p).unwrap();
        let centre = |r: std::ops::Range<usize>| {
            let m = r.len() as f32;
            r.fold([0f32; 2], |acc, i| [acc[0] + e[2 * i] / m, acc[1] + e[2 * i + 1] / m])
        };
        let (c0, c1) = (centre(0..n / 2), centre(n / 2..n));
        let gap = ((c0[0] - c1[0]).powi(2) + (c0[1] - c1[1]).powi(2)).sqrt();
        let spread = (0..n / 2).map(|i| ((e[2 * i] - c0[0]).powi(2) + (e[2 * i + 1] - c0[1]).powi(2)).sqrt()).sum::<f32>() / (n / 2) as f32;
        assert!(e.iter().all(|v| v.is_finite()));
        assert!(gap > 3.0 * spread, "gap {gap} spread {spread}");
    }
}
