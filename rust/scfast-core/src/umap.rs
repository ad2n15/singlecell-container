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

/// Interleaves the bits of two 16-bit coordinates (Z-order curve).
fn morton(x: u32, y: u32) -> u32 {
    let spread = |mut v: u32| {
        v &= 0xFFFF;
        v = (v | (v << 8)) & 0x00FF_00FF;
        v = (v | (v << 4)) & 0x0F0F_0F0F;
        v = (v | (v << 2)) & 0x3333_3333;
        (v | (v << 1)) & 0x5555_5555
    };
    spread(x) | (spread(y) << 1)
}

/// Hints the cache to load the line holding `p`; negative samples are drawn ahead and
/// prefetched, so their random reads over the embedding overlap.
#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0)
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags))
    }
}

/// Negative samples drawn this many draws before they are used.
const NEG_AHEAD: usize = 16;

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
    let (mut lo, mut hi) = ([f32::INFINITY; DIM], [f32::NEG_INFINITY; DIM]);
    for r in init.chunks_exact(DIM) {
        for d in 0..DIM {
            lo[d] = lo[d].min(r[d]);
            hi[d] = hi[d].max(r[d]);
        }
    }
    let span: Vec<f32> = (0..DIM).map(|d| if hi[d] > lo[d] { hi[d] - lo[d] } else { 1.0 }).collect();
    // Cells are relabelled in Z order of their start positions, so graph neighbours sit close
    // in memory and each thread's run of heads covers one region of the layout.
    let mut order: Vec<u32> = (0..n as u32).collect();
    let code = |i: usize| {
        let q = |d: usize| ((init[i * DIM + d] - lo[d]) / span[d] * 65535.0) as u32;
        morton(q(0), q(1))
    };
    let codes: Vec<u32> = (0..n).into_par_iter().map(code).collect();
    order.par_sort_unstable_by_key(|&i| (codes[i as usize], i));
    let mut rank = vec![0u32; n];
    for (r, &i) in order.iter().enumerate() {
        rank[i as usize] = r as u32;
    }

    // Edge e is sampled at the epochs where floor(epoch / eps_e) steps up, with eps_e =
    // wmax / w_e, and draws `negative_sample_rate` negatives each time: umap-learn's schedule
    // without its per-edge counters, so an epoch reads 8 bytes per edge and writes none.
    // Heads are implicit in the row offsets of the relabelled graph.
    let wmax = graph.weights.iter().copied().fold(0f32, f32::max);
    let cutoff = wmax / p.n_epochs as f32;
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0usize);
    let mut tails = Vec::new();
    let mut inv_eps = Vec::new();
    for &i in &order {
        let i = i as usize;
        for q in graph.indptr[i]..graph.indptr[i + 1] {
            let wq = graph.weights[q];
            if wq >= cutoff && wq > 0.0 {
                tails.push(rank[graph.indices[q] as usize]);
                inv_eps.push(wq / wmax);
            }
        }
        offsets.push(tails.len());
    }
    // Tasks are runs of whole rows of about CHUNK edges, so each head is written by one task.
    const CHUNK: usize = 4096;
    let mut tasks = vec![0usize];
    for r in 1..=n {
        if offsets[r] - offsets[*tasks.last().unwrap()] >= CHUNK || r == n {
            tasks.push(r);
        }
    }
    let tasks: Vec<(usize, usize)> = tasks.windows(2).map(|w| (w[0], w[1])).collect();

    let emb: Vec<AtomicU32> = order.iter().flat_map(|&i| {
        let span = &span;
        (0..DIM).map(move |d| AtomicU32::new((10.0 * (init[i as usize * DIM + d] - lo[d]) / span[d]).to_bits()))
    }).collect();

    let (a, b) = (p.a, p.b);
    let n_neg = p.negative_sample_rate;
    let get = |i: usize| [f32::from_bits(emb[i * DIM].load(Relaxed)), f32::from_bits(emb[i * DIM + 1].load(Relaxed))];
    let put = |i: usize, v: [f32; DIM]| {
        emb[i * DIM].store(v[0].to_bits(), Relaxed);
        emb[i * DIM + 1].store(v[1].to_bits(), Relaxed);
    };

    for epoch in 1..p.n_epochs {
        let (ep, prev) = (epoch as f32, (epoch - 1) as f32);
        let alpha = p.learning_rate * (1.0 - ep / p.n_epochs as f32);
        tasks.par_iter().enumerate().with_min_len(MIN_TASK_EDGES / CHUNK).for_each(|(c, &(r0, r1))| {
            let mut rng = (p.seed ^ 0x9E37_79B9_7F4A_7C15).wrapping_mul(epoch as u64 + 1).wrapping_add(c as u64 * 0x2545_F491_4F6C_DD1D) | 1;
            let draw = |rng: &mut u64| {
                let q = (((xorshift(rng) >> 32) * n as u64) >> 32) as usize;
                prefetch(&emb[q * DIM]);
                q
            };
            let mut ahead = [0usize; NEG_AHEAD];
            for q in ahead.iter_mut() {
                *q = draw(&mut rng);
            }
            let mut slot = 0;
            for j in r0..r1 {
                let (cur0, mut delta) = (get(j), [0f32; DIM]);
                let mut touched = false;
                for e in offsets[j]..offsets[j + 1] {
                    let ie = inv_eps[e];
                    if (ep * ie).floor() <= (prev * ie).floor() {
                        continue;
                    }
                    touched = true;
                    let oth = get(tails[e] as usize);
                    let dx = [cur0[0] - oth[0], cur0[1] - oth[1]];
                    let d2 = dx[0] * dx[0] + dx[1] * dx[1];
                    if d2 > 0.0 {
                        let pb = d2.powf(b);
                        let gc = -2.0 * a * b * pb / (d2 * (a * pb + 1.0));
                        for d in 0..DIM {
                            delta[d] += 2.0 * clip(gc * dx[d]) * alpha;
                        }
                    }
                    for _ in 0..n_neg {
                        let q = std::mem::replace(&mut ahead[slot], draw(&mut rng));
                        slot = (slot + 1) % NEG_AHEAD;
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
                }
                if touched {
                    put(j, [cur0[0] + delta[0], cur0[1] + delta[1]]);
                }
            }
        });
    }
    let mut out = vec![0f32; n * DIM];
    for (r, &i) in order.iter().enumerate() {
        for d in 0..DIM {
            out[i as usize * DIM + d] = f32::from_bits(emb[r * DIM + d].load(Relaxed));
        }
    }
    Ok(out)
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
        let d = |i: usize, c: [f32; 2]| (e[2 * i] - c[0]).powi(2) + (e[2 * i + 1] - c[1]).powi(2);
        assert!(e.iter().all(|v| v.is_finite()));
        let wrong = (0..n).filter(|&i| (i < n / 2) != (d(i, c0) < d(i, c1))).count();
        assert_eq!(wrong, 0, "{wrong} cells lie nearer the other blob's centre");
    }
}
