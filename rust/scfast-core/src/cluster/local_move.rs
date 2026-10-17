//! Batched parallel local moving (RB objective), replacing single-clustering's sequential
//! `local_move` in [`super::leiden_rb`].
//!
//! Same queue discipline as single-clustering: every node starts queued in a shuffled
//! order, and a node that moves re-queues its neighbours outside its new community. Nodes
//! leave the queue in batches; each node of a batch picks its best community in parallel
//! against the state at the start of the batch (single-clustering's gain and tie rules),
//! then the moves are applied in queue order. Batch boundaries depend only on the queue, so
//! the result is the same for any thread count. Before a move is applied its gain is
//! recomputed against the current state (weights to the old and the chosen community,
//! current strengths) and the move is dropped unless it still beats the floor, so every
//! applied move improves the objective and the loop terminates, as the sequential one does.

use std::cell::RefCell;
use std::collections::VecDeque;

use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use single_clustering::community_search::leiden::{NeighborWeights, Partition};
use single_clustering::network::CSRNetwork;

/// single-clustering's improvement floor, relative to the total edge weight.
const MIN_GAIN_RELATIVE: f64 = 1e-12;
/// single-clustering's move cap per node (a safety valve; moves strictly improve).
const MAX_MOVES_PER_NODE: usize = 100;
/// Batches of n / 16 nodes, at most this many. Smaller batches mean more fork-join
/// barriers: on 500k cells n / 64 (capped at 4096) took 2.9 s against 2.6 s.
const MAX_BATCH: usize = 65_536;
/// Stands for "a fresh empty community" in a proposal.
const EMPTY: u32 = u32::MAX;

thread_local! {
    static WEIGHTS: RefCell<NeighborWeights> = RefCell::new(NeighborWeights::with_capacity(0));
}

/// Moves nodes of `graph` between the communities of `start` until no single move improves
/// the RB objective by more than the floor. Returns the new membership.
pub(super) fn local_move(graph: &CSRNetwork, start: &Partition, resolution: f64, rng: &mut ChaCha8Rng) -> Vec<usize> {
    let n = graph.node_count();
    let four_m = 4.0 * start.total_weight();
    let threshold = MIN_GAIN_RELATIVE * start.total_weight().abs().max(1.0);
    let strength: Vec<f64> = (0..n).into_par_iter().map(|v| graph.strength(v)).collect();
    let self_loop: Vec<f64> = (0..n).into_par_iter().map(|v| graph.self_loop_weight(v)).collect();
    let mut membership: Vec<u32> = (0..n).map(|v| start.membership(v) as u32).collect();
    let slots = start.slots().max(n) + 1;
    let mut kc = vec![0f64; slots];
    let mut size = vec![0u32; slots];
    for (v, &c) in membership.iter().enumerate() {
        kc[c as usize] += strength[v];
        size[c as usize] += 1;
    }
    let mut free: Vec<u32> = (0..slots as u32).rev().filter(|&c| size[c as usize] == 0).collect();

    let mut order: Vec<u32> = (0..n as u32).collect();
    order.shuffle(rng);
    let mut queue: VecDeque<u32> = order.into();
    let mut queued = vec![true; n];
    let batch_len = (n / 16).clamp(1, MAX_BATCH);
    let mut moves = 0usize;
    let budget = n.saturating_mul(MAX_MOVES_PER_NODE);
    if four_m == 0.0 {
        return membership.iter().map(|&c| c as usize).collect();
    }

    let gain = |w: f64, kv: f64, sl: f64, k_c: f64| (w + sl) - resolution * kv * (2.0 * k_c + kv) / four_m;
    let mut batch = Vec::with_capacity(batch_len);
    let mut proposals = Vec::with_capacity(batch_len);
    let mut wake = Vec::new();
    while !queue.is_empty() && moves < budget {
        batch.clear();
        batch.extend(queue.drain(..batch_len.min(queue.len())));
        for &v in &batch {
            queued[v as usize] = false;
        }
        batch
            .par_iter()
            .map(|&v| {
                let v = v as usize;
                WEIGHTS.with_borrow_mut(|nw| {
                    nw.ensure_capacity(slots);
                    nw.collect(graph, &membership, v);
                    let (kv, sl) = (strength[v], self_loop[v]);
                    let old = membership[v];
                    let base = gain(nw.weight_to(old as usize), kv, sl, kc[old as usize] - kv);
                    let (mut best, mut best_gain) = (old, base);
                    for &c in nw.touched() {
                        if c as u32 == old {
                            continue;
                        }
                        let g = gain(nw.weight_to(c), kv, sl, kc[c]);
                        if g > best_gain || (g == best_gain && (c as u32) < best) {
                            (best, best_gain) = (c as u32, g);
                        }
                    }
                    if size[old as usize] > 1 {
                        let g = gain(0.0, kv, sl, 0.0);
                        if g > best_gain {
                            (best, best_gain) = (EMPTY, g);
                        }
                    }
                    (best != old && best_gain - base > threshold).then_some(best)
                })
            })
            .collect_into_vec(&mut proposals);

        for (&v, &p) in batch.iter().zip(&proposals) {
            let Some(mut target) = p else { continue };
            let v = v as usize;
            let old = membership[v];
            let (kv, sl) = (strength[v], self_loop[v]);
            let (mut w_old, mut w_target) = (0.0, 0.0);
            wake.clear();
            for (u, w) in graph.neighbors(v) {
                if u != v {
                    let c = membership[u];
                    if c == old {
                        w_old += w;
                    } else if c == target {
                        w_target += w;
                    }
                    // only neighbours outside the new community can have gone unstable
                    if c != target && !queued[u] {
                        wake.push(u as u32);
                    }
                }
            }
            let k_target = if target == EMPTY { 0.0 } else { kc[target as usize] };
            if gain(w_target, kv, sl, k_target) - gain(w_old, kv, sl, kc[old as usize] - kv) <= threshold {
                continue;
            }
            if target == EMPTY {
                if size[old as usize] == 1 {
                    continue;
                }
                target = loop {
                    let c = free.pop().expect("a non-singleton community leaves an empty slot");
                    if size[c as usize] == 0 {
                        break c;
                    }
                };
            }
            membership[v] = target;
            kc[old as usize] -= strength[v];
            kc[target as usize] += strength[v];
            size[old as usize] -= 1;
            size[target as usize] += 1;
            if size[old as usize] == 0 {
                free.push(old);
            }
            moves += 1;
            for &u in &wake {
                queued[u as usize] = true;
            }
            queue.extend(&wake);
        }
    }
    membership.iter().map(|&c| c as usize).collect()
}
