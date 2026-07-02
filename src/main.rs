//! NeuraRank Index -- Benchmark & Demo
use neura_rank_index::{NeuraRankIndex, NRIConfig, cosine_similarity, l2_norm};
use std::time::Instant;

fn random_vec(dims: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut v: Vec<f32> = (0..dims).map(|_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }).collect();
    let norm = l2_norm(&v).max(1e-10);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

fn brute_force_search(corpus: &[Vec<f32>], query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = corpus.iter().enumerate()
        .map(|(i, v)| (i, cosine_similarity(v, query)))
        .collect();
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scores.truncate(top_k);
    scores
}

fn recall_at_k(nri_results: &[(u32, f32)], brute_results: &[(usize, f32)], k: usize) -> f32 {
    let nri_ids: std::collections::HashSet<usize> = nri_results.iter()
        .take(k).map(|(id, _)| *id as usize).collect();
    let brute_ids: std::collections::HashSet<usize> = brute_results.iter()
        .take(k).map(|(id, _)| *id).collect();
    nri_ids.intersection(&brute_ids).count() as f32 / k as f32
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           NeuraRank Index (NRI) — Benchmark Suite           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let dims = 128;
    let n_queries = 20;
    let top_k = 10;

    for &n_vectors in &[500usize, 2_000, 5_000] {
        println!("━━━ Corpus Size: {:>5} vectors | dims={} ━━━", n_vectors, dims);

        let corpus: Vec<Vec<f32>> = (0..n_vectors as u64)
            .map(|i| random_vec(dims, i * 31337 + 17))
            .collect();

        // High-recall config: wide beam, more edges
        let cfg = NRIConfig {
            k_neighbors: 32,
            beam_width: 256,
            n_projections: 128,
            centrality_weight: 0.08,
            damping: 0.85,
            centrality_iters: 3,
        };
        let mut idx = NeuraRankIndex::new(cfg);

        let t0 = Instant::now();
        for (i, v) in corpus.iter().enumerate() {
            idx.insert(v.clone(), Some(format!("doc-{}", i)));
        }
        let build_time = t0.elapsed();
        println!("  [NRI] Build time  : {:.2}ms", build_time.as_secs_f64() * 1000.0);

        let stats = idx.stats();
        println!("  [NRI] Avg edges   : {:.1} | Buckets: {} | MaxCentrality: {:.4}",
            stats.avg_edges_per_node, stats.bucket_count, stats.max_centrality);

        let queries: Vec<Vec<f32>> = (0..n_queries as u64)
            .map(|i| random_vec(dims, i * 99991 + 777_777))
            .collect();

        let t1 = Instant::now();
        let nri_results_all: Vec<Vec<(u32, f32)>> = queries.iter()
            .map(|q| idx.search(q, top_k))
            .collect();
        let nri_time = t1.elapsed();

        let t2 = Instant::now();
        let brute_results_all: Vec<Vec<(usize, f32)>> = queries.iter()
            .map(|q| brute_force_search(&corpus, q, top_k))
            .collect();
        let brute_time = t2.elapsed();

        let avg_recall: f32 = nri_results_all.iter().zip(brute_results_all.iter())
            .map(|(nri, brute)| recall_at_k(nri, brute, top_k))
            .sum::<f32>() / n_queries as f32;

        let nri_us = nri_time.as_micros() / n_queries as u128;
        let brute_us = brute_time.as_micros() / n_queries as u128;
        let speedup = brute_us as f64 / nri_us as f64;

        println!("  [NRI]   Query avg : {}µs/query", nri_us);
        println!("  [Brute] Query avg : {}µs/query", brute_us);
        println!("  Speedup           : {:.1}x", speedup);
        println!("  Recall@{}          : {:.1}%", top_k, avg_recall * 100.0);
        println!();
    }

    // Quality demo
    println!("━━━ Quality Demo: Known Nearest Neighbours ━━━");
    let mut idx2 = NeuraRankIndex::with_defaults();
    let anchor = random_vec(dims, 42);
    let anchor_id = idx2.insert(anchor.clone(), Some("ANCHOR".into()));

    for i in 0..5u64 {
        let mut v = anchor.clone();
        let noise = 0.05 * (i + 1) as f32;
        v.iter_mut().enumerate().for_each(|(j, x)| *x += noise * ((j as f32).sin()));
        let norm = l2_norm(&v).max(1e-10);
        v.iter_mut().for_each(|x| *x /= norm);
        idx2.insert(v, Some(format!("near-clone-{}", i)));
    }
    for i in 0..200u64 {
        idx2.insert(random_vec(dims, i * 12345 + 99), None);
    }

    let results = idx2.search(&anchor, 5);
    println!("  Query: ANCHOR vector");
    for (rank, (id, score)) in results.iter().enumerate() {
        let label = idx2.get_node(*id)
            .and_then(|n| n.label.clone())
            .unwrap_or("unlabeled".into());
        println!("  Rank {}: id={:<4} score={:.4}  [{}]", rank+1, id, score, label);
    }
    let found = results.iter().any(|(id, _)| *id == anchor_id);
    println!("\n  Anchor in top-5: {}", if found { "YES ✓" } else { "NO ✗" });

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  NRI: 5 Core Novelties                                       ║");
    println!("║  1. SimHash fingerprint (128-bit) — O(1) approx distance     ║");
    println!("║  2. Centrality diffusion — PageRank-style hub authority       ║");
    println!("║  3. Beam-walk graph retrieval — sub-linear query traversal    ║");
    println!("║  4. Coarse-bucket entry routing — 16-bit prefix O(1) seed    ║");
    println!("║  5. NRI composite score: cosine + centrality blending         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
