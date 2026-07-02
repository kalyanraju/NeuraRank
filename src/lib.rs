use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::fs::File;
use std::path::Path;

use serde::{Deserialize, Serialize};
use rayon::prelude::*;

#[cfg(feature = "python")]
use pyo3::prelude::*;

// --- Types -------------------------------------------------------------------

pub type NodeId = u32;
pub type Dims = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SimFingerprint(pub u128);

impl SimFingerprint {
    #[inline(always)]
    pub fn hamming_dist(&self, other: &SimFingerprint) -> u32 {
        (self.0 ^ other.0).count_ones()
    }

    #[inline(always)]
    pub fn similarity(&self, other: &SimFingerprint) -> f32 {
        1.0 - (self.hamming_dist(other) as f32 / 128.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub vector: Vec<f32>,
    pub fingerprint: SimFingerprint,
    pub edges: Vec<NodeId>,
    pub centrality: f32,
    pub label: Option<String>,
}

#[derive(Clone)]
struct Candidate { score: f32, id: NodeId }
impl PartialEq for Candidate { fn eq(&self, o: &Self) -> bool { self.score == o.score } }
impl Eq for Candidate {}
impl PartialOrd for Candidate { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for Candidate { fn cmp(&self, o: &Self) -> Ordering { self.score.partial_cmp(&o.score).unwrap_or(Ordering::Equal) } }


// --- Config ------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NRIConfig {
    pub k_neighbors: usize,
    pub beam_width: usize,
    pub damping: f32,
    pub n_projections: usize,
    pub centrality_weight: f32,
    pub centrality_iters: usize,
}

impl Default for NRIConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 16,
            beam_width: 64,
            damping: 0.85,
            n_projections: 128,
            centrality_weight: 0.15,
            centrality_iters: 5,
        }
    }
}

// --- Stats -------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub node_count: usize,
    pub deleted_count: usize,
    pub bucket_count: usize,
    pub avg_edges_per_node: f32,
    pub avg_centrality: f32,
    pub max_centrality: f32,
    pub dims: usize,
}

// --- SIMD dot product --------------------------------------------------------

#[inline(always)]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let mut acc = _mm256_setzero_ps();
    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    let lo  = _mm256_castps256_ps128(acc);
    let hi  = _mm256_extractf128_ps(acc, 1);
    let s4  = _mm_add_ps(lo, hi);
    let sh  = _mm_movehdup_ps(s4);
    let s2  = _mm_add_ps(s4, sh);
    let sh2 = _mm_movehl_ps(sh, s2);
    let mut result = _mm_cvtss_f32(_mm_add_ss(s2, sh2));
    for i in (chunks * 8)..n {
        result += a[i] * b[i];
    }
    result
}

// --- SimHasher ---------------------------------------------------------------

pub struct SimHasher {
    planes: Vec<Vec<f32>>,
    pub dims: usize,
}

impl SimHasher {
    pub fn new(dims: usize, n_projections: usize, seed: u64) -> Self {
        let planes = (0..n_projections)
            .map(|i| Self::random_plane(dims, seed.wrapping_add((i as u64).wrapping_mul(6364136223846793005))))
            .collect();
        SimHasher { planes, dims }
    }

    fn random_plane(dims: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut plane = Vec::with_capacity(dims);
        let mut i = 0;
        while i < dims {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (state >> 11) as f32 / (1u64 << 53) as f32;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (state >> 11) as f32 / (1u64 << 53) as f32;
            let r = (-2.0 * (u1 + 1e-10).ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            plane.push(r * theta.cos());
            i += 1;
            if i < dims {
                plane.push(r * theta.sin());
                i += 1;
            }
        }
        let norm = plane.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        plane.iter_mut().for_each(|x| *x /= norm);
        plane
    }

    pub fn fingerprint(&self, vec: &[f32]) -> SimFingerprint {
        let mut bits: u128 = 0;
        for (bit_idx, plane) in self.planes.iter().enumerate() {
            if dot_product(plane, vec) >= 0.0 {
                bits |= 1u128 << bit_idx;
            }
        }
        SimFingerprint(bits)
    }
}

// --- Serialization helper ----------------------------------------------------

#[derive(Serialize, Deserialize)]
struct PersistedIndex {
    config: NRIConfig,
    nodes: HashMap<NodeId, Node>,
    coarse_buckets: HashMap<u16, Vec<NodeId>>,
    next_id: NodeId,
    dims: Option<usize>,
    deleted: HashSet<NodeId>,
}

// --- Main Index --------------------------------------------------------------

pub struct NeuraRankIndex {
    pub config: NRIConfig,
    nodes: HashMap<NodeId, Node>,
    coarse_buckets: HashMap<u16, Vec<NodeId>>,
    hasher: SimHasher,
    next_id: NodeId,
    dims: Option<Dims>,
    deleted: HashSet<NodeId>,
}

impl NeuraRankIndex {
    pub fn new(config: NRIConfig) -> Self {
        let hasher = SimHasher::new(512, config.n_projections, 0xDEAD_BEEF_CAFE_BABE);
        NeuraRankIndex {
            config,
            nodes: HashMap::new(),
            coarse_buckets: HashMap::new(),
            hasher,
            next_id: 0,
            dims: None,
            deleted: HashSet::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(NRIConfig::default())
    }

    fn reinit_hasher(&mut self, dims: usize) {
        self.hasher = SimHasher::new(dims, self.config.n_projections, 0xDEAD_BEEF_CAFE_BABE);
    }

    pub fn insert(&mut self, vector: Vec<f32>, label: Option<String>) -> NodeId {
        let dims = vector.len();
        if self.dims.is_none() {
            self.dims = Some(dims);
            self.reinit_hasher(dims);
        }
        assert_eq!(dims, self.dims.unwrap(), "Vector dimension mismatch");

        let id = self.next_id;
        self.next_id += 1;

        let fp = self.hasher.fingerprint(&vector);
        let edges = self.find_k_nearest_by_fingerprint(fp, self.config.k_neighbors, Some(id));

        let damping = self.config.damping;
        let k = self.config.k_neighbors as f32;
        for &nb in &edges {
            if let Some(node) = self.nodes.get_mut(&nb) {
                node.centrality += damping / k.max(1.0);
            }
        }

        let bucket_key = (fp.0 >> 112) as u16;
        self.coarse_buckets.entry(bucket_key).or_default().push(id);
        self.nodes.insert(id, Node {
            id, vector, fingerprint: fp, edges,
            centrality: 1.0 - self.config.damping,
            label,
        });
        id
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(NodeId, f32)> {
        if self.nodes.is_empty() { return vec![]; }
        let dims = self.dims.unwrap();
        assert_eq!(query.len(), dims, "Query dimension mismatch");

        let query_fp = self.hasher.fingerprint(query);
        let query_norm = l2_norm(query);

        let entry_bucket = (query_fp.0 >> 112) as u16;
        let mut entry_candidates: Vec<NodeId> = self.coarse_buckets
            .get(&entry_bucket).cloned().unwrap_or_default()
            .into_iter().filter(|id| !self.deleted.contains(id)).collect();

        if entry_candidates.is_empty() {
            for delta in 1u16..=8 {
                for &b in &[entry_bucket.wrapping_add(delta), entry_bucket.wrapping_sub(delta)] {
                    if let Some(v) = self.coarse_buckets.get(&b) {
                        entry_candidates.extend(v.iter().filter(|id| !self.deleted.contains(id)));
                    }
                }
                if !entry_candidates.is_empty() { break; }
            }
        }
        if entry_candidates.is_empty() {
            if let Some(&fallback) = self.nodes.keys().find(|id| !self.deleted.contains(id)) {
                entry_candidates.push(fallback);
            } else {
                return vec![];
            }
        }

        let beam_width = self.config.beam_width;
        let cw = self.config.centrality_weight;
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut frontier: BinaryHeap<Candidate> = BinaryHeap::new();

        for &eid in entry_candidates.iter().take(beam_width) {
            if let Some(node) = self.nodes.get(&eid) {
                frontier.push(Candidate { score: query_fp.similarity(&node.fingerprint), id: eid });
            }
        }

        let mut iters = 0usize;
        let max_iters = beam_width * 8;

        while let Some(Candidate { id: cur_id, .. }) = frontier.pop() {
            if visited.contains(&cur_id) { continue; }
            visited.insert(cur_id);
            iters += 1;
            if iters > max_iters { break; }
            if self.deleted.contains(&cur_id) { continue; }

            let node = match self.nodes.get(&cur_id) { Some(n) => n, None => continue };

            let dot = dot_product(&node.vector, query);
            let node_norm = l2_norm(&node.vector);
            let cosine = if query_norm > 0.0 && node_norm > 0.0 {
                (dot / (query_norm * node_norm)).clamp(-1.0, 1.0)
            } else { 0.0 };

            results.push(Candidate { score: (1.0 - cw) * cosine + cw * node.centrality.tanh(), id: cur_id });

            for &nb_id in &node.edges {
                if !visited.contains(&nb_id) && !self.deleted.contains(&nb_id) {
                    if let Some(nb) = self.nodes.get(&nb_id) {
                        frontier.push(Candidate { score: query_fp.similarity(&nb.fingerprint), id: nb_id });
                    }
                }
            }

            if frontier.len() > beam_width {
                let mut top_w: Vec<Candidate> = frontier.drain().collect();
                top_w.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                top_w.truncate(beam_width);
                frontier = top_w.into_iter().collect();
            }
        }

        let mut out: Vec<Candidate> = results.drain().collect();
        out.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        out.truncate(top_k);
        out.into_iter().map(|c| (c.id, c.score)).collect()
    }

    /// Run multiple queries in parallel using rayon.
    pub fn batch_search(&self, queries: &[Vec<f32>], top_k: usize) -> Vec<Vec<(NodeId, f32)>> {
        queries.par_iter().map(|q| self.search(q, top_k)).collect()
    }

    /// Mark a node as deleted (tombstone). Silently ignores nonexistent IDs.
    /// Call compact() to reclaim memory and rewire edges.
    pub fn delete(&mut self, id: NodeId) {
        if self.nodes.contains_key(&id) {
            self.deleted.insert(id);
        }
    }

    /// Remove all tombstoned nodes, rebuild buckets, rewire edges. Returns count removed.
    pub fn compact(&mut self) -> usize {
        let removed = self.deleted.len();
        for id in &self.deleted {
            self.nodes.remove(id);
        }
        self.coarse_buckets.clear();
        for (&id, node) in &self.nodes {
            let bk = (node.fingerprint.0 >> 112) as u16;
            self.coarse_buckets.entry(bk).or_default().push(id);
        }
        let deleted = std::mem::take(&mut self.deleted);
        for node in self.nodes.values_mut() {
            node.edges.retain(|id| !deleted.contains(id));
        }
        removed
    }

    /// True PageRank power iteration to recompute centrality scores from graph structure.
    /// Dangling-node mass (nodes with no live out-edges) is redistributed uniformly,
    /// so scores sum to 1.0 at convergence.
    pub fn rerank(&mut self, iters: usize) {
        let ids: Vec<NodeId> = self.nodes.keys()
            .filter(|id| !self.deleted.contains(id))
            .copied().collect();
        let n = ids.len() as f32;
        if n == 0.0 { return; }
        let d = self.config.damping;

        let mut in_edges: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for &id in &ids {
            for &nb in &self.nodes[&id].edges {
                if !self.deleted.contains(&nb) {
                    in_edges.entry(nb).or_default().push(id);
                }
            }
        }

        // Pre-compute which nodes are dangling (no live out-edges).
        let dangling: HashSet<NodeId> = ids.iter().copied()
            .filter(|id| {
                self.nodes[id].edges.iter().all(|nb| self.deleted.contains(nb))
            })
            .collect();

        let mut ranks: HashMap<NodeId, f32> = ids.iter().map(|&id| (id, 1.0 / n)).collect();
        for _ in 0..iters {
            // Collect mass lost to dangling nodes and redistribute it uniformly.
            let dangling_mass: f32 = dangling.iter().map(|id| ranks[id]).sum();
            let new_ranks: HashMap<NodeId, f32> = ids.iter().map(|&id| {
                let mut rank = (1.0 - d) / n + d * dangling_mass / n;
                if let Some(srcs) = in_edges.get(&id) {
                    for &src in srcs {
                        if !dangling.contains(&src) {
                            let out_deg = self.nodes[&src].edges.len() as f32;
                            if out_deg > 0.0 { rank += d * ranks[&src] / out_deg; }
                        }
                    }
                }
                (id, rank)
            }).collect();
            ranks = new_ranks;
        }

        for (&id, node) in &mut self.nodes {
            if let Some(&r) = ranks.get(&id) {
                node.centrality = r;
            }
        }
    }

    /// Persist the index to disk using bincode.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let persisted = PersistedIndex {
            config: self.config.clone(),
            nodes: self.nodes.clone(),
            coarse_buckets: self.coarse_buckets.clone(),
            next_id: self.next_id,
            dims: self.dims,
            deleted: self.deleted.clone(),
        };
        let data = bincode::serialize(&persisted)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let mut writer = BufWriter::new(File::create(path)?);
        writer.write_all(&data)
    }

    /// Load a previously saved index from disk.
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;
        let p: PersistedIndex = bincode::deserialize(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let dims = p.dims.unwrap_or(512);
        let hasher = SimHasher::new(dims, p.config.n_projections, 0xDEAD_BEEF_CAFE_BABE);
        Ok(NeuraRankIndex {
            config: p.config,
            nodes: p.nodes,
            coarse_buckets: p.coarse_buckets,
            hasher,
            next_id: p.next_id,
            dims: p.dims,
            deleted: p.deleted,
        })
    }

    fn find_k_nearest_by_fingerprint(&self, fp: SimFingerprint, k: usize, exclude: Option<NodeId>) -> Vec<NodeId> {
        let mut candidates: Vec<(f32, NodeId)> = self.nodes
            .par_iter()
            .filter(|(&id, _)| exclude != Some(id) && !self.deleted.contains(&id))
            .map(|(&id, node)| {
                let sim = 1.0 - (fp.hamming_dist(&node.fingerprint) as f32 / 128.0);
                (sim, id)
            })
            .collect();
        candidates.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        candidates.truncate(k);
        candidates.into_iter().map(|(_, id)| id).collect()
    }

    pub fn get_node(&self, id: NodeId) -> Option<&Node> { self.nodes.get(&id) }
    pub fn len(&self) -> usize { self.nodes.len().saturating_sub(self.deleted.len()) }
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn stats(&self) -> IndexStats {
        let live: Vec<&Node> = self.nodes.values()
            .filter(|n| !self.deleted.contains(&n.id)).collect();
        let centralities: Vec<f32> = live.iter().map(|n| n.centrality).collect();
        let avg_c = if centralities.is_empty() { 0.0 }
            else { centralities.iter().sum::<f32>() / centralities.len() as f32 };
        let max_c = centralities.iter().cloned().fold(0.0f32, f32::max);
        let avg_e = if live.is_empty() { 0.0 }
            else { live.iter().map(|n| n.edges.len()).sum::<usize>() as f32 / live.len() as f32 };
        IndexStats {
            node_count: live.len(),
            deleted_count: self.deleted.len(),
            bucket_count: self.coarse_buckets.len(),
            avg_edges_per_node: avg_e,
            avg_centrality: avg_c,
            max_centrality: max_c,
            dims: self.dims.unwrap_or(0),
        }
    }
}

// --- Public helpers ----------------------------------------------------------

#[inline(always)]
pub fn l2_norm(v: &[f32]) -> f32 {
    dot_product(v, v).sqrt()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na > 0.0 && nb > 0.0 { (dot / (na * nb)).clamp(-1.0, 1.0) } else { 0.0 }
}

// --- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet as StdHashSet;

    // ── Fixtures ──────────────────────────────────────────────────────────────

    fn lcg_vec(dims: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..dims).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        }).collect()
    }

    fn unit_vec(dims: usize, seed: u64) -> Vec<f32> {
        let mut v = lcg_vec(dims, seed);
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        v.iter_mut().for_each(|x| *x /= n);
        v
    }

    fn brute_top_k(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = corpus.iter().enumerate()
            .map(|(i, v)| (i, cosine_similarity(v, query)))
            .collect();
        scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);
        scores
    }

    fn recall_at_k(results: &[(NodeId, f32)], ground_truth: &[(usize, f32)], k: usize) -> f32 {
        let found: StdHashSet<usize> = results.iter().take(k).map(|(id, _)| *id as usize).collect();
        let truth: StdHashSet<usize> = ground_truth.iter().take(k).map(|(id, _)| *id).collect();
        found.intersection(&truth).count() as f32 / k as f32
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(name)
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // SimFingerprint
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_fingerprint_identical_vectors_equal() {
        let h = SimHasher::new(64, 128, 1);
        let v = lcg_vec(64, 42);
        assert_eq!(h.fingerprint(&v), h.fingerprint(&v));
    }

    #[test]
    fn test_fingerprint_deterministic_across_instances() {
        let v = lcg_vec(64, 7);
        let fp1 = SimHasher::new(64, 128, 99).fingerprint(&v);
        let fp2 = SimHasher::new(64, 128, 99).fingerprint(&v);
        assert_eq!(fp1, fp2, "Same seed + same vector must always produce the same fingerprint");
    }

    #[test]
    fn test_fingerprint_different_seeds_differ() {
        let v = lcg_vec(64, 7);
        let fp1 = SimHasher::new(64, 128, 1).fingerprint(&v);
        let fp2 = SimHasher::new(64, 128, 2).fingerprint(&v);
        assert_ne!(fp1, fp2, "Different seeds should produce different projection sets");
    }

    #[test]
    fn test_hamming_dist_self_is_zero() {
        let fp = SimFingerprint(0xDEAD_BEEF_1234_5678_u128);
        assert_eq!(fp.hamming_dist(&fp), 0);
    }

    #[test]
    fn test_hamming_dist_upper_bound() {
        let h = SimHasher::new(128, 128, 0);
        for i in 0..20u64 {
            let d = h.fingerprint(&lcg_vec(128, i))
                     .hamming_dist(&h.fingerprint(&lcg_vec(128, i + 100)));
            assert!(d <= 128, "Hamming distance must be ≤ 128, got {d}");
        }
    }

    #[test]
    fn test_fingerprint_similarity_in_unit_interval() {
        let h = SimHasher::new(128, 128, 0);
        for i in 0..20u64 {
            let s = h.fingerprint(&lcg_vec(128, i))
                     .similarity(&h.fingerprint(&lcg_vec(128, i + 50)));
            assert!(s >= 0.0 && s <= 1.0, "Similarity {s} must be in [0, 1]");
        }
    }

    #[test]
    fn test_fingerprint_similarity_identical_is_one() {
        let h = SimHasher::new(64, 128, 5);
        let fp = h.fingerprint(&lcg_vec(64, 99));
        assert_eq!(fp.similarity(&fp), 1.0);
    }

    #[test]
    fn test_fingerprint_near_identical_lower_hamming_than_random() {
        // Statistical: run 30 trials — average hamming for near-identical pairs
        // should be significantly less than for random pairs.
        let h = SimHasher::new(128, 128, 42);
        let (mut near_sum, mut rand_sum) = (0u32, 0u32);
        for seed in 0u64..30 {
            let base = unit_vec(128, seed);
            let mut near = base.clone();
            near[0] += 0.01; // tiny nudge — keeps cos-sim > 0.99
            let n = near.iter().map(|x| x * x).sum::<f32>().sqrt();
            near.iter_mut().for_each(|x| *x /= n);
            let random = unit_vec(128, seed + 10_000);
            let fp_b = h.fingerprint(&base);
            near_sum += fp_b.hamming_dist(&h.fingerprint(&near));
            rand_sum += fp_b.hamming_dist(&h.fingerprint(&random));
        }
        assert!(near_sum < rand_sum,
            "Near-identical avg hamming ({}) should be less than random ({})",
            near_sum / 30, rand_sum / 30);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // dot_product / SIMD
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dot_product_known_values() {
        assert!((dot_product(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!( dot_product(&[1.0, 0.0],       &[0.0, 1.0]      ).abs()         < 1e-6);
        assert!((dot_product(&[2.0, 3.0],        &[4.0, 5.0]      ) - 23.0).abs() < 1e-6);
        assert!( dot_product(&[-1.0, 1.0],       &[1.0, 1.0]      ).abs()         < 1e-6);
    }

    #[test]
    fn test_dot_product_matches_scalar_for_all_sizes() {
        // Covers: multiples of 8, non-multiples (SIMD remainder path), large arrays
        for &len in &[1usize, 7, 8, 9, 15, 16, 31, 64, 128, 255, 256] {
            let a = lcg_vec(len, 1);
            let b = lcg_vec(len, 2);
            let simd:   f32 = dot_product(&a, &b);
            let scalar: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            assert!((simd - scalar).abs() < 1e-3,
                "len={len}: SIMD={simd:.6} scalar={scalar:.6} diff={:.2e}", (simd-scalar).abs());
        }
    }

    #[test]
    fn test_dot_product_remainder_path() {
        // 13 elements → 1 SIMD chunk of 8 + 5 scalar
        let a = vec![1.0f32; 13];
        let b = vec![2.0f32; 13];
        assert!((dot_product(&a, &b) - 26.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_norm_unit_vector() {
        assert!((l2_norm(&unit_vec(128, 1)) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_norm_zero_vector() {
        assert_eq!(l2_norm(&vec![0.0f32; 64]), 0.0);
    }

    #[test]
    fn test_l2_norm_known_value() {
        assert!((l2_norm(&[3.0f32, 4.0]) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_identical_is_one() {
        let v = unit_vec(64, 7);
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal_is_zero() {
        assert!(cosine_similarity(&[1.0f32, 0.0], &[0.0f32, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_antipodal_is_minus_one() {
        let v = unit_vec(64, 9);
        let neg: Vec<f32> = v.iter().map(|x| -x).collect();
        assert!((cosine_similarity(&v, &neg) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector_is_zero() {
        let z = vec![0.0f32; 8];
        let v = unit_vec(8, 1);
        assert_eq!(cosine_similarity(&z, &v), 0.0);
        assert_eq!(cosine_similarity(&v, &z), 0.0);
    }

    #[test]
    fn test_cosine_similarity_result_clamped() {
        // Floating-point drift could push past ±1; must be clamped
        let v = unit_vec(512, 42);
        let s = cosine_similarity(&v, &v);
        assert!(s <= 1.0 && s >= -1.0, "cosine_similarity must be in [-1, 1], got {s}");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Insert
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_insert_first_id_is_zero() {
        let mut idx = NeuraRankIndex::with_defaults();
        assert_eq!(idx.insert(lcg_vec(32, 1), None), 0);
    }

    #[test]
    fn test_insert_ids_monotonically_increasing() {
        let mut idx = NeuraRankIndex::with_defaults();
        let ids: Vec<u32> = (0..10u64).map(|i| idx.insert(lcg_vec(32, i), None)).collect();
        for w in ids.windows(2) {
            assert!(w[1] == w[0] + 1, "IDs must be consecutive: {:?}", ids);
        }
    }

    #[test]
    fn test_insert_label_retrievable() {
        let mut idx = NeuraRankIndex::with_defaults();
        let id = idx.insert(lcg_vec(32, 1), Some("hello".into()));
        assert_eq!(idx.get_node(id).unwrap().label.as_deref(), Some("hello"));
    }

    #[test]
    fn test_insert_none_label_retrievable() {
        let mut idx = NeuraRankIndex::with_defaults();
        let id = idx.insert(lcg_vec(32, 1), None);
        assert!(idx.get_node(id).unwrap().label.is_none());
    }

    #[test]
    fn test_insert_locks_dims_after_first() {
        let mut idx = NeuraRankIndex::with_defaults();
        idx.insert(lcg_vec(77, 1), None);
        assert_eq!(idx.dims, Some(77));
    }

    #[test]
    #[should_panic(expected = "Vector dimension mismatch")]
    fn test_insert_dimension_mismatch_panics() {
        let mut idx = NeuraRankIndex::with_defaults();
        idx.insert(lcg_vec(32, 1), None);
        idx.insert(lcg_vec(64, 2), None);
    }

    #[test]
    fn test_insert_edges_at_most_k() {
        let cfg = NRIConfig { k_neighbors: 8, ..Default::default() };
        let mut idx = NeuraRankIndex::new(cfg);
        for i in 0..50u64 { idx.insert(lcg_vec(32, i), None); }
        for node in idx.nodes.values() {
            assert!(node.edges.len() <= 8,
                "Node {} has {} edges but k_neighbors=8", node.id, node.edges.len());
        }
    }

    #[test]
    fn test_insert_centrality_is_positive() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..20u64 { idx.insert(lcg_vec(32, i), None); }
        for node in idx.nodes.values() {
            assert!(node.centrality > 0.0,
                "Node {} centrality={} should be > 0", node.id, node.centrality);
        }
    }

    #[test]
    fn test_insert_every_node_in_exactly_one_bucket() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..20u64 { idx.insert(lcg_vec(64, i), None); }
        let total: usize = idx.coarse_buckets.values().map(|v| v.len()).sum();
        assert_eq!(total, 20, "Every inserted node must appear in exactly one bucket");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Search
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_search_empty_index_returns_empty() {
        assert!(NeuraRankIndex::with_defaults().search(&lcg_vec(32, 1), 10).is_empty());
    }

    #[test]
    fn test_search_top_k_zero_returns_empty() {
        let mut idx = NeuraRankIndex::with_defaults();
        idx.insert(lcg_vec(32, 1), None);
        assert!(idx.search(&lcg_vec(32, 99), 0).is_empty());
    }

    #[test]
    fn test_search_result_count_matches_top_k() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..50u64 { idx.insert(lcg_vec(64, i), None); }
        assert_eq!(idx.search(&lcg_vec(64, 999), 10).len(), 10);
    }

    #[test]
    fn test_search_top_k_larger_than_corpus_returns_all() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..5u64 { idx.insert(lcg_vec(32, i), None); }
        assert_eq!(idx.search(&lcg_vec(32, 99), 100).len(), 5,
            "top_k > N should return all N nodes");
    }

    #[test]
    fn test_search_results_sorted_descending() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..50u64 { idx.insert(lcg_vec(64, i), None); }
        let results = idx.search(&lcg_vec(64, 999), 20);
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1, "Out of order: {} < {}", w[0].1, w[1].1);
        }
    }

    #[test]
    fn test_search_no_duplicate_ids() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..50u64 { idx.insert(lcg_vec(64, i), None); }
        let results = idx.search(&lcg_vec(64, 999), 20);
        let ids: StdHashSet<NodeId> = results.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids.len(), results.len(), "Duplicate node IDs in search results");
    }

    #[test]
    fn test_search_scores_in_valid_range() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..50u64 { idx.insert(unit_vec(64, i), None); }
        for (id, score) in idx.search(&unit_vec(64, 999), 20) {
            assert!(score >= -1.0 && score <= 1.0,
                "Score {score} for id {id} outside [-1, 1]");
        }
    }

    #[test]
    fn test_search_exact_match_is_top_result() {
        let mut idx = NeuraRankIndex::with_defaults();
        let anchor = unit_vec(128, 42);
        let anchor_id = idx.insert(anchor.clone(), Some("anchor".into()));
        for i in 1..50u64 { idx.insert(lcg_vec(128, i * 9999), None); }
        let results = idx.search(&anchor, 1);
        assert_eq!(results[0].0, anchor_id, "Exact query vector must be the top result");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Recall quality (structured corpus)
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_recall_near_clones_dominate_top_k() {
        // Gold-standard quality test: insert 1 anchor + 9 near-clones (cosine > 0.9999)
        // + 90 random distractors. The near-clones must dominate top-10 results.
        // This verifies that NRI correctly identifies vectors known to be nearest neighbors.
        let dims = 128;
        let cfg = NRIConfig { k_neighbors: 16, beam_width: 128, centrality_weight: 0.1, ..Default::default() };
        let mut idx = NeuraRankIndex::new(cfg);
        let anchor = unit_vec(dims, 42);
        let anchor_id = idx.insert(anchor.clone(), Some("anchor".into()));
        let mut near_set: StdHashSet<u32> = [anchor_id].into_iter().collect();
        for i in 1..=9u64 {
            let mut v = anchor.clone();
            v[i as usize % dims] += 0.003 * i as f32; // tiny per-dim nudge, cos-sim > 0.9999
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= n);
            near_set.insert(idx.insert(v, None));
        }
        for i in 0..90u64 { idx.insert(lcg_vec(dims, i * 99991 + 777), None); }
        let results = idx.search(&anchor, 10);
        let found = results.iter().filter(|(id, _)| near_set.contains(id)).count();
        assert!(found >= 7,
            "At least 7 of the 10 near-clones must appear in top-10, found {found}");
    }

    #[test]
    fn test_recall_vs_brute_force_dense_small_corpus() {
        // With a small corpus (50 nodes), wide k (k=32 ≈ 64% of N) and a wide beam,
        // the graph is nearly complete and NRI should approach brute-force recall.
        let dims = 128;
        let cfg = NRIConfig { k_neighbors: 32, beam_width: 200, centrality_weight: 0.0, ..Default::default() };
        let mut idx = NeuraRankIndex::new(cfg);
        let mut corpus: Vec<Vec<f32>> = Vec::new();
        for i in 0..50u64 {
            let v = unit_vec(dims, i * 31337 + 17);
            corpus.push(v.clone());
            idx.insert(v, None);
        }
        let mut total_recall = 0.0f32;
        let (n_q, top_k) = (10, 10);
        for qi in 0..n_q {
            let q = unit_vec(dims, qi * 99991 + 777_777);
            let nri   = idx.search(&q, top_k);
            let brute = brute_top_k(&corpus, &q, top_k);
            total_recall += recall_at_k(&nri, &brute, top_k);
        }
        let avg = total_recall / n_q as f32;
        assert!(avg >= 0.60,
            "Recall@10 on small dense corpus (N=50, k=32) should be ≥60%, got {:.1}%", avg * 100.0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // centrality_weight = 0  →  pure cosine ordering
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_pure_cosine_mode_scores_equal_cosine() {
        // With centrality_weight=0, NRI_score(v) = (1-0)*cosine + 0*tanh(c) = cosine.
        // Every result's reported score must equal its true cosine similarity.
        let dims = 32;
        let cfg = NRIConfig { k_neighbors: 8, beam_width: 512, centrality_weight: 0.0, ..Default::default() };
        let mut idx = NeuraRankIndex::new(cfg);
        let mut corpus: Vec<Vec<f32>> = Vec::new();
        for i in 0..20u64 {
            let v = unit_vec(dims, i);
            corpus.push(v.clone());
            idx.insert(v, None);
        }
        let q = unit_vec(dims, 9999);
        let results = idx.search(&q, 10);
        for (id, nri_score) in &results {
            let true_cosine = cosine_similarity(&corpus[*id as usize], &q);
            assert!((*nri_score - true_cosine).abs() < 1e-5,
                "Node {id}: NRI score {nri_score:.6} should equal cosine {true_cosine:.6}");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Delete + Compact
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_delete_reduces_len() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..10u64 { idx.insert(lcg_vec(32, i), None); }
        idx.delete(0); assert_eq!(idx.len(), 9);
        idx.delete(1); assert_eq!(idx.len(), 8);
    }

    #[test]
    fn test_deleted_node_invisible_in_search_immediately() {
        let mut idx = NeuraRankIndex::with_defaults();
        let anchor = unit_vec(64, 1);
        let target_id = idx.insert(anchor.clone(), None);
        for i in 2..50u64 { idx.insert(lcg_vec(64, i * 9999), None); }
        idx.delete(target_id);
        let results = idx.search(&anchor, 10);
        assert!(results.iter().all(|(id, _)| *id != target_id),
            "Deleted node {target_id} appeared in search results before compact");
    }

    #[test]
    fn test_delete_nonexistent_id_is_safe() {
        let mut idx = NeuraRankIndex::with_defaults();
        idx.insert(lcg_vec(32, 1), None);
        idx.delete(9999); // must not panic
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_delete_all_then_search_empty() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..5u64 { idx.insert(lcg_vec(32, i), None); }
        for id in 0..5u32 { idx.delete(id); }
        assert!(idx.is_empty());
        assert!(idx.search(&lcg_vec(32, 99), 5).is_empty(),
            "Search on all-deleted index must return empty");
    }

    #[test]
    fn test_compact_removes_from_nodes_map() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..10u64 { idx.insert(lcg_vec(32, i), None); }
        idx.delete(3); idx.delete(7);
        let removed = idx.compact();
        assert_eq!(removed, 2);
        assert!(!idx.nodes.contains_key(&3), "Node 3 must be gone after compact");
        assert!(!idx.nodes.contains_key(&7), "Node 7 must be gone after compact");
        assert_eq!(idx.deleted.len(), 0, "deleted set must be empty after compact");
    }

    #[test]
    fn test_compact_rewires_edges() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..20u64 { idx.insert(lcg_vec(32, i), None); }
        idx.delete(0); idx.delete(5);
        idx.compact();
        for node in idx.nodes.values() {
            assert!(!node.edges.contains(&0),
                "Node {} still has edge → deleted node 0", node.id);
            assert!(!node.edges.contains(&5),
                "Node {} still has edge → deleted node 5", node.id);
        }
    }

    #[test]
    fn test_compact_zero_deletes_returns_zero() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..5u64 { idx.insert(lcg_vec(32, i), None); }
        assert_eq!(idx.compact(), 0);
        assert_eq!(idx.len(), 5);
    }

    #[test]
    fn test_search_after_compact_excludes_deleted() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..30u64 { idx.insert(lcg_vec(64, i), None); }
        for id in [2u32, 5, 10, 15] { idx.delete(id); }
        idx.compact();
        let results = idx.search(&lcg_vec(64, 9999), 10);
        assert_eq!(results.len(), 10);
        for (id, _) in &results {
            assert!(!matches!(*id, 2 | 5 | 10 | 15),
                "Compact-deleted id {id} appeared in results");
        }
    }

    #[test]
    fn test_double_compact_is_idempotent() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..10u64 { idx.insert(lcg_vec(32, i), None); }
        idx.delete(3);
        idx.compact();
        assert_eq!(idx.compact(), 0, "Second compact should remove nothing");
        assert_eq!(idx.len(), 9);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Rerank (PageRank power iteration)
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_rerank_centralities_are_positive() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..20u64 { idx.insert(lcg_vec(64, i), None); }
        idx.rerank(5);
        for node in idx.nodes.values() {
            assert!(node.centrality > 0.0,
                "Node {} centrality {} must be > 0 after rerank", node.id, node.centrality);
        }
    }

    #[test]
    fn test_rerank_centralities_sum_to_one() {
        // With dangling-node redistribution, PageRank is mass-conserving: Σ rank = 1.0.
        let mut idx = NeuraRankIndex::new(NRIConfig { k_neighbors: 16, ..Default::default() });
        for i in 0..50u64 { idx.insert(lcg_vec(64, i), None); }
        idx.rerank(30);
        let sum: f32 = idx.nodes.values().map(|n| n.centrality).sum();
        assert!((sum - 1.0).abs() < 0.02,
            "PageRank scores must sum to ≈1.0, got {sum:.4}");
    }

    #[test]
    fn test_rerank_hub_higher_than_leaf() {
        // Hub: 20 near-copies all link to it → many inbound edges.
        // Leaf: single random vector, far from everything → minimal inbound edges.
        let dims = 64;
        let hub = unit_vec(dims, 0);
        let mut idx = NeuraRankIndex::new(NRIConfig { k_neighbors: 4, ..Default::default() });
        let hub_id = idx.insert(hub.clone(), Some("hub".into()));
        for i in 1..=20u64 {
            let mut v = hub.clone();
            v[0] += 0.001 * i as f32;
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= n);
            idx.insert(v, None);
        }
        let leaf_id = idx.insert(unit_vec(dims, 99_999), Some("leaf".into()));
        idx.rerank(10);
        let hub_rank  = idx.get_node(hub_id ).unwrap().centrality;
        let leaf_rank = idx.get_node(leaf_id).unwrap().centrality;
        assert!(hub_rank > leaf_rank,
            "Hub centrality ({hub_rank:.6}) should exceed leaf ({leaf_rank:.6})");
    }

    #[test]
    fn test_rerank_empty_index_is_noop() {
        NeuraRankIndex::with_defaults().rerank(5); // must not panic
    }

    #[test]
    fn test_rerank_preserves_search_correctness() {
        let mut idx = NeuraRankIndex::with_defaults();
        let anchor = unit_vec(64, 0);
        let anchor_id = idx.insert(anchor.clone(), None);
        for i in 1..30u64 { idx.insert(lcg_vec(64, i * 9999), None); }
        idx.rerank(5);
        assert!(idx.search(&anchor, 5).iter().any(|(id, _)| *id == anchor_id),
            "Anchor must remain in top-5 after rerank");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch search
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_batch_search_matches_sequential_exactly() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..50u64 { idx.insert(lcg_vec(64, i), None); }
        let queries: Vec<Vec<f32>> = (0..8u64).map(|i| lcg_vec(64, i + 500)).collect();
        let batch = idx.batch_search(&queries, 5);
        for (qi, q) in queries.iter().enumerate() {
            assert_eq!(batch[qi], idx.search(q, 5),
                "batch_search[{qi}] differs from sequential search()");
        }
    }

    #[test]
    fn test_batch_search_empty_queries_returns_empty() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..10u64 { idx.insert(lcg_vec(32, i), None); }
        assert!(idx.batch_search(&[], 5).is_empty());
    }

    #[test]
    fn test_batch_search_correct_result_count_per_query() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..50u64 { idx.insert(lcg_vec(64, i), None); }
        let queries: Vec<Vec<f32>> = (0..4u64).map(|i| lcg_vec(64, i + 999)).collect();
        let batch = idx.batch_search(&queries, 7);
        assert_eq!(batch.len(), 4);
        for results in &batch { assert_eq!(results.len(), 7); }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Save / Load
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_save_load_search_results_identical() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..30u64 { idx.insert(lcg_vec(64, i), Some(format!("doc-{i}"))); }
        let path = tmp("nri_search.bin");
        idx.save(&path).unwrap();
        let loaded = NeuraRankIndex::load(&path).unwrap();
        let q = lcg_vec(64, 12345);
        assert_eq!(idx.search(&q, 10), loaded.search(&q, 10));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_load_config_preserved() {
        let cfg = NRIConfig {
            k_neighbors: 8, beam_width: 32, damping: 0.7,
            n_projections: 64, centrality_weight: 0.3, centrality_iters: 3,
        };
        let mut idx = NeuraRankIndex::new(cfg);
        idx.insert(lcg_vec(32, 1), None);
        let path = tmp("nri_config.bin");
        idx.save(&path).unwrap();
        let loaded = NeuraRankIndex::load(&path).unwrap();
        assert_eq!(loaded.config.k_neighbors, 8);
        assert_eq!(loaded.config.beam_width, 32);
        assert!((loaded.config.damping - 0.7).abs() < 1e-6);
        assert_eq!(loaded.config.n_projections, 64);
        assert!((loaded.config.centrality_weight - 0.3).abs() < 1e-6);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_load_labels_preserved() {
        let mut idx = NeuraRankIndex::with_defaults();
        idx.insert(lcg_vec(32, 1), Some("alpha".into()));
        idx.insert(lcg_vec(32, 2), Some("beta".into()));
        idx.insert(lcg_vec(32, 3), None);
        let path = tmp("nri_labels.bin");
        idx.save(&path).unwrap();
        let loaded = NeuraRankIndex::load(&path).unwrap();
        assert_eq!(loaded.get_node(0).unwrap().label.as_deref(), Some("alpha"));
        assert_eq!(loaded.get_node(1).unwrap().label.as_deref(), Some("beta"));
        assert!(loaded.get_node(2).unwrap().label.is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_load_tombstones_preserved() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..10u64 { idx.insert(lcg_vec(32, i), None); }
        idx.delete(3); idx.delete(7);
        let path = tmp("nri_tombstones.bin");
        idx.save(&path).unwrap();
        let loaded = NeuraRankIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 8, "Loaded index should show 8 live nodes");
        let results = loaded.search(&lcg_vec(32, 9999), 8);
        assert!(results.iter().all(|(id, _)| *id != 3 && *id != 7),
            "Tombstoned nodes must stay hidden after save/load");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_load_next_id_preserved() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..5u64 { idx.insert(lcg_vec(32, i), None); }
        let path = tmp("nri_nextid.bin");
        idx.save(&path).unwrap();
        let mut loaded = NeuraRankIndex::load(&path).unwrap();
        let new_id = loaded.insert(lcg_vec(32, 99), None);
        assert_eq!(new_id, 5, "First post-load insert should get ID 5 (not 0)");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_nonexistent_file_returns_err() {
        assert!(NeuraRankIndex::load("/nonexistent/path/nri_does_not_exist.bin").is_err());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Stats
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_stats_empty_index() {
        let s = NeuraRankIndex::with_defaults().stats();
        assert_eq!(s.node_count, 0);
        assert_eq!(s.deleted_count, 0);
        assert_eq!(s.dims, 0);
        assert_eq!(s.avg_centrality, 0.0);
        assert_eq!(s.max_centrality, 0.0);
    }

    #[test]
    fn test_stats_node_count_correct() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..15u64 { idx.insert(lcg_vec(32, i), None); }
        assert_eq!(idx.stats().node_count, 15);
    }

    #[test]
    fn test_stats_deleted_count_correct() {
        let mut idx = NeuraRankIndex::with_defaults();
        for i in 0..10u64 { idx.insert(lcg_vec(32, i), None); }
        idx.delete(2); idx.delete(4);
        let s = idx.stats();
        assert_eq!(s.node_count, 8);
        assert_eq!(s.deleted_count, 2);
    }

    #[test]
    fn test_stats_dims_reflects_inserted_dimension() {
        let mut idx = NeuraRankIndex::with_defaults();
        idx.insert(lcg_vec(77, 1), None);
        assert_eq!(idx.stats().dims, 77);
    }

    #[test]
    fn test_stats_avg_edges_in_range() {
        let cfg = NRIConfig { k_neighbors: 8, ..Default::default() };
        let mut idx = NeuraRankIndex::new(cfg);
        for i in 0..20u64 { idx.insert(lcg_vec(32, i), None); }
        let avg = idx.stats().avg_edges_per_node;
        assert!(avg > 0.0 && avg <= 8.0,
            "avg_edges_per_node={avg} should be in (0, k_neighbors=8]");
    }
}

// --- Python bindings (feature = "python") ------------------------------------

#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::exceptions::PyIOError;

    #[pyclass(name = "NeuraRankIndex")]
    pub struct PyNeuraRankIndex {
        inner: NeuraRankIndex,
    }

    #[pymethods]
    impl PyNeuraRankIndex {
        #[new]
        #[pyo3(signature = (k_neighbors=16, beam_width=64, damping=0.85, n_projections=128, centrality_weight=0.15))]
        fn new(k_neighbors: usize, beam_width: usize, damping: f32, n_projections: usize, centrality_weight: f32) -> Self {
            Self {
                inner: NeuraRankIndex::new(NRIConfig {
                    k_neighbors, beam_width, damping, n_projections, centrality_weight, centrality_iters: 5,
                }),
            }
        }

        fn insert(&mut self, vector: Vec<f32>, label: Option<String>) -> u32 {
            self.inner.insert(vector, label)
        }

        fn search(&self, query: Vec<f32>, top_k: usize) -> Vec<(u32, f32)> {
            self.inner.search(&query, top_k)
        }

        fn batch_search(&self, queries: Vec<Vec<f32>>, top_k: usize) -> Vec<Vec<(u32, f32)>> {
            self.inner.batch_search(&queries, top_k)
        }

        fn delete(&mut self, id: u32) { self.inner.delete(id); }

        fn compact(&mut self) -> usize { self.inner.compact() }

        #[pyo3(signature = (iters=5))]
        fn rerank(&mut self, iters: usize) { self.inner.rerank(iters); }

        fn __len__(&self) -> usize { self.inner.len() }
        fn is_empty(&self) -> bool { self.inner.is_empty() }

        fn save(&self, path: String) -> PyResult<()> {
            self.inner.save(&path).map_err(|e| PyIOError::new_err(e.to_string()))
        }

        #[staticmethod]
        fn load(path: String) -> PyResult<Self> {
            NeuraRankIndex::load(&path)
                .map(|inner| Self { inner })
                .map_err(|e| PyIOError::new_err(e.to_string()))
        }

        fn stats(&self) -> std::collections::HashMap<String, f64> {
            let s = self.inner.stats();
            [
                ("node_count".into(),        s.node_count as f64),
                ("deleted_count".into(),     s.deleted_count as f64),
                ("bucket_count".into(),      s.bucket_count as f64),
                ("avg_edges_per_node".into(),s.avg_edges_per_node as f64),
                ("avg_centrality".into(),    s.avg_centrality as f64),
                ("max_centrality".into(),    s.max_centrality as f64),
                ("dims".into(),              s.dims as f64),
            ].into_iter().collect()
        }
    }

    #[pymodule]
    fn neura_rank_index(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyNeuraRankIndex>()?;
        Ok(())
    }
}
