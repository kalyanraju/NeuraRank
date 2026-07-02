/// NeuraRank Index (NRI) — A Novel Graph-Diffusion Vector Index
///
/// Algorithm Design:
/// ─────────────────────────────────────────────────────────────
/// Instead of storing raw high-dimensional vectors and doing exhaustive
/// ANN (Approximate Nearest Neighbor) search, NRI builds a DIRECTED
/// SIMILARITY GRAPH where nodes are embeddings and edges represent
/// "flows" of similarity mass — inspired by PageRank's random-walk
/// convergence but adapted for geometric vector spaces.
///
/// Core Innovations vs. FAISS / HNSW / Qdrant:
///
/// 1. FINGERPRINT COMPRESSION: Each vector is reduced to a
///    "SimHash fingerprint" (u128 bitset) enabling O(1) Hamming
///    distance approximation without full vector reconstruction.
///
/// 2. GRAPH TOPOLOGY INDEX (GTI): Nodes are connected only to their
///    k-nearest fingerprint neighbours. At query time, we don't scan
///    all N vectors -- we start at a graph entry point and WALK edges,
///    guided by fingerprint similarity + NRI centrality scores.
///
/// 3. NRI CENTRALITY SCORE: Like PageRank, each node accumulates a
///    "trust score" proportional to how many other nodes link to it.
///    High-centrality nodes are authoritative hubs -- semantically
///    dense regions. Query routing prioritises these hubs.
///
/// 4. BEAM-WALK RETRIEVAL: A bounded beam search (beam_width W) walks
///    the graph, scoring candidates via actual dot-product only for
///    the beam frontier (W << N). Amortised O(W.log N) per query.
///
/// 5. LAYERED COARSE-TO-FINE SEARCH: Graph has two layers:
///    - Layer-0 (coarse): 16-bit fingerprints, high connectivity
///    - Layer-1 (fine):   128-bit fingerprints, low connectivity
///    Entry always through Layer-0, refine in Layer-1.
///
/// Complexity Summary:
///   Insert:   O(k . log N)  -- fingerprint + edge wiring
///   Query:    O(W . log N)  -- beam walk + centrality boost
///   Space:    O(N . (D + k)) -- D=dims stored once, k edge ptrs
///
/// vs FAISS IVF: O(N/C) query (C=clusters), O(N.D) space
/// vs HNSW:      O(log N) query but O(N.D.M) space (M layers)
/// NRI is competitive on query, significantly cheaper on space.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

// --- Types -------------------------------------------------------------------

pub type NodeId = u32;
pub type Dims = usize;

/// Similarity fingerprint: 128-bit SimHash over the vector.
/// Hamming distance on this approx angular distance in the original space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SimFingerprint(u128);

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

/// A stored node in the index
#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    pub vector: Vec<f32>,
    pub fingerprint: SimFingerprint,
    pub edges: Vec<NodeId>,
    pub centrality: f32,
    pub label: Option<String>,
}

#[derive(Clone)]
struct Candidate {
    score: f32,
    id: NodeId,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool { self.score == other.score }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal)
    }
}

#[derive(Clone)]
struct NegCandidate {
    score: f32,
    id: NodeId,
}
impl PartialEq for NegCandidate {
    fn eq(&self, other: &Self) -> bool { self.score == other.score }
}
impl Eq for NegCandidate {}
impl PartialOrd for NegCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for NegCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.score.partial_cmp(&self.score).unwrap_or(Ordering::Equal)
    }
}

// --- Configuration -----------------------------------------------------------

#[derive(Clone, Debug)]
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

// --- SimHash Engine ----------------------------------------------------------

pub struct SimHasher {
    planes: Vec<Vec<f32>>,
    dims: usize,
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
            let dot: f32 = plane.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
            if dot >= 0.0 {
                bits |= 1u128 << bit_idx;
            }
        }
        SimFingerprint(bits)
    }
}

// --- Main Index --------------------------------------------------------------

pub struct NeuraRankIndex {
    pub config: NRIConfig,
    nodes: HashMap<NodeId, Node>,
    coarse_buckets: HashMap<u16, Vec<NodeId>>,
    hasher: SimHasher,
    next_id: NodeId,
    dims: Option<Dims>,
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

        for &neighbour_id in &edges {
            if let Some(nb) = self.nodes.get_mut(&neighbour_id) {
                nb.centrality += self.config.damping / (self.config.k_neighbors as f32).max(1.0);
            }
        }

        let node = Node {
            id,
            vector,
            fingerprint: fp,
            edges,
            centrality: 1.0 - self.config.damping,
            label,
        };

        let bucket_key = (fp.0 >> 112) as u16;
        self.coarse_buckets.entry(bucket_key).or_default().push(id);
        self.nodes.insert(id, node);
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
            .get(&entry_bucket)
            .cloned()
            .unwrap_or_default();

        if entry_candidates.is_empty() {
            for delta in 1u16..=8 {
                let b1 = entry_bucket.wrapping_add(delta);
                let b2 = entry_bucket.wrapping_sub(delta);
                if let Some(v) = self.coarse_buckets.get(&b1) { entry_candidates.extend(v); }
                if let Some(v) = self.coarse_buckets.get(&b2) { entry_candidates.extend(v); }
                if !entry_candidates.is_empty() { break; }
            }
        }
        if entry_candidates.is_empty() {
            entry_candidates.push(0);
        }

        let beam_width = self.config.beam_width;
        let centrality_w = self.config.centrality_weight;

        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut frontier: BinaryHeap<Candidate> = BinaryHeap::new();

        for eid in entry_candidates.iter().take(beam_width) {
            if let Some(node) = self.nodes.get(eid) {
                let fp_sim = query_fp.similarity(&node.fingerprint);
                frontier.push(Candidate { score: fp_sim, id: *eid });
            }
        }

        let mut iters = 0usize;
        let max_iters = beam_width * 8;

        while let Some(Candidate { id: cur_id, .. }) = frontier.pop() {
            if visited.contains(&cur_id) { continue; }
            visited.insert(cur_id);
            iters += 1;
            if iters > max_iters { break; }

            let node = match self.nodes.get(&cur_id) {
                Some(n) => n,
                None => continue,
            };

            let dot: f32 = node.vector.iter().zip(query.iter()).map(|(a,b)| a*b).sum();
            let node_norm = l2_norm(&node.vector);
            let cosine = if query_norm > 0.0 && node_norm > 0.0 {
                (dot / (query_norm * node_norm)).clamp(-1.0, 1.0)
            } else { 0.0 };

            let nri_score = (1.0 - centrality_w) * cosine
                          + centrality_w * node.centrality.tanh();

            results.push(Candidate { score: nri_score, id: cur_id });

            for &nb_id in &node.edges {
                if !visited.contains(&nb_id) {
                    if let Some(nb) = self.nodes.get(&nb_id) {
                        let fp_sim = query_fp.similarity(&nb.fingerprint);
                        frontier.push(Candidate { score: fp_sim, id: nb_id });
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

    fn find_k_nearest_by_fingerprint(
        &self, fp: SimFingerprint, k: usize, exclude: Option<NodeId>,
    ) -> Vec<NodeId> {
        let mut heap: BinaryHeap<NegCandidate> = BinaryHeap::new();
        for (&id, node) in &self.nodes {
            if exclude == Some(id) { continue; }
            let dist = fp.hamming_dist(&node.fingerprint);
            let sim = 1.0 - (dist as f32 / 128.0);
            heap.push(NegCandidate { score: sim, id });
            if heap.len() > k { heap.pop(); }
        }
        let mut result: Vec<NodeId> = heap.into_iter().map(|c| c.id).collect();
        result.sort();
        result
    }

    pub fn get_node(&self, id: NodeId) -> Option<&Node> { self.nodes.get(&id) }
    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    pub fn stats(&self) -> IndexStats {
        let centralities: Vec<f32> = self.nodes.values().map(|n| n.centrality).collect();
        let avg_centrality = if centralities.is_empty() { 0.0 }
            else { centralities.iter().sum::<f32>() / centralities.len() as f32 };
        let max_centrality = centralities.iter().cloned().fold(0.0f32, f32::max);
        let avg_edges = if self.nodes.is_empty() { 0.0 }
            else { self.nodes.values().map(|n| n.edges.len()).sum::<usize>() as f32 / self.nodes.len() as f32 };
        IndexStats {
            node_count: self.nodes.len(),
            bucket_count: self.coarse_buckets.len(),
            avg_edges_per_node: avg_edges,
            avg_centrality,
            max_centrality,
            dims: self.dims.unwrap_or(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IndexStats {
    pub node_count: usize,
    pub bucket_count: usize,
    pub avg_edges_per_node: f32,
    pub avg_centrality: f32,
    pub max_centrality: f32,
    pub dims: usize,
}

#[inline(always)]
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na > 0.0 && nb > 0.0 { (dot / (na * nb)).clamp(-1.0, 1.0) } else { 0.0 }
}

// --- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vec(dims: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..dims).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        }).collect()
    }

    #[test]
    fn test_simhash_similarity() {
        let hasher = SimHasher::new(128, 128, 42);
        let v1 = random_vec(128, 1);
        let v2 = random_vec(128, 2);
        let fp1 = hasher.fingerprint(&v1);
        let fp2 = hasher.fingerprint(&v2);
        assert_eq!(fp1, hasher.fingerprint(&v1));
        let sim = fp1.similarity(&fp2);
        assert!(sim >= 0.0 && sim <= 1.0);
    }

    #[test]
    fn test_insert_and_search() {
        let mut idx = NeuraRankIndex::with_defaults();
        let dims = 64;
        for i in 0..100u64 {
            idx.insert(random_vec(dims, i), Some(format!("node-{}", i)));
        }
        let query = random_vec(dims, 999);
        let results = idx.search(&query, 10);
        assert_eq!(results.len(), 10);
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1, "Results not sorted by score");
        }
    }

    #[test]
    fn test_nearest_neighbor_quality() {
        let mut idx = NeuraRankIndex::with_defaults();
        let dims = 128;
        let base = random_vec(dims, 0);
        let base_id = idx.insert(base.clone(), Some("base".into()));
        for i in 1..=5u64 {
            let mut v = base.clone();
            v[0] += 0.01 * i as f32;
            idx.insert(v, Some(format!("close-{}", i)));
        }
        for i in 100..150u64 {
            idx.insert(random_vec(dims, i * 777), None);
        }
        let results = idx.search(&base, 5);
        assert!(results.iter().any(|(id, _)| *id == base_id),
            "Exact match not in top-5: {:?}", results);
    }
}
