//! NRI HTTP Server — drop-in vector DB over REST.
//!
//! Run:  cargo run --release --bin nri-server --features server
//!       NRI_PORT=7700 (default)
//!
//! Endpoints:
//!   GET  /health
//!   GET  /stats
//!   POST /insert          {"vector":[...], "label":"..."}  → {"id":0}
//!   POST /search          {"query":[...], "top_k":10}      → {"results":[{"id":0,"score":0.9}]}
//!   POST /batch_search    {"queries":[[...]], "top_k":10}  → {"results":[[...]]}
//!   DELETE /node/:id                                       → {"ok":true}
//!   POST /compact                                          → {"ok":true,"removed":5}
//!   POST /rerank          {"iters":5}                      → {"ok":true}
//!   POST /save            {"path":"index.nri"}             → {"ok":true}
//!   POST /load            {"path":"index.nri"}             → {"ok":true,"node_count":100}

use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{delete, get, post},
    Json, Router,
};
use neura_rank_index::{NeuraRankIndex, NRIConfig};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::RwLock;

type AppState = Arc<RwLock<NeuraRankIndex>>;

// --- Request types ---

#[derive(Deserialize)]
struct InsertReq {
    vector: Vec<f32>,
    label: Option<String>,
}

#[derive(Deserialize)]
struct SearchReq {
    query: Vec<f32>,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

#[derive(Deserialize)]
struct BatchSearchReq {
    queries: Vec<Vec<f32>>,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

#[derive(Deserialize)]
struct RerankReq {
    #[serde(default = "default_iters")]
    iters: usize,
}

#[derive(Deserialize)]
struct FileReq {
    path: String,
}

#[derive(Deserialize)]
struct InitReq {
    k_neighbors: Option<usize>,
    beam_width: Option<usize>,
    damping: Option<f32>,
    centrality_weight: Option<f32>,
}

fn default_top_k() -> usize { 10 }
fn default_iters() -> usize { 5 }

// --- Response helper ---------------------------------------------------------

fn err(msg: impl ToString) -> (StatusCode, Json<Value>) {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"ok": false, "error": msg.to_string()})))
}

// --- Handlers ----------------------------------------------------------------

async fn health() -> Json<Value> {
    Json(json!({"ok": true, "version": env!("CARGO_PKG_VERSION"), "name": "NeuraRank Index Server"}))
}

async fn stats(State(s): State<AppState>) -> Json<Value> {
    let idx = s.read().await;
    let st = idx.stats();
    Json(json!({
        "node_count": st.node_count,
        "deleted_count": st.deleted_count,
        "bucket_count": st.bucket_count,
        "avg_edges_per_node": st.avg_edges_per_node,
        "avg_centrality": st.avg_centrality,
        "max_centrality": st.max_centrality,
        "dims": st.dims,
    }))
}

async fn insert(
    State(s): State<AppState>,
    Json(req): Json<InsertReq>,
) -> Json<Value> {
    let mut idx = s.write().await;
    let id = idx.insert(req.vector, req.label);
    Json(json!({"id": id}))
}

async fn search(
    State(s): State<AppState>,
    Json(req): Json<SearchReq>,
) -> Json<Value> {
    let idx = s.read().await;
    let results = idx.search(&req.query, req.top_k);
    let out: Vec<Value> = results.iter()
        .map(|(id, score)| json!({"id": id, "score": score}))
        .collect();
    Json(json!({"results": out}))
}

async fn batch_search(
    State(s): State<AppState>,
    Json(req): Json<BatchSearchReq>,
) -> Json<Value> {
    let idx = s.read().await;
    let all = idx.batch_search(&req.queries, req.top_k);
    let out: Vec<Value> = all.iter().map(|results| {
        let r: Vec<Value> = results.iter()
            .map(|(id, score)| json!({"id": id, "score": score}))
            .collect();
        json!(r)
    }).collect();
    Json(json!({"results": out}))
}

async fn delete_node(
    State(s): State<AppState>,
    Path(id): Path<u32>,
) -> Json<Value> {
    let mut idx = s.write().await;
    idx.delete(id);
    Json(json!({"ok": true, "id": id}))
}

async fn compact(State(s): State<AppState>) -> Json<Value> {
    let mut idx = s.write().await;
    let removed = idx.compact();
    Json(json!({"ok": true, "removed": removed}))
}

async fn rerank(
    State(s): State<AppState>,
    Json(req): Json<RerankReq>,
) -> Json<Value> {
    let mut idx = s.write().await;
    idx.rerank(req.iters);
    Json(json!({"ok": true}))
}

async fn save_index(
    State(s): State<AppState>,
    Json(req): Json<FileReq>,
) -> Json<Value> {
    let idx = s.read().await;
    match idx.save(&req.path) {
        Ok(_) => Json(json!({"ok": true, "path": req.path})),
        Err(e) => Json(json!({"ok": false, "error": e.to_string()})),
    }
}

async fn load_index(
    State(s): State<AppState>,
    Json(req): Json<FileReq>,
) -> Json<Value> {
    match NeuraRankIndex::load(&req.path) {
        Ok(new_idx) => {
            let node_count = new_idx.len();
            let mut idx = s.write().await;
            *idx = new_idx;
            Json(json!({"ok": true, "node_count": node_count}))
        }
        Err(e) => Json(json!({"ok": false, "error": e.to_string()})),
    }
}

async fn reinit(
    State(s): State<AppState>,
    Json(req): Json<InitReq>,
) -> Json<Value> {
    let cfg = {
        let idx = s.read().await;
        let mut c = idx.config.clone();
        if let Some(v) = req.k_neighbors { c.k_neighbors = v; }
        if let Some(v) = req.beam_width  { c.beam_width = v; }
        if let Some(v) = req.damping     { c.damping = v; }
        if let Some(v) = req.centrality_weight { c.centrality_weight = v; }
        c
    };
    let mut idx = s.write().await;
    *idx = NeuraRankIndex::new(cfg);
    Json(json!({"ok": true}))
}

// --- Main --------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let port = std::env::var("NRI_PORT").unwrap_or_else(|_| "7700".into());
    let idx: AppState = Arc::new(RwLock::new(NeuraRankIndex::with_defaults()));

    let app = Router::new()
        .route("/health",       get(health))
        .route("/stats",        get(stats))
        .route("/insert",       post(insert))
        .route("/search",       post(search))
        .route("/batch_search", post(batch_search))
        .route("/node/:id",     delete(delete_node))
        .route("/compact",      post(compact))
        .route("/rerank",       post(rerank))
        .route("/save",         post(save_index))
        .route("/load",         post(load_index))
        .route("/reinit",       post(reinit))
        .with_state(idx);

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await
        .unwrap_or_else(|e| panic!("Failed to bind {}: {}", addr, e));

    println!("NeuraRank Index Server v{}", env!("CARGO_PKG_VERSION"));
    println!("Listening on http://{}", addr);
    println!("POST /insert  POST /search  POST /batch_search  DELETE /node/:id");
    println!("POST /compact POST /rerank  POST /save          POST /load");

    axum::serve(listener, app).await.unwrap();
}
