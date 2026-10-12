// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! PLAID index for multivector (late interaction, ColBERT-style) search.
//!
//! A multivector row is a document made of token embeddings; its score for a
//! query (also a set of token embeddings) is MaxSim: the sum over query tokens
//! of the best inner product with any document token. PLAID (Santhanam et
//! al., "PLAID: An Efficient Engine for Late Interaction Retrieval", 2022)
//! answers top-k MaxSim queries without scanning every token:
//!
//! 1. every token is stored as its nearest centroid plus a residual quantized
//!    to `num_bits` bits per dimension,
//! 2. a query probes the `n_ivf_probe` best centroids of each query token and
//!    takes the documents of their inverted lists as candidates,
//! 3. candidates are scored with every token replaced by its centroid, and
//! 4. the best `n_full_scores / 4` are rescored exactly on decompressed tokens.
//!
//! A segment stores three files:
//!
//! ```text
//! segment (_indices/{uuid}/)
//! ├── plaid_centroids.lance   one row per centroid
//! │     ├── centroid     FixedSizeList<Float32, dimension>  (L2-normalized)
//! │     └── ivf_length   UInt32  number of documents in its inverted list
//! ├── plaid_ivf.lance         inverted lists, concatenated in centroid order
//! │     └── doc_id       UInt32  row number in plaid_docs.lance
//! └── plaid_docs.lance        one row per indexed document
//!       ├── _rowid       UInt64
//!       ├── codes        List<UInt32>  centroid id of every token
//!       └── residuals    Binary        packed residuals, dimension * num_bits / 8 bytes per token
//!       (every file repeats the codec metadata under the `lance:plaid` schema metadata key)
//! ```
//!
//! The algorithm is adapted from lightonai/next-plaid (Apache-2.0).

mod builder;
mod codec;
mod index;

pub use builder::PlaidIndexWriter;
pub use codec::{ResidualCodec, kmeans_params};
pub use index::PlaidIndex;

use std::sync::Arc;

use async_trait::async_trait;
use lance_core::cache::LanceCache;
use lance_core::{Error, Result};
use serde::{Deserialize, Serialize};

use crate::pb;
use crate::scalar::expression::ScalarQueryParser;
use crate::scalar::registry::ScalarIndexPlugin;
use crate::scalar::{IndexStore, RowIdRemapper, ScalarIndex};

pub(crate) const PLAID_INDEX_VERSION: u32 = 0;

/// Parameters to build a PLAID index.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PlaidIndexParams {
    /// Bits per quantized residual dimension: 1, 2, 4 or 8. More bits make
    /// the exact rescoring closer to the original embeddings and the index larger.
    pub num_bits: u8,
    /// Number of centroids. `None` picks `2^floor(log2(16 * sqrt(num_tokens)))`,
    /// the PLAID default.
    pub num_centroids: Option<usize>,
    /// Number of k-means iterations to train the centroids.
    pub max_iters: u32,
    /// Number of token embeddings sampled per centroid to train k-means.
    pub sample_rate: usize,
    /// Seed of the k-means initialization; `None` draws one from the OS.
    pub seed: Option<u64>,
}

impl Default for PlaidIndexParams {
    fn default() -> Self {
        Self {
            num_bits: 4,
            num_centroids: None,
            max_iters: 4,
            sample_rate: 64,
            seed: None,
        }
    }
}

impl PlaidIndexParams {
    pub fn from_json(json: &str) -> Result<Self> {
        let params: Self = serde_json::from_str(json).map_err(|e| {
            Error::invalid_input(format!("invalid PLAID index parameters {json}: {e}"))
        })?;
        if !matches!(params.num_bits, 1 | 2 | 4 | 8) {
            return Err(Error::invalid_input(format!(
                "PLAID num_bits must be 1, 2, 4 or 8, got {}",
                params.num_bits
            )));
        }
        if params.num_centroids == Some(0) || params.max_iters == 0 || params.sample_rate == 0 {
            return Err(Error::invalid_input(format!(
                "PLAID num_centroids, max_iters and sample_rate must be positive, got {params:?}"
            )));
        }
        Ok(params)
    }

    /// The number of centroids to train for `num_tokens` token embeddings.
    pub fn num_centroids_for(&self, num_tokens: usize) -> usize {
        self.num_centroids.unwrap_or_else(|| {
            let target = 16.0 * (num_tokens as f64).sqrt();
            1usize << target.log2().floor().max(0.0) as u32
        })
    }
}

/// Parameters of a PLAID search.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaidSearchParams {
    /// Number of centroids probed per query token.
    pub n_ivf_probe: usize,
    /// Number of candidates kept after the approximate centroid scoring; the
    /// best quarter of them is rescored exactly.
    pub n_full_scores: usize,
    /// Probed centroids whose best score over all query tokens is below this
    /// threshold are skipped. `None` keeps every probed centroid.
    pub centroid_score_threshold: Option<f32>,
}

impl Default for PlaidSearchParams {
    fn default() -> Self {
        Self {
            n_ivf_probe: 8,
            n_full_scores: 4096,
            centroid_score_threshold: Some(0.4),
        }
    }
}

pub(crate) fn details_any(codec: &ResidualCodec) -> prost_types::Any {
    prost_types::Any::from_msg(&pb::PlaidIndexDetails {
        num_bits: codec.num_bits() as u32,
        num_centroids: codec.num_centroids() as u32,
        dimension: codec.dimension() as u32,
    })
    .expect("encoding a protobuf message into a buffer cannot fail")
}

#[derive(Debug, Default)]
pub struct PlaidIndexPlugin;

#[async_trait]
impl ScalarIndexPlugin for PlaidIndexPlugin {
    // PLAID trains on a sample before it encodes every row, which a single
    // stream of training data cannot provide; the dataset builds it.

    fn name(&self) -> &str {
        "Plaid"
    }

    fn provides_exact_answer(&self) -> bool {
        false
    }

    fn version(&self) -> u32 {
        PLAID_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        _index_name: String,
        _index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        None
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        _index_details: &prost_types::Any,
        _index_version: u32,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        _cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        Ok(PlaidIndex::load(index_store, frag_reuse_index).await? as Arc<dyn ScalarIndex>)
    }

    fn details_as_json(&self, details: &prost_types::Any) -> Result<serde_json::Value> {
        let details = details.to_msg::<pb::PlaidIndexDetails>()?;
        Ok(serde_json::json!({
            "num_bits": details.num_bits,
            "num_centroids": details.num_centroids,
            "dimension": details.dimension,
        }))
    }
}

#[cfg(test)]
mod tests;
