// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A loaded PLAID index segment and its search.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, UInt32Type, UInt64Type};
use arrow_array::{Array, RecordBatch};
use async_trait::async_trait;
use datafusion::execution::SendableRecordBatchStream;
use futures::{StreamExt, TryStreamExt};
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder, LanceCache, WeakLanceCache};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::{Error, Result};
use lance_select::RowAddrMask;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use roaring::RoaringBitmap;

use super::builder::{
    CENTROID_COLUMN, CENTROIDS_FILE, CODES_COLUMN, DOC_ID_COLUMN, DOCS_FILE, FILE_FORMAT_VERSION,
    INV_NORMS_COLUMN, IVF_FILE, IVF_LENGTH_COLUMN, METADATA_KEY, PlaidFileMetadata,
    PlaidIndexWriter, RESIDUALS_COLUMN, with_row_ids,
};
use super::codec::{ResidualCodec, dot_i8, normalize_rows, quantize_query};
use super::{PlaidIndexParams, PlaidSearchParams, details_any};
use crate::metrics::MetricsCollector;
use crate::scalar::{
    AnyQuery, BuiltinIndexType, CreatedIndex, IndexReader, IndexStore, OldIndexDataFilter,
    RowIdRemapper, ScalarIndex, ScalarIndexParams, SearchResult, UpdateCriteria,
};
use crate::scalar::{TrainingCriteria, TrainingOrdering};
use crate::{Index, IndexType};

/// Rows of the docs file read at a time when rewriting a segment.
const REWRITE_BATCH_ROWS: usize = 8192;

/// Marks a document whose row was removed by fragment reuse remapping.
const REMOVED_ROW: u64 = u64::MAX;

/// Documents per residual chunk, the unit residuals are read and cached in.
const RESIDUAL_CHUNK_DOCS: usize = 1024;

/// Packed residuals and inverse reconstruction norms of the tokens of
/// `RESIDUAL_CHUNK_DOCS` consecutive documents.
#[derive(Debug, DeepSizeOf)]
pub struct ResidualChunk {
    residuals: Vec<u8>,
    inv_norms: Vec<f32>,
}

#[derive(Debug, Clone)]
struct ResidualChunkKey {
    chunk: u32,
}

impl CacheKey for ResidualChunkKey {
    type ValueType = ResidualChunk;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("residual-chunk-{}", self.chunk).into()
    }

    fn type_name() -> &'static str {
        "PlaidResidualChunk"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.vector.plaid.residual-chunk-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u32(self.chunk);
    }
}

/// One segment of a PLAID index.
///
/// The centroids, inverted lists, row ids and token centroid ids are resident;
/// residuals are read in chunks through the index cache, and only for the
/// documents a query rescores.
pub struct PlaidIndex {
    store: Arc<dyn IndexStore>,
    codec: Arc<ResidualCodec>,
    /// `ivf_offsets[c]..ivf_offsets[c + 1]` is the range of `ivf_doc_ids`
    /// holding the documents with a token assigned to centroid `c`.
    ivf_offsets: Vec<usize>,
    ivf_doc_ids: Vec<u32>,
    /// Row id of every document, or [`REMOVED_ROW`].
    row_ids: Vec<u64>,
    /// `token_offsets[d]..token_offsets[d + 1]` are the tokens of document `d`.
    token_offsets: Vec<usize>,
    /// Centroid id of every token.
    codes: Vec<u32>,
    docs_reader: Arc<dyn IndexReader>,
    cache: WeakLanceCache,
}

/// The documents of a query worth rescoring exactly, with what the exact
/// rescoring reuses from the earlier stages.
pub struct Shortlist {
    /// L2-normalized query tokens, concatenated.
    query: Vec<f32>,
    /// Row-major `[num_centroids, num_query_tokens]` centroid scores.
    centroid_scores: Vec<f32>,
    docs: Vec<u32>,
}

impl Shortlist {
    /// Ids of the residual chunks holding the shortlisted documents.
    fn chunks(&self) -> Vec<u32> {
        let mut chunks: Vec<u32> = self
            .docs
            .iter()
            .map(|&doc| doc / RESIDUAL_CHUNK_DOCS as u32)
            .collect();
        chunks.sort_unstable();
        chunks.dedup();
        chunks
    }
}

impl std::fmt::Debug for PlaidIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlaidIndex")
            .field("num_bits", &self.codec.num_bits())
            .field("num_centroids", &self.codec.num_centroids())
            .field("num_docs", &self.row_ids.len())
            .field("num_tokens", &self.codes.len())
            .finish()
    }
}

impl DeepSizeOf for PlaidIndex {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        // Residual chunks live in the index cache, which sizes them itself.
        self.codec.centroids().len() * size_of::<f32>()
            + self.ivf_offsets.len() * size_of::<usize>()
            + self.ivf_doc_ids.len() * size_of::<u32>()
            + self.row_ids.len() * size_of::<u64>()
            + self.token_offsets.len() * size_of::<usize>()
            + self.codes.len() * size_of::<u32>()
    }
}

fn read_metadata(reader: &dyn IndexReader, file: &str) -> Result<PlaidFileMetadata> {
    let value = reader.schema().metadata.get(METADATA_KEY).ok_or_else(|| {
        Error::corrupt_file_named(file, format!("missing {METADATA_KEY} schema metadata"))
    })?;
    let metadata: PlaidFileMetadata = serde_json::from_str(value)
        .map_err(|e| Error::corrupt_file_named(file, format!("invalid {METADATA_KEY}: {e}")))?;
    if metadata.version != FILE_FORMAT_VERSION {
        return Err(Error::not_supported(format!(
            "PLAID index file {file} has format version {} but this Lance reads version {FILE_FORMAT_VERSION}",
            metadata.version
        )));
    }
    Ok(metadata)
}

/// Insert `(score, centroid)` into `top`, kept sorted by descending score and
/// at most `capacity` long.
fn push_top(top: &mut Vec<(f32, u32)>, capacity: usize, score: f32, centroid: u32) {
    if top.len() == capacity && top.last().is_some_and(|&(worst, _)| score <= worst) {
        return;
    }
    let position = top.partition_point(|&(other, _)| other >= score);
    top.insert(position, (score, centroid));
    top.truncate(capacity);
}

impl PlaidIndex {
    pub async fn load(
        store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<Self>> {
        let (centroids_reader, ivf_reader, docs_reader) = futures::try_join!(
            store.open_index_file(CENTROIDS_FILE),
            store.open_index_file(IVF_FILE),
            store.open_index_file(DOCS_FILE),
        )?;
        let metadata = read_metadata(centroids_reader.as_ref(), CENTROIDS_FILE)?;
        for (reader, file) in [(&ivf_reader, IVF_FILE), (&docs_reader, DOCS_FILE)] {
            if read_metadata(reader.as_ref(), file)? != metadata {
                return Err(Error::corrupt_file_named(
                    file,
                    format!("{METADATA_KEY} does not match {CENTROIDS_FILE}"),
                ));
            }
        }
        if docs_reader.num_rows() != metadata.num_docs {
            return Err(Error::corrupt_file_named(
                DOCS_FILE,
                format!(
                    "has {} rows but the index records {} documents",
                    docs_reader.num_rows(),
                    metadata.num_docs
                ),
            ));
        }

        let centroids_batch = centroids_reader
            .read_range(0..centroids_reader.num_rows(), None)
            .await?;
        let centroid_values = centroids_batch[CENTROID_COLUMN]
            .as_fixed_size_list()
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        let centroids = Array2::from_shape_vec(
            (metadata.num_centroids, metadata.dimension),
            centroid_values,
        )
        .map_err(|e| Error::corrupt_file_named(CENTROIDS_FILE, e.to_string()))?;
        let codec = Arc::new(ResidualCodec::try_new(
            metadata.num_bits,
            centroids,
            metadata.bucket_cutoffs.clone(),
            metadata.bucket_weights.clone(),
        )?);

        let mut ivf_offsets = Vec::with_capacity(metadata.num_centroids + 1);
        ivf_offsets.push(0);
        for len in centroids_batch[IVF_LENGTH_COLUMN]
            .as_primitive::<UInt32Type>()
            .values()
        {
            ivf_offsets.push(ivf_offsets.last().copied().unwrap_or(0) + *len as usize);
        }
        let ivf_batch = ivf_reader
            .read_range(0..ivf_reader.num_rows(), None)
            .await?;
        let ivf_doc_ids = ivf_batch[DOC_ID_COLUMN]
            .as_primitive::<UInt32Type>()
            .values()
            .to_vec();
        if ivf_offsets.last().copied() != Some(ivf_doc_ids.len()) {
            return Err(Error::corrupt_file_named(
                IVF_FILE,
                format!(
                    "has {} doc ids but the inverted list lengths sum to {:?}",
                    ivf_doc_ids.len(),
                    ivf_offsets.last()
                ),
            ));
        }

        let docs = docs_reader
            .read_range(
                0..docs_reader.num_rows(),
                Some(&[lance_core::ROW_ID, CODES_COLUMN]),
            )
            .await?;
        let mut row_ids = docs[lance_core::ROW_ID]
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();
        if let Some(remapper) = frag_reuse_index {
            for row_id in &mut row_ids {
                *row_id = remapper.remap_row_id(*row_id).unwrap_or(REMOVED_ROW);
            }
        }
        let codes_list = docs[CODES_COLUMN].as_list::<i32>();
        let first_offset = codes_list.value_offsets()[0];
        let token_offsets: Vec<usize> = codes_list
            .value_offsets()
            .iter()
            .map(|o| (*o - first_offset) as usize)
            .collect();
        let num_tokens = token_offsets.last().copied().unwrap_or(0);
        let codes = codes_list.values().as_primitive::<UInt32Type>().values()
            [first_offset as usize..][..num_tokens]
            .to_vec();
        if codes
            .iter()
            .any(|&code| code as usize >= metadata.num_centroids)
        {
            return Err(Error::corrupt_file_named(
                DOCS_FILE,
                format!(
                    "has a centroid id >= num_centroids={}",
                    metadata.num_centroids
                ),
            ));
        }

        Ok(Arc::new(Self {
            store,
            codec,
            ivf_offsets,
            ivf_doc_ids,
            row_ids,
            token_offsets,
            codes,
            docs_reader,
            cache: WeakLanceCache::from(cache),
        }))
    }

    pub fn codec(&self) -> &Arc<ResidualCodec> {
        &self.codec
    }

    pub fn num_docs(&self) -> usize {
        self.row_ids.len()
    }

    pub fn dimension(&self) -> usize {
        self.codec.dimension()
    }

    /// Return the `k` documents with the highest MaxSim score to `query`
    /// (the query token embeddings concatenated) among the rows `mask`
    /// selects, as `(row_id, distance)` ordered by ascending distance.
    ///
    /// The distance is `num_query_tokens - MaxSim` over L2-normalized tokens,
    /// the multivector cosine distance of a flat search.
    ///
    /// This runs [`Self::shortlist`], [`Self::load_residuals`] and
    /// [`Self::rescore`] in turn; callers that schedule CPU work themselves
    /// call those directly.
    pub async fn search(
        &self,
        query: &[f32],
        k: usize,
        params: &PlaidSearchParams,
        mask: &RowAddrMask,
    ) -> Result<Vec<(u64, f32)>> {
        let shortlist = self.shortlist(query, k, params, mask)?;
        let chunks = self.load_residuals(&shortlist).await?;
        self.rescore(&shortlist, &chunks, k)
    }

    /// The first three stages of a search: probe the centroids of every query
    /// token, gather the documents of their inverted lists that `mask`
    /// selects, and keep the best of them by approximate (centroid) MaxSim.
    pub fn shortlist(
        &self,
        query: &[f32],
        k: usize,
        params: &PlaidSearchParams,
        mask: &RowAddrMask,
    ) -> Result<Shortlist> {
        let dimension = self.dimension();
        if query.is_empty() || !query.len().is_multiple_of(dimension) {
            return Err(Error::invalid_input(format!(
                "PLAID query has {} values, not a positive multiple of the index dimension {dimension}",
                query.len()
            )));
        }
        let query = ArrayView2::from_shape((query.len() / dimension, dimension), query)
            .map_err(|e| Error::invalid_input(format!("invalid PLAID query shape: {e}")))?;
        let query = normalize_rows(query.to_owned()).into_raw_vec_and_offset().0;
        let num_query_tokens = query.len() / dimension;
        let num_centroids = self.codec.num_centroids();
        let empty = |query, centroid_scores| Shortlist {
            query,
            centroid_scores,
            docs: Vec::new(),
        };
        if k == 0 || self.num_docs() == 0 {
            return Ok(empty(query, Vec::new()));
        }

        // Stage 1: probe the best centroids of every query token.
        let scores = self.codec.centroid_scores(&query);
        let num_probes = params.n_ivf_probe.clamp(1, num_centroids);
        let mut top: Vec<Vec<(f32, u32)>> =
            vec![Vec::with_capacity(num_probes + 1); num_query_tokens];
        for (centroid, row) in scores.chunks_exact(num_query_tokens).enumerate() {
            for (top, &score) in top.iter_mut().zip(row) {
                push_top(top, num_probes, score, centroid as u32);
            }
        }
        let mut probed = vec![false; num_centroids];
        for top in &top {
            for (rank, &(_, centroid)) in top.iter().enumerate() {
                let row = &scores[centroid as usize * num_query_tokens..][..num_query_tokens];
                // The best centroid of every query token survives the score
                // threshold, so that a query far from every centroid still has
                // candidates.
                let keep = rank == 0
                    || params.centroid_score_threshold.is_none_or(|threshold| {
                        row.iter().copied().fold(f32::NEG_INFINITY, f32::max) >= threshold
                    });
                probed[centroid as usize] |= keep;
            }
        }

        // Stage 2: candidates are the documents of the probed inverted lists.
        let mut candidates: Vec<u32> = probed
            .iter()
            .enumerate()
            .filter(|(_, is_probed)| **is_probed)
            .flat_map(|(c, _)| &self.ivf_doc_ids[self.ivf_offsets[c]..self.ivf_offsets[c + 1]])
            .copied()
            .collect();
        candidates.sort_unstable();
        candidates.dedup();
        candidates.retain(|&doc| {
            let row_id = self.row_ids[doc as usize];
            row_id != REMOVED_ROW && mask.selected(row_id)
        });

        // Stage 3: approximate MaxSim with every token replaced by its centroid.
        let mut approx: Vec<(u32, f32)> = candidates
            .par_iter()
            .map_init(
                || vec![f32::NEG_INFINITY; num_query_tokens],
                |best, &doc| {
                    best.fill(f32::NEG_INFINITY);
                    let doc = doc as usize;
                    for &code in &self.codes[self.token_offsets[doc]..self.token_offsets[doc + 1]] {
                        let row = &scores[code as usize * num_query_tokens..][..num_query_tokens];
                        for (b, s) in best.iter_mut().zip(row) {
                            *b = b.max(*s);
                        }
                    }
                    (doc as u32, best.iter().sum())
                },
            )
            .collect();
        // Like PLAID, only the best quarter of the `n_full_scores` shortlist
        // is worth decompressing.
        let num_decompress = (params.n_full_scores / 4).max(k).min(approx.len());
        top_by_score(&mut approx, num_decompress);
        let mut docs: Vec<u32> = approx.into_iter().map(|(doc, _)| doc).collect();
        docs.sort_unstable();
        Ok(Shortlist {
            query,
            centroid_scores: scores,
            docs,
        })
    }

    /// Read (or take from the index cache) the residual chunks of the shortlisted documents.
    pub async fn load_residuals(
        &self,
        shortlist: &Shortlist,
    ) -> Result<HashMap<u32, Arc<ResidualChunk>>> {
        let chunks = shortlist.chunks();
        let loaded: Vec<_> = futures::stream::iter(chunks.into_iter().map(|chunk| async move {
            let residuals = self
                .cache
                .get_or_insert_with_key(ResidualChunkKey { chunk }, || {
                    self.read_residual_chunk(chunk)
                })
                .await?;
            Ok::<_, Error>((chunk, residuals))
        }))
        .buffer_unordered(self.store.io_parallelism().max(1))
        .try_collect()
        .await?;
        Ok(loaded.into_iter().collect())
    }

    async fn read_residual_chunk(&self, chunk: u32) -> Result<ResidualChunk> {
        let start = chunk as usize * RESIDUAL_CHUNK_DOCS;
        let end = (start + RESIDUAL_CHUNK_DOCS).min(self.num_docs());
        let batch = self
            .docs_reader
            .read_range(start..end, Some(&[RESIDUALS_COLUMN, INV_NORMS_COLUMN]))
            .await?;
        let residuals = batch[RESIDUALS_COLUMN].as_binary::<i32>();
        let residual_bytes = &residuals.value_data()[residuals.value_offsets()[0] as usize
            ..residuals.value_offsets()[residuals.len()] as usize];
        let inv_norms = batch[INV_NORMS_COLUMN].as_list::<i32>();
        let inv_norm_values = &inv_norms.values().as_primitive::<Float32Type>().values()[inv_norms
            .value_offsets()[0]
            as usize
            ..inv_norms.value_offsets()[inv_norms.len()] as usize];
        let num_tokens = self.token_offsets[end] - self.token_offsets[start];
        if residual_bytes.len() != num_tokens * self.codec.packed_dim()
            || inv_norm_values.len() != num_tokens
        {
            return Err(Error::corrupt_file_named(
                DOCS_FILE,
                format!(
                    "documents {start}..{end} have {} residual bytes and {} norms for {num_tokens} tokens of {} bytes",
                    residual_bytes.len(),
                    inv_norm_values.len(),
                    self.codec.packed_dim()
                ),
            ));
        }
        Ok(ResidualChunk {
            residuals: residual_bytes.to_vec(),
            inv_norms: inv_norm_values.to_vec(),
        })
    }

    /// Stage 4: exact MaxSim of the shortlisted documents on their decoded
    /// tokens, returning the best `k` as `(row_id, distance)`.
    ///
    /// Each token scores `(q . centroid + q . residual) / ||centroid + residual||`:
    /// the centroid term comes from stage 1 and the residual term is an int8
    /// inner product, so no token is decompressed to floats.
    pub fn rescore(
        &self,
        shortlist: &Shortlist,
        chunks: &HashMap<u32, Arc<ResidualChunk>>,
        k: usize,
    ) -> Result<Vec<(u64, f32)>> {
        let dimension = self.dimension();
        let num_query_tokens = shortlist.query.len() / dimension;
        let packed_dim = self.codec.packed_dim();
        let (query_i8, query_scales) = quantize_query(&shortlist.query, dimension);
        let residual_scales: Vec<f32> = query_scales
            .iter()
            .map(|scale| scale * self.codec.weight_scale())
            .collect();
        let scores = &shortlist.centroid_scores;
        let mut exact: Vec<(u32, f32)> = shortlist
            .docs
            .par_iter()
            .map_init(
                || {
                    (
                        vec![0i8; dimension],
                        vec![f32::NEG_INFINITY; num_query_tokens],
                    )
                },
                |(residual, best), &doc| {
                    best.fill(f32::NEG_INFINITY);
                    let chunk_id = doc / RESIDUAL_CHUNK_DOCS as u32;
                    let chunk = &chunks[&chunk_id];
                    let chunk_first_token =
                        self.token_offsets[chunk_id as usize * RESIDUAL_CHUNK_DOCS];
                    let doc = doc as usize;
                    for token in self.token_offsets[doc]..self.token_offsets[doc + 1] {
                        let local = token - chunk_first_token;
                        self.codec.expand_residual_i8(
                            &chunk.residuals[local * packed_dim..][..packed_dim],
                            residual,
                        );
                        let inv_norm = chunk.inv_norms[local];
                        let centroid_scores = &scores
                            [self.codes[token] as usize * num_query_tokens..][..num_query_tokens];
                        for (((b, query_token), centroid_score), scale) in best
                            .iter_mut()
                            .zip(query_i8.chunks_exact(dimension))
                            .zip(centroid_scores)
                            .zip(&residual_scales)
                        {
                            let residual_score = dot_i8(query_token, residual) as f32 * scale;
                            *b = b.max((centroid_score + residual_score) * inv_norm);
                        }
                    }
                    (doc as u32, best.iter().sum())
                },
            )
            .collect();
        let num_results = k.min(exact.len());
        top_by_score(&mut exact, num_results);
        Ok(exact
            .into_iter()
            .map(|(doc, score)| (self.row_ids[doc as usize], num_query_tokens as f32 - score))
            .collect())
    }

    /// Rewrite this segment's documents into a new segment with the same
    /// codec, mapping each row id through `map_row_id` (`None` drops the
    /// document), then append `new_data`.
    async fn rewrite(
        &self,
        dest_store: &dyn IndexStore,
        map_row_id: impl Fn(u64) -> Option<u64>,
        new_data: Option<SendableRecordBatchStream>,
    ) -> Result<CreatedIndex> {
        let docs_reader = self.store.open_index_file(DOCS_FILE).await?;
        let mut writer = PlaidIndexWriter::try_new(self.codec.clone(), dest_store).await?;
        let num_docs = self.num_docs();
        for start in (0..num_docs).step_by(REWRITE_BATCH_ROWS) {
            let end = (start + REWRITE_BATCH_ROWS).min(num_docs);
            let batch: RecordBatch = docs_reader.read_range(start..end, None).await?;
            let row_ids = self.row_ids[start..end]
                .iter()
                .map(|&row_id| {
                    if row_id == REMOVED_ROW {
                        None
                    } else {
                        map_row_id(row_id)
                    }
                })
                .collect();
            writer.write_encoded(with_row_ids(&batch, row_ids)?).await?;
        }
        if let Some(new_data) = new_data {
            writer.write_stream(new_data).await?;
        }
        let files = writer.finish().await?;
        Ok(CreatedIndex {
            index_details: details_any(&self.codec),
            index_version: super::PLAID_INDEX_VERSION,
            files,
        })
    }
}

/// Keep the `n` highest scores of `scores`, sorted by descending score.
fn top_by_score(scores: &mut Vec<(u32, f32)>, n: usize) {
    if n == 0 {
        scores.clear();
        return;
    }
    if scores.len() > n {
        scores.select_nth_unstable_by(n - 1, |a, b| b.1.total_cmp(&a.1));
        scores.truncate(n);
    }
    scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
}

#[async_trait]
impl Index for PlaidIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    async fn prewarm(&self) -> Result<()> {
        let num_chunks = self.num_docs().div_ceil(RESIDUAL_CHUNK_DOCS) as u32;
        futures::stream::iter((0..num_chunks).map(|chunk| async move {
            self.cache
                .get_or_insert_with_key(ResidualChunkKey { chunk }, || {
                    self.read_residual_chunk(chunk)
                })
                .await
        }))
        .buffer_unordered(self.store.io_parallelism().max(1))
        .try_collect::<Vec<_>>()
        .await?;
        Ok(())
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "type": "Plaid",
            "num_bits": self.codec.num_bits(),
            "num_centroids": self.codec.num_centroids(),
            "dimension": self.dimension(),
            "num_docs": self.num_docs(),
            "num_tokens": self.codes.len(),
        }))
    }

    fn index_type(&self) -> IndexType {
        IndexType::Plaid
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        Ok(self
            .row_ids
            .iter()
            .filter(|&&row_id| row_id != REMOVED_ROW)
            .map(|row_id| (row_id >> 32) as u32)
            .collect())
    }
}

#[async_trait]
impl ScalarIndex for PlaidIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        _metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        Err(Error::not_supported(format!(
            "PLAID index cannot evaluate scalar filter {query:?}; query it with a nearest neighbor search instead"
        )))
    }

    fn can_remap(&self) -> bool {
        true
    }

    fn stored_fragments(&self) -> Option<RoaringBitmap> {
        Some(
            self.row_ids
                .iter()
                .filter(|&&row_id| row_id != REMOVED_ROW)
                .map(|row_id| (row_id >> 32) as u32)
                .collect(),
        )
    }

    async fn remap(
        &self,
        mapping: &RowAddrRemap,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        self.rewrite(
            dest_store,
            |row_id| mapping.get(row_id).unwrap_or(Some(row_id)),
            None,
        )
        .await
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        old_data_filter: Option<OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        let kept: Option<HashMap<u64, ()>> = match &old_data_filter {
            None => None,
            Some(filter) => {
                let live = self
                    .row_ids
                    .iter()
                    .copied()
                    .filter(|&row_id| row_id != REMOVED_ROW)
                    .collect::<Vec<_>>();
                let row_ids = arrow_array::UInt64Array::from(live);
                let keep = filter.filter_row_ids(&row_ids);
                Some(
                    row_ids
                        .values()
                        .iter()
                        .zip(keep.iter())
                        .filter(|(_, keep)| keep.unwrap_or(false))
                        .map(|(row_id, _)| (*row_id, ()))
                        .collect(),
                )
            }
        };
        self.rewrite(
            dest_store,
            |row_id| match &kept {
                Some(kept) if !kept.contains_key(&row_id) => None,
                _ => Some(row_id),
            },
            Some(new_data),
        )
        .await
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(TrainingCriteria::new(TrainingOrdering::None).with_row_id())
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Ok(
            ScalarIndexParams::for_builtin(BuiltinIndexType::Plaid).with_params(
                &PlaidIndexParams {
                    num_bits: self.codec.num_bits(),
                    num_centroids: Some(self.codec.num_centroids()),
                    ..Default::default()
                },
            ),
        )
    }
}
