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
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::{Error, Result};
use lance_select::RowAddrMask;
use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;
use roaring::RoaringBitmap;

use super::builder::{
    CENTROID_COLUMN, CENTROIDS_FILE, CODES_COLUMN, DOC_ID_COLUMN, DOCS_FILE, FILE_FORMAT_VERSION,
    IVF_FILE, IVF_LENGTH_COLUMN, METADATA_KEY, PlaidFileMetadata, PlaidIndexWriter,
    RESIDUALS_COLUMN, with_row_ids,
};
use super::codec::{ResidualCodec, normalize_rows};
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

/// One segment of a PLAID index, fully resident in memory.
///
/// TODO: keep the residuals in the index cache in chunks instead of loading
/// them all; only the `n_full_scores / 4` best candidates of a query read them.
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
    /// Packed residual of every token, [`ResidualCodec::packed_dim`] bytes each.
    residuals: Vec<u8>,
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
        self.codec.centroids().len() * size_of::<f32>()
            + self.ivf_offsets.len() * size_of::<usize>()
            + self.ivf_doc_ids.len() * size_of::<u32>()
            + self.row_ids.len() * size_of::<u64>()
            + self.token_offsets.len() * size_of::<usize>()
            + self.codes.len() * size_of::<u32>()
            + self.residuals.len()
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

impl PlaidIndex {
    pub async fn load(
        store: Arc<dyn IndexStore>,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
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
            .read_range(0..docs_reader.num_rows(), None)
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
        let token_offsets: Vec<usize> = codes_list
            .value_offsets()
            .iter()
            .map(|o| (*o - codes_list.value_offsets()[0]) as usize)
            .collect();
        let codes = codes_list.values().as_primitive::<UInt32Type>().values()
            [codes_list.value_offsets()[0] as usize..]
            [..token_offsets.last().copied().unwrap_or(0)]
            .to_vec();
        let residuals_array = docs[RESIDUALS_COLUMN].as_binary::<i32>();
        let residual_start = residuals_array.value_offsets()[0] as usize;
        let residual_end = residuals_array.value_offsets()[residuals_array.len()] as usize;
        let residuals = residuals_array.value_data()[residual_start..residual_end].to_vec();
        if residuals.len() != codes.len() * codec.packed_dim() {
            return Err(Error::corrupt_file_named(
                DOCS_FILE,
                format!(
                    "has {} residual bytes for {} tokens of {} bytes",
                    residuals.len(),
                    codes.len(),
                    codec.packed_dim()
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
            residuals,
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
    /// (one query token per row) among the rows `mask` selects, as
    /// `(row_id, distance)` ordered by ascending distance.
    ///
    /// The distance is `num_query_tokens - MaxSim` over L2-normalized tokens,
    /// the multivector cosine distance of a flat search.
    pub fn search(
        &self,
        query: ArrayView2<f32>,
        k: usize,
        params: &PlaidSearchParams,
        mask: &RowAddrMask,
    ) -> Result<Vec<(u64, f32)>> {
        if query.ncols() != self.dimension() {
            return Err(Error::invalid_input(format!(
                "PLAID query has vectors of dimension {} but the index has dimension {}",
                query.ncols(),
                self.dimension()
            )));
        }
        if k == 0 || query.nrows() == 0 || self.num_docs() == 0 {
            return Ok(Vec::new());
        }
        let query = normalize_rows(query.to_owned());
        let num_query_tokens = query.nrows();
        let num_centroids = self.codec.num_centroids();

        // Stage 1: probe the best centroids of every query token.
        let scores = self.codec.centroid_scores(query.view());
        let scores = scores
            .as_slice()
            .ok_or_else(|| Error::internal("centroid scores are not contiguous".to_string()))?;
        let score =
            |centroid: u32, token: usize| scores[centroid as usize * num_query_tokens + token];
        let num_probes = params.n_ivf_probe.clamp(1, num_centroids);
        let mut probed = vec![false; num_centroids];
        let mut order: Vec<u32> = (0..num_centroids as u32).collect();
        for token in 0..num_query_tokens {
            order.select_nth_unstable_by(num_probes - 1, |&a, &b| {
                score(b, token).total_cmp(&score(a, token))
            });
            for &centroid in &order[..num_probes] {
                probed[centroid as usize] = true;
            }
        }
        if let Some(threshold) = params.centroid_score_threshold {
            for (centroid, is_probed) in probed.iter_mut().enumerate() {
                if *is_probed {
                    let row = &scores[centroid * num_query_tokens..][..num_query_tokens];
                    *is_probed = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) >= threshold;
                }
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
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

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

        // Stage 4: exact MaxSim on decompressed tokens.
        let dimension = self.dimension();
        let packed_dim = self.codec.packed_dim();
        let mut exact: Vec<(u32, f32)> = approx
            .par_iter()
            .map(|&(doc, _)| {
                let tokens = self.token_offsets[doc as usize]..self.token_offsets[doc as usize + 1];
                let mut decoded = Array2::<f32>::zeros((tokens.len(), dimension));
                for (mut row, token) in decoded.axis_iter_mut(Axis(0)).zip(tokens) {
                    let packed = &self.residuals[token * packed_dim..][..packed_dim];
                    self.codec.decode_into(
                        self.codes[token],
                        packed,
                        row.as_slice_mut()
                            .expect("rows of a standard layout array are contiguous"),
                    );
                }
                let similarities = query.dot(&decoded.t());
                let score: f32 = similarities
                    .axis_iter(Axis(0))
                    .map(|row| row.iter().copied().fold(f32::NEG_INFINITY, f32::max))
                    .sum();
                (doc, score)
            })
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
        // The whole segment is loaded when it is opened.
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
