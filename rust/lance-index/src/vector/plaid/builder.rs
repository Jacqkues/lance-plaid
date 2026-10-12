// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Writes the files of a PLAID index segment.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{BinaryBuilder, Float32Builder, ListBuilder, UInt32Builder};
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{
    Array, ArrayRef, BinaryArray, FixedSizeListArray, Float32Array, ListArray, RecordBatch,
    UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use datafusion::execution::SendableRecordBatchStream;
use futures::TryStreamExt;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, ROW_ID, Result};
use serde::{Deserialize, Serialize};

use super::codec::{ResidualCodec, fsl_to_array2, normalize_rows};
use crate::scalar::registry::VALUE_COLUMN_NAME;
use crate::scalar::{IndexFile, IndexStore, IndexWriter};

/// Centroids, one row per centroid, with the length of its inverted list.
pub const CENTROIDS_FILE: &str = "plaid_centroids.lance";
/// Inverted lists: the doc ids of every centroid, concatenated in centroid order.
pub const IVF_FILE: &str = "plaid_ivf.lance";
/// One row per indexed document: row id, centroid id and packed residual of each token.
pub const DOCS_FILE: &str = "plaid_docs.lance";

pub(crate) const METADATA_KEY: &str = "lance:plaid";
pub(crate) const FILE_FORMAT_VERSION: u32 = 1;

pub(crate) const CENTROID_COLUMN: &str = "centroid";
pub(crate) const IVF_LENGTH_COLUMN: &str = "ivf_length";
pub(crate) const DOC_ID_COLUMN: &str = "doc_id";
pub(crate) const CODES_COLUMN: &str = "codes";
pub(crate) const RESIDUALS_COLUMN: &str = "residuals";
pub(crate) const INV_NORMS_COLUMN: &str = "inv_norms";

/// Rows per batch when writing the inverted lists.
const IVF_WRITE_ROWS: usize = 1 << 20;

/// Everything about a segment that is not stored as rows, kept in the schema
/// metadata of the centroids file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct PlaidFileMetadata {
    pub version: u32,
    pub num_bits: u8,
    pub dimension: usize,
    pub num_centroids: usize,
    pub num_docs: usize,
    pub num_tokens: usize,
    pub bucket_cutoffs: Vec<f32>,
    pub bucket_weights: Vec<f32>,
}

pub(crate) fn docs_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(ROW_ID, DataType::UInt64, false),
        Field::new(
            CODES_COLUMN,
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, false))),
            false,
        ),
        Field::new(RESIDUALS_COLUMN, DataType::Binary, false),
        Field::new(
            INV_NORMS_COLUMN,
            DataType::List(Arc::new(Field::new("item", DataType::Float32, false))),
            false,
        ),
    ]))
}

/// Encodes documents with a trained [`ResidualCodec`] and writes a PLAID segment.
///
/// Documents are written as they arrive; the inverted lists are kept in memory
/// (one `u32` per distinct (centroid, document) pair) and written by [`Self::finish`].
pub struct PlaidIndexWriter<'a> {
    codec: Arc<ResidualCodec>,
    store: &'a dyn IndexStore,
    docs_writer: Box<dyn IndexWriter>,
    inverted_lists: Vec<Vec<u32>>,
    num_docs: u32,
    num_tokens: usize,
}

impl<'a> PlaidIndexWriter<'a> {
    pub async fn try_new(codec: Arc<ResidualCodec>, store: &'a dyn IndexStore) -> Result<Self> {
        let docs_writer = store.new_index_file(DOCS_FILE, docs_schema()).await?;
        let inverted_lists = vec![Vec::new(); codec.num_centroids()];
        Ok(Self {
            codec,
            store,
            docs_writer,
            inverted_lists,
            num_docs: 0,
            num_tokens: 0,
        })
    }

    /// Encode and write a stream of `value` (`List<FixedSizeList<Float32>>`)
    /// and `_rowid` batches. Null and empty multivectors are not indexed.
    pub async fn write_stream(&mut self, mut data: SendableRecordBatchStream) -> Result<()> {
        while let Some(batch) = data.try_next().await? {
            self.write_raw_batch(batch).await?;
        }
        Ok(())
    }

    pub async fn write_raw_batch(&mut self, batch: RecordBatch) -> Result<()> {
        let codec = self.codec.clone();
        let encoded = spawn_cpu(move || encode_batch(&codec, &batch)).await?;
        self.write_encoded(encoded).await
    }

    /// Write documents that were already encoded with this writer's codec.
    pub(crate) async fn write_encoded(&mut self, docs: RecordBatch) -> Result<()> {
        if docs.num_rows() == 0 {
            return Ok(());
        }
        let codes = docs[CODES_COLUMN].as_list::<i32>();
        let next_doc = self
            .num_docs
            .checked_add(docs.num_rows() as u32)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "a PLAID index segment holds at most {} documents",
                    u32::MAX
                ))
            })?;
        for (offset, doc_codes) in codes.iter().enumerate() {
            let doc_id = self.num_docs + offset as u32;
            let doc_codes = doc_codes.ok_or_else(|| {
                Error::invalid_input("encoded PLAID documents must not be null".to_string())
            })?;
            for &code in doc_codes
                .as_primitive::<arrow_array::types::UInt32Type>()
                .values()
            {
                let list = &mut self.inverted_lists[code as usize];
                if list.last() != Some(&doc_id) {
                    list.push(doc_id);
                }
            }
        }
        self.num_tokens += codes.values().len();
        self.num_docs = next_doc;
        self.docs_writer.write_record_batch(docs).await?;
        Ok(())
    }

    pub async fn finish(mut self) -> Result<Vec<IndexFile>> {
        let metadata = PlaidFileMetadata {
            version: FILE_FORMAT_VERSION,
            num_bits: self.codec.num_bits(),
            dimension: self.codec.dimension(),
            num_centroids: self.codec.num_centroids(),
            num_docs: self.num_docs as usize,
            num_tokens: self.num_tokens,
            bucket_cutoffs: self.codec.bucket_cutoffs().to_vec(),
            bucket_weights: self.codec.bucket_weights().to_vec(),
        };
        let metadata =
            HashMap::from([(METADATA_KEY.to_string(), serde_json::to_string(&metadata)?)]);

        let docs_file = self
            .docs_writer
            .finish_with_metadata(metadata.clone())
            .await?;

        let ivf_schema = Arc::new(Schema::new(vec![Field::new(
            DOC_ID_COLUMN,
            DataType::UInt32,
            false,
        )]));
        let mut ivf_writer = self
            .store
            .new_index_file(IVF_FILE, ivf_schema.clone())
            .await?;
        let mut pending: Vec<u32> = Vec::with_capacity(IVF_WRITE_ROWS);
        for list in &self.inverted_lists {
            pending.extend_from_slice(list);
            if pending.len() >= IVF_WRITE_ROWS {
                let column = UInt32Array::from(std::mem::take(&mut pending));
                ivf_writer
                    .write_record_batch(RecordBatch::try_new(
                        ivf_schema.clone(),
                        vec![Arc::new(column)],
                    )?)
                    .await?;
            }
        }
        if !pending.is_empty() {
            let column = UInt32Array::from(pending);
            ivf_writer
                .write_record_batch(RecordBatch::try_new(
                    ivf_schema.clone(),
                    vec![Arc::new(column)],
                )?)
                .await?;
        }
        let ivf_file = ivf_writer.finish_with_metadata(metadata.clone()).await?;

        let dimension = self.codec.dimension();
        let centroids_schema = Arc::new(Schema::new(vec![
            Field::new(
                CENTROID_COLUMN,
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dimension as i32,
                ),
                false,
            ),
            Field::new(IVF_LENGTH_COLUMN, DataType::UInt32, false),
        ]));
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from_iter_values(self.codec.centroids().iter().copied()),
            dimension as i32,
        )?;
        let lengths =
            UInt32Array::from_iter_values(self.inverted_lists.iter().map(|l| l.len() as u32));
        let mut centroids_writer = self
            .store
            .new_index_file(CENTROIDS_FILE, centroids_schema.clone())
            .await?;
        centroids_writer
            .write_record_batch(RecordBatch::try_new(
                centroids_schema,
                vec![Arc::new(centroids), Arc::new(lengths)],
            )?)
            .await?;
        let centroids_file = centroids_writer.finish_with_metadata(metadata).await?;

        Ok(vec![centroids_file, ivf_file, docs_file])
    }
}

/// Encode one batch of raw multivectors into the rows of the docs file.
fn encode_batch(codec: &ResidualCodec, batch: &RecordBatch) -> Result<RecordBatch> {
    let row_ids = batch
        .column_by_name(ROW_ID)
        .and_then(|c| c.as_primitive_opt::<UInt64Type>())
        .ok_or_else(|| {
            Error::invalid_input("PLAID training data must have a _rowid column".to_string())
        })?;
    let values = batch.column_by_name(VALUE_COLUMN_NAME).ok_or_else(|| {
        Error::invalid_input("PLAID training data must have a value column".to_string())
    })?;
    let lists = values.as_list_opt::<i32>().ok_or_else(|| {
        Error::invalid_input(format!(
            "PLAID index requires a List<FixedSizeList<Float32>> column, got {}",
            values.data_type()
        ))
    })?;
    let tokens = lists.values().as_fixed_size_list_opt().ok_or_else(|| {
        Error::invalid_input(format!(
            "PLAID index requires a List<FixedSizeList<Float32>> column, got {}",
            values.data_type()
        ))
    })?;
    if tokens.value_length() as usize != codec.dimension() {
        return Err(Error::invalid_input(format!(
            "PLAID index has dimension {} but the column has vectors of dimension {}",
            codec.dimension(),
            tokens.value_length()
        )));
    }

    let offsets = lists.value_offsets();
    let first = offsets[0] as usize;
    let last = offsets[lists.len()] as usize;
    let tokens = normalize_rows(fsl_to_array2(&tokens.slice(first, last - first))?);
    let codes = codec.assign(tokens.view());
    let mut packed = Vec::new();
    codec.encode_residuals(tokens.view(), &codes, &mut packed);
    let inv_norms = codec.inv_norms(&codes, &packed);

    let packed_dim = codec.packed_dim();
    let mut out_row_ids = Vec::with_capacity(lists.len());
    let mut out_codes =
        ListBuilder::with_capacity(UInt32Builder::with_capacity(codes.len()), lists.len())
            .with_field(Arc::new(Field::new("item", DataType::UInt32, false)));
    let mut out_residuals = BinaryBuilder::with_capacity(lists.len(), packed.len());
    let mut out_inv_norms =
        ListBuilder::with_capacity(Float32Builder::with_capacity(codes.len()), lists.len())
            .with_field(Arc::new(Field::new("item", DataType::Float32, false)));
    for row in 0..lists.len() {
        let start = offsets[row] as usize - first;
        let end = offsets[row + 1] as usize - first;
        if lists.is_null(row) || start == end || row_ids.is_null(row) {
            continue;
        }
        out_row_ids.push(row_ids.value(row));
        out_codes.values().append_slice(&codes[start..end]);
        out_codes.append(true);
        out_residuals.append_value(&packed[start * packed_dim..end * packed_dim]);
        out_inv_norms.values().append_slice(&inv_norms[start..end]);
        out_inv_norms.append(true);
    }
    Ok(RecordBatch::try_new(
        docs_schema(),
        vec![
            Arc::new(UInt64Array::from(out_row_ids)) as ArrayRef,
            Arc::new(out_codes.finish()) as ArrayRef,
            Arc::new(out_residuals.finish()) as ArrayRef,
            Arc::new(out_inv_norms.finish()) as ArrayRef,
        ],
    )?)
}

/// The docs of `batch` (docs file rows) for which `keep` is true.
pub(crate) fn filter_docs(
    batch: &RecordBatch,
    keep: &arrow_array::BooleanArray,
) -> Result<RecordBatch> {
    Ok(arrow_select::filter::filter_record_batch(batch, keep)?)
}

/// Replace the row ids of encoded docs, dropping docs whose new row id is `None`.
pub(crate) fn with_row_ids(batch: &RecordBatch, row_ids: Vec<Option<u64>>) -> Result<RecordBatch> {
    let keep = arrow_array::BooleanArray::from_iter(row_ids.iter().map(|id| Some(id.is_some())));
    let row_ids = UInt64Array::from_iter_values(row_ids.into_iter().flatten());
    let kept = filter_docs(batch, &keep)?;
    let codes: &ListArray = kept[CODES_COLUMN].as_list::<i32>();
    let residuals: &BinaryArray = kept[RESIDUALS_COLUMN].as_binary::<i32>();
    let inv_norms: &ListArray = kept[INV_NORMS_COLUMN].as_list::<i32>();
    Ok(RecordBatch::try_new(
        docs_schema(),
        vec![
            Arc::new(row_ids) as ArrayRef,
            Arc::new(codes.clone()) as ArrayRef,
            Arc::new(residuals.clone()) as ArrayRef,
            Arc::new(inv_norms.clone()) as ArrayRef,
        ],
    )?)
}
