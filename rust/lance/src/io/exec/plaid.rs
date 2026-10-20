// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Execution node for PLAID multivector search.
//!
//! Opens every segment of the PLAID index on the queried column, searches each
//! one under the shared prefilter and merges the per-segment top-k by
//! multivector cosine distance. Each segment rescores its candidates exactly
//! (on decompressed tokens), so distances are comparable across segments.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Float32Type};
use arrow_array::{Array, ArrayRef, Float32Array, RecordBatch, UInt64Array};
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use datafusion_physical_expr::{Distribution, EquivalenceProperties, Partitioning};
use futures::{StreamExt, future::try_join_all, stream};
use lance_core::utils::tokio::spawn_cpu;
use lance_core::utils::tracing::StreamTracingExt;
use lance_core::{Error, Result};
use lance_index::prefilter::PreFilter;
use lance_index::scalar::ScalarIndex;
use lance_index::vector::Query;
use lance_index::vector::plaid::{PlaidIndex, PlaidSearchParams};
use lance_select::RowAddrMask;
use lance_table::format::IndexMetadata;
use tracing::instrument;

use super::PreFilterSource;
use super::knn::KNN_INDEX_SCHEMA;
use super::utils::{IndexMetrics, PreFilterMasks, build_prefilter};
use crate::Dataset;
use crate::index::DatasetIndexInternalExt;

/// Whether `segment` belongs to a PLAID index.
pub fn is_plaid_segment(segment: &IndexMetadata) -> bool {
    segment
        .index_details
        .as_ref()
        .is_some_and(|details| details.type_url.ends_with("PlaidIndexDetails"))
}

/// The PLAID search parameters of a nearest neighbor query: `nprobes` is the
/// number of centroids probed per query token and `ef` the number of
/// candidates kept after the approximate centroid scoring.
pub fn plaid_search_params(query: &Query) -> PlaidSearchParams {
    let defaults = PlaidSearchParams::default();
    PlaidSearchParams {
        n_ivf_probe: query.maximum_nprobes.unwrap_or(defaults.n_ivf_probe),
        n_full_scores: query.ef.unwrap_or(defaults.n_full_scores),
        ..defaults
    }
}

/// The query tokens of a multivector query as a flat `Float32` array.
///
/// The scanner casts the query to the column's element type, so a float16
/// column hands us a float16 query: it is widened back here, since the codec
/// scores in f32.
fn query_values(key: &ArrayRef) -> Result<Float32Array> {
    let values = match key.data_type() {
        arrow_schema::DataType::List(_) => key.as_list::<i32>().values().clone(),
        arrow_schema::DataType::FixedSizeList(_, _) => key.as_fixed_size_list().values().clone(),
        _ => key.clone(),
    };
    match values.data_type() {
        arrow_schema::DataType::Float32 => Ok(values.as_primitive::<Float32Type>().clone()),
        arrow_schema::DataType::Float16 => Ok(Float32Array::from_iter_values(
            values
                .as_primitive::<Float16Type>()
                .values()
                .iter()
                .map(|v| v.to_f32()),
        )),
        other => Err(Error::invalid_input(format!(
            "PLAID search only supports float16 and float32 query vectors, got {other}"
        ))),
    }
}

/// Searches every segment of a PLAID index and emits the `k` rows with the
/// smallest multivector distance as `_distance` and `_rowid`.
#[derive(Debug)]
pub struct PlaidSearchExec {
    dataset: Arc<Dataset>,
    query: Query,
    k: usize,
    params: PlaidSearchParams,
    segments: Arc<[IndexMetadata]>,
    prefilter_source: PreFilterSource,
    /// Rows whose indexed value was replaced by a data overlay after the
    /// index was built; they are excluded from the results.
    overlay_block: Option<RowAddrMask>,
    external_mask: Option<Arc<RowAddrMask>>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl PlaidSearchExec {
    pub fn new(
        dataset: Arc<Dataset>,
        query: Query,
        k: usize,
        segments: Vec<IndexMetadata>,
        prefilter_source: PreFilterSource,
        overlay_block: Option<RowAddrMask>,
        external_mask: Option<Arc<RowAddrMask>>,
    ) -> Self {
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(KNN_INDEX_SCHEMA.clone()),
            Partitioning::RoundRobinBatch(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        let params = plaid_search_params(&query);
        Self {
            dataset,
            query,
            k,
            params,
            segments: Arc::from(segments),
            prefilter_source,
            overlay_block,
            external_mask,
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for PlaidSearchExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "PlaidSearch: column={}, k={}, nprobes={}, n_full_scores={}, segments={}",
                self.query.column,
                self.k,
                self.params.n_ivf_probe,
                self.params.n_full_scores,
                self.segments.len()
            ),
            DisplayFormatType::TreeRender => {
                write!(f, "PlaidSearch\ncolumn={}\nk={}", self.query.column, self.k)
            }
        }
    }
}

impl ExecutionPlan for PlaidSearchExec {
    fn name(&self) -> &str {
        "PlaidSearchExec"
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.prefilter_source.execution_plan().into_iter().collect()
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        self.children()
            .iter()
            .map(|_| Distribution::SinglePartition)
            .collect()
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let prefilter_source = match children.len() {
            0 => {
                if !matches!(self.prefilter_source, PreFilterSource::None) {
                    return Err(DataFusionError::Internal(
                        "PlaidSearchExec with a prefilter requires one child".to_string(),
                    ));
                }
                PreFilterSource::None
            }
            1 => {
                let child = children.pop().ok_or_else(|| {
                    DataFusionError::Internal("child vanished after length check".to_string())
                })?;
                self.prefilter_source.with_execution_plan(child)?
            }
            other => {
                return Err(DataFusionError::Internal(format!(
                    "PlaidSearchExec accepts at most one child, got {other}"
                )));
            }
        };
        Ok(Arc::new(Self {
            dataset: self.dataset.clone(),
            query: self.query.clone(),
            k: self.k,
            params: self.params.clone(),
            segments: self.segments.clone(),
            prefilter_source,
            overlay_block: self.overlay_block.clone(),
            external_mask: self.external_mask.clone(),
            properties: self.properties.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    #[instrument(name = "plaid_search_exec", level = "debug", skip_all)]
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let dataset = self.dataset.clone();
        let query = self.query.clone();
        let k = self.k;
        let params = self.params.clone();
        let segments = self.segments.clone();
        let prefilter_source = self.prefilter_source.clone();
        let overlay_block = self.overlay_block.clone();
        let external_mask = self.external_mask.clone();
        let metrics = self.metrics.clone();
        let index_metrics = Arc::new(IndexMetrics::new(&self.metrics, partition));
        let baseline_metrics = BaselineMetrics::new(&self.metrics, partition);
        let stream = stream::once(async move {
            let _timer = baseline_metrics.elapsed_compute().timer();
            let indices = try_join_all(segments.iter().map(|segment| {
                dataset.open_scalar_index(&query.column, &segment.uuid, index_metrics.as_ref())
            }))
            .await?;
            let pre_filter = build_prefilter(
                context,
                partition,
                &prefilter_source,
                dataset,
                &segments,
                PreFilterMasks {
                    overlay_block,
                    external_mask,
                },
                &metrics,
            )?;
            pre_filter.wait_for_ready().await?;
            let mask = pre_filter.mask();
            let query_values = Arc::new(query_values(&query.key)?);

            let mut hits: Vec<(u64, f32)> = Vec::new();
            for index in indices {
                let (mask, params, query_values) =
                    (mask.clone(), params.clone(), query_values.clone());
                let index_for_cpu = index.clone();
                let shortlist = spawn_cpu(move || {
                    plaid_segment(index_for_cpu.as_ref())?.shortlist(
                        query_values.values(),
                        k,
                        &params,
                        &mask,
                    )
                })
                .await?;
                let chunks = plaid_segment(index.as_ref())?
                    .load_residuals(&shortlist)
                    .await?;
                let segment_hits = spawn_cpu(move || {
                    plaid_segment(index.as_ref())?.rescore(&shortlist, &chunks, k)
                })
                .await?;
                hits.extend(segment_hits);
            }
            hits.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            hits.truncate(k);
            index_metrics.flush_io();
            baseline_metrics.record_output(hits.len());
            hits_batch(&hits)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            KNN_INDEX_SCHEMA.clone(),
            stream.stream_in_current_span().boxed(),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
}

fn plaid_segment(index: &dyn ScalarIndex) -> Result<&PlaidIndex> {
    index.as_any().downcast_ref::<PlaidIndex>().ok_or_else(|| {
        Error::invalid_input(format!(
            "opened index is a {} index, not a PLAID index",
            index.index_type()
        ))
    })
}

fn hits_batch(hits: &[(u64, f32)]) -> DataFusionResult<RecordBatch> {
    let distances = Float32Array::from_iter_values(hits.iter().map(|(_, distance)| *distance));
    let row_ids = UInt64Array::from_iter_values(hits.iter().map(|(row_id, _)| *row_id));
    Ok(RecordBatch::try_new(
        KNN_INDEX_SCHEMA.clone(),
        vec![
            Arc::new(distances) as ArrayRef,
            Arc::new(row_ids) as ArrayRef,
        ],
    )?)
}
