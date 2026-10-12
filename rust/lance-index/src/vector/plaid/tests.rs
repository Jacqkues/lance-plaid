// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, ListArray, RecordBatch, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema};
use datafusion::execution::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use lance_core::ROW_ID;
use lance_core::cache::LanceCache;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::utils::tempfile::TempObjDir;
use lance_io::object_store::ObjectStore;
use lance_select::{RowAddrMask, RowAddrTreeMap};
use ndarray::{Array2, Axis};
use rand::{Rng, SeedableRng, rngs::StdRng};

use lance_arrow::FixedSizeListArrayExt;

use super::codec::normalize_rows;
use super::*;
use crate::scalar::OldIndexDataFilter;
use crate::scalar::lance_format::LanceIndexStore;
use crate::scalar::registry::VALUE_COLUMN_NAME;

const DIM: usize = 32;
const TOKENS_PER_DOC: usize = 12;

fn test_store() -> (TempObjDir, Arc<LanceIndexStore>) {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::local()),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    (tmpdir, store)
}

/// Documents whose tokens are noisy copies of a few of 64 random topics, so
/// that MaxSim neighbors are well defined but not trivially separated.
struct Corpus {
    topics: Array2<f32>,
    docs: Vec<Option<Array2<f32>>>,
    rng: StdRng,
}

impl Corpus {
    fn new(num_docs: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let topics = Array2::from_shape_fn((64, DIM), |_| rng.random_range(-1.0f32..1.0));
        let mut corpus = Self {
            topics,
            docs: Vec::new(),
            rng,
        };
        corpus.docs = (0..num_docs)
            .map(|_| Some(corpus.sample(TOKENS_PER_DOC)))
            .collect();
        corpus
    }

    fn sample(&mut self, num_tokens: usize) -> Array2<f32> {
        let doc_topics: Vec<usize> = (0..3).map(|_| self.rng.random_range(0..64)).collect();
        let mut tokens = Array2::zeros((num_tokens, DIM));
        for mut token in tokens.axis_iter_mut(Axis(0)) {
            let topic = doc_topics[self.rng.random_range(0..doc_topics.len())];
            for (j, v) in token.iter_mut().enumerate() {
                *v = self.topics[[topic, j]] + self.rng.random_range(-0.6f32..0.6);
            }
        }
        normalize_rows(tokens)
    }

    fn batch(&self, rows: std::ops::Range<usize>) -> RecordBatch {
        let item = Arc::new(Field::new("item", DataType::Float32, false));
        let token_field = Arc::new(Field::new(
            "item",
            DataType::FixedSizeList(item.clone(), DIM as i32),
            false,
        ));
        let docs = &self.docs[rows.clone()];
        let lengths = docs.iter().map(|d| d.as_ref().map_or(0, |d| d.nrows()));
        let values: Vec<f32> = docs
            .iter()
            .flatten()
            .flat_map(|d| d.iter().copied())
            .collect();
        let tokens =
            FixedSizeListArray::new(item, DIM as i32, Arc::new(Float32Array::from(values)), None);
        let nulls = NullBuffer::from(docs.iter().map(Option::is_some).collect::<Vec<_>>());
        let lists = ListArray::new(
            token_field,
            OffsetBuffer::from_lengths(lengths),
            Arc::new(tokens),
            Some(nulls),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new(VALUE_COLUMN_NAME, lists.data_type().clone(), true),
            Field::new(ROW_ID, DataType::UInt64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(lists) as ArrayRef,
                Arc::new(UInt64Array::from_iter_values(rows.map(|r| r as u64))) as ArrayRef,
            ],
        )
        .unwrap()
    }

    fn stream(&self, rows: std::ops::Range<usize>, batch_rows: usize) -> SendableRecordBatchStream {
        let batches: Vec<_> = rows
            .clone()
            .step_by(batch_rows)
            .map(|start| Ok(self.batch(start..(start + batch_rows).min(rows.end))))
            .collect();
        let schema = self.batch(0..1).schema();
        Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(batches),
        ))
    }

    fn sample_tokens(&self) -> FixedSizeListArray {
        let values: Vec<f32> = self
            .docs
            .iter()
            .flatten()
            .flat_map(|d| d.iter().copied())
            .collect();
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), DIM as i32).unwrap()
    }

    /// Row ids of the `k` best documents by exact MaxSim.
    fn exact_top_k(&self, query: &Array2<f32>, k: usize) -> Vec<u64> {
        let mut scored: Vec<(u64, f32)> = self
            .docs
            .iter()
            .enumerate()
            .filter_map(|(i, d)| d.as_ref().map(|d| (i as u64, d)))
            .map(|(i, d)| {
                let sims = query.dot(&d.t());
                let score = sims
                    .axis_iter(Axis(0))
                    .map(|r| r.iter().copied().fold(f32::NEG_INFINITY, f32::max))
                    .sum();
                (i, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.into_iter().take(k).map(|(i, _)| i).collect()
    }
}

async fn build(corpus: &Corpus, store: &LanceIndexStore, num_bits: u8) -> Arc<PlaidIndex> {
    let codec = ResidualCodec::train(
        &corpus.sample_tokens(),
        32,
        num_bits,
        &kmeans_params(10, Some(7)),
    )
    .unwrap();
    let mut writer = PlaidIndexWriter::try_new(Arc::new(codec), store)
        .await
        .unwrap();
    writer
        .write_stream(corpus.stream(0..corpus.docs.len(), 100))
        .await
        .unwrap();
    writer.finish().await.unwrap();
    PlaidIndex::load(Arc::new(store.clone()), None)
        .await
        .unwrap()
}

fn recall(
    index: &PlaidIndex,
    corpus: &mut Corpus,
    params: &PlaidSearchParams,
    mask: &RowAddrMask,
) -> f32 {
    let mut hits = 0;
    for _ in 0..20 {
        let query = corpus.sample(8);
        let expected = corpus.exact_top_k(&query, 10);
        let found: Vec<u64> = index
            .search(query.as_slice().unwrap(), 10, params, mask)
            .unwrap()
            .into_iter()
            .map(|(row_id, _)| row_id)
            .collect();
        hits += expected.iter().filter(|r| found.contains(r)).count();
    }
    hits as f32 / 200.0
}

/// Exhaustive search on 8-bit residuals finds the exact MaxSim neighbors.
#[tokio::test]
async fn test_exhaustive_search_is_exact() {
    let mut corpus = Corpus::new(500, 1);
    let (_dir, store) = test_store();
    let index = build(&corpus, &store, 8).await;
    assert_eq!(index.num_docs(), 500);
    let exhaustive = PlaidSearchParams {
        n_ivf_probe: 32,
        n_full_scores: 4 * 500,
        centroid_score_threshold: None,
    };
    let recall = recall(&index, &mut corpus, &exhaustive, &RowAddrMask::all_rows());
    assert!(recall >= 0.98, "recall@10 {recall} < 0.98");
}

/// With the default pruning, recall is bounded by the approximate centroid
/// scoring and, at low bit widths, by the residual quantization.
#[rstest::rstest]
#[case::two_bits(2)]
#[case::four_bits(4)]
#[tokio::test]
async fn test_search_recall_against_exact_maxsim(#[case] num_bits: u8) {
    let mut corpus = Corpus::new(500, 1);
    let (_dir, store) = test_store();
    let index = build(&corpus, &store, num_bits).await;
    let params = PlaidSearchParams {
        n_ivf_probe: 4,
        n_full_scores: 256,
        centroid_score_threshold: None,
    };
    let recall = recall(&index, &mut corpus, &params, &RowAddrMask::all_rows());
    eprintln!("num_bits={num_bits} recall={recall}");
    assert!(recall >= 0.7, "recall@10 {recall} < 0.7");
}

#[tokio::test]
async fn test_distance_is_num_query_tokens_minus_maxsim() {
    let mut corpus = Corpus::new(200, 2);
    let (_dir, store) = test_store();
    let index = build(&corpus, &store, 8).await;
    let query = corpus.sample(8);
    let results = index
        .search(
            query.as_slice().unwrap(),
            5,
            &PlaidSearchParams::default(),
            &RowAddrMask::all_rows(),
        )
        .unwrap();
    assert!(results.windows(2).all(|w| w[0].1 <= w[1].1), "{results:?}");
    let (row_id, distance) = results[0];
    let doc = corpus.docs[row_id as usize].as_ref().unwrap();
    let maxsim: f32 = query
        .dot(&doc.t())
        .axis_iter(Axis(0))
        .map(|r| r.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        .sum();
    // 8-bit residuals reconstruct the tokens almost exactly.
    assert!(
        (distance - (8.0 - maxsim)).abs() < 0.05,
        "{distance} vs {}",
        8.0 - maxsim
    );
}

#[tokio::test]
async fn test_null_and_empty_documents_are_not_indexed() {
    let mut corpus = Corpus::new(300, 3);
    corpus.docs[5] = None;
    corpus.docs[6] = Some(Array2::zeros((0, DIM)));
    let (_dir, store) = test_store();
    let index = build(&corpus, &store, 2).await;
    assert_eq!(index.num_docs(), 298);
    let query = corpus.sample(8);
    let results = index
        .search(
            query.as_slice().unwrap(),
            300,
            &PlaidSearchParams {
                n_ivf_probe: 32,
                n_full_scores: 1200,
                centroid_score_threshold: None,
            },
            &RowAddrMask::all_rows(),
        )
        .unwrap();
    assert_eq!(results.len(), 298);
    assert!(
        results
            .iter()
            .all(|(row_id, _)| *row_id != 5 && *row_id != 6)
    );
}

#[tokio::test]
async fn test_score_threshold_keeps_best_centroid_of_each_token() {
    let mut corpus = Corpus::new(200, 8);
    let (_dir, store) = test_store();
    let index = build(&corpus, &store, 4).await;
    let query = corpus.sample(4);
    // No centroid can reach this score: only each token's best one is probed.
    let params = PlaidSearchParams {
        centroid_score_threshold: Some(10.0),
        ..Default::default()
    };
    let results = index
        .search(
            query.as_slice().unwrap(),
            5,
            &params,
            &RowAddrMask::all_rows(),
        )
        .unwrap();
    assert_eq!(results.len(), 5);
}

#[tokio::test]
async fn test_search_respects_mask() {
    let mut corpus = Corpus::new(400, 4);
    let (_dir, store) = test_store();
    let index = build(&corpus, &store, 8).await;
    let allowed: RowAddrTreeMap = (0..400u64).filter(|r| r % 3 == 0).collect();
    let mask = RowAddrMask::from_allowed(allowed);
    for _ in 0..5 {
        let query = corpus.sample(8);
        let results = index
            .search(
                query.as_slice().unwrap(),
                10,
                &PlaidSearchParams::default(),
                &mask,
            )
            .unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|(row_id, _)| row_id % 3 == 0));
    }
    // Exact recall still holds within the allowed rows.
    corpus
        .docs
        .iter_mut()
        .enumerate()
        .filter(|(i, _)| i % 3 != 0)
        .for_each(|(_, d)| *d = None);
    let mask = RowAddrMask::from_allowed((0..400u64).filter(|r| r % 3 == 0).collect());
    let exhaustive = PlaidSearchParams {
        n_ivf_probe: 32,
        n_full_scores: 4 * 400,
        centroid_score_threshold: None,
    };
    let recall = recall(&index, &mut corpus, &exhaustive, &mask);
    assert!(recall >= 0.98, "recall@10 {recall} < 0.98");
}

#[tokio::test]
async fn test_update_appends_and_filters_old_rows() {
    let corpus = Corpus::new(400, 5);
    let (_dir, store) = test_store();
    let codec =
        ResidualCodec::train(&corpus.sample_tokens(), 32, 4, &kmeans_params(10, Some(7))).unwrap();
    let mut writer = PlaidIndexWriter::try_new(Arc::new(codec), store.as_ref())
        .await
        .unwrap();
    writer
        .write_stream(corpus.stream(0..300, 100))
        .await
        .unwrap();
    writer.finish().await.unwrap();
    let index = PlaidIndex::load(store.clone(), None).await.unwrap();

    let (_dest_dir, dest) = test_store();
    let keep: RowAddrTreeMap = (100..300u64).collect();
    index
        .update(
            corpus.stream(300..400, 64),
            dest.as_ref(),
            Some(OldIndexDataFilter::RowIds(keep)),
        )
        .await
        .unwrap();
    let updated = PlaidIndex::load(dest.clone(), None).await.unwrap();
    assert_eq!(updated.num_docs(), 300);

    // A document of the new data is its own nearest neighbor; dropped rows never come back.
    let query = corpus.docs[350].clone().unwrap();
    let results = updated
        .search(
            query.as_slice().unwrap(),
            400,
            &PlaidSearchParams {
                n_ivf_probe: 32,
                n_full_scores: 1600,
                centroid_score_threshold: None,
            },
            &RowAddrMask::all_rows(),
        )
        .unwrap();
    assert_eq!(results[0].0, 350);
    assert!(results.iter().all(|(row_id, _)| *row_id >= 100));
    assert_eq!(results.len(), 300);
}

#[tokio::test]
async fn test_remap_moves_and_drops_rows() {
    let corpus = Corpus::new(200, 6);
    let (_dir, store) = test_store();
    let index = build(&corpus, &store, 4).await;
    let mapping = RowAddrRemap::direct(
        (0..200u64)
            .map(|r| (r, if r < 50 { None } else { Some(r + 1000) }))
            .collect(),
    );
    let (_dest_dir, dest) = test_store();
    index.remap(&mapping, dest.as_ref()).await.unwrap();
    let remapped = PlaidIndex::load(dest.clone(), None).await.unwrap();
    assert_eq!(remapped.num_docs(), 150);
    let query = corpus.docs[120].clone().unwrap();
    let results = remapped
        .search(
            query.as_slice().unwrap(),
            5,
            &PlaidSearchParams::default(),
            &RowAddrMask::all_rows(),
        )
        .unwrap();
    assert_eq!(results[0].0, 1120);
}

#[test]
fn test_params_validation_and_default_num_centroids() {
    let params = PlaidIndexParams::from_json(r#"{"num_bits": 2}"#).unwrap();
    assert_eq!(params.num_bits, 2);
    // 16 * sqrt(7.7M tokens) ~ 44k -> 32768, PLAID's choice for BEIR FiQA.
    assert_eq!(params.num_centroids_for(7_695_260), 32768);
    let err = PlaidIndexParams::from_json(r#"{"num_bits": 3}"#).unwrap_err();
    assert!(
        err.to_string().contains("num_bits must be 1, 2, 4 or 8"),
        "{err}"
    );
    let err = PlaidIndexParams::from_json(r#"{"nbits": 4}"#).unwrap_err();
    assert!(err.to_string().contains("unknown field"), "{err}");
}
