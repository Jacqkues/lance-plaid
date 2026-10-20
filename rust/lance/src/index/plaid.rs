// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Building a PLAID multivector index over a dataset column.

use std::sync::Arc;

use arrow_array::Array;
use arrow_schema::DataType;
use lance_core::utils::tokio::spawn_cpu;
use lance_index::progress::IndexBuildProgress;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{CreatedIndex, TrainingCriteria, TrainingOrdering};
use lance_index::vector::plaid::{
    PLAID_INDEX_VERSION, PlaidIndexParams, PlaidIndexWriter, ResidualCodec, details_any,
    kmeans_params,
};
use tracing::instrument;
use uuid::Uuid;

use crate::dataset::index::LanceIndexStoreExt;
use crate::index::scalar::load_training_data;
use crate::index::vector::utils::{
    estimate_multivector_vectors_per_row, filter_finite_training_data, maybe_sample_training_data,
};
use crate::{Dataset, Error, Result};

/// Train a PLAID codec on a sample of `column` and encode every row of it.
///
/// Two passes over the column: k-means and the residual buckets are trained
/// on a sample of `num_centroids * sample_rate` token embeddings, then every
/// row is streamed through the frozen codec.
#[instrument(level = "debug", skip_all)]
pub(super) async fn build_plaid_index(
    dataset: &Dataset,
    column: &str,
    uuid: Uuid,
    params: &PlaidIndexParams,
    train: bool,
    fragment_ids: Option<Vec<u32>>,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<CreatedIndex> {
    let field = dataset
        .schema()
        .field(column)
        .ok_or_else(|| Error::invalid_input(format!("No column with name {column}")))?;
    let is_multivector = matches!(
        field.data_type(),
        DataType::List(item) if matches!(
            item.data_type(),
            DataType::FixedSizeList(value, _)
                if matches!(value.data_type(), DataType::Float16 | DataType::Float32)
        )
    );
    if !is_multivector {
        return Err(Error::invalid_input(format!(
            "PLAID index requires a List<FixedSizeList<Float16 | Float32>> column, column {column} has type {}",
            field.data_type()
        )));
    }
    if !train {
        return Err(Error::not_supported(
            "PLAID index cannot be created empty: its centroids are trained on the column"
                .to_string(),
        ));
    }

    progress.stage_start("train_codec", None, "tokens").await?;
    let num_rows = match &fragment_ids {
        Some(ids) => dataset.count_rows_in_fragments(ids).await?,
        None => dataset.count_rows(None).await?,
    };
    let tokens_per_row =
        estimate_multivector_vectors_per_row(dataset, column, num_rows, fragment_ids.as_deref())
            .await?;
    let num_centroids = params.num_centroids_for(num_rows * tokens_per_row);
    let sample = maybe_sample_training_data(
        dataset,
        column,
        num_centroids.saturating_mul(params.sample_rate),
        fragment_ids.as_deref(),
    )
    .await?;
    let sample = filter_finite_training_data(sample)?;
    if sample.len() < 2 {
        return Err(Error::invalid_input(format!(
            "PLAID index needs at least 2 token embeddings to train on, column {column} has {}",
            sample.len()
        )));
    }
    // A tiny column cannot support the default number of centroids.
    let num_centroids = num_centroids.min(sample.len() - 1);
    let (num_bits, kmeans) = (
        params.num_bits,
        kmeans_params(params.max_iters, params.seed),
    );
    let codec =
        spawn_cpu(move || ResidualCodec::train(&sample, num_centroids, num_bits, &kmeans)).await?;
    progress.stage_complete("train_codec").await?;

    progress.stage_start("encode", None, "rows").await?;
    let store = LanceIndexStore::from_dataset_for_new(dataset, &uuid)?;
    let codec = Arc::new(codec);
    let mut writer = PlaidIndexWriter::try_new(codec.clone(), &store).await?;
    let criteria = TrainingCriteria::new(TrainingOrdering::None).with_row_id();
    let data = load_training_data(dataset, column, &criteria, None, true, fragment_ids).await?;
    writer.write_stream(data).await?;
    let files = writer.finish().await?;
    progress.stage_complete("encode").await?;

    Ok(CreatedIndex {
        index_details: details_any(&codec),
        index_version: PLAID_INDEX_VERSION,
        files,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float32Type, Int32Type};
    use arrow_array::{
        ArrayRef, FixedSizeListArray, Float16Array, Float32Array, Int32Array, ListArray,
        RecordBatch, RecordBatchIterator,
    };
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::{DataType, Field, Schema};
    use lance_index::IndexType;
    use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
    use lance_linalg::distance::DistanceType;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rstest::rstest;

    use crate::Dataset;
    use crate::dataset::{WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;

    const DIM: usize = 32;

    /// `element` is the type of the token values: Float32 or Float16.
    fn schema(element: &DataType) -> Arc<Schema> {
        let token = Field::new(
            "item",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", element.clone(), true)),
                DIM as i32,
            ),
            true,
        );
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("vec", DataType::List(Arc::new(token)), true),
        ]))
    }

    /// Normalized tokens drawn around 3 of 16 random topics per document.
    fn tokens(rng: &mut StdRng, topics: &[Vec<f32>], num_tokens: usize) -> Vec<f32> {
        let doc_topics: Vec<usize> = (0..3).map(|_| rng.random_range(0..topics.len())).collect();
        let mut values = Vec::with_capacity(num_tokens * DIM);
        for _ in 0..num_tokens {
            let topic = &topics[doc_topics[rng.random_range(0..3)]];
            let token: Vec<f32> = topic
                .iter()
                .map(|v| v + rng.random_range(-0.6..0.6))
                .collect();
            let norm = token.iter().map(|v| v * v).sum::<f32>().sqrt();
            values.extend(token.iter().map(|v| v / norm));
        }
        values
    }

    struct Data {
        rng: StdRng,
        topics: Vec<Vec<f32>>,
        element: DataType,
    }

    impl Data {
        fn new(element: DataType) -> Self {
            let mut rng = StdRng::seed_from_u64(42);
            let topics = (0..16)
                .map(|_| (0..DIM).map(|_| rng.random_range(-1.0..1.0)).collect())
                .collect();
            Self {
                rng,
                topics,
                element,
            }
        }

        fn schema(&self) -> Arc<Schema> {
            schema(&self.element)
        }

        fn batch(&mut self, ids: std::ops::Range<i32>) -> RecordBatch {
            let lengths: Vec<usize> = ids.clone().map(|_| self.rng.random_range(4..16)).collect();
            let values: Vec<f32> = lengths
                .iter()
                .flat_map(|&n| tokens(&mut self.rng, &self.topics, n))
                .collect();
            let token_field = match self.schema().field(1).data_type() {
                DataType::List(field) => field.clone(),
                _ => unreachable!(),
            };
            let values: ArrayRef = match self.element {
                DataType::Float16 => Arc::new(Float16Array::from_iter_values(
                    values.into_iter().map(half::f16::from_f32),
                )),
                _ => Arc::new(Float32Array::from(values)),
            };
            let fsl = FixedSizeListArray::try_new(
                Arc::new(Field::new("item", self.element.clone(), true)),
                DIM as i32,
                values,
                None,
            )
            .unwrap();
            let lists = ListArray::new(
                token_field,
                OffsetBuffer::from_lengths(lengths),
                Arc::new(fsl),
                None,
            );
            RecordBatch::try_new(
                self.schema(),
                vec![
                    Arc::new(Int32Array::from_iter_values(ids)) as ArrayRef,
                    Arc::new(lists) as ArrayRef,
                ],
            )
            .unwrap()
        }

        fn query(&mut self) -> ListArray {
            let values = tokens(&mut self.rng, &self.topics, 6);
            ListArray::from_iter_primitive::<Float32Type, _, _>(
                values
                    .chunks(DIM)
                    .map(|t| Some(t.iter().map(|v| Some(*v)).collect::<Vec<_>>())),
            )
        }
    }

    async fn dataset(data: &mut Data, params: serde_json::Value) -> Dataset {
        let batch = data.batch(0..300);
        let reader = RecordBatchIterator::new(vec![Ok(batch)], data.schema());
        let write = WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, "memory://", Some(write))
            .await
            .unwrap();
        assert_eq!(dataset.fragments().len(), 3);
        let params = ScalarIndexParams::for_builtin(BuiltinIndexType::Plaid).with_params(&params);
        dataset
            .create_index(&["vec"], IndexType::Plaid, None, &params, true)
            .await
            .unwrap();
        dataset
    }

    /// `(id, _distance)` of a nearest neighbor search, configured by `configure`.
    async fn search(
        dataset: &Dataset,
        query: &ListArray,
        k: usize,
        configure: impl FnOnce(&mut crate::dataset::scanner::Scanner),
    ) -> Vec<(i32, f32)> {
        let mut scan = dataset.scan();
        scan.nearest("vec", query, k)
            .unwrap()
            .project(&["id"])
            .unwrap();
        configure(&mut scan);
        let batch = scan.try_into_batch().await.unwrap();
        let ids = batch["id"].as_primitive::<Int32Type>().values().to_vec();
        let distances = batch["_distance"]
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        ids.into_iter().zip(distances).collect()
    }

    fn overlap(a: &[(i32, f32)], b: &[(i32, f32)]) -> usize {
        a.iter()
            .filter(|(id, _)| b.iter().any(|(other, _)| other == id))
            .count()
    }

    #[rstest]
    #[case::float32(DataType::Float32)]
    #[case::float16(DataType::Float16)]
    #[tokio::test]
    async fn test_plaid_search_matches_flat_search(#[case] element: DataType) {
        let mut data = Data::new(element);
        let dataset = dataset(
            &mut data,
            serde_json::json!({"num_bits": 8, "num_centroids": 32, "seed": 1}),
        )
        .await;
        let stats: serde_json::Value =
            serde_json::from_str(&dataset.index_statistics("vec_idx").await.unwrap()).unwrap();
        assert_eq!(stats["index_type"], "Plaid", "{stats}");

        let (mut exhaustive_hits, mut default_hits) = (0, 0);
        for _ in 0..10 {
            let query = data.query();
            let flat = search(&dataset, &query, 10, |s| {
                s.use_index(false).distance_metric(DistanceType::Cosine);
            })
            .await;
            let exhaustive = search(&dataset, &query, 10, |s| {
                s.nprobes(32).ef(4 * 300);
            })
            .await;
            let default = search(&dataset, &query, 10, |_| {}).await;
            // 8-bit residuals: the index distance is the flat MaxSim distance.
            for (id, distance) in &exhaustive {
                if let Some((_, flat_distance)) = flat.iter().find(|(other, _)| other == id) {
                    assert!(
                        (distance - flat_distance).abs() < 0.05,
                        "{distance} vs {flat_distance}"
                    );
                }
            }
            exhaustive_hits += overlap(&exhaustive, &flat);
            default_hits += overlap(&default, &flat);
        }
        assert!(
            exhaustive_hits >= 98,
            "exhaustive recall@10 {exhaustive_hits}/100"
        );
        assert!(default_hits >= 70, "default recall@10 {default_hits}/100");
    }

    #[rstest]
    #[case::float32(DataType::Float32)]
    #[case::float16(DataType::Float16)]
    #[tokio::test]
    async fn test_plaid_search_with_filter_refine_deletion_and_append(#[case] element: DataType) {
        let mut data = Data::new(element);
        let mut dataset = dataset(
            &mut data,
            serde_json::json!({"num_bits": 4, "num_centroids": 32, "seed": 1}),
        )
        .await;
        let query = data.query();

        let filtered = search(&dataset, &query, 10, |s| {
            s.filter("id % 2 = 0").unwrap().prefilter(true);
        })
        .await;
        assert_eq!(filtered.len(), 10);
        assert!(filtered.iter().all(|(id, _)| id % 2 == 0), "{filtered:?}");

        // Refined distances are computed on the original vectors.
        let refined = search(&dataset, &query, 5, |s| {
            s.refine(4);
        })
        .await;
        let flat = search(&dataset, &query, 300, |s| {
            s.use_index(false).distance_metric(DistanceType::Cosine);
        })
        .await;
        for (id, distance) in &refined {
            let (_, flat_distance) = flat.iter().find(|(other, _)| other == id).unwrap();
            assert!(
                (distance - flat_distance).abs() < 1e-4,
                "{distance} vs {flat_distance}"
            );
        }

        let best = refined[0].0;
        dataset.delete(&format!("id = {best}")).await.unwrap();
        let after_delete = search(&dataset, &query, 10, |_| {}).await;
        assert!(after_delete.iter().all(|(id, _)| *id != best));

        // Rows appended after the index was built are searched exhaustively.
        let appended = data.batch(300..310);
        let new_doc = appended["vec"].as_list::<i32>().value(3);
        // The query is always float32, whatever the column type.
        let new_query = ListArray::from_iter_primitive::<Float32Type, _, _>(
            new_doc.as_fixed_size_list().iter().map(|t| {
                let token = arrow_cast::cast(&t.unwrap(), &DataType::Float32).unwrap();
                Some(
                    token
                        .as_primitive::<Float32Type>()
                        .values()
                        .iter()
                        .map(|v| Some(*v))
                        .collect::<Vec<_>>(),
                )
            }),
        );
        let reader = RecordBatchIterator::new(vec![Ok(appended)], data.schema());
        let append = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, Arc::new(dataset), Some(append))
            .await
            .unwrap();
        let hits = search(&dataset, &new_query, 3, |_| {}).await;
        assert_eq!(hits[0].0, 303, "{hits:?}");
    }
}
