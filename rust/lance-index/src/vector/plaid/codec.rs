// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Residual codec of a PLAID index.
//!
//! Every token embedding is stored as the id of its nearest centroid plus its
//! residual (embedding minus centroid) quantized to `num_bits` bits per
//! dimension. The quantization buckets are shared by every dimension: their
//! boundaries are quantiles of the residual values of a held-out sample, and a
//! bucket decodes to the residual quantile at its middle.
//!
//! Adapted from lightonai/next-plaid (Apache-2.0).

use arrow_array::{Array, FixedSizeListArray, Float32Array, cast::AsArray, types::Float32Type};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;

use crate::vector::kmeans::{KMeans, KMeansParams};

/// Rows of tokens assigned to centroids per matrix multiplication.
///
/// Bounds the `[rows, num_centroids]` score matrix to 32 MiB at 32k centroids.
const ASSIGN_CHUNK_ROWS: usize = 256;

/// Maximum number of held-out residuals used to fit the quantization buckets.
const MAX_HELDOUT_TOKENS: usize = 50_000;

/// Quantizes token embeddings as a centroid id and bucketized residuals.
#[derive(Debug, Clone)]
pub struct ResidualCodec {
    num_bits: u8,
    /// L2-normalized centroids, `[num_centroids, dimension]`.
    centroids: Array2<f32>,
    /// Upper bounds (exclusive) of the first `2^num_bits - 1` buckets, ascending.
    bucket_cutoffs: Vec<f32>,
    /// Residual value each of the `2^num_bits` buckets decodes to.
    bucket_weights: Vec<f32>,
    /// For each byte value, the decoded residuals of the `8 / num_bits`
    /// dimensions packed in it, most significant bits first.
    byte_lut: Vec<f32>,
}

impl ResidualCodec {
    pub fn try_new(
        num_bits: u8,
        centroids: Array2<f32>,
        bucket_cutoffs: Vec<f32>,
        bucket_weights: Vec<f32>,
    ) -> Result<Self> {
        validate_num_bits(num_bits)?;
        let num_buckets = 1usize << num_bits;
        if bucket_cutoffs.len() != num_buckets - 1 || bucket_weights.len() != num_buckets {
            return Err(Error::invalid_input(format!(
                "PLAID codec with num_bits={num_bits} needs {} bucket cutoffs and {num_buckets} bucket weights, got {} and {}",
                num_buckets - 1,
                bucket_cutoffs.len(),
                bucket_weights.len()
            )));
        }
        let dimension = centroids.ncols();
        if dimension == 0 || (dimension * num_bits as usize) % 8 != 0 {
            return Err(Error::invalid_input(format!(
                "PLAID index needs dimension * num_bits to be a positive multiple of 8, got dimension={dimension} and num_bits={num_bits}"
            )));
        }
        let values_per_byte = 8 / num_bits as usize;
        let mask = (num_buckets - 1) as u8;
        let mut byte_lut = Vec::with_capacity(256 * values_per_byte);
        for byte in 0..=255u8 {
            for slot in 0..values_per_byte {
                let shift = 8 - num_bits as usize * (slot + 1);
                byte_lut.push(bucket_weights[((byte >> shift) & mask) as usize]);
            }
        }
        Ok(Self {
            num_bits,
            centroids,
            bucket_cutoffs,
            bucket_weights,
            byte_lut,
        })
    }

    /// Train centroids with k-means on `sample` and fit the residual buckets.
    ///
    /// `sample` holds one token embedding per row; it must have more rows than
    /// `num_centroids`.
    pub fn train(
        sample: &FixedSizeListArray,
        num_centroids: usize,
        num_bits: u8,
        kmeans_params: &KMeansParams,
    ) -> Result<Self> {
        validate_num_bits(num_bits)?;
        let dimension = sample.value_length() as usize;
        let sample = normalize_rows(fsl_to_array2(sample)?);
        let num_tokens = sample.nrows();
        if num_tokens <= num_centroids {
            return Err(Error::invalid_input(format!(
                "PLAID training needs more sample tokens than centroids, got {num_tokens} tokens for num_centroids={num_centroids}"
            )));
        }

        let training = FixedSizeListArray::try_new_from_values(
            Float32Array::from_iter_values(sample.iter().copied()),
            dimension as i32,
        )?;
        let kmeans = KMeans::new_with_params(&training, num_centroids, kmeans_params)?;
        let centroids = kmeans
            .centroids
            .as_primitive::<Float32Type>()
            .values()
            .to_vec();
        let centroids = normalize_rows(
            Array2::from_shape_vec((num_centroids, dimension), centroids)
                .map_err(|e| Error::index(format!("PLAID centroids have an invalid shape: {e}")))?,
        );

        // Fit the buckets on the tail of the sample, which k-means sampled last.
        let heldout_rows = MAX_HELDOUT_TOKENS.min(num_tokens);
        let heldout = sample.slice(ndarray::s![num_tokens - heldout_rows.., ..]);
        let codes = assign(centroids.view(), heldout);
        let mut residuals = Vec::with_capacity(heldout_rows * dimension);
        for (token, &code) in heldout.axis_iter(Axis(0)).zip(&codes) {
            let centroid = centroids.row(code as usize);
            residuals.extend(token.iter().zip(centroid).map(|(t, c)| t - c));
        }
        residuals.retain(|v| v.is_finite());
        residuals.sort_unstable_by(f32::total_cmp);
        let num_buckets = 1usize << num_bits;
        let bucket_cutoffs = (1..num_buckets)
            .map(|i| quantile(&residuals, i as f64 / num_buckets as f64))
            .collect();
        let bucket_weights = (0..num_buckets)
            .map(|i| quantile(&residuals, (i as f64 + 0.5) / num_buckets as f64))
            .collect();
        Self::try_new(num_bits, centroids, bucket_cutoffs, bucket_weights)
    }

    pub fn num_bits(&self) -> u8 {
        self.num_bits
    }

    pub fn num_centroids(&self) -> usize {
        self.centroids.nrows()
    }

    pub fn dimension(&self) -> usize {
        self.centroids.ncols()
    }

    pub fn centroids(&self) -> ArrayView2<'_, f32> {
        self.centroids.view()
    }

    pub fn bucket_cutoffs(&self) -> &[f32] {
        &self.bucket_cutoffs
    }

    pub fn bucket_weights(&self) -> &[f32] {
        &self.bucket_weights
    }

    /// Bytes of one packed residual.
    pub fn packed_dim(&self) -> usize {
        self.dimension() * self.num_bits as usize / 8
    }

    /// Id of the nearest centroid (by inner product) of every row of `tokens`.
    pub fn assign(&self, tokens: ArrayView2<f32>) -> Vec<u32> {
        assign(self.centroids.view(), tokens)
    }

    /// Append the packed quantized residual of every row of `tokens` to `out`.
    pub fn encode_residuals(&self, tokens: ArrayView2<f32>, codes: &[u32], out: &mut Vec<u8>) {
        let packed_dim = self.packed_dim();
        let values_per_byte = 8 / self.num_bits as usize;
        let start = out.len();
        out.resize(start + tokens.nrows() * packed_dim, 0);
        out[start..]
            .par_chunks_mut(packed_dim)
            .enumerate()
            .for_each(|(row, packed)| {
                let token = tokens.row(row);
                let centroid = self.centroids.row(codes[row] as usize);
                for (dim, (t, c)) in token.iter().zip(centroid).enumerate() {
                    let residual = t - c;
                    let bucket = self.bucket_cutoffs.partition_point(|&cut| cut < residual) as u8;
                    let slot = dim % values_per_byte;
                    packed[dim / values_per_byte] |=
                        bucket << (8 - self.num_bits as usize * (slot + 1));
                }
            });
    }

    /// Reconstruct one L2-normalized token embedding into `out`.
    pub fn decode_into(&self, code: u32, packed: &[u8], out: &mut [f32]) {
        let values_per_byte = 8 / self.num_bits as usize;
        let centroid = self.centroids.row(code as usize);
        for (byte_idx, &byte) in packed.iter().enumerate() {
            let lut = &self.byte_lut[byte as usize * values_per_byte..][..values_per_byte];
            let dims = byte_idx * values_per_byte..(byte_idx + 1) * values_per_byte;
            for ((o, c), w) in out[dims.clone()]
                .iter_mut()
                .zip(centroid.slice(ndarray::s![dims]))
                .zip(lut)
            {
                *o = c + w;
            }
        }
        let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        out.iter_mut().for_each(|v| *v /= norm);
    }

    /// Inner product of every centroid with every query token, `[num_centroids, num_query_tokens]`.
    pub fn centroid_scores(&self, query: ArrayView2<f32>) -> Array2<f32> {
        self.centroids.dot(&query.t())
    }
}

fn validate_num_bits(num_bits: u8) -> Result<()> {
    if matches!(num_bits, 1 | 2 | 4 | 8) {
        Ok(())
    } else {
        Err(Error::invalid_input(format!(
            "PLAID num_bits must be 1, 2, 4 or 8, got {num_bits}"
        )))
    }
}

fn assign(centroids: ArrayView2<f32>, tokens: ArrayView2<f32>) -> Vec<u32> {
    let chunks: Vec<_> = tokens
        .axis_chunks_iter(Axis(0), ASSIGN_CHUNK_ROWS)
        .collect();
    chunks
        .into_par_iter()
        .flat_map_iter(|chunk| {
            let scores = chunk.dot(&centroids.t());
            scores
                .axis_iter(Axis(0))
                .map(|row| argmax(row))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn argmax(row: ArrayView1<f32>) -> u32 {
    let mut best = 0;
    let mut best_score = f32::NEG_INFINITY;
    for (i, &score) in row.iter().enumerate() {
        if score > best_score {
            best_score = score;
            best = i;
        }
    }
    best as u32
}

/// Linear-interpolated quantile of sorted `values`, as numpy's default method.
fn quantile(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = (pos - lo as f64) as f32;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

/// Copy a `FixedSizeList<Float32>` into a `[len, dimension]` matrix.
pub(crate) fn fsl_to_array2(array: &FixedSizeListArray) -> Result<Array2<f32>> {
    let dimension = array.value_length() as usize;
    let values = array
        .values()
        .as_primitive_opt::<Float32Type>()
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "PLAID index only supports float32 vectors, got {}",
                array.value_type()
            ))
        })?;
    let start = array.offset() * dimension;
    let values = &values.values()[start..start + array.len() * dimension];
    Array2::from_shape_vec((array.len(), dimension), values.to_vec())
        .map_err(|e| Error::index(format!("invalid vector shape: {e}")))
}

pub(crate) fn normalize_rows(mut matrix: Array2<f32>) -> Array2<f32> {
    for mut row in matrix.axis_iter_mut(Axis(0)) {
        let norm = row.dot(&row).sqrt().max(1e-12);
        row /= norm;
    }
    matrix
}

/// The k-means parameters PLAID trains its centroids with.
pub fn kmeans_params(max_iters: u32, seed: Option<u64>) -> KMeansParams {
    KMeansParams {
        max_iters,
        distance_type: DistanceType::L2,
        seed,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rstest::rstest;

    fn random_tokens(rows: usize, dimension: usize, seed: u64) -> FixedSizeListArray {
        let mut rng = StdRng::seed_from_u64(seed);
        let values: Vec<f32> = (0..rows * dimension)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), dimension as i32)
            .unwrap()
    }

    #[rstest]
    #[case::one_bit(1)]
    #[case::two_bits(2)]
    #[case::four_bits(4)]
    #[case::eight_bits(8)]
    fn test_round_trip_error_shrinks_with_bits(#[case] num_bits: u8) {
        let dimension = 16;
        let sample = random_tokens(2_000, dimension, 42);
        let codec =
            ResidualCodec::train(&sample, 16, num_bits, &kmeans_params(10, Some(1))).unwrap();
        let tokens = normalize_rows(fsl_to_array2(&random_tokens(200, dimension, 7)).unwrap());
        let codes = codec.assign(tokens.view());
        let mut packed = Vec::new();
        codec.encode_residuals(tokens.view(), &codes, &mut packed);
        assert_eq!(packed.len(), 200 * codec.packed_dim());

        let mut decoded = vec![0.0; dimension];
        let mut mean_cos = 0.0;
        for (i, token) in tokens.axis_iter(Axis(0)).enumerate() {
            let row = &packed[i * codec.packed_dim()..][..codec.packed_dim()];
            codec.decode_into(codes[i], row, &mut decoded);
            mean_cos += token.iter().zip(&decoded).map(|(a, b)| a * b).sum::<f32>();
        }
        mean_cos /= 200.0;
        let min_cos = match num_bits {
            1 => 0.75,
            2 => 0.85,
            4 => 0.95,
            _ => 0.99,
        };
        assert!(
            mean_cos > min_cos,
            "num_bits={num_bits}: mean cosine {mean_cos} <= {min_cos}"
        );
    }

    #[test]
    fn test_packing_layout() {
        // Two dimensions per byte with 4 bits: first dimension in the high nibble.
        let centroids = Array2::zeros((1, 2));
        let cutoffs: Vec<f32> = (1..16).map(|i| i as f32).collect();
        let weights: Vec<f32> = (0..16).map(|i| i as f32 + 0.5).collect();
        let codec = ResidualCodec::try_new(4, centroids, cutoffs, weights).unwrap();
        let tokens = Array2::from_shape_vec((1, 2), vec![3.5, 12.5]).unwrap();
        let mut packed = Vec::new();
        codec.encode_residuals(tokens.view(), &[0], &mut packed);
        assert_eq!(packed, vec![0x3C]);
    }

    #[test]
    fn test_invalid_num_bits() {
        let err = ResidualCodec::try_new(3, Array2::zeros((1, 8)), vec![], vec![]).unwrap_err();
        assert!(err.to_string().contains("num_bits must be 1, 2, 4 or 8"));
    }
}
