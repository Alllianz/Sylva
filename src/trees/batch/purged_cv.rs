/// Partición de índices para validación cruzada depurada (Purged Split)
#[derive(Clone, Debug)]
pub struct PurgedCvSplit {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
}

/// Generador de particiones Purged & Embargoed Cross-Validation
pub struct PurgedCrossValidator {
    pub n_splits: usize,
    pub embargo_pct: f32, // Porcentaje de embargo tras el bloque de prueba (ej. 0.01 = 1%)
}

impl PurgedCrossValidator {
    pub fn new(n_splits: usize, embargo_pct: f32) -> Self {
        Self {
            n_splits: n_splits.max(2),
            embargo_pct: embargo_pct.clamp(0.0, 0.2),
        }
    }

    /// Genera los folds purgados respetando el horizonte forward y el embargo temporal
    pub fn split(&self, n_samples: usize, target_horizon: usize) -> Vec<PurgedCvSplit> {
        let mut splits = Vec::with_capacity(self.n_splits);
        let fold_size = n_samples / self.n_splits;
        let embargo_size = ((n_samples as f32) * self.embargo_pct) as usize;

        for fold_idx in 0..self.n_splits {
            let test_start = fold_idx * fold_size;
            let test_end = if fold_idx == self.n_splits - 1 {
                n_samples
            } else {
                (fold_idx + 1) * fold_size
            };

            let test_indices: Vec<usize> = (test_start..test_end).collect();

            // Purging: Eliminar de entrenamiento las muestras cuya etiqueta forward se solape con el inicio de test
            let purge_start = test_start.saturating_sub(target_horizon);

            // Embargoing: Eliminar de entrenamiento las muestras inmediatamente posteriores al final de test
            let embargo_end = (test_end + embargo_size).min(n_samples);

            let mut train_indices = Vec::with_capacity(n_samples - test_indices.len());

            for i in 0..n_samples {
                // Si la muestra está antes del bloque de test pero su etiqueta toca test, PURGE
                if i >= purge_start && i < test_start {
                    continue;
                }
                // Si la muestra está dentro de test, TEST
                if i >= test_start && i < test_end {
                    continue;
                }
                // Si la muestra está en la zona de EMBARGO posterior a test, EMBARGO
                if i >= test_end && i < embargo_end {
                    continue;
                }

                train_indices.push(i);
            }

            splits.push(PurgedCvSplit {
                train_indices,
                test_indices,
            });
        }

        splits
    }
}

/// Métricas de validación cruzada fuera de muestra
#[derive(Clone, Debug, Default)]
pub struct CrossValidationMetrics {
    pub mean_oos_mse: f32,
    pub mean_oos_hit_ratio: f32, // Mean Directional Accuracy (MDA)
    pub mean_oos_rank_ic: f32,   // Spearman Rank Correlation
    pub fold_metrics: Vec<(f32, f32, f32)>, // (MSE, MDA, RankIC)
}

/// Evalúa predicciones vs targets calculando MSE, MDA y Rank IC
pub fn evaluate_predictions(predictions: &[f32], targets: &[f32]) -> (f32, f32, f32) {
    let n = predictions.len();
    if n == 0 || n != targets.len() {
        return (0.0, 0.5, 0.0);
    }

    // 1. Mean Squared Error (MSE)
    let mut sum_sq_err = 0.0f32;
    let mut correct_dir = 0;
    let mut dir_count = 0;

    for i in 0..n {
        let diff = predictions[i] - targets[i];
        sum_sq_err += diff * diff;

        if predictions[i].abs() > 1e-7 && targets[i].abs() > 1e-7 {
            if (predictions[i] > 0.0 && targets[i] > 0.0) || (predictions[i] < 0.0 && targets[i] < 0.0) {
                correct_dir += 1;
            }
            dir_count += 1;
        }
    }

    let mse = sum_sq_err / (n as f32);
    let mda = if dir_count > 0 {
        correct_dir as f32 / dir_count as f32
    } else {
        0.5
    };

    // 2. Spearman Rank Correlation (Rank IC)
    let rank_ic = compute_rank_ic(predictions, targets);

    (mse, mda, rank_ic)
}

fn compute_rank_ic(preds: &[f32], targets: &[f32]) -> f32 {
    let n = preds.len();
    if n < 3 {
        return 0.0;
    }

    let ranks_x = get_ranks(preds);
    let ranks_y = get_ranks(targets);

    let mean_x = (n as f32 + 1.0) * 0.5;
    let mean_y = mean_x;

    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;

    for i in 0..n {
        let dx = ranks_x[i] - mean_x;
        let dy = ranks_y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom > 1e-9 {
        cov / denom
    } else {
        0.0
    }
}

fn get_ranks(vals: &[f32]) -> Vec<f32> {
    let n = vals.len();
    let mut indexed: Vec<(usize, f32)> = vals.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && (indexed[j + 1].1 - indexed[j].1).abs() < 1e-9 {
            j += 1;
        }
        let avg_rank = (i + j + 2) as f32 * 0.5;
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }
    ranks
}
