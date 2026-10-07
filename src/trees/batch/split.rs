use crate::features::TOTAL_FEATURES;
use crate::trees::dataset::DatasetSample;

/// Representa el candidato óptimo de división para un nodo
#[derive(Clone, Debug)]
pub struct SplitCandidate {
    pub feature_idx: usize,
    pub threshold: f32,
    pub gain: f32,
    pub left_val: f32,
    pub right_val: f32,
    pub left_indices: Vec<usize>,
    pub right_indices: Vec<usize>,
}

/// Calcula el valor óptimo de una hoja con regularización L1 (alpha) y L2 (lambda)
#[inline(always)]
pub fn calculate_leaf_weight(sum_g: f32, sum_h: f32, l1_reg: f32, l2_reg: f32) -> f32 {
    if sum_h + l2_reg <= 1e-7 {
        return 0.0;
    }

    if l1_reg > 0.0 {
        if sum_g > l1_reg {
            -(sum_g - l1_reg) / (sum_h + l2_reg)
        } else if sum_g < -l1_reg {
            -(sum_g + l1_reg) / (sum_h + l2_reg)
        } else {
            0.0
        }
    } else {
        -sum_g / (sum_h + l2_reg)
    }
}

/// Calcula la ganancia de puntuación de un nodo individual
#[inline(always)]
pub fn calculate_node_score(sum_g: f32, sum_h: f32, l1_reg: f32, l2_reg: f32) -> f32 {
    if sum_h + l2_reg <= 1e-7 {
        return 0.0;
    }

    let g_adj = if l1_reg > 0.0 {
        if sum_g > l1_reg {
            sum_g - l1_reg
        } else if sum_g < -l1_reg {
            sum_g + l1_reg
        } else {
            0.0
        }
    } else {
        sum_g
    };

    (g_adj * g_adj) / (sum_h + l2_reg)
}

/// Encuentra la mejor división en un subconjunto de muestras evaluando las características permitidas
pub fn find_best_split(
    samples: &[DatasetSample],
    sample_indices: &[usize],
    gradients: &[f32],
    hessians: &[f32],
    candidate_features: &[usize],
    min_samples_leaf: usize,
    l1_reg: f32,
    l2_reg: f32,
    gamma: f32,
) -> Option<SplitCandidate> {
    if sample_indices.len() < min_samples_leaf * 2 {
        return None;
    }

    // Sumar gradientes totales del nodo actual
    let mut total_g = 0.0f32;
    let mut total_h = 0.0f32;
    for &idx in sample_indices {
        total_g += gradients[idx];
        total_h += hessians[idx];
    }

    let root_score = calculate_node_score(total_g, total_h, l1_reg, l2_reg);
    let mut best_gain = 0.0f32;
    let mut best_candidate: Option<SplitCandidate> = None;

    for &feat_idx in candidate_features {
        if feat_idx >= TOTAL_FEATURES {
            continue;
        }

        // Extraer y ordenar pares (valor_feature, sample_index)
        let mut feat_values: Vec<(f32, usize)> = sample_indices
            .iter()
            .map(|&idx| (samples[idx].features[feat_idx], idx))
            .collect();

        feat_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut left_g = 0.0f32;
        let mut left_h = 0.0f32;
        let n_items = feat_values.len();

        for i in 0..(n_items - 1) {
            let (val_curr, idx_curr) = feat_values[i];
            left_g += gradients[idx_curr];
            left_h += hessians[idx_curr];

            let right_count = n_items - (i + 1);
            let left_count = i + 1;

            if left_count < min_samples_leaf || right_count < min_samples_leaf {
                continue;
            }

            let (val_next, _) = feat_values[i + 1];
            // No dividir si los valores consecutivos son idénticos
            if (val_next - val_curr).abs() < 1e-7 {
                continue;
            }

            let right_g = total_g - left_g;
            let right_h = total_h - left_h;

            let left_score = calculate_node_score(left_g, left_h, l1_reg, l2_reg);
            let right_score = calculate_node_score(right_g, right_h, l1_reg, l2_reg);

            // Ganancia de Taylor de segundo orden: 0.5 * (Score_L + Score_R - Score_Root) - gamma
            let gain = 0.5 * (left_score + right_score - root_score) - gamma;

            if gain > best_gain {
                best_gain = gain;
                let threshold = (val_curr + val_next) * 0.5;

                let left_indices: Vec<usize> = feat_values[..=i].iter().map(|&(_, idx)| idx).collect();
                let right_indices: Vec<usize> = feat_values[(i + 1)..].iter().map(|&(_, idx)| idx).collect();

                let left_val = calculate_leaf_weight(left_g, left_h, l1_reg, l2_reg);
                let right_val = calculate_leaf_weight(right_g, right_h, l1_reg, l2_reg);

                best_candidate = Some(SplitCandidate {
                    feature_idx: feat_idx,
                    threshold,
                    gain,
                    left_val,
                    right_val,
                    left_indices,
                    right_indices,
                });
            }
        }
    }

    best_candidate
}
