use crate::features::TOTAL_FEATURES;
use crate::trees::online::types::{OnlineGbdtConfig, OnlineLeafStats};

/// Nodo de un árbol de decisión streaming online
#[derive(Clone, Debug)]
pub enum OnlineTreeNode {
    Leaf {
        stats: OnlineLeafStats,
        weight: f32,
    },
    Internal {
        feature_idx: usize,
        threshold: f32,
        gain: f32,
        left: Box<OnlineTreeNode>,
        right: Box<OnlineTreeNode>,
    },
}

impl OnlineTreeNode {
    /// Crea una nueva hoja con estadísticas iniciales vacías
    pub fn new_leaf() -> Self {
        OnlineTreeNode::Leaf {
            stats: OnlineLeafStats::default(),
            weight: 0.0,
        }
    }

    /// Inferencia para una observación de variables
    pub fn predict(&self, features: &[f32; TOTAL_FEATURES]) -> f32 {
        match self {
            OnlineTreeNode::Leaf { weight, .. } => *weight,
            OnlineTreeNode::Internal {
                feature_idx,
                threshold,
                left,
                right,
                ..
            } => {
                let val = if *feature_idx < TOTAL_FEATURES {
                    features[*feature_idx]
                } else {
                    0.0
                };

                if val <= *threshold {
                    left.predict(features)
                } else {
                    right.predict(features)
                }
            }
        }
    }

    /// Actualiza la hoja con una nueva muestra y evalúa si procede dividir usando la cota de Hoeffding
    pub fn update(
        &mut self,
        features: &[f32; TOTAL_FEATURES],
        g: f32,
        h: f32,
        cfg: &OnlineGbdtConfig,
        depth: usize,
        active_features: &[usize],
    ) -> bool {
        match self {
            OnlineTreeNode::Internal {
                feature_idx,
                threshold,
                left,
                right,
                ..
            } => {
                let val = if *feature_idx < TOTAL_FEATURES {
                    features[*feature_idx]
                } else {
                    0.0
                };

                if val <= *threshold {
                    left.update(features, g, h, cfg, depth + 1, active_features)
                } else {
                    right.update(features, g, h, cfg, depth + 1, active_features)
                }
            }
            OnlineTreeNode::Leaf { stats, weight } => {
                // 1. Actualizar estadísticas de la hoja con olvido exponencial
                stats.update(features, g, h, cfg.decay_factor);
                *weight = stats.compute_weight(cfg.l1_reg, cfg.l2_reg);

                // 2. Si ya alcanzó la profundidad máxima o pocas muestras, no intentar dividir
                if depth >= cfg.max_depth || stats.sample_count < cfg.min_samples_split {
                    return false;
                }

                // 3. Evaluar división solo cada `grace_period` muestras
                if stats.samples_since_split_eval < cfg.grace_period {
                    return false;
                }
                stats.samples_since_split_eval = 0;

                // 4. Buscar los dos mejores candidatos de división entre las variables activas
                let root_score = stats.compute_score(cfg.l1_reg, cfg.l2_reg);
                let mut best_gain = 0.0f32;
                let mut best_feat = 0usize;
                let mut best_threshold = 0.0f32;
                let mut best_left_stats = OnlineLeafStats::default();
                let mut best_right_stats = OnlineLeafStats::default();

                let mut second_best_gain = 0.0f32;

                for &feat_idx in active_features {
                    if feat_idx >= TOTAL_FEATURES {
                        continue;
                    }

                    let min_v = stats.feature_min[feat_idx];
                    let max_v = stats.feature_max[feat_idx];
                    let range = max_v - min_v;
                    if range <= 1e-6 {
                        continue;
                    }

                    for bin in 0..8 {
                        let threshold = min_v + range * ((bin as f32 + 1.0) / 9.0);
                        let left_g = stats.split_g_left[feat_idx][bin];
                        let left_h = stats.split_h_left[feat_idx][bin];
                        let right_g = stats.split_g_right[feat_idx][bin];
                        let right_h = stats.split_h_right[feat_idx][bin];

                        if left_h < 1.0 || right_h < 1.0 {
                            continue;
                        }

                        let left_score = compute_score_static(left_g, left_h, cfg.l1_reg, cfg.l2_reg);
                        let right_score = compute_score_static(right_g, right_h, cfg.l1_reg, cfg.l2_reg);

                        // Ganancia de segundo orden según formulación Taylor
                        let gain = 0.5 * (left_score + right_score - root_score) - cfg.gamma;

                        if gain > best_gain {
                            second_best_gain = best_gain;
                            best_gain = gain;
                            best_feat = feat_idx;
                            best_threshold = threshold;

                            // Inicializar estadísticas de los hijos
                            let mut ls = OnlineLeafStats::default();
                            ls.sum_g = left_g;
                            ls.sum_h = left_h;
                            ls.sample_count = (stats.sample_count / 2).max(1);

                            let mut rs = OnlineLeafStats::default();
                            rs.sum_g = right_g;
                            rs.sum_h = right_h;
                            rs.sample_count = (stats.sample_count / 2).max(1);

                            best_left_stats = ls;
                            best_right_stats = rs;
                        } else if gain > second_best_gain {
                            second_best_gain = gain;
                        }
                    }
                }

                if best_gain <= 0.0 {
                    return false;
                }

                // 5. Cota de Hoeffding: \epsilon = \sqrt{ \frac{R^2 \ln(1/\delta)}{2 n} }
                // R = rango máximo de ganancia heurística estimada
                let n = stats.sample_count as f32;
                let delta = cfg.split_confidence.clamp(1e-5, 0.5);
                let range_r = (root_score.abs() + 1.0).min(100.0);
                let epsilon = ((range_r * range_r * (1.0f32 / delta).ln()) / (2.0 * n)).sqrt();

                // Criterio de división de Hoeffding: (G1 - G2 > \epsilon) o (\epsilon < \tau)
                let delta_gain = best_gain - second_best_gain;
                let should_split = delta_gain > epsilon || epsilon < cfg.tie_threshold;

                if should_split && best_gain > cfg.gamma {
                    let left_weight = best_left_stats.compute_weight(cfg.l1_reg, cfg.l2_reg);
                    let right_weight = best_right_stats.compute_weight(cfg.l1_reg, cfg.l2_reg);

                    *self = OnlineTreeNode::Internal {
                        feature_idx: best_feat,
                        threshold: best_threshold,
                        gain: best_gain,
                        left: Box::new(OnlineTreeNode::Leaf {
                            stats: best_left_stats,
                            weight: left_weight,
                        }),
                        right: Box::new(OnlineTreeNode::Leaf {
                            stats: best_right_stats,
                            weight: right_weight,
                        }),
                    };
                    return true;
                }

                false
            }
        }
    }

    /// Acumula la importancia de características basada en ganancia
    pub fn compute_feature_importance(&self, importances: &mut [f32; TOTAL_FEATURES]) {
        match self {
            OnlineTreeNode::Leaf { .. } => {}
            OnlineTreeNode::Internal {
                feature_idx,
                gain,
                left,
                right,
                ..
            } => {
                if *feature_idx < TOTAL_FEATURES {
                    importances[*feature_idx] += *gain;
                }
                left.compute_feature_importance(importances);
                right.compute_feature_importance(importances);
            }
        }
    }

    /// Profundidad del subárbol
    pub fn depth(&self) -> usize {
        match self {
            OnlineTreeNode::Leaf { .. } => 0,
            OnlineTreeNode::Internal { left, right, .. } => 1 + left.depth().max(right.depth()),
        }
    }

    /// Número total de hojas en el subárbol
    pub fn n_leaves(&self) -> usize {
        match self {
            OnlineTreeNode::Leaf { .. } => 1,
            OnlineTreeNode::Internal { left, right, .. } => left.n_leaves() + right.n_leaves(),
        }
    }
}

#[inline(always)]
fn compute_score_static(sum_g: f32, sum_h: f32, l1_reg: f32, l2_reg: f32) -> f32 {
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
