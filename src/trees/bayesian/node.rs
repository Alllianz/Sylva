use crate::features::TOTAL_FEATURES;
use crate::trees::bayesian::leaf::BayesianLeaf;
use crate::trees::bayesian::types::BayesianTreeConfig;

/// Nodo de un árbol de decisión bayesiano online
#[derive(Clone, Debug)]
pub enum BayesianTreeNode {
    Leaf {
        leaf: BayesianLeaf,
    },
    Internal {
        feature_idx: usize,
        threshold: f32,
        gain: f32,
        left: Box<BayesianTreeNode>,
        right: Box<BayesianTreeNode>,
    },
}

impl BayesianTreeNode {
    /// Crea una nueva hoja bayesiana
    pub fn new_leaf(cfg: &BayesianTreeConfig) -> Self {
        BayesianTreeNode::Leaf {
            leaf: BayesianLeaf::new(
                cfg.prior_mean,
                cfg.prior_precision,
                cfg.prior_shape,
                cfg.prior_scale,
            ),
        }
    }

    /// Inferencia bayesiana: retorna (media_esperada, varianza_epistemica)
    pub fn predict(&self, features: &[f32; TOTAL_FEATURES]) -> (f32, f32) {
        match self {
            BayesianTreeNode::Leaf { leaf } => leaf.predict(),
            BayesianTreeNode::Internal {
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

    /// Muestreo de Thompson a través del árbol
    pub fn sample_posterior(&self, features: &[f32; TOTAL_FEATURES], rng_state: &mut u64) -> f32 {
        match self {
            BayesianTreeNode::Leaf { leaf } => leaf.sample_posterior(rng_state),
            BayesianTreeNode::Internal {
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
                    left.sample_posterior(features, rng_state)
                } else {
                    right.sample_posterior(features, rng_state)
                }
            }
        }
    }

    /// Actualiza la hoja con una nueva muestra
    pub fn update(
        &mut self,
        features: &[f32; TOTAL_FEATURES],
        residual: f32,
        cfg: &BayesianTreeConfig,
        depth: usize,
        active_features: &[usize],
    ) -> bool {
        match self {
            BayesianTreeNode::Internal {
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
                    left.update(features, residual, cfg, depth + 1, active_features)
                } else {
                    right.update(features, residual, cfg, depth + 1, active_features)
                }
            }
            BayesianTreeNode::Leaf { leaf } => {
                // Actualizar distribución posterior conjugada NIG
                leaf.update(residual, cfg.decay_factor);

                // División estocástica bayesiana si hay suficiente evidencia estadística
                if depth < cfg.max_depth && leaf.sample_count >= cfg.min_samples_leaf * 2 && leaf.sample_count % 20 == 0 {
                    let (_, current_var) = leaf.predict();
                    // Si la incertidumbre es alta y hay variabilidad en las características
                    if current_var > 0.001 && !active_features.is_empty() {
                        let feat_idx = active_features[leaf.sample_count % active_features.len()];
                        let threshold = features[feat_idx];

                        let left_leaf = BayesianLeaf::new(
                            leaf.mu,
                            leaf.kappa * 0.5,
                            leaf.alpha * 0.5,
                            leaf.beta * 0.5,
                        );
                        let right_leaf = BayesianLeaf::new(
                            leaf.mu,
                            leaf.kappa * 0.5,
                            leaf.alpha * 0.5,
                            leaf.beta * 0.5,
                        );

                        *self = BayesianTreeNode::Internal {
                            feature_idx: feat_idx,
                            threshold,
                            gain: current_var,
                            left: Box::new(BayesianTreeNode::Leaf { leaf: left_leaf }),
                            right: Box::new(BayesianTreeNode::Leaf { leaf: right_leaf }),
                        };
                        return true;
                    }
                }

                false
            }
        }
    }

    /// Acumula la ganancia de importancia por variable
    pub fn compute_feature_importance(&self, importances: &mut [f32; TOTAL_FEATURES]) {
        match self {
            BayesianTreeNode::Leaf { .. } => {}
            BayesianTreeNode::Internal {
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

    pub fn depth(&self) -> usize {
        match self {
            BayesianTreeNode::Leaf { .. } => 0,
            BayesianTreeNode::Internal { left, right, .. } => 1 + left.depth().max(right.depth()),
        }
    }

    pub fn n_leaves(&self) -> usize {
        match self {
            BayesianTreeNode::Leaf { .. } => 1,
            BayesianTreeNode::Internal { left, right, .. } => left.n_leaves() + right.n_leaves(),
        }
    }
}
