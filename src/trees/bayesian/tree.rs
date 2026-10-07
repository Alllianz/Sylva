use crate::features::TOTAL_FEATURES;
use crate::trees::bayesian::node::BayesianTreeNode;
use crate::trees::bayesian::types::BayesianTreeConfig;

/// Árbol de decisión individual con inferencia bayesiana por hoja
#[derive(Clone, Debug)]
pub struct BayesianDecisionTree {
    pub root: BayesianTreeNode,
    pub tree_idx: usize,
    pub active_features: Vec<usize>,
}

impl BayesianDecisionTree {
    pub fn new(tree_idx: usize, active_features: Vec<usize>, cfg: &BayesianTreeConfig) -> Self {
        Self {
            root: BayesianTreeNode::new_leaf(cfg),
            tree_idx,
            active_features,
        }
    }

    #[inline(always)]
    pub fn predict(&self, features: &[f32; TOTAL_FEATURES]) -> (f32, f32) {
        self.root.predict(features)
    }

    #[inline(always)]
    pub fn sample_posterior(&self, features: &[f32; TOTAL_FEATURES], rng_state: &mut u64) -> f32 {
        self.root.sample_posterior(features, rng_state)
    }

    pub fn update(
        &mut self,
        features: &[f32; TOTAL_FEATURES],
        residual: f32,
        cfg: &BayesianTreeConfig,
    ) -> bool {
        self.root.update(features, residual, cfg, 0, &self.active_features)
    }

    pub fn compute_feature_importance(&self, importances: &mut [f32; TOTAL_FEATURES]) {
        self.root.compute_feature_importance(importances);
    }

    pub fn depth(&self) -> usize {
        self.root.depth()
    }

    pub fn n_leaves(&self) -> usize {
        self.root.n_leaves()
    }
}
