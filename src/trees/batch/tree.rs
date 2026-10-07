use crate::features::TOTAL_FEATURES;

/// Nodo de un árbol de decisión de regresión
#[derive(Clone, Debug)]
pub enum TreeNode {
    Leaf {
        value: f32,
        weight: f32,
        n_samples: usize,
    },
    Internal {
        feature_idx: usize,
        threshold: f32,
        gain: f32,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    pub fn predict(&self, features: &[f32; TOTAL_FEATURES]) -> f32 {
        match self {
            TreeNode::Leaf { value, .. } => *value,
            TreeNode::Internal {
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

    pub fn compute_feature_importance(&self, importances: &mut [f32; TOTAL_FEATURES]) {
        match self {
            TreeNode::Leaf { .. } => {}
            TreeNode::Internal {
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
            TreeNode::Leaf { .. } => 0,
            TreeNode::Internal { left, right, .. } => 1 + left.depth().max(right.depth()),
        }
    }

    pub fn n_leaves(&self) -> usize {
        match self {
            TreeNode::Leaf { .. } => 1,
            TreeNode::Internal { left, right, .. } => left.n_leaves() + right.n_leaves(),
        }
    }
}

/// Árbol de decisión individual en el ensamble GBDT
#[derive(Clone, Debug)]
pub struct DecisionTree {
    pub root: TreeNode,
}

impl DecisionTree {
    pub fn new(root: TreeNode) -> Self {
        Self { root }
    }

    #[inline(always)]
    pub fn predict(&self, features: &[f32; TOTAL_FEATURES]) -> f32 {
        self.root.predict(features)
    }

    pub fn compute_feature_importance(&self, importances: &mut [f32; TOTAL_FEATURES]) {
        self.root.compute_feature_importance(importances);
    }
}
