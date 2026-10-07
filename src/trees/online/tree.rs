use crate::features::TOTAL_FEATURES;
use crate::trees::online::node::OnlineTreeNode;
use crate::trees::online::types::OnlineGbdtConfig;

/// Árbol de decisión streaming individual dentro del ensamble de boosting online
#[derive(Clone, Debug)]
pub struct OnlineDecisionTree {
    pub root: OnlineTreeNode,
    pub tree_idx: usize,
    pub active_features: Vec<usize>,
}

impl OnlineDecisionTree {
    /// Crea un nuevo árbol online con una raíz de hoja
    pub fn new(tree_idx: usize, active_features: Vec<usize>) -> Self {
        Self {
            root: OnlineTreeNode::new_leaf(),
            tree_idx,
            active_features,
        }
    }

    /// Realiza inferencia sobre un vector de características
    #[inline(always)]
    pub fn predict(&self, features: &[f32; TOTAL_FEATURES]) -> f32 {
        self.root.predict(features)
    }

    /// Actualiza el árbol online con una nueva observación
    pub fn update(
        &mut self,
        features: &[f32; TOTAL_FEATURES],
        g: f32,
        h: f32,
        cfg: &OnlineGbdtConfig,
    ) -> bool {
        self.root.update(features, g, h, cfg, 0, &self.active_features)
    }

    /// Acumula la ganancia de importancia por característica
    pub fn compute_feature_importance(&self, importances: &mut [f32; TOTAL_FEATURES]) {
        self.root.compute_feature_importance(importances);
    }

    /// Profundidad máxima actual del árbol
    pub fn depth(&self) -> usize {
        self.root.depth()
    }

    /// Número total de hojas en el árbol
    pub fn n_leaves(&self) -> usize {
        self.root.n_leaves()
    }
}
