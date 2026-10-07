use crate::trees::batch::TreeNode;
use crate::trees::gpu::types::{GpuNode, GpuTreeMeta};
use crate::trees::online::gbdt::OnlineGbdtModel;
use crate::trees::online::node::OnlineTreeNode;
use crate::trees::online::tree::OnlineDecisionTree;

/// Representación plana contigua de un ensamble completo de árboles para VRAM
#[derive(Clone, Debug, Default)]
pub struct FlatEnsemble {
    pub nodes: Vec<GpuNode>,
    pub tree_metas: Vec<GpuTreeMeta>,
    pub base_score: f32,
    pub learning_rate: f32,
}

impl FlatEnsemble {
    pub fn new(base_score: f32, learning_rate: f32) -> Self {
        Self {
            nodes: Vec::new(),
            tree_metas: Vec::new(),
            base_score,
            learning_rate,
        }
    }

    /// Total de nodos en el búfer plano
    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Total de árboles compilados
    pub fn total_trees(&self) -> usize {
        self.tree_metas.len()
    }

    /// Compila un modelo completo `OnlineGbdtModel` a memoria lineal de GPU
    pub fn from_online_gbdt(model: &OnlineGbdtModel) -> Self {
        let mut ensemble = Self::new(model.base_score, model.config.learning_rate);
        for tree in &model.trees {
            ensemble.add_online_tree(tree);
        }
        ensemble
    }

    /// Agrega un árbol de decisión online al búfer plano
    pub fn add_online_tree(&mut self, tree: &OnlineDecisionTree) {
        let root_idx = self.nodes.len() as u32;
        let count_before = self.nodes.len();
        self.flatten_online_node(&tree.root);
        let node_count = (self.nodes.len() - count_before) as u32;

        self.tree_metas.push(GpuTreeMeta {
            root_idx,
            node_count,
            _pad0: 0,
            _pad1: 0,
        });
    }

    fn flatten_online_node(&mut self, node: &OnlineTreeNode) -> u32 {
        let current_idx = self.nodes.len() as u32;

        match node {
            OnlineTreeNode::Leaf { weight, .. } => {
                self.nodes.push(GpuNode::new_leaf(*weight));
                current_idx
            }
            OnlineTreeNode::Internal {
                feature_idx,
                threshold,
                left,
                right,
                ..
            } => {
                // Reservar slot temporalmente
                self.nodes.push(GpuNode::default());

                // Compilar subárbol izquierdo y derecho
                let left_idx = self.flatten_online_node(left);
                let right_idx = self.flatten_online_node(right);

                // Escribir nodo interno con los punteros de índice correctos
                self.nodes[current_idx as usize] = GpuNode::new_internal(
                    *feature_idx as u32,
                    *threshold,
                    left_idx,
                    right_idx,
                );

                current_idx
            }
        }
    }

    /// Agrega un árbol batch `TreeNode` al búfer plano
    pub fn add_batch_node(&mut self, node: &TreeNode) {
        let root_idx = self.nodes.len() as u32;
        let count_before = self.nodes.len();
        self.flatten_batch_node(node);
        let node_count = (self.nodes.len() - count_before) as u32;

        self.tree_metas.push(GpuTreeMeta {
            root_idx,
            node_count,
            _pad0: 0,
            _pad1: 0,
        });
    }

    fn flatten_batch_node(&mut self, node: &TreeNode) -> u32 {
        let current_idx = self.nodes.len() as u32;

        match node {
            TreeNode::Leaf { weight, .. } => {
                self.nodes.push(GpuNode::new_leaf(*weight));
                current_idx
            }
            TreeNode::Internal {
                feature_idx,
                threshold,
                left,
                right,
                ..
            } => {
                self.nodes.push(GpuNode::default());
                let left_idx = self.flatten_batch_node(left);
                let right_idx = self.flatten_batch_node(right);

                self.nodes[current_idx as usize] = GpuNode::new_internal(
                    *feature_idx as u32,
                    *threshold,
                    left_idx,
                    right_idx,
                );

                current_idx
            }
        }
    }
}
