use bytemuck::{Pod, Zeroable};

/// Nodo lineal aplanado para almacenamiento en memoria continua de VRAM (WGSL Storage Buffer)
/// Alineado a 32 bytes (16-byte alignment en WGSL)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GpuNode {
    pub feature_idx: u32,
    pub threshold: f32,
    pub left_child: u32,
    pub right_child: u32,
    pub leaf_weight: f32,
    pub is_leaf: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl GpuNode {
    pub fn new_leaf(weight: f32) -> Self {
        Self {
            feature_idx: 0,
            threshold: 0.0,
            left_child: 0,
            right_child: 0,
            leaf_weight: weight,
            is_leaf: 1,
            _pad0: 0,
            _pad1: 0,
        }
    }

    pub fn new_internal(feature_idx: u32, threshold: f32, left_child: u32, right_child: u32) -> Self {
        Self {
            feature_idx,
            threshold,
            left_child,
            right_child,
            leaf_weight: 0.0,
            is_leaf: 0,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

/// Metadatos por árbol dentro del búfer global de nodos del ensamble (16 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GpuTreeMeta {
    pub root_idx: u32,
    pub node_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Uniformes para el Compute Shader de Inferencia de Árboles (16 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GpuInferenceUniforms {
    pub n_samples: u32,
    pub n_trees: u32,
    pub base_score: f32,
    pub learning_rate: f32,
}

/// Parámetros de umbral para barrido masivo de candidatos en GPU (16 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GpuThresholdParams {
    pub threshold_long: f32,
    pub threshold_short: f32,
    pub fee_rate: f32,
    pub n_samples: u32,
}

/// Métricas de resultado de simulación de un umbral calculadas en GPU (32 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GpuThresholdResult {
    pub total_trades: u32,
    pub win_trades: u32,
    pub net_pnl_pct: f32,
    pub max_drawdown_pct: f32,
    pub profit_factor: f32,
    pub total_longs: u32,
    pub total_shorts: u32,
    pub _pad: u32,
}

/// Parámetros uniformes para entrenamiento de árboles en GPU (32 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GpuTrainUniforms {
    pub n_samples: u32,
    pub n_features: u32,
    pub target_node: u32,
    pub min_samples_leaf: u32,
    pub learning_rate: f32,
    pub l2_reg: f32,
    pub gamma: f32,
    pub _pad: f32,
}

/// Candidato de división evaluado por un Workgroup en GPU (32 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GpuSplitCandidate {
    pub gain: f32,
    pub threshold: f32,
    pub feature_idx: u32,
    pub sample_count_left: u32,
    pub left_weight: f32,
    pub right_weight: f32,
    pub valid: u32,
    pub _pad: u32,
}

