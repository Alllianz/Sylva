// ===================================================================================================
// Sylva GPU Tree Trainer WGSL Compute Shader
// Entrenamiento nativo de Gradient Boosted Decision Trees directamente en VRAM (AMD RDNA 3)
// 0% CPU, 0% Look-Ahead Bias, 100% Aceleración Paralela en GPU
// ===================================================================================================

struct TrainUniforms {
    n_samples: u32,
    n_features: u32,
    target_node: u32,
    min_samples_leaf: u32,
    learning_rate: f32,
    l2_reg: f32,
    gamma: f32,
    _pad: f32,
};

struct SplitCandidate {
    gain: f32,
    threshold: f32,
    feature_idx: u32,
    sample_count_left: u32,
    left_weight: f32,
    right_weight: f32,
    valid: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: TrainUniforms;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read> targets: array<f32>;
@group(0) @binding(3) var<storage, read> predictions: array<f32>;
@group(0) @binding(4) var<storage, read> sample_nodes: array<u32>;
@group(0) @binding(5) var<storage, read_write> split_candidates: array<SplitCandidate>;

const TOTAL_FEATURES: u32 = 32u;
const NUM_BINS: u32 = 16u;

// Calcula la ganancia cuadrática de segundo orden para un nodo (XGBoost objective)
fn compute_node_score(sum_g: f32, sum_h: f32, l2: f32) -> f32 {
    if (sum_h <= 1e-4) {
        return 0.0;
    }
    return (sum_g * sum_g) / (sum_h + l2);
}

// Cada Workgroup evalúa una característica (feature) para el nodo objetivo
@compute @workgroup_size(64)
fn evaluate_splits(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let feat_idx = wg_id.x;
    if (feat_idx >= params.n_features) {
        return;
    }

    // Paso 1: Encontrar min y max de la característica para las muestras pertenecientes al nodo objetivo
    var f_min: f32 = 1e30;
    var f_max: f32 = -1e30;
    var node_sample_count: u32 = 0u;
    var total_g: f32 = 0.0;
    var total_h: f32 = 0.0;

    for (var i: u32 = 0u; i < params.n_samples; i = i + 1u) {
        if (sample_nodes[i] == params.target_node) {
            let val = features[i * TOTAL_FEATURES + feat_idx];
            f_min = min(f_min, val);
            f_max = max(f_max, val);
            node_sample_count = node_sample_count + 1u;

            let residual = predictions[i] - targets[i];
            total_g = total_g + residual;
            total_h = total_h + 1.0;
        }
    }

    // Si hay menos muestras que el mínimo requerido o el rango es cero, no es divisible
    let f_range = f_max - f_min;
    if (node_sample_count < params.min_samples_leaf * 2u || f_range <= 1e-5) {
        if (local_id.x == 0u) {
            split_candidates[feat_idx].gain = -1.0;
            split_candidates[feat_idx].valid = 0u;
        }
        return;
    }

    let root_score = compute_node_score(total_g, total_h, params.l2_reg);

    var best_gain: f32 = -1.0;
    var best_threshold: f32 = 0.0;
    var best_left_count: u32 = 0u;
    var best_left_w: f32 = 0.0;
    var best_right_w: f32 = 0.0;

    // Paso 2: Evaluar 16 puntos de corte (bins) a lo largo del rango
    for (var b: u32 = 1u; b < NUM_BINS; b = b + 1u) {
        let thresh = f_min + f_range * (f32(b) / f32(NUM_BINS));
        var left_g: f32 = 0.0;
        var left_h: f32 = 0.0;
        var left_count: u32 = 0u;

        for (var i: u32 = 0u; i < params.n_samples; i = i + 1u) {
            if (sample_nodes[i] == params.target_node) {
                let val = features[i * TOTAL_FEATURES + feat_idx];
                if (val <= thresh) {
                    let residual = predictions[i] - targets[i];
                    left_g = left_g + residual;
                    left_h = left_h + 1.0;
                    left_count = left_count + 1u;
                }
            }
        }

        let right_count = node_sample_count - left_count;
        if (left_count < params.min_samples_leaf || right_count < params.min_samples_leaf) {
            continue;
        }

        let right_g = total_g - left_g;
        let right_h = total_h - left_h;

        let left_score = compute_node_score(left_g, left_h, params.l2_reg);
        let right_score = compute_node_score(right_g, right_h, params.l2_reg);

        let gain = 0.5 * (left_score + right_score - root_score) - params.gamma;

        if (gain > best_gain) {
            best_gain = gain;
            best_threshold = thresh;
            best_left_count = left_count;
            best_left_w = -left_g / (left_h + params.l2_reg);
            best_right_w = -right_g / (right_h + params.l2_reg);
        }
    }

    if (local_id.x == 0u) {
        split_candidates[feat_idx].gain = best_gain;
        split_candidates[feat_idx].threshold = best_threshold;
        split_candidates[feat_idx].feature_idx = feat_idx;
        split_candidates[feat_idx].sample_count_left = best_left_count;
        split_candidates[feat_idx].left_weight = best_left_w;
        split_candidates[feat_idx].right_weight = best_right_w;
        split_candidates[feat_idx].valid = select(0u, 1u, best_gain > 0.0);
    }
}
