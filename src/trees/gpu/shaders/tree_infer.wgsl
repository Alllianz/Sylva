struct GpuNode {
    feature_idx: u32,
    threshold: f32,
    left_child: u32,
    right_child: u32,
    leaf_weight: f32,
    is_leaf: u32,
    _pad0: u32,
    _pad1: u32,
};

struct GpuTreeMeta {
    root_idx: u32,
    node_count: u32,
    _pad0: u32,
    _pad1: u32,
};

struct Uniforms {
    n_samples: u32,
    n_trees: u32,
    base_score: f32,
    learning_rate: f32,
};

@group(0) @binding(0) var<uniform> params: Uniforms;
@group(0) @binding(1) var<storage, read> features: array<f32>;
@group(0) @binding(2) var<storage, read> nodes: array<GpuNode>;
@group(0) @binding(3) var<storage, read> tree_metas: array<GpuTreeMeta>;
@group(0) @binding(4) var<storage, read_write> predictions: array<f32>;

const TOTAL_FEATURES: u32 = 32u;
const MAX_TREE_DEPTH: u32 = 32u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sample_idx = global_id.x;
    if (sample_idx >= params.n_samples) {
        return;
    }

    let feat_offset = sample_idx * TOTAL_FEATURES;
    var pred: f32 = params.base_score;

    for (var t: u32 = 0u; t < params.n_trees; t = t + 1u) {
        let t_info = tree_metas[t];
        var curr_idx = t_info.root_idx;
        var depth: u32 = 0u;

        while (depth < MAX_TREE_DEPTH) {
            let node = nodes[curr_idx];
            if (node.is_leaf == 1u) {
                pred = pred + params.learning_rate * node.leaf_weight;
                break;
            }

            let feat_val = features[feat_offset + node.feature_idx];
            if (feat_val <= node.threshold) {
                curr_idx = node.left_child;
            } else {
                curr_idx = node.right_child;
            }
            depth = depth + 1u;
        }
    }

    predictions[sample_idx] = pred;
}
