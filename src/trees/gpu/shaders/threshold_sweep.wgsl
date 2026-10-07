struct ThresholdPair {
    thr_long: f32,
    thr_short: f32,
    fee_rate: f32,
    _pad: f32,
};

struct SweepResult {
    total_trades: u32,
    win_trades: u32,
    net_pnl_pct: f32,
    max_drawdown_pct: f32,
    profit_factor: f32,
    total_longs: u32,
    total_shorts: u32,
    _pad: u32,
};

struct SweepUniforms {
    n_samples: u32,
    n_pairs: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: SweepUniforms;
@group(0) @binding(1) var<storage, read> pairs: array<ThresholdPair>;
@group(0) @binding(2) var<storage, read> predictions: array<f32>;
@group(0) @binding(3) var<storage, read> bar_returns: array<f32>;
@group(0) @binding(4) var<storage, read_write> results: array<SweepResult>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    if (pair_idx >= params.n_pairs) {
        return;
    }

    let p = pairs[pair_idx];
    var current_pos: f32 = 0.0;
    var total_trades: u32 = 0u;
    var win_trades: u32 = 0u;
    var longs_count: u32 = 0u;
    var shorts_count: u32 = 0u;
    var gross_win: f32 = 0.0;
    var gross_loss: f32 = 0.0;

    var equity: f32 = 1.0;
    var peak_equity: f32 = 1.0;
    var max_dd: f32 = 0.0;

    for (var i: u32 = 0u; i < params.n_samples; i = i + 1u) {
        let alpha = predictions[i];
        let ret = bar_returns[i];

        var next_pos: f32 = current_pos;
        if (alpha > p.thr_long) {
            next_pos = 1.0;
        } else if (alpha < p.thr_short) {
            next_pos = -1.0;
        }

        if (next_pos != current_pos) {
            total_trades = total_trades + 1u;
            if (next_pos == 1.0) {
                longs_count = longs_count + 1u;
            } else if (next_pos == -1.0) {
                shorts_count = shorts_count + 1u;
            }
            equity = equity * (1.0 - p.fee_rate * 2.0);
            current_pos = next_pos;
        }

        if (current_pos != 0.0) {
            let bar_pnl = current_pos * ret;
            if (bar_pnl > 0.0) {
                gross_win = gross_win + bar_pnl;
                win_trades = win_trades + 1u;
            } else {
                gross_loss = gross_loss + abs(bar_pnl);
            }
            equity = equity * (1.0 + bar_pnl);
        }

        if (equity > peak_equity) {
            peak_equity = equity;
        } else {
            let dd = (peak_equity - equity) / peak_equity;
            if (dd > max_dd) {
                max_dd = dd;
            }
        }
    }

    var pf: f32 = 0.0;
    if (gross_loss > 1e-6) {
        pf = gross_win / gross_loss;
    } else if (gross_win > 0.0) {
        pf = 50.0;
    }

    results[pair_idx].total_trades = total_trades;
    results[pair_idx].win_trades = win_trades;
    results[pair_idx].net_pnl_pct = (equity - 1.0) * 100.0;
    results[pair_idx].max_drawdown_pct = max_dd * 100.0;
    results[pair_idx].profit_factor = pf;
    results[pair_idx].total_longs = longs_count;
    results[pair_idx].total_shorts = shorts_count;
    results[pair_idx]._pad = 0u;
}
