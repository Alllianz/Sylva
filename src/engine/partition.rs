use crate::data::db::Kline;

/// Divide el dataset de forma estrictamente causal en In-Sample (80%) y Out-Of-Sample (20%)
/// respetando el período inicial de warm-up (por defecto 100 velas)
pub fn get_dataset_partition_indices(klines: &[Kline], warmup_bars: usize) -> (usize, usize) {
    let total_len = klines.len();
    if total_len <= warmup_bars {
        return (0, total_len);
    }

    let available = total_len - warmup_bars;
    let is_len = ((available as f64) * 0.80).round() as usize;

    let is_start_idx = warmup_bars;
    let oos_start_idx = warmup_bars + is_len;

    (is_start_idx, oos_start_idx)
}
