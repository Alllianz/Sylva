use crate::data::db::Kline;

/// Timestamp obligatorio para inicio de trading y partición de backtest:
/// 2020-04-20 00:00:00 UTC (1587340800000 ms).
pub const MANDATORY_BACKTEST_START_TS: i64 = 1587340800000;

/// Divide el dataset de forma estrictamente causal en In-Sample (80%) y Out-Of-Sample (20%),
/// garantizando que el inicio de operaciones (In-Sample) comience obligatoriamente a partir del 20-04-2020 (1587340800000 ms UTC).
/// Las velas previas a dicha fecha se utilizan única y exclusivamente como período causal de warm-up
/// para la estabilización de indicadores técnicos sin Look-Ahead Bias.
pub fn get_dataset_partition_indices(klines: &[Kline], warmup_bars: usize) -> (usize, usize) {
    let total_len = klines.len();
    if total_len == 0 {
        return (0, 0);
    }
    if total_len <= warmup_bars {
        return (0, total_len);
    }

    // Localizar la primera vela con timestamp >= 2020-04-20 00:00:00 UTC
    let mandatory_idx_opt = klines.iter().position(|k| k.timestamp >= MANDATORY_BACKTEST_START_TS);

    let is_start_idx = match mandatory_idx_opt {
        Some(idx) => {
            // Si la fecha 2020-04-20 existe en el dataset, aseguramos que haya al menos warmup_bars
            // previos para la estabilización causal de indicadores técnicos.
            // Si el dataset ya cuenta con >= warmup_bars antes del 20-04-2020, arrancamos exactamente en dicha fecha.
            idx.max(warmup_bars)
        }
        None => {
            // Si todas las velas fueran anteriores a la fecha obligatoria (caso anómalo)
            warmup_bars
        }
    };

    if is_start_idx >= total_len {
        return (0, total_len);
    }

    let available = total_len - is_start_idx;
    let is_len = ((available as f64) * 0.80).round() as usize;
    let oos_start_idx = is_start_idx + is_len;

    (is_start_idx, oos_start_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dummy_kline(ts: i64) -> Kline {
        Kline {
            timestamp: ts,
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 102.0,
            volume: 1000.0,
        }
    }

    #[test]
    fn test_partition_strictly_starts_at_2020_04_20() {
        let mut klines = Vec::new();
        let step = 86400 * 1000; // 1 día en ms
        let start_ts = MANDATORY_BACKTEST_START_TS - 500 * step;
        for i in 0..1000 {
            klines.push(make_dummy_kline(start_ts + i * step));
        }

        let (is_start, oos_start) = get_dataset_partition_indices(&klines, 100);
        // Debe empezar exactamente en el índice 500 (2020-04-20)
        assert_eq!(is_start, 500);
        assert_eq!(klines[is_start].timestamp, MANDATORY_BACKTEST_START_TS);

        // In-Sample debe ser el 80% de las 500 velas restantes (400 velas)
        assert_eq!(oos_start, 500 + 400);
    }

    #[test]
    fn test_partition_fallback_when_after_2020() {
        let mut klines = Vec::new();
        let step = 86400 * 1000;
        let start_ts = MANDATORY_BACKTEST_START_TS + 10 * step;
        for i in 0..500 {
            klines.push(make_dummy_kline(start_ts + i * step));
        }

        let (is_start, oos_start) = get_dataset_partition_indices(&klines, 100);
        assert_eq!(is_start, 100);
        assert_eq!(oos_start, 100 + 320);
    }
}
