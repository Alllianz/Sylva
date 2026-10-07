use crate::metrics::backtest_report::BacktestReport;

/// Métricas de linealidad R² para evaluación de la curva de equidad
#[derive(Clone, Debug, Default)]
pub struct LinearityMetrics {
    pub r2_global: f64,
    pub r2_is: f64,
    pub r2_oos: f64,
    pub slope_global: f64,
    pub composite_linearity_score: f64,
}

pub fn calculate_series_r2(curve: &[(i64, f64)]) -> (f64, f64) {
    let n = curve.len();
    if n < 2 {
        return (0.0, 0.0);
    }

    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    for (i, &(_, eq)) in curve.iter().enumerate() {
        sum_x += i as f64;
        sum_y += eq;
    }
    let mean_x = sum_x / (n as f64);
    let mean_y = sum_y / (n as f64);

    let mut s_xx = 0.0f64;
    let mut s_yy = 0.0f64;
    let mut s_xy = 0.0f64;

    for (i, &(_, eq)) in curve.iter().enumerate() {
        let dx = (i as f64) - mean_x;
        let dy = eq - mean_y;
        s_xx += dx * dx;
        s_yy += dy * dy;
        s_xy += dx * dy;
    }

    if s_xx <= 1e-12 || s_yy <= 1e-12 {
        return (0.0, 0.0);
    }

    let slope = s_xy / s_xx;
    if slope <= 0.0 {
        return (0.0, slope);
    }

    let r2 = ((s_xy * s_xy) / (s_xx * s_yy)).clamp(0.0, 1.0);
    (r2, slope)
}

pub fn evaluate_equity_linearity(
    curve: &[(i64, f64)],
    oos_start_idx: usize,
    sim_start_idx: usize,
) -> LinearityMetrics {
    if curve.len() < 4 {
        return LinearityMetrics::default();
    }

    let (r2_global, slope_global) = calculate_series_r2(curve);

    let is_slice_len = if oos_start_idx > sim_start_idx {
        oos_start_idx - sim_start_idx
    } else {
        curve.len() / 2
    };

    let is_slice = if is_slice_len <= curve.len() {
        &curve[0..is_slice_len]
    } else {
        &curve[..]
    };

    let oos_slice = if is_slice_len < curve.len() {
        &curve[is_slice_len..]
    } else {
        &curve[..]
    };

    let (r2_is, _) = calculate_series_r2(is_slice);
    let (r2_oos, _) = calculate_series_r2(oos_slice);

    let min_seg_r2 = r2_is.min(r2_oos);
    let avg_seg_r2 = (r2_is + r2_oos) / 2.0;
    let composite = 0.40 * r2_global + 0.35 * avg_seg_r2 + 0.25 * min_seg_r2;

    LinearityMetrics {
        r2_global,
        r2_is,
        r2_oos,
        slope_global,
        composite_linearity_score: composite.clamp(0.0, 1.0),
    }
}

pub fn calculate_slope_ratio(slope_is: f64, slope_oos: f64) -> f64 {
    if slope_is > 0.0 && slope_oos > 0.0 {
        let ratio = slope_is / slope_oos;
        if ratio < 1.0 {
            ratio
        } else {
            1.0 / ratio
        }
    } else {
        0.0
    }
}

pub fn calculate_capital_max_dd_ratio(final_cap: f64, max_dd_nom: f64) -> f64 {
    let profit = (final_cap - 10000.0).max(0.0);
    if max_dd_nom > 0.0 {
        (profit / max_dd_nom).clamp(0.0, 50.0)
    } else if profit > 0.0 {
        50.0
    } else {
        0.0
    }
}

/// Autotuning Score Institucional idéntico al de Astro Evo Max:
/// Combina la consistencia de pendiente IS/OOS, el ratio de retorno vs Max Drawdown y la linealidad compuesta R².
pub fn calculate_autotuning_score(
    slope_ratio: f64,
    capital_dd_ratio: f64,
    composite_linearity: f64,
) -> f64 {
    let cap_dd_norm = (capital_dd_ratio / 5.0).clamp(0.0, 1.0);
    (0.35 * slope_ratio + 0.35 * cap_dd_norm + 0.30 * composite_linearity) * 100.0
}

/// Evalúa el Fitness institucional exacto de Astro EVO Max sobre un reporte de backtest
pub fn evaluate_report_astro_fitness(
    report: &BacktestReport,
    oos_start_idx: usize,
    is_start_idx: usize,
) -> f64 {
    let n_bars = oos_start_idx.saturating_sub(is_start_idx);
    let min_trades = if n_bars > 1000 {
        (n_bars / 250).max(15)
    } else {
        (n_bars / 30).max(5)
    };
    if report.total_trades < min_trades {
        return -100.0;
    }
    let lin_eval = evaluate_equity_linearity(&report.equity_curve, oos_start_idx, is_start_idx);
    let mid_is = (report.equity_curve.len() / 2).max(1);
    let half1_profit = report.equity_curve[mid_is - 1].1 - report.initial_capital;
    let half2_profit = report.final_capital - report.equity_curve[mid_is - 1].1;
    let slope1 = half1_profit / (mid_is as f64);
    let slope2 = half2_profit / ((report.equity_curve.len() - mid_is) as f64);
    let slope_ratio = calculate_slope_ratio(slope1, slope2);

    let cap_dd_ratio = calculate_capital_max_dd_ratio(report.final_capital, report.max_drawdown_amount);
    calculate_autotuning_score(slope_ratio, cap_dd_ratio, lin_eval.composite_linearity_score)
}
