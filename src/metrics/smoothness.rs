/// Módulo de cálculo de suavidad de curva de equidad, Sortino Ratio y penalización de varianza de drawdowns (Astro EVO).

#[derive(Clone, Debug, Default)]
pub struct SmoothnessMetrics {
    pub sortino_ratio: f64,
    pub downside_deviation: f64,
    pub smoothness_score: f64,
    pub ulcer_index: f64,
}

/// Calcula el Ulcer Index y métricas de suavidad sobre la curva de equidad
pub fn calculate_equity_smoothness(
    curve: &[(i64, f64)],
    composite_linearity: f64,
) -> SmoothnessMetrics {
    if curve.len() < 2 {
        return SmoothnessMetrics::default();
    }

    // 1. Cálculo de Ulcer Index sobre la curva de equidad
    let mut peak = curve[0].1;
    let mut sum_sq_dd = 0.0f64;

    for &(_, eq) in curve {
        if eq > peak {
            peak = eq;
        }
        let dd_pct = if peak > 0.0 {
            ((peak - eq) / peak) * 100.0
        } else {
            0.0
        };
        sum_sq_dd += dd_pct * dd_pct;
    }
    let ulcer_index = (sum_sq_dd / (curve.len() as f64)).sqrt();

    // 2. Cálculo de Sortino Ratio sobre los retornos entre pasos de equidad
    let mut returns = Vec::with_capacity(curve.len() - 1);
    for i in 1..curve.len() {
        let prev = curve[i - 1].1;
        let curr = curve[i].1;
        if prev > 0.0 {
            returns.push((curr - prev) / prev);
        }
    }

    let (sortino_ratio, downside_dev) = if returns.is_empty() {
        (0.0, 0.0)
    } else {
        let mean_ret = returns.iter().sum::<f64>() / (returns.len() as f64);
        let mut sum_downside_sq = 0.0f64;
        let mut downside_count = 0;

        for &r in &returns {
            if r < 0.0 {
                sum_downside_sq += r * r;
                downside_count += 1;
            }
        }

        let downside_dev = if downside_count > 0 {
            (sum_downside_sq / (returns.len() as f64)).sqrt()
        } else {
            0.0
        };

        let sortino = if downside_dev > 1e-6 {
            (mean_ret / downside_dev) * (365.0f64).sqrt() // Anualizado aproximado
        } else if mean_ret > 0.0 {
            10.0
        } else {
            0.0
        };

        (sortino.max(0.0), downside_dev)
    };

    // 3. Score de Suavidad Compuesto
    // Combina alta linealidad (R²) castigando el Ulcer Index (duración y profundidad del drawdown)
    let smoothness = (composite_linearity / (1.0 + (ulcer_index / 10.0))).clamp(0.0, 1.0);

    SmoothnessMetrics {
        sortino_ratio,
        downside_deviation: downside_dev,
        smoothness_score: smoothness,
        ulcer_index,
    }
}
