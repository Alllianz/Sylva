use std::collections::HashMap;

/// Estructura con la información multidimensional de ranking evolutivo por Puestos (Sylva EVO Max)
#[derive(Clone, Debug)]
pub struct CandidateRankScore {
    pub candidate_id: String,
    pub rank_slope: usize,
    pub rank_cap_dd: usize,
    pub rank_r2: usize,
    pub rank_smoothness: usize,
    pub weighted_avg_rank: f64,
    pub sylva_rank_fitness: f64, // Fitness normalizado para maximización (mayor es mejor)
    pub astro_rank_fitness: f64, // Alias retrocompatible
}

/// Computa el ranking multicriterio exacto de Sylva EVO integrando:
/// 1. Consistencia de Pendiente IS/OOS (Slope Ratio más cercano a 1.0)
/// 2. Ratio Capital / Max Drawdown (Mayor es mejor)
/// 3. Linealidad R² Compuesta (Mayor es mejor)
/// 4. Suavidad y Resistencia a Drawdowns (Ulcer Index penalizado, Mayor es mejor)
///
/// Cada candidato es evaluado por su PUESTO en cada categoría (1º, 2º, 3º...) eliminando distorsiones monetarias.
pub fn compute_multicriteria_rankings<T: Clone>(
    items: &[T],
    get_key: impl Fn(&T) -> String,
    get_slope_ratio: impl Fn(&T) -> f64,
    get_cap_dd_ratio: impl Fn(&T) -> f64,
    get_r2_score: impl Fn(&T) -> f64,
    get_smoothness_score: impl Fn(&T) -> f64,
) -> HashMap<String, CandidateRankScore> {
    let mut results_map = HashMap::new();
    if items.is_empty() {
        return results_map;
    }

    let n = items.len();

    // 1. Ranking de Pendiente IS/OOS (Slope Ratio más cercano a 1.0)
    let mut slope_sorted: Vec<(String, f64)> = items
        .iter()
        .map(|item| (get_key(item), get_slope_ratio(item)))
        .collect();
    slope_sorted.sort_by(|a, b| {
        let diff_a = (a.1 - 1.0).abs();
        let diff_b = (b.1 - 1.0).abs();
        diff_a
            .partial_cmp(&diff_b)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let mut rank_slope_map = HashMap::new();
    for (idx, (key, _)) in slope_sorted.into_iter().enumerate() {
        rank_slope_map.insert(key, idx + 1);
    }

    // 2. Ranking de Capital / Max Drawdown (Mayor es mejor)
    let mut cap_dd_sorted: Vec<(String, f64)> = items
        .iter()
        .map(|item| (get_key(item), get_cap_dd_ratio(item)))
        .collect();
    cap_dd_sorted.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let mut rank_cap_dd_map = HashMap::new();
    for (idx, (key, _)) in cap_dd_sorted.into_iter().enumerate() {
        rank_cap_dd_map.insert(key, idx + 1);
    }

    // 3. Ranking de Linealidad R² Compuesta (Mayor es mejor)
    let mut r2_sorted: Vec<(String, f64)> = items
        .iter()
        .map(|item| (get_key(item), get_r2_score(item)))
        .collect();
    r2_sorted.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let mut rank_r2_map = HashMap::new();
    for (idx, (key, _)) in r2_sorted.into_iter().enumerate() {
        rank_r2_map.insert(key, idx + 1);
    }

    // 4. Ranking de Suavidad y Resistencia a Drawdowns (Mayor es mejor)
    let mut smooth_sorted: Vec<(String, f64)> = items
        .iter()
        .map(|item| (get_key(item), get_smoothness_score(item)))
        .collect();
    smooth_sorted.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let mut rank_smooth_map = HashMap::new();
    for (idx, (key, _)) in smooth_sorted.into_iter().enumerate() {
        rank_smooth_map.insert(key, idx + 1);
    }

    // 5. Ponderación final del Rank (Sylva EVO Max):
    // Pesos: Slope (1.0), Cap/DD (1.0), Linealidad R² (2.0), Suavidad (1.0) -> Total divisor = 5.0
    for item in items {
        let key = get_key(item);
        let r_slope = *rank_slope_map.get(&key).unwrap_or(&n);
        let r_cap_dd = *rank_cap_dd_map.get(&key).unwrap_or(&n);
        let r_r2 = *rank_r2_map.get(&key).unwrap_or(&n);
        let r_smooth = *rank_smooth_map.get(&key).unwrap_or(&n);

        let weighted_rank = (1.0 * (r_slope as f64)
            + 1.0 * (r_cap_dd as f64)
            + 2.0 * (r_r2 as f64)
            + 1.0 * (r_smooth as f64))
            / 5.0;

        // Fitness para maximización: 100 * (N - weighted_rank + 1) / N
        let sylva_rank_fitness = ((n as f64 - weighted_rank + 1.0) / (n as f64)) * 100.0;

        results_map.insert(
            key.clone(),
            CandidateRankScore {
                candidate_id: key,
                rank_slope: r_slope,
                rank_cap_dd: r_cap_dd,
                rank_r2: r_r2,
                rank_smoothness: r_smooth,
                weighted_avg_rank: weighted_rank,
                sylva_rank_fitness,
                astro_rank_fitness: sylva_rank_fitness,
            },
        );
    }

    results_map
}
