use crate::data::db::Kline;
use crate::engine::equation_model::EquationModel;
use crate::engine::simulator::BacktestSimulator;
use crate::engine::types::BacktestConfig;
use crate::metrics::autotuning::evaluate_report_astro_fitness;

/// Resultado de la calibración de umbrales con Fitness Astro EVO
#[derive(Debug, Clone)]
pub struct TunerResult {
    pub best_threshold_long: f32,
    pub best_threshold_short: f32,
    pub best_fitness: f64,
    pub total_candidates_tested: usize,
}

/// Calibrador Universal de Umbrales Óptimos mediante la función de Fitness de Astro EVO
pub struct ThresholdTuner {
    pub backtest_config: BacktestConfig,
}

impl ThresholdTuner {
    pub fn new(backtest_config: BacktestConfig) -> Self {
        Self { backtest_config }
    }

    /// Calibra los umbrales óptimos buscando maximizar el Fitness de Astro EVO exclusivamente en In-Sample
    pub fn tune<F, M>(
        &self,
        klines: &[Kline],
        is_start_idx: usize,
        oos_start_idx: usize,
        long_grid: &[f32],
        short_grid: &[f32],
        mut model_factory: F,
    ) -> TunerResult
    where
        F: FnMut(f32, f32) -> M,
        M: EquationModel,
    {
        let mut best_fitness = -1000.0f64;
        let mut best_thr_long = if !long_grid.is_empty() { long_grid[0] } else { 0.0005 };
        let mut best_thr_short = if !short_grid.is_empty() { short_grid[0] } else { -0.0005 };
        let mut total_tested = 0;

        let sim = BacktestSimulator::new(self.backtest_config.clone());

        for &thr_l in long_grid {
            for &thr_s in short_grid {
                if thr_l <= thr_s {
                    continue;
                }
                total_tested += 1;
                let mut model = model_factory(thr_l, thr_s);
                let report = sim.run_range(&mut model, klines, is_start_idx, oos_start_idx);
                let fitness = evaluate_report_astro_fitness(&report, oos_start_idx, is_start_idx);

                if fitness > best_fitness {
                    best_fitness = fitness;
                    best_thr_long = thr_l;
                    best_thr_short = thr_s;
                }
            }
        }

        if best_fitness <= -100.0 && !long_grid.is_empty() && !short_grid.is_empty() {
            best_thr_long = long_grid[long_grid.len() / 2];
            best_thr_short = short_grid[short_grid.len() / 2];
        }

        TunerResult {
            best_threshold_long: best_thr_long,
            best_threshold_short: best_thr_short,
            best_fitness,
            total_candidates_tested: total_tested,
        }
    }
}
