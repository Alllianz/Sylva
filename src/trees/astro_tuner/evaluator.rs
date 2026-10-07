use crate::dashboard::grid_dashboard::GbdtGridCandidateReport;
use crate::data::db::Kline;
use crate::engine::precomputed_model::PrecomputedAlphaModel;
use crate::engine::simulator::BacktestSimulator;
use crate::engine::threshold_tuner::ThresholdTuner;
use crate::engine::types::BacktestConfig;
use crate::features::CachedIndicators;
use crate::features::FeatureMask;
use crate::trees::astro_tuner::types::{AstroTreeCandidateConfig, AstroTreeModelType, AstroTreeTrial};
use crate::trees::batch::purged_cv::evaluate_predictions;
use crate::trees::bayesian::model::BayesianOnlineGbdtTrainer;
use crate::trees::dataset::TabularDataset;
use crate::trees::gpu::GpuTreeEngine;
use crate::trees::online::gbdt::OnlineGbdtTrainer;
use std::error::Error;
use std::sync::Arc;

/// Evaluador causal de candidatos para la auto-optimización Astro EVO con aceleración GPU opcional
pub struct AstroTreeEvaluator {
    pub backtest_config_nom: BacktestConfig,
    pub backtest_config_pct: BacktestConfig,
    pub rolling_window: usize,
    pub cached: Arc<CachedIndicators>,
    pub gpu_engine: Option<Arc<GpuTreeEngine>>,
}

impl AstroTreeEvaluator {
    pub fn new(
        backtest_config_nom: BacktestConfig,
        backtest_config_pct: BacktestConfig,
        rolling_window: usize,
        cached: Arc<CachedIndicators>,
    ) -> Self {
        Self {
            backtest_config_nom,
            backtest_config_pct,
            rolling_window,
            cached,
            gpu_engine: None,
        }
    }

    pub fn with_gpu_engine(mut self, gpu: Arc<GpuTreeEngine>) -> Self {
        self.gpu_engine = Some(gpu);
        self
    }

    /// Evalúa un candidato individual sobre la partición In-Sample y calcula las métricas Astro EVO
    pub fn evaluate_candidate(
        &self,
        candidate: &AstroTreeCandidateConfig,
        trial_idx: usize,
        generation_idx: usize,
        klines: &[Kline],
        is_dataset: &TabularDataset,
        is_start_idx: usize,
        oos_start_idx: usize,
        mask: Option<&FeatureMask>,
        color: String,
    ) -> Result<AstroTreeTrial, Box<dyn Error + Send + Sync>> {
        let sim_nom = BacktestSimulator::new(self.backtest_config_nom.clone());
        let sim_pct = BacktestSimulator::new(self.backtest_config_pct.clone());
        let tuner = ThresholdTuner::new(self.backtest_config_nom.clone());

        let (rep_nom, rep_pct, tune_long, tune_short, fitness, mse, mda, rank_ic) = match candidate.model_type {
            AstroTreeModelType::OnlineConventional => {
                let online_cfg = candidate.to_online_config();
                let trainer = OnlineGbdtTrainer::new(online_cfg);
                let trained_model = trainer.fit_stream(is_dataset, mask)?;

                // Inferencia en GPU (o CPU como respaldo)
                let preds = if let Some(ref gpu) = self.gpu_engine {
                    gpu.predict_dataset(&trained_model, is_dataset)
                        .unwrap_or_else(|_| trained_model.predict_dataset(is_dataset))
                } else {
                    trained_model.predict_dataset(is_dataset)
                };

                let targets: Vec<f32> = is_dataset.samples.iter().map(|s| s.target).collect();
                let (eval_mse, eval_mda, eval_ic) = evaluate_predictions(&preds, &targets);

                // Calibración de umbrales con PrecomputedAlphaModel (0 recálculo de árboles, velocidad instantánea)
                let alphas_arc = Arc::new(preds);
                let effective_start = is_start_idx.max(self.rolling_window);

                let tune_res = tuner.tune(
                    klines,
                    is_start_idx,
                    oos_start_idx,
                    &[0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012, 0.0018, 0.0025, 0.0035],
                    &[-0.0001, -0.0002, -0.0003, -0.0005, -0.0008, -0.0012, -0.0018, -0.0025, -0.0035],
                    |tl, ts| {
                        PrecomputedAlphaModel::new(
                            Arc::clone(&alphas_arc),
                            effective_start,
                            tl,
                            ts,
                        )
                    },
                );

                // Simulación ultra-rápida sin recálculo de árboles en CPU (0% Look-Ahead Bias, 100% velocidad nativa)
                let eval_end = is_start_idx + alphas_arc.len();
                let mut model_nom = PrecomputedAlphaModel::new(
                    Arc::clone(&alphas_arc),
                    effective_start,
                    tune_res.best_threshold_long,
                    tune_res.best_threshold_short,
                );
                let mut model_pct = PrecomputedAlphaModel::new(
                    Arc::clone(&alphas_arc),
                    effective_start,
                    tune_res.best_threshold_long,
                    tune_res.best_threshold_short,
                );

                let r_nom = sim_nom.run_range(&mut model_nom, klines, is_start_idx, eval_end);
                let r_pct = sim_pct.run_range(&mut model_pct, klines, is_start_idx, eval_end);

                (r_nom, r_pct, tune_res.best_threshold_long, tune_res.best_threshold_short, tune_res.best_fitness, eval_mse, eval_mda, eval_ic)
            }
            AstroTreeModelType::BayesianOnline => {
                let bayes_cfg = candidate.to_bayesian_config();
                let trainer = BayesianOnlineGbdtTrainer::new(bayes_cfg);
                let trained_model = trainer.fit_stream(is_dataset, mask)?;

                let preds = trained_model.predict_dataset(is_dataset);
                let pred_means: Vec<f32> = preds.iter().map(|p| p.mean).collect();
                let targets: Vec<f32> = is_dataset.samples.iter().map(|s| s.target).collect();
                let (eval_mse, eval_mda, eval_ic) = evaluate_predictions(&pred_means, &targets);

                let alphas_arc = Arc::new(pred_means);
                let effective_start = is_start_idx.max(self.rolling_window);

                let tune_res = tuner.tune(
                    klines,
                    is_start_idx,
                    oos_start_idx,
                    &[0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012, 0.0018, 0.0025, 0.0035],
                    &[-0.0001, -0.0002, -0.0003, -0.0005, -0.0008, -0.0012, -0.0018, -0.0025, -0.0035],
                    |tl, ts| {
                        PrecomputedAlphaModel::new(
                            Arc::clone(&alphas_arc),
                            effective_start,
                            tl,
                            ts,
                        )
                    },
                );

                let eval_end = is_start_idx + alphas_arc.len();
                let mut model_nom = PrecomputedAlphaModel::new(
                    Arc::clone(&alphas_arc),
                    effective_start,
                    tune_res.best_threshold_long,
                    tune_res.best_threshold_short,
                );
                let mut model_pct = PrecomputedAlphaModel::new(
                    Arc::clone(&alphas_arc),
                    effective_start,
                    tune_res.best_threshold_long,
                    tune_res.best_threshold_short,
                );

                let r_nom = sim_nom.run_range(&mut model_nom, klines, is_start_idx, eval_end);
                let r_pct = sim_pct.run_range(&mut model_pct, klines, is_start_idx, eval_end);

                (r_nom, r_pct, tune_res.best_threshold_long, tune_res.best_threshold_short, tune_res.best_fitness, eval_mse, eval_mda, eval_ic)
            }
        };

        // 1. Calcular de inmediato las métricas escalares exactas para el ranking multicriterio:
        let raw_slope_ratio = {
            let mid_is = (rep_nom.equity_curve.len() / 2).max(1);
            if mid_is > 0 && mid_is <= rep_nom.equity_curve.len() {
                let half1 = rep_nom.equity_curve[mid_is - 1].1 - rep_nom.initial_capital;
                let half2 = rep_nom.final_capital - rep_nom.equity_curve[mid_is - 1].1;
                let s1 = half1 / (mid_is as f64);
                let s2 = half2 / ((rep_nom.equity_curve.len() - mid_is) as f64);
                crate::metrics::autotuning::calculate_slope_ratio(s1, s2)
            } else {
                0.0
            }
        };
        let raw_cap_dd_ratio = crate::metrics::autotuning::calculate_capital_max_dd_ratio(rep_nom.final_capital, rep_nom.max_drawdown_amount);
        let raw_r2_score = crate::metrics::autotuning::evaluate_equity_linearity(&rep_nom.equity_curve, oos_start_idx, is_start_idx).composite_linearity_score;
        let raw_smoothness_score = crate::metrics::smoothness::calculate_equity_smoothness(&rep_nom.equity_curve, raw_r2_score).smoothness_score;

        // 2. Podar y submuestrear inmediatamente las curvas de equidad para no saturar la RAM (ahorro de 28 GB):
        let mut light_nom = rep_nom;
        light_nom.downsample_for_summary(50);
        let mut light_pct = rep_pct;
        light_pct.downsample_for_summary(50);

        let cand_rep = GbdtGridCandidateReport {
            max_depth: candidate.max_depth,
            min_samples_leaf: candidate.min_samples_leaf,
            n_trees: candidate.n_trees,
            cv_mse: mse,
            cv_mda: mda * 100.0,
            cv_ic: rank_ic,
            best_thr_long: tune_long,
            best_thr_short: tune_short,
            is_astro_fitness: fitness,
            rank_slope: 0,
            rank_cap_dd: 0,
            rank_r2: 0,
            rank_smoothness: 0,
            weighted_avg_rank: 0.0,
            astro_rank_fitness: 0.0,
            raw_slope_ratio,
            raw_cap_dd_ratio,
            raw_r2_score,
            raw_smoothness_score,
            report_nom: light_nom.clone(),
            report_pct: light_pct.clone(),
            color,
        };

        Ok(AstroTreeTrial {
            trial_idx,
            generation_idx,
            config: candidate.clone(),
            best_threshold_long: tune_long,
            best_threshold_short: tune_short,
            is_astro_fitness: fitness,
            report_nom: light_nom,
            report_pct: light_pct,
            candidate_report: cand_rep,
        })
    }
}
