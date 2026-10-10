use crate::dashboard::grid_dashboard::GbdtGridCandidateReport;
use crate::data::db::Kline;
use crate::engine::dynamic_threshold::ThresholdMode;
use crate::engine::precomputed_model::PrecomputedAlphaModel;
use crate::engine::simulator::BacktestSimulator;
use crate::engine::threshold_tuner::ThresholdTuner;
use crate::engine::types::BacktestConfig;
use crate::features::CachedIndicators;
use crate::features::FeatureMask;
use crate::trees::sylva_tuner::types::{SylvaTreeCandidateConfig, SylvaTreeModelType, SylvaTreeTrial};
use crate::trees::batch::purged_cv::evaluate_predictions;
use crate::trees::bayesian::model::BayesianOnlineGbdtTrainer;
use crate::trees::bayesian::model_wrapper::BayesianGbdtEquationModel;
use crate::trees::dataset::TabularDataset;
use crate::trees::gpu::GpuTreeEngine;
use crate::trees::online::gbdt::OnlineGbdtTrainer;
use crate::trees::online::model_wrapper::OnlineGbdtEquationModel;
use std::error::Error;
use std::sync::Arc;

/// Evaluador causal de candidatos para la auto-optimización Sylva EVO con aceleración GPU opcional
pub struct SylvaTreeEvaluator {
    pub backtest_config_nom: BacktestConfig,
    pub backtest_config_pct: BacktestConfig,
    pub rolling_window: usize,
    pub cached: Arc<CachedIndicators>,
    pub gpu_engine: Option<Arc<GpuTreeEngine>>,
    pub threshold_mode: ThresholdMode,
}

pub type AstroTreeEvaluator = SylvaTreeEvaluator;

impl SylvaTreeEvaluator {
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
            threshold_mode: ThresholdMode::default(),
        }
    }

    pub fn with_gpu_engine(mut self, gpu: Arc<GpuTreeEngine>) -> Self {
        self.gpu_engine = Some(gpu);
        self
    }

    pub fn with_threshold_mode(mut self, mode: ThresholdMode) -> Self {
        self.threshold_mode = mode;
        self
    }

    /// Evalúa un candidato individual sobre la partición In-Sample y calcula las métricas Sylva EVO
    pub fn evaluate_candidate(
        &self,
        candidate: &SylvaTreeCandidateConfig,
        trial_idx: usize,
        generation_idx: usize,
        klines: &[Kline],
        datasets_by_h: &std::collections::HashMap<usize, TabularDataset>,
        is_start_idx: usize,
        oos_start_idx: usize,
        mask: Option<&FeatureMask>,
        color: String,
    ) -> Result<SylvaTreeTrial, Box<dyn Error + Send + Sync>> {
        let is_dataset = datasets_by_h
            .get(&candidate.target_horizon)
            .or_else(|| datasets_by_h.values().next())
            .expect("Dataset tabular no disponible para el horizonte");

        let mut cfg_nom = self.backtest_config_nom.clone();
        cfg_nom.max_holding_bars = candidate.target_horizon;
        let mut cfg_pct = self.backtest_config_pct.clone();
        cfg_pct.max_holding_bars = candidate.target_horizon;

        let sim_nom = BacktestSimulator::new(cfg_nom.clone());
        let sim_pct = BacktestSimulator::new(cfg_pct.clone());
        let tuner = ThresholdTuner::new(cfg_nom);

        let (rep_nom, rep_pct, tune_long, tune_short, fitness, mse, mda, rank_ic) = match candidate.model_type {
            SylvaTreeModelType::OnlineConventional => {
                let online_cfg = candidate.to_online_config();
                let trainer = OnlineGbdtTrainer::new(online_cfg);
                let trained_model = trainer.fit_stream(is_dataset, mask)?;

                if let Some(ref gpu) = self.gpu_engine {
                    // --- MODO GPU ACELERADO (VRAM & Compute Shaders) ---
                    let preds = gpu.predict_dataset(&trained_model, is_dataset)
                        .unwrap_or_else(|_| trained_model.predict_dataset(is_dataset));
                    let targets: Vec<f32> = is_dataset.samples.iter().map(|s| s.target).collect();
                    let (eval_mse, eval_mda, eval_ic) = evaluate_predictions(&preds, &targets);

                    let alphas_arc = Arc::new(preds);
                    let effective_start = is_start_idx.max(self.rolling_window);
                    let vol_ratios_vec: Vec<f32> = (effective_start..effective_start + alphas_arc.len())
                        .map(|idx| crate::engine::dynamic_threshold::calculate_volatility_factor(idx, Some(&self.cached)))
                        .collect();
                    let vol_ratios_arc = Arc::new(vol_ratios_vec);

                    let vr_for_tune = Arc::clone(&vol_ratios_arc);
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
                            ).with_vol_ratios(Arc::clone(&vr_for_tune))
                        },
                    );

                    let eval_end = is_start_idx + alphas_arc.len();
                    let mut model_nom = PrecomputedAlphaModel::new(
                        Arc::clone(&alphas_arc),
                        effective_start,
                        tune_res.best_threshold_long,
                        tune_res.best_threshold_short,
                    ).with_vol_ratios(Arc::clone(&vol_ratios_arc));
                    let mut model_pct = PrecomputedAlphaModel::new(
                        Arc::clone(&alphas_arc),
                        effective_start,
                        tune_res.best_threshold_long,
                        tune_res.best_threshold_short,
                    ).with_vol_ratios(Arc::clone(&vol_ratios_arc));

                    let r_nom = sim_nom.run_range(&mut model_nom, klines, is_start_idx, eval_end);
                    let r_pct = sim_pct.run_range(&mut model_pct, klines, is_start_idx, eval_end);

                    (r_nom, r_pct, tune_res.best_threshold_long, tune_res.best_threshold_short, tune_res.best_fitness, eval_mse, eval_mda, eval_ic)
                } else {
                    // --- MODO CPU PURO DETERMINISTA (32 Hilos Rayon - Motor Nativo) ---
                    let cached_clone = Arc::clone(&self.cached);
                    let tune_res = tuner.tune(
                        klines,
                        is_start_idx,
                        oos_start_idx,
                        &[0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012, 0.0018, 0.0025, 0.0035],
                        &[-0.0001, -0.0002, -0.0003, -0.0005, -0.0008, -0.0012, -0.0018, -0.0025, -0.0035],
                        |tl, ts| {
                            let mut m = OnlineGbdtEquationModel::new(
                                trained_model.clone(),
                                tl,
                                ts,
                                self.rolling_window,
                            ).with_cached_indicators(Arc::clone(&cached_clone));
                            if let Some(m_mask) = mask {
                                m = m.with_mask((*m_mask).clone());
                            }
                            m
                        },
                    );

                    let mut model_nom = OnlineGbdtEquationModel::new(
                        trained_model.clone(),
                        tune_res.best_threshold_long,
                        tune_res.best_threshold_short,
                        self.rolling_window,
                    ).with_cached_indicators(Arc::clone(&self.cached));
                    let mut model_pct = OnlineGbdtEquationModel::new(
                        trained_model.clone(),
                        tune_res.best_threshold_long,
                        tune_res.best_threshold_short,
                        self.rolling_window,
                    ).with_cached_indicators(Arc::clone(&self.cached));
                    if let Some(m_mask) = mask {
                        model_nom = model_nom.with_mask((*m_mask).clone());
                        model_pct = model_pct.with_mask((*m_mask).clone());
                    }

                    let r_nom = sim_nom.run_range(&mut model_nom, klines, is_start_idx, oos_start_idx);
                    let r_pct = sim_pct.run_range(&mut model_pct, klines, is_start_idx, oos_start_idx);

                    let preds = trained_model.predict_dataset(is_dataset);
                    let targets: Vec<f32> = is_dataset.samples.iter().map(|s| s.target).collect();
                    let (eval_mse, eval_mda, eval_ic) = evaluate_predictions(&preds, &targets);

                    (r_nom, r_pct, tune_res.best_threshold_long, tune_res.best_threshold_short, tune_res.best_fitness, eval_mse, eval_mda, eval_ic)
                }
            }
            SylvaTreeModelType::BayesianOnline => {
                let bayes_cfg = candidate.to_bayesian_config();
                let trainer = BayesianOnlineGbdtTrainer::new(bayes_cfg);
                let trained_model = trainer.fit_stream(is_dataset, mask)?;

                if let Some(ref _gpu) = self.gpu_engine {
                    // --- MODO GPU ACELERADO (VRAM & Compute Shaders) ---
                    let preds = trained_model.predict_dataset(is_dataset);
                    let pred_means: Vec<f32> = preds.iter().map(|p| p.mean).collect();
                    let targets: Vec<f32> = is_dataset.samples.iter().map(|s| s.target).collect();
                    let (eval_mse, eval_mda, eval_ic) = evaluate_predictions(&pred_means, &targets);

                    let alphas_arc = Arc::new(pred_means);
                    let effective_start = is_start_idx.max(self.rolling_window);
                    let vol_ratios_vec: Vec<f32> = (effective_start..effective_start + alphas_arc.len())
                        .map(|idx| crate::engine::dynamic_threshold::calculate_volatility_factor(idx, Some(&self.cached)))
                        .collect();
                    let vol_ratios_arc = Arc::new(vol_ratios_vec);

                    let vr_for_tune = Arc::clone(&vol_ratios_arc);
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
                            ).with_vol_ratios(Arc::clone(&vr_for_tune))
                        },
                    );

                    let eval_end = is_start_idx + alphas_arc.len();
                    let mut model_nom = PrecomputedAlphaModel::new(
                        Arc::clone(&alphas_arc),
                        effective_start,
                        tune_res.best_threshold_long,
                        tune_res.best_threshold_short,
                    ).with_vol_ratios(Arc::clone(&vol_ratios_arc));
                    let mut model_pct = PrecomputedAlphaModel::new(
                        Arc::clone(&alphas_arc),
                        effective_start,
                        tune_res.best_threshold_long,
                        tune_res.best_threshold_short,
                    ).with_vol_ratios(Arc::clone(&vol_ratios_arc));

                    let r_nom = sim_nom.run_range(&mut model_nom, klines, is_start_idx, eval_end);
                    let r_pct = sim_pct.run_range(&mut model_pct, klines, is_start_idx, eval_end);

                    (r_nom, r_pct, tune_res.best_threshold_long, tune_res.best_threshold_short, tune_res.best_fitness, eval_mse, eval_mda, eval_ic)
                } else {
                    // --- MODO CPU PURO DETERMINISTA (32 Hilos Rayon - Motor Nativo) ---
                    let cached_clone = Arc::clone(&self.cached);
                    let tune_res = tuner.tune(
                        klines,
                        is_start_idx,
                        oos_start_idx,
                        &[0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012, 0.0018, 0.0025, 0.0035],
                        &[-0.0001, -0.0002, -0.0003, -0.0005, -0.0008, -0.0012, -0.0018, -0.0025, -0.0035],
                        |tl, ts| {
                            let mut m = BayesianGbdtEquationModel::new(
                                trained_model.clone(),
                                tl,
                                ts,
                                self.rolling_window,
                            ).with_cached_indicators(Arc::clone(&cached_clone));
                            if let Some(m_mask) = mask {
                                m = m.with_mask((*m_mask).clone());
                            }
                            m
                        },
                    );

                    let mut model_nom = BayesianGbdtEquationModel::new(
                        trained_model.clone(),
                        tune_res.best_threshold_long,
                        tune_res.best_threshold_short,
                        self.rolling_window,
                    ).with_cached_indicators(Arc::clone(&self.cached));
                    let mut model_pct = BayesianGbdtEquationModel::new(
                        trained_model.clone(),
                        tune_res.best_threshold_long,
                        tune_res.best_threshold_short,
                        self.rolling_window,
                    ).with_cached_indicators(Arc::clone(&self.cached));
                    if let Some(m_mask) = mask {
                        model_nom = model_nom.with_mask((*m_mask).clone());
                        model_pct = model_pct.with_mask((*m_mask).clone());
                    }

                    let r_nom = sim_nom.run_range(&mut model_nom, klines, is_start_idx, oos_start_idx);
                    let r_pct = sim_pct.run_range(&mut model_pct, klines, is_start_idx, oos_start_idx);

                    let preds = trained_model.predict_dataset(is_dataset);
                    let pred_means: Vec<f32> = preds.iter().map(|p| p.mean).collect();
                    let targets: Vec<f32> = is_dataset.samples.iter().map(|s| s.target).collect();
                    let (eval_mse, eval_mda, eval_ic) = evaluate_predictions(&pred_means, &targets);

                    (r_nom, r_pct, tune_res.best_threshold_long, tune_res.best_threshold_short, tune_res.best_fitness, eval_mse, eval_mda, eval_ic)
                }
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
            target_horizon: candidate.target_horizon,
            cv_mse: mse,
            cv_mda: mda * 100.0,
            cv_ic: rank_ic,
            best_thr_long: tune_long,
            best_thr_short: tune_short,
            is_sylva_fitness: fitness,
            rank_slope: 0,
            rank_cap_dd: 0,
            rank_r2: 0,
            rank_smoothness: 0,
            weighted_avg_rank: 0.0,
            sylva_rank_fitness: 0.0,
            raw_slope_ratio,
            raw_cap_dd_ratio,
            raw_r2_score,
            raw_smoothness_score,
            report_nom: light_nom.clone(),
            report_pct: light_pct.clone(),
            color,
        };

        Ok(SylvaTreeTrial {
            trial_idx,
            generation_idx,
            config: candidate.clone(),
            best_threshold_long: tune_long,
            best_threshold_short: tune_short,
            is_sylva_fitness: fitness,
            report_nom: light_nom,
            report_pct: light_pct,
            candidate_report: cand_rep,
        })
    }
}
