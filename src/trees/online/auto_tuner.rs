use crate::data::db::Kline;
use crate::dashboard::grid_dashboard::{generate_gbdt_grid_dashboard, GbdtGridCandidateReport, GbdtGridDashboardSummary};
use crate::engine::simulator::BacktestSimulator;
use crate::engine::threshold_tuner::ThresholdTuner;
use crate::engine::types::BacktestConfig;
use crate::features::FeatureMask;
use crate::metrics::autotuning::{calculate_capital_max_dd_ratio, calculate_slope_ratio, evaluate_equity_linearity};
use crate::metrics::ranking::compute_multicriteria_rankings;
use crate::metrics::smoothness::calculate_equity_smoothness;
use crate::trees::dataset::TabularDataset;
use crate::trees::online::gbdt::{OnlineGbdtModel, OnlineGbdtTrainer};
use crate::trees::online::model_wrapper::OnlineGbdtEquationModel;
use crate::trees::online::types::OnlineGbdtConfig;
use rayon::prelude::*;
use std::error::Error;

/// Configuración de búsqueda para el optimizador de Online GBDT
#[derive(Clone, Debug)]
pub struct OnlineGbdtAutoTuningConfig {
    pub candidate_configs: Vec<OnlineGbdtConfig>,
    pub rolling_window: usize,
    pub target_horizon: usize,
}

impl Default for OnlineGbdtAutoTuningConfig {
    fn default() -> Self {
        let depths = [2, 3, 4];
        let n_trees_list = [20, 30, 40];
        let lrs = [0.02, 0.04];
        let decays = [0.990, 0.995, 0.999];

        let mut configs = Vec::new();
        for &d in &depths {
            for &trees in &n_trees_list {
                for &lr in &lrs {
                    for &dec in &decays {
                        configs.push(OnlineGbdtConfig {
                            n_trees: trees,
                            max_depth: d,
                            learning_rate: lr,
                            decay_factor: dec,
                            l2_reg: if d >= 4 { 2.0 } else { 1.0 },
                            l1_reg: 0.01,
                            min_samples_split: 25,
                            grace_period: 15,
                            split_confidence: 0.05,
                            tie_threshold: 0.05,
                            colsample_bytree: 0.8,
                            gamma: 0.0001,
                        });
                    }
                }
            }
        }

        Self {
            candidate_configs: configs,
            rolling_window: 100,
            target_horizon: 1,
        }
    }
}

/// Resultado de la optimización del modelo Online
pub struct OnlineGbdtAutoTuningResult {
    pub champion_model: OnlineGbdtModel,
    pub champion_config: OnlineGbdtConfig,
    pub champion_thr_long: f32,
    pub champion_thr_short: f32,
    pub candidate_reports: Vec<GbdtGridCandidateReport>,
    pub total_evaluated: usize,
}

pub struct OnlineGbdtAutoTuner {
    pub config: OnlineGbdtAutoTuningConfig,
    pub backtest_config_nom: BacktestConfig,
    pub backtest_config_pct: BacktestConfig,
    pub cached: std::sync::Arc<crate::features::CachedIndicators>,
}

impl OnlineGbdtAutoTuner {
    pub fn new(
        config: OnlineGbdtAutoTuningConfig,
        backtest_config_nom: BacktestConfig,
        backtest_config_pct: BacktestConfig,
        cached: std::sync::Arc<crate::features::CachedIndicators>,
    ) -> Self {
        Self {
            config,
            backtest_config_nom,
            backtest_config_pct,
            cached,
        }
    }

    /// Ejecuta el proceso de selección y optimización multicriterio
    pub fn auto_optimize(
        &self,
        klines: &[Kline],
        is_dataset: &TabularDataset,
        is_start_idx: usize,
        oos_start_idx: usize,
        tf: &str,
        mask: Option<&FeatureMask>,
    ) -> Result<OnlineGbdtAutoTuningResult, Box<dyn Error + Send + Sync>> {
        let total_cands = self.config.candidate_configs.len();
        let num_threads = rayon::current_num_threads();
        println!(
            "\n🚀 INICIANDO AUTO-OPTIMIZACIÓN ONLINE GBDT (Streaming Hoeffding Trees - {} Candidatos en {} Hilos CPU)...",
            total_cands, num_threads
        );

        let sim_nom = BacktestSimulator::new(self.backtest_config_nom.clone());
        let sim_pct = BacktestSimulator::new(self.backtest_config_pct.clone());
        let tuner = ThresholdTuner::new(self.backtest_config_nom.clone());

        let colors = [
            "#10b981", "#3b82f6", "#8b5cf6", "#f59e0b", "#ec4899",
            "#06b6d4", "#f97316", "#14b8a6", "#6366f1", "#a855f7",
            "#84cc16", "#ef4444", "#eab308", "#0ea5e9", "#d946ef",
            "#22c55e", "#64748b", "#fb7185", "#38bdf8", "#4ade80",
        ];

        let indexed_configs: Vec<(usize, OnlineGbdtConfig, String)> = self.config.candidate_configs
            .iter()
            .enumerate()
            .map(|(idx, cfg)| (idx, cfg.clone(), colors[idx % colors.len()].to_string()))
            .collect();

        let cached_ref = std::sync::Arc::clone(&self.cached);

        let mut candidate_results: Vec<(GbdtGridCandidateReport, OnlineGbdtModel)> = indexed_configs
            .into_par_iter()
            .map(|(_idx, cfg, color)| -> Result<(GbdtGridCandidateReport, OnlineGbdtModel), Box<dyn Error + Send + Sync>> {
                let trainer = OnlineGbdtTrainer::new(cfg.clone());
                let trained_model = trainer.fit_stream(is_dataset, mask)?;

                // Calibrar umbrales óptimos con el wrapper online
                let tune_res = tuner.tune(
                    klines,
                    is_start_idx,
                    oos_start_idx,
                    &[0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012, 0.0018, 0.0025, 0.0035],
                    &[-0.0001, -0.0002, -0.0003, -0.0005, -0.0008, -0.0012, -0.0018, -0.0025, -0.0035],
                    |thr_l, thr_s| {
                        let mut m = OnlineGbdtEquationModel::new(
                            trained_model.clone(),
                            thr_l,
                            thr_s,
                            self.config.rolling_window,
                        ).with_cached_indicators(std::sync::Arc::clone(&cached_ref));
                        if let Some(fmask) = mask {
                            m = m.with_mask(fmask.clone());
                        }
                        m
                    },
                );

                let mut model_nom = OnlineGbdtEquationModel::new(
                    trained_model.clone(),
                    tune_res.best_threshold_long,
                    tune_res.best_threshold_short,
                    self.config.rolling_window,
                ).with_cached_indicators(std::sync::Arc::clone(&cached_ref));
                let mut model_pct = OnlineGbdtEquationModel::new(
                    trained_model.clone(),
                    tune_res.best_threshold_long,
                    tune_res.best_threshold_short,
                    self.config.rolling_window,
                ).with_cached_indicators(std::sync::Arc::clone(&cached_ref));

                if let Some(fmask) = mask {
                    model_nom = model_nom.with_mask(fmask.clone());
                    model_pct = model_pct.with_mask(fmask.clone());
                }

                let rep_nom = sim_nom.run_range(&mut model_nom, klines, is_start_idx, klines.len());
                let rep_pct = sim_pct.run_range(&mut model_pct, klines, is_start_idx, klines.len());

                // Métricas de predicción OOS inicial
                let preds = trained_model.predict_dataset(is_dataset);
                let targets: Vec<f32> = is_dataset.samples.iter().map(|s| s.target).collect();
                let (mse, mda, rank_ic) = crate::trees::batch::purged_cv::evaluate_predictions(&preds, &targets);

                let report = GbdtGridCandidateReport {
                    max_depth: cfg.max_depth,
                    min_samples_leaf: cfg.min_samples_split,
                    n_trees: cfg.n_trees,
                    cv_mse: mse,
                    cv_mda: mda * 100.0,
                    cv_ic: rank_ic,
                    best_thr_long: tune_res.best_threshold_long,
                    best_thr_short: tune_res.best_threshold_short,
                    is_astro_fitness: tune_res.best_fitness,
                    rank_slope: 0,
                    rank_cap_dd: 0,
                    rank_r2: 0,
                    rank_smoothness: 0,
                    weighted_avg_rank: 0.0,
                    astro_rank_fitness: 0.0,
                    raw_slope_ratio: 0.0,
                    raw_cap_dd_ratio: 0.0,
                    raw_r2_score: 0.0,
                    raw_smoothness_score: 0.0,
                    report_nom: rep_nom,
                    report_pct: rep_pct,
                    color,
                };

                Ok((report, trained_model))
            })
            .collect::<Result<Vec<_>, _>>()?;

        for (idx, (report, _)) in candidate_results.iter().enumerate() {
            if (idx + 1) % 5 == 0 || idx == total_cands - 1 {
                println!(
                    "  • Candidato [{:>2}/{}]: Depth: {} | Trees: {:>2} | LR: {:.2} | Decay: {:.3} ➔ Fitness: {:>6.2} | Profit: +${:.2} ({:+.2}%)",
                    idx + 1, total_cands, report.max_depth, report.n_trees, self.config.candidate_configs[idx].learning_rate, self.config.candidate_configs[idx].decay_factor, report.is_astro_fitness, report.report_nom.net_profit, report.report_nom.total_return_pct
                );
            }
        }

        // Ranking Multicriterio Astro EVO
        let cand_reports: Vec<GbdtGridCandidateReport> = candidate_results.iter().map(|(c, _)| c.clone()).collect();
        let rankings = compute_multicriteria_rankings(
            &cand_reports,
            |c| format!("{}-{}-{}-{:.4}", c.max_depth, c.min_samples_leaf, c.n_trees, c.best_thr_long),
            |c| {
                let mid_is = (c.report_nom.equity_curve.len() / 2).max(1);
                let half1 = c.report_nom.equity_curve[mid_is - 1].1 - c.report_nom.initial_capital;
                let half2 = c.report_nom.final_capital - c.report_nom.equity_curve[mid_is - 1].1;
                let s1 = half1 / (mid_is as f64);
                let s2 = half2 / ((c.report_nom.equity_curve.len() - mid_is) as f64);
                calculate_slope_ratio(s1, s2)
            },
            |c| calculate_capital_max_dd_ratio(c.report_nom.final_capital, c.report_nom.max_drawdown_amount),
            |c| evaluate_equity_linearity(&c.report_nom.equity_curve, oos_start_idx, is_start_idx).composite_linearity_score,
            |c| {
                let lin = evaluate_equity_linearity(&c.report_nom.equity_curve, oos_start_idx, is_start_idx);
                let smooth = calculate_equity_smoothness(&c.report_nom.equity_curve, lin.composite_linearity_score);
                smooth.smoothness_score
            },
        );

        for (cand, _) in &mut candidate_results {
            let key = format!("{}-{}-{}-{:.4}", cand.max_depth, cand.min_samples_leaf, cand.n_trees, cand.best_thr_long);
            if let Some(r) = rankings.get(&key) {
                cand.rank_slope = r.rank_slope;
                cand.rank_cap_dd = r.rank_cap_dd;
                cand.rank_r2 = r.rank_r2;
                cand.rank_smoothness = r.rank_smoothness;
                cand.weighted_avg_rank = r.weighted_avg_rank;
                cand.astro_rank_fitness = r.astro_rank_fitness;
            }
        }

        candidate_results.sort_by(|a, b| a.0.weighted_avg_rank.partial_cmp(&b.0.weighted_avg_rank).unwrap_or(std::cmp::Ordering::Equal));

        println!("\n=======================================================================================================================================");
        println!("                         🏆 LEADERBOARD DE AUTO-OPTIMIZACIÓN ONLINE GBDT (STREAMING TREES - PUESTOS ASTRO EVO)                          ");
        println!("=======================================================================================================================================");
        println!("  {:<5} | {:<5} | {:<10} | {:<5} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12} | {:>15}",
            "Puesto", "Depth", "Grace/Split", "Trees", "Rank Pond.", "Rank R²(2x)", "Rank Slope", "Rank Cap/DD", "Rank Smooth", "Net Profit", "Umbrales (L/S)"
        );
        println!("---------------------------------------------------------------------------------------------------------------------------------------");

        for (rank, (cand, _)) in candidate_results.iter().take(10).enumerate() {
            let medal = match rank {
                0 => "🥇 1º",
                1 => "🥈 2º",
                2 => "🥉 3º",
                _ => "   ",
            };
            println!(
                "  {:<2} {} | {:>5} | {:>10} | {:>5} | {:>10.2} | {:>10} | {:>10} | {:>10} | {:>10} | ${:>+10.2} | {:+.4}/{:+.4}",
                rank + 1, medal, cand.max_depth, cand.min_samples_leaf, cand.n_trees, cand.weighted_avg_rank, format!("#{}", cand.rank_r2), format!("#{}", cand.rank_slope), format!("#{}", cand.rank_cap_dd), format!("#{}", cand.rank_smoothness), cand.report_nom.net_profit, cand.best_thr_long, cand.best_thr_short
            );
        }
        println!("=======================================================================================================================================");

        let summary_grid = GbdtGridDashboardSummary {
            tf: tf.to_string(),
            is_start_idx,
            oos_start_idx,
            initial_capital: self.backtest_config_nom.initial_capital,
            candidates: candidate_results.iter().map(|(c, _)| c.clone()).collect(),
        };
        let _ = generate_gbdt_grid_dashboard(&summary_grid, klines);

        let (champion_rep, champion_mod) = candidate_results.remove(0);
        let champ_cfg = self.config.candidate_configs.iter().find(|c| c.max_depth == champion_rep.max_depth && c.n_trees == champion_rep.n_trees).cloned().unwrap_or_default();

        Ok(OnlineGbdtAutoTuningResult {
            champion_model: champion_mod,
            champion_config: champ_cfg,
            champion_thr_long: champion_rep.best_thr_long,
            champion_thr_short: champion_rep.best_thr_short,
            candidate_reports: vec![champion_rep],
            total_evaluated: total_cands,
        })
    }
}
