use crate::data::db::Kline;
use crate::dashboard::grid_dashboard::{generate_gbdt_grid_dashboard, GbdtGridCandidateReport, GbdtGridDashboardSummary};
use crate::engine::simulator::BacktestSimulator;
use crate::engine::threshold_tuner::ThresholdTuner;
use crate::engine::types::BacktestConfig;
use crate::features::FeatureMask;
use crate::metrics::autotuning::{calculate_capital_max_dd_ratio, calculate_slope_ratio, evaluate_equity_linearity};
use crate::metrics::ranking::compute_multicriteria_rankings;
use crate::metrics::smoothness::calculate_equity_smoothness;
use crate::trees::bayesian::gp::acquisition::{evaluate_acquisition, AcquisitionType};
use crate::trees::bayesian::gp::gaussian_process::GaussianProcessRegressor;
use crate::trees::bayesian::gp::kernel::Matern52Kernel;
use crate::trees::bayesian::gp::sampler::{ContinuousBound, LatinHypercubeSampler};
use crate::trees::bayesian::model::{BayesianOnlineGbdtModel, BayesianOnlineGbdtTrainer};
use crate::trees::bayesian::model_wrapper::BayesianGbdtEquationModel;
use crate::trees::bayesian::types::BayesianTreeConfig;
use crate::trees::dataset::TabularDataset;
use rayon::prelude::*;
use std::error::Error;

/// Configuración de la Optimización Bayesiana para Hiperparámetros de Árboles
#[derive(Clone, Debug)]
pub struct BayesianTreeOptimizerConfig {
    pub initial_samples: usize,
    pub total_iterations: usize,
    pub candidates_per_iter: usize,
    pub exploration_xi: f64,
    pub rolling_window: usize,
}

impl Default for BayesianTreeOptimizerConfig {
    fn default() -> Self {
        Self {
            initial_samples: 12,
            total_iterations: 24,
            candidates_per_iter: 500,
            exploration_xi: 0.01,
            rolling_window: 100,
        }
    }
}

pub struct BayesianTreeOptimizationResult {
    pub champion_model: BayesianOnlineGbdtModel,
    pub champion_config: BayesianTreeConfig,
    pub champion_thr_long: f32,
    pub champion_thr_short: f32,
    pub candidate_reports: Vec<GbdtGridCandidateReport>,
    pub total_evaluated: usize,
}

/// Optimizador Bayesiano mediante Procesos Gaussianos para Árboles Online
pub struct BayesianTreeOptimizer {
    pub config: BayesianTreeOptimizerConfig,
    pub backtest_config_nom: BacktestConfig,
    pub backtest_config_pct: BacktestConfig,
    pub cached: std::sync::Arc<crate::features::CachedIndicators>,
}

impl BayesianTreeOptimizer {
    pub fn new(
        config: BayesianTreeOptimizerConfig,
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

    /// Ejecuta el ciclo de Optimización Bayesiana (GP Matérn 5/2 + Expected Improvement)
    pub fn optimize(
        &self,
        klines: &[Kline],
        is_dataset: &TabularDataset,
        is_start_idx: usize,
        oos_start_idx: usize,
        tf: &str,
        mask: Option<&FeatureMask>,
    ) -> Result<BayesianTreeOptimizationResult, Box<dyn Error + Send + Sync>> {
        println!(
            "\n🔬 INICIANDO OPTIMIZACIÓN BAYESIANA DE ÁRBOLES ONLINE (GP Matérn 5/2 + Expected Improvement)..."
        );
        println!(
            "  Muestras Iniciales Latin Hypercube: {} | Iteraciones Bayesianas: {} | Candidatos/Paso: {}\n",
            self.config.initial_samples, self.config.total_iterations, self.config.candidates_per_iter
        );

        // 1. Definir hiperespacio de búsqueda continuo
        // [0]: n_trees (15..45)
        // [1]: max_depth (2..5)
        // [2]: learning_rate (0.01..0.08)
        // [3]: decay_factor (0.980..0.9995)
        // [4]: prior_precision (0.2..4.0)
        // [5]: uncertainty_penalty_kappa (0.0..2.0)
        let bounds = vec![
            ContinuousBound { min: 15.0, max: 45.0 },
            ContinuousBound { min: 2.0, max: 5.0 },
            ContinuousBound { min: 0.01, max: 0.08 },
            ContinuousBound { min: 0.980, max: 0.9995 },
            ContinuousBound { min: 0.2, max: 4.0 },
            ContinuousBound { min: 0.0, max: 2.0 },
        ];

        let initial_points = LatinHypercubeSampler::sample(&bounds, self.config.initial_samples, 42);

        let sim_nom = BacktestSimulator::new(self.backtest_config_nom.clone());
        let sim_pct = BacktestSimulator::new(self.backtest_config_pct.clone());
        let tuner = ThresholdTuner::new(self.backtest_config_nom.clone());

        let colors = [
            "#10b981", "#3b82f6", "#8b5cf6", "#f59e0b", "#ec4899",
            "#06b6d4", "#f97316", "#14b8a6", "#6366f1", "#a855f7",
            "#84cc16", "#ef4444", "#eab308", "#0ea5e9", "#d946ef",
            "#22c55e", "#64748b", "#fb7185", "#38bdf8", "#4ade80",
        ];

        let mut x_history: Vec<Vec<f64>> = Vec::new();
        let mut y_history: Vec<f64> = Vec::new();
        let mut evaluated_results: Vec<(GbdtGridCandidateReport, BayesianOnlineGbdtModel, BayesianTreeConfig)> = Vec::new();

        let cached_ref = std::sync::Arc::clone(&self.cached);

        let evaluate_point = |point: &[f64], idx: usize| -> Result<(GbdtGridCandidateReport, BayesianOnlineGbdtModel, BayesianTreeConfig), Box<dyn Error + Send + Sync>> {
            let n_trees = point[0].round() as usize;
            let max_depth = point[1].round() as usize;
            let learning_rate = point[2] as f32;
            let decay_factor = point[3] as f32;
            let prior_precision = point[4] as f32;
            let uncertainty_penalty_kappa = point[5] as f32;

            let b_cfg = BayesianTreeConfig {
                n_trees,
                max_depth,
                learning_rate,
                decay_factor,
                prior_mean: 0.0,
                prior_precision,
                prior_shape: 2.5,
                prior_scale: 1.0,
                min_samples_leaf: 20,
                uncertainty_penalty_kappa,
                use_thompson_sampling: false,
                confidence_level: 0.95,
                colsample_bytree: 0.8,
            };

            let trainer = BayesianOnlineGbdtTrainer::new(b_cfg.clone());
            let trained_model = trainer.fit_stream(is_dataset, mask)?;

            let tune_res = tuner.tune(
                klines,
                is_start_idx,
                oos_start_idx,
                &[0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012, 0.0018, 0.0025, 0.0035],
                &[-0.0001, -0.0002, -0.0003, -0.0005, -0.0008, -0.0012, -0.0018, -0.0025, -0.0035],
                |thr_l, thr_s| {
                    let mut m = BayesianGbdtEquationModel::new(
                        trained_model.clone(),
                        thr_l,
                        thr_s,
                        100,
                    ).with_cached_indicators(std::sync::Arc::clone(&cached_ref));
                    if let Some(fmask) = mask {
                        m = m.with_mask(fmask.clone());
                    }
                    m
                },
            );

            let mut model_nom = BayesianGbdtEquationModel::new(
                trained_model.clone(),
                tune_res.best_threshold_long,
                tune_res.best_threshold_short,
                100,
            ).with_cached_indicators(std::sync::Arc::clone(&cached_ref));
            let mut model_pct = BayesianGbdtEquationModel::new(
                trained_model.clone(),
                tune_res.best_threshold_long,
                tune_res.best_threshold_short,
                100,
            ).with_cached_indicators(std::sync::Arc::clone(&cached_ref));

            if let Some(fmask) = mask {
                model_nom = model_nom.with_mask(fmask.clone());
                model_pct = model_pct.with_mask(fmask.clone());
            }

            let rep_nom = sim_nom.run_range(&mut model_nom, klines, is_start_idx, klines.len());
            let rep_pct = sim_pct.run_range(&mut model_pct, klines, is_start_idx, klines.len());

            let preds = trained_model.predict_dataset(is_dataset);
            let pred_means: Vec<f32> = preds.iter().map(|p| p.mean).collect();
            let targets: Vec<f32> = is_dataset.samples.iter().map(|s| s.target).collect();
            let (mse, mda, rank_ic) = crate::trees::batch::purged_cv::evaluate_predictions(&pred_means, &targets);

            let color = colors[idx % colors.len()].to_string();

            let report = GbdtGridCandidateReport {
                max_depth,
                min_samples_leaf: 20,
                n_trees,
                target_horizon: 1,
                cv_mse: mse,
                cv_mda: mda * 100.0,
                cv_ic: rank_ic,
                best_thr_long: tune_res.best_threshold_long,
                best_thr_short: tune_res.best_threshold_short,
                is_sylva_fitness: tune_res.best_fitness,
                rank_slope: 0,
                rank_cap_dd: 0,
                rank_r2: 0,
                rank_smoothness: 0,
                weighted_avg_rank: 0.0,
                sylva_rank_fitness: 0.0,
                raw_slope_ratio: 0.0,
                raw_cap_dd_ratio: 0.0,
                raw_r2_score: 0.0,
                raw_smoothness_score: 0.0,
                report_nom: rep_nom,
                report_pct: rep_pct,
                color,
            };

            Ok((report, trained_model, b_cfg))
        };

        // 2. Evaluar puntos iniciales Latin Hypercube en paralelo
        let num_threads = rayon::current_num_threads();
        println!("--- FASE 1: Exploración Inicial Latin Hypercube Determinista en Paralelo ({} Puntos en {} Hilos CPU) ---", initial_points.len(), num_threads);

        let indexed_points: Vec<(usize, Vec<f64>)> = initial_points
            .iter()
            .enumerate()
            .map(|(i, pt)| (i, pt.clone()))
            .collect();

        let lhc_results: Vec<(GbdtGridCandidateReport, BayesianOnlineGbdtModel, BayesianTreeConfig)> = indexed_points
            .into_par_iter()
            .map(|(i, pt)| evaluate_point(&pt, i))
            .collect::<Result<Vec<_>, _>>()?;

        for (i, (rep, mod_obj, cfg_obj)) in lhc_results.into_iter().enumerate() {
            let pt = &initial_points[i];
            let fitness = rep.is_sylva_fitness;
            println!(
                "  • LHC [{:>2}/{}]: Trees: {:>2} | Depth: {} | LR: {:.3} | Decay: {:.4} | Kappa: {:.2} ➔ Fitness: {:>6.2} | Profit: +${:.2}",
                i + 1, initial_points.len(), cfg_obj.n_trees, cfg_obj.max_depth, cfg_obj.learning_rate, cfg_obj.decay_factor, cfg_obj.uncertainty_penalty_kappa, fitness, rep.report_nom.net_profit
            );

            x_history.push(pt.to_vec());
            y_history.push(fitness);
            evaluated_results.push((rep, mod_obj, cfg_obj));
        }

        // 3. Bucle Bayesiano Activo: Actualizar GP y optimizar Expected Improvement
        println!("\n--- FASE 2: Optimización Activa con Gaussian Process Matérn 5/2 ---");
        let kernel = Box::new(Matern52Kernel::with_ard(
            vec![10.0, 1.0, 0.02, 0.005, 1.0, 0.5],
            1.0,
        ));
        let mut gp = GaussianProcessRegressor::new(kernel, 1e-4);

        for iter in 0..self.config.total_iterations {
            gp.fit(x_history.clone(), y_history.clone())?;

            let mut best_y = f64::MIN;
            for &y in &y_history {
                if y > best_y {
                    best_y = y;
                }
            }

            // Muestrear candidatos densos
            let candidate_points = LatinHypercubeSampler::generate_acquisition_candidates(
                &bounds,
                &x_history,
                self.config.candidates_per_iter,
                0.15,
                (iter + 100) as u64,
            );

            let best_candidate = candidate_points
                .par_iter()
                .map(|cand| {
                    let (mu, var) = gp.predict(cand);
                    let acq = evaluate_acquisition(
                        AcquisitionType::ExpectedImprovement,
                        mu,
                        var,
                        best_y,
                        self.config.exploration_xi,
                        2.0,
                    );
                    (acq, cand)
                })
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(acq, cand)| (acq, cand.clone()))
                .unwrap_or((0.0, candidate_points[0].clone()));

            let best_acq = best_candidate.0;
            let best_point = best_candidate.1;

            let curr_idx = evaluated_results.len();
            let (rep, mod_obj, cfg_obj) = evaluate_point(&best_point, curr_idx)?;
            let fitness = rep.is_sylva_fitness;

            println!(
                "  • BO Paso [{:>2}/{}]: Trees: {:>2} | Depth: {} | LR: {:.3} | Decay: {:.4} | EI: {:.5} ➔ Fitness: {:>6.2} | Retorno: +${:.2}",
                iter + 1, self.config.total_iterations, cfg_obj.n_trees, cfg_obj.max_depth, cfg_obj.learning_rate, cfg_obj.decay_factor, best_acq, fitness, rep.report_nom.net_profit
            );

            x_history.push(best_point);
            y_history.push(fitness);
            evaluated_results.push((rep, mod_obj, cfg_obj));
        }

        // 4. Ranking Multicriterio Sylva EVO
        let cand_reports: Vec<GbdtGridCandidateReport> = evaluated_results.iter().map(|(c, _, _)| c.clone()).collect();
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

        for (cand, _, _) in &mut evaluated_results {
            let key = format!("{}-{}-{}-{:.4}", cand.max_depth, cand.min_samples_leaf, cand.n_trees, cand.best_thr_long);
            if let Some(r) = rankings.get(&key) {
                cand.rank_slope = r.rank_slope;
                cand.rank_cap_dd = r.rank_cap_dd;
                cand.rank_r2 = r.rank_r2;
                cand.rank_smoothness = r.rank_smoothness;
                cand.weighted_avg_rank = r.weighted_avg_rank;
                cand.sylva_rank_fitness = r.sylva_rank_fitness;
            }
        }

        evaluated_results.sort_by(|a, b| a.0.weighted_avg_rank.partial_cmp(&b.0.weighted_avg_rank).unwrap_or(std::cmp::Ordering::Equal));

        println!("\n=======================================================================================================================================");
        println!("                     🏆 LEADERBOARD DE OPTIMIZACIÓN BAYESIANA DE ÁRBOLES ONLINE (PUESTOS SYLVA EVO)                                   ");
        println!("=======================================================================================================================================");
        println!("  {:<5} | {:<5} | {:<7} | {:<6} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>12} | {:>15}",
            "Puesto", "Depth", "Trees", "Decay", "Rank Pond.", "Rank R²(2x)", "Rank Slope", "Rank Cap/DD", "Rank Smooth", "Net Profit", "Umbrales (L/S)"
        );
        println!("---------------------------------------------------------------------------------------------------------------------------------------");

        for (rank, (cand, _, cfg)) in evaluated_results.iter().take(10).enumerate() {
            let medal = match rank {
                0 => "🥇 1º",
                1 => "🥈 2º",
                2 => "🥉 3º",
                _ => "   ",
            };
            println!(
                "  {:<2} {} | {:>5} | {:>7} | {:>6.4} | {:>10.2} | {:>10} | {:>10} | {:>10} | {:>10} | ${:>+10.2} | {:+.4}/{:+.4}",
                rank + 1, medal, cand.max_depth, cand.n_trees, cfg.decay_factor, cand.weighted_avg_rank, format!("#{}", cand.rank_r2), format!("#{}", cand.rank_slope), format!("#{}", cand.rank_cap_dd), format!("#{}", cand.rank_smoothness), cand.report_nom.net_profit, cand.best_thr_long, cand.best_thr_short
            );
        }
        println!("=======================================================================================================================================");

        let summary_grid = GbdtGridDashboardSummary {
            tf: tf.to_string(),
            is_start_idx,
            oos_start_idx,
            initial_capital: self.backtest_config_nom.initial_capital,
            candidates: evaluated_results.iter().map(|(c, _, _)| c.clone()).collect(),
        };
        let _ = generate_gbdt_grid_dashboard(&summary_grid, klines);

        let (champion_rep, champion_mod, champion_cfg) = evaluated_results.remove(0);

        Ok(BayesianTreeOptimizationResult {
            champion_model: champion_mod,
            champion_config: champion_cfg,
            champion_thr_long: champion_rep.best_thr_long,
            champion_thr_short: champion_rep.best_thr_short,
            candidate_reports: vec![champion_rep],
            total_evaluated: evaluated_results.len() + 1,
        })
    }
}
