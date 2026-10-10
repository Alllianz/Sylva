use crate::data::db::Kline;
use crate::dashboard::grid_dashboard::{generate_gbdt_grid_dashboard, GbdtGridCandidateReport, GbdtGridDashboardSummary};
use crate::engine::simulator::BacktestSimulator;
use crate::engine::types::BacktestConfig;
use crate::features::CachedIndicators;
use crate::features::FeatureMask;
use crate::metrics::ranking::compute_multicriteria_rankings;
use crate::trees::sylva_tuner::evaluator::SylvaTreeEvaluator;
use crate::trees::sylva_tuner::sampler::SylvaTreeSampler;
use crate::trees::sylva_tuner::types::{
    SylvaTreeAutoTuningConfig, SylvaTreeAutoTuningResult, SylvaTreeCandidateConfig,
    SylvaTreeModelType, SylvaTreeTrial,
};
use crate::trees::dataset::TabularDataset;
use crate::trees::gpu::GpuTreeEngine;
use crate::trees::online::gbdt::OnlineGbdtTrainer;
use crate::trees::online::model_wrapper::OnlineGbdtEquationModel;
use rayon::prelude::*;
use std::error::Error;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Optimizador Evolutivo Automático para Árboles (Sylva EVO Tree Engine)
pub struct SylvaTreeOptimizer {
    pub config: SylvaTreeAutoTuningConfig,
    pub backtest_config_nom: BacktestConfig,
    pub backtest_config_pct: BacktestConfig,
    pub cached: Arc<CachedIndicators>,
    pub gpu_engine: Option<Arc<GpuTreeEngine>>,
}

pub type AstroTreeOptimizer = SylvaTreeOptimizer;

impl SylvaTreeOptimizer {
    pub fn new(
        config: SylvaTreeAutoTuningConfig,
        backtest_config_nom: BacktestConfig,
        backtest_config_pct: BacktestConfig,
        cached: Arc<CachedIndicators>,
    ) -> Self {
        let use_gpu = config.use_gpu;
        Self::new_with_engine(config, backtest_config_nom, backtest_config_pct, cached, use_gpu)
    }

    pub fn new_with_engine(
        config: SylvaTreeAutoTuningConfig,
        backtest_config_nom: BacktestConfig,
        backtest_config_pct: BacktestConfig,
        cached: Arc<CachedIndicators>,
        use_gpu: bool,
    ) -> Self {
        let gpu_engine = if use_gpu {
            match GpuTreeEngine::new() {
                Ok(engine) => {
                    println!("  🚀 Motor GPU inicializado exitosamente: {}", engine.hardware_info());
                    Some(Arc::new(engine))
                }
                Err(err) => {
                    eprintln!("  ⚠️ No se pudo inicializar GPU ({}), recurriendo a CPU multihilo Rayon", err);
                    None
                }
            }
        } else {
            None
        };

        Self {
            config,
            backtest_config_nom,
            backtest_config_pct,
            cached,
            gpu_engine,
        }
    }

    /// Ejecuta el proceso completo de auto-optimización evolutiva determinista acelerada por GPU
    pub fn optimize(
        &self,
        klines: &[Kline],
        datasets_by_h: &std::collections::HashMap<usize, TabularDataset>,
        is_start_idx: usize,
        oos_start_idx: usize,
        tf: &str,
        mask: Option<&FeatureMask>,
    ) -> Result<SylvaTreeAutoTuningResult, Box<dyn Error + Send + Sync>> {
        let num_threads = rayon::current_num_threads();
        let model_label = match self.config.model_type {
            SylvaTreeModelType::OnlineConventional => "Online GBDT (Streaming Hoeffding Trees)",
            SylvaTreeModelType::BayesianOnline => "Bayesian Online Trees (NIG Posterior & Incertidumbre)",
        };

        let feature_label = match mask {
            Some(m) if m.active_count() <= 5 => "Microestructura Pura (F1..F5)",
            Some(m) if m.active_count() <= 30 => "Conjunto Reducido de 30 Features (F1..F30 Mercado Puro)",
            _ => "Red Completa de 32 Variables Cuantitativas",
        };

        let hardware_status = match &self.gpu_engine {
            Some(gpu) => format!("🎮 ACELERACIÓN GPU: {}", gpu.hardware_info()),
            None => format!("💻 CPU MULTIHILO ({} CORES)", num_threads),
        };

        println!("\n=======================================================================================================================================");
        println!("               🧬 AUTO-OPTIMIZACIÓN EVOLUTIVA SYLVA EVO - {}", hardware_status);
        println!("=======================================================================================================================================");
        println!("  • Modelo: {}", model_label);
        println!("  • Espacio de Variables: {}", feature_label);
        println!("  • Horizontes Causales Explorados: {:?}", self.config.candidate_horizons);
        println!("  • Generaciones: {} | Población por Gen: {} | Pruebas Iniciales Exploratorias: {} | Hilos Rayon: {}",
            self.config.generations, self.config.population_size, self.config.initial_exploratory_trials, num_threads
        );
        if let Some(gpu) = &self.gpu_engine {
            println!("  • Pipeline Gráfico: Inferencia Vectorial en VRAM + Calibración de Umbrales Instantánea ({})", gpu.hardware_info());
        }
        println!("---------------------------------------------------------------------------------------------------------------------------------------");

        let mut evaluator = SylvaTreeEvaluator::new(
            self.backtest_config_nom.clone(),
            self.backtest_config_pct.clone(),
            self.config.rolling_window,
            Arc::clone(&self.cached),
        ).with_threshold_mode(self.config.threshold_mode);
        if let Some(ref gpu) = self.gpu_engine {
            evaluator = evaluator.with_gpu_engine(Arc::clone(gpu));
        }

        let evaluator_arc = Arc::new(evaluator);
        let mut sampler = SylvaTreeSampler::new(
            self.config.seed,
            self.config.model_type,
            self.config.candidate_horizons.clone(),
        );

        let mut all_trials: Vec<SylvaTreeTrial> = Vec::new();
        let trial_counter = Arc::new(AtomicUsize::new(0));

        // 1. --- FASE 1: POBLACIÓN INICIAL EXPLORATORIA (LHS & BLOQUES DIVERSIFICADOS) ---
        println!("\n--- FASE 1: Exploración Inicial de Alta Cobertura ({} Candidatos) ---", self.config.initial_exploratory_trials);

        let colors = ["#38bdf8", "#818cf8", "#c084fc", "#f472b6", "#fb7185", "#34d399", "#fbbf24", "#a3e635"];
        let exploratory_candidates: Vec<(usize, SylvaTreeCandidateConfig, String)> = (0..self.config.initial_exploratory_trials)
            .map(|i| {
                let cand = sampler.sample_exploratory();
                let color = colors[i % colors.len()].to_string();
                (i + 1, cand, color)
            })
            .collect();

        let exploratory_trials: Vec<SylvaTreeTrial> = exploratory_candidates
            .into_par_iter()
            .map(|(trial_idx, cand, color)| {
                let eval = Arc::clone(&evaluator_arc);
                let trial_res = eval.evaluate_candidate(
                    &cand,
                    trial_idx,
                    0,
                    klines,
                    datasets_by_h,
                    is_start_idx,
                    oos_start_idx,
                    mask,
                    color,
                ).expect("Error evaluando candidato exploratorio");
                trial_res
            })
            .collect();

        for trial in exploratory_trials {
            println!(
                "  [Init #{:02}/{:02}] H={} | d={} | trees={} | lr={:.3} | decay={:.4} => Fitness={:+.2} | NetNom=+${:.2} ({:+.2}%)",
                trial.trial_idx, self.config.initial_exploratory_trials, trial.config.target_horizon, trial.config.max_depth, trial.config.n_trees, trial.config.learning_rate, trial.config.decay_factor, trial.is_sylva_fitness, trial.report_nom.net_profit, trial.report_nom.total_return_pct
            );
            all_trials.push(trial);
        }

        // 2. --- FASE 2: BUCLE EVOLUTIVO DETERMINISTA PARALELO SYLVA EVO ---
        println!("\n--- FASE 2: Bucle Evolutivo Determinista Sylva EVO ({} Generaciones x {} Individuos) ---", self.config.generations, self.config.population_size);

        for generation in 1..=self.config.generations {
            all_trials.sort_by(|a, b| {
                b.is_sylva_fitness
                    .partial_cmp(&a.is_sylva_fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let top_parents: Vec<SylvaTreeCandidateConfig> = all_trials.iter().take(4).map(|t| t.config.clone()).collect();
            let best_historical_fitness = all_trials[0].is_sylva_fitness;

            println!("\n=== GENERACIÓN {}/{} (Mejor Fitness Histórico: {:+.2}) ===", generation, self.config.generations, best_historical_fitness);

            let gen_candidates: Vec<(usize, SylvaTreeCandidateConfig, String)> = (0..self.config.population_size)
                .map(|ind_idx| {
                    let parent_idx = (sampler.next_u32() as usize) % top_parents.len();
                    let parent = &top_parents[parent_idx];
                    let mut child = if sampler.next_f32() < self.config.mutation_rate {
                        sampler.mutate_parent(parent)
                    } else {
                        parent.clone()
                    };
                    if sampler.next_f32() < 0.15 {
                        child = sampler.sample_exploratory();
                    }
                    let t_idx = trial_counter.fetch_add(1, Ordering::SeqCst) + 1;
                    let color = colors[ind_idx % colors.len()].to_string();
                    (t_idx, child, color)
                })
                .collect();

            let gen_trials: Vec<SylvaTreeTrial> = gen_candidates
                .into_par_iter()
                .map(|(trial_idx, cand, color)| {
                    let eval = Arc::clone(&evaluator_arc);
                    eval.evaluate_candidate(
                        &cand,
                        trial_idx,
                        generation,
                        klines,
                        datasets_by_h,
                        is_start_idx,
                        oos_start_idx,
                        mask,
                        color,
                    ).expect("Error evaluando candidato generacional")
                })
                .collect();

            for (ind_idx, trial) in gen_trials.into_iter().enumerate() {
                let is_new_record = trial.is_sylva_fitness > best_historical_fitness;
                let record_badge = if is_new_record { " 🔥 NUEVO RÉCORD!" } else { "" };
                println!(
                    "    • [G{:02} #{:02}/{:02}] H={} | d={} | trees={} | lr={:.3} | decay={:.4} => Fitness={:+.2} | NetNom=+${:.2}{}",
                    generation, ind_idx + 1, self.config.population_size, trial.config.target_horizon, trial.config.max_depth, trial.config.n_trees, trial.config.learning_rate, trial.config.decay_factor, trial.is_sylva_fitness, trial.report_nom.net_profit, record_badge
                );
                all_trials.push(trial);
            }
        }

        // 3. --- FASE 3: RANKING MULTICRITERIO POR PUESTOS DE SYLVA EVO ---
        println!("\n  📊 Calculando Ranking Multicriterio por Puestos Sylva EVO (R² 2x, Slope Ratio, Cap/DD, Smoothness)...");
        let mut candidate_reports: Vec<GbdtGridCandidateReport> = all_trials.iter().map(|t| t.candidate_report.clone()).collect();

        let rankings = compute_multicriteria_rankings(
            &candidate_reports,
            |c| format!("{}-{}-{}-{}-{:.4}", c.target_horizon, c.max_depth, c.min_samples_leaf, c.n_trees, c.best_thr_long),
            |c| c.raw_slope_ratio,
            |c| c.raw_cap_dd_ratio,
            |c| c.raw_r2_score,
            |c| c.raw_smoothness_score,
        );

        for (idx, trial) in all_trials.iter_mut().enumerate() {
            let key = format!("{}-{}-{}-{}-{:.4}", trial.config.target_horizon, trial.config.max_depth, trial.config.min_samples_leaf, trial.config.n_trees, trial.best_threshold_long);
            if let Some(r_score) = rankings.get(&key) {
                trial.candidate_report.rank_slope = r_score.rank_slope;
                trial.candidate_report.rank_cap_dd = r_score.rank_cap_dd;
                trial.candidate_report.rank_r2 = r_score.rank_r2;
                trial.candidate_report.rank_smoothness = r_score.rank_smoothness;
                trial.candidate_report.weighted_avg_rank = r_score.weighted_avg_rank;
                trial.candidate_report.sylva_rank_fitness = r_score.sylva_rank_fitness;
                candidate_reports[idx] = trial.candidate_report.clone();
            }
        }

        all_trials.sort_by(|a, b| {
            a.candidate_report
                .weighted_avg_rank
                .partial_cmp(&b.candidate_report.weighted_avg_rank)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.trial_idx.cmp(&b.trial_idx))
        });

        let mut champion = all_trials[0].clone();

        // Para el Campeón Absoluto, reconstruir la curva completa de alta resolución y lista de trades
        // con max_holding_bars acoplado 100% a su target_horizon:
        let mut champ_cfg_nom = self.backtest_config_nom.clone();
        champ_cfg_nom.max_holding_bars = champion.config.target_horizon;
        let mut champ_cfg_pct = self.backtest_config_pct.clone();
        champ_cfg_pct.max_holding_bars = champion.config.target_horizon;
        let sim_nom = BacktestSimulator::new(champ_cfg_nom);
        let sim_pct = BacktestSimulator::new(champ_cfg_pct);

        let champ_dataset = datasets_by_h.get(&champion.config.target_horizon)
            .or_else(|| datasets_by_h.values().next())
            .expect("Dataset tabular no encontrado para el campeón");

        let champion_online_cfg = champion.config.to_online_config();
        if let Ok(champ_trainer) = OnlineGbdtTrainer::new(champion_online_cfg).fit_stream(champ_dataset, mask) {
            let mut model_nom = OnlineGbdtEquationModel::new(
                champ_trainer.clone(),
                champion.best_threshold_long,
                champion.best_threshold_short,
                self.config.rolling_window,
            ).with_cached_indicators(Arc::clone(&self.cached));
            let mut model_pct = OnlineGbdtEquationModel::new(
                champ_trainer,
                champion.best_threshold_long,
                champion.best_threshold_short,
                self.config.rolling_window,
            ).with_cached_indicators(Arc::clone(&self.cached));
            if let Some(fmask) = mask {
                model_nom = model_nom.with_mask(fmask.clone());
                model_pct = model_pct.with_mask(fmask.clone());
            }
            champion.report_nom = sim_nom.run_range(&mut model_nom, klines, is_start_idx, klines.len());
            champion.report_pct = sim_pct.run_range(&mut model_pct, klines, is_start_idx, klines.len());
        }

        println!("\n=======================================================================================================================================");
        println!("                             🏆 CAMPEÓN ABSOLUTO SYLVA EVO MAX (MEJOR PUESTO PONDERADO MULTICRITERIO)                                 ");
        println!("=======================================================================================================================================");
        println!("  • Horizonte Causal H: {} velas (predicción forward acoplada a holding de {} barras)", champion.config.target_horizon, champion.config.target_horizon);
        println!("  • Arquitectura: Profundidad={} | Árboles={} | Muestras Mínimas Hoja={}", champion.config.max_depth, champion.config.n_trees, champion.config.min_samples_leaf);
        println!("  • Hiperparámetros: Learning Rate={:.3} | Factor Olvido δ={:.4} | Regularización L2={:.2}", champion.config.learning_rate, champion.config.decay_factor, champion.config.l2_reg);
        println!("  • Umbrales Calibrados: Long={:+.4} | Short={:+.4}", champion.best_threshold_long, champion.best_threshold_short);
        println!("  • Métricas IS: Puesto Ponderado={:.2} (R²: #{} | Slope: #{} | DD: #{} | Smooth: #{})",
            champion.candidate_report.weighted_avg_rank, champion.candidate_report.rank_r2, champion.candidate_report.rank_slope, champion.candidate_report.rank_cap_dd, champion.candidate_report.rank_smoothness
        );
        println!("  • Rendimiento Nominal: +${:.2} ({:+.2}%) | WinRate: {:.1}% | ProfitFactor: {:.2} | MaxDD: {:.2}%",
            champion.report_nom.net_profit, champion.report_nom.total_return_pct, champion.report_nom.win_rate_pct, champion.report_nom.profit_factor, champion.report_nom.max_drawdown_pct
        );
        println!("  • Rendimiento Compuesto: +${:.2} ({:+.2}%) | MaxDD: {:.2}% | Liquidaciones: {}",
            champion.report_pct.net_profit, champion.report_pct.total_return_pct, champion.report_pct.max_drawdown_pct, champion.report_pct.liquidations
        );
        println!("=======================================================================================================================================\n");

        let total_evaluated = all_trials.len();
        let top_candidates: Vec<GbdtGridCandidateReport> = candidate_reports.iter().take(30).cloned().collect();
        let grid_summary = GbdtGridDashboardSummary {
            tf: tf.to_string(),
            is_start_idx,
            oos_start_idx,
            initial_capital: self.backtest_config_nom.initial_capital,
            candidates: top_candidates,
        };
        let _ = generate_gbdt_grid_dashboard(&grid_summary, klines);

        Ok(SylvaTreeAutoTuningResult {
            champion_trial: champion,
            all_trials,
            candidate_reports,
            total_evaluated,
        })
    }
}
