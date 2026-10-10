use crate::dashboard::{
    generate_dual_dashboard, generate_gbdt_grid_dashboard, GbdtGridCandidateReport,
    GbdtGridDashboardSummary,
};
use crate::data::db::{get_all_klines_closed_only, Connection};
use crate::engine::dynamic_threshold::ThresholdMode;
use crate::engine::partition::get_dataset_partition_indices;
use crate::engine::simulator::BacktestSimulator;
use crate::engine::threshold_tuner::ThresholdTuner;
use crate::engine::types::BacktestConfig;
use crate::features::{CachedIndicators, FeatureMask, TOTAL_FEATURES};
use crate::metrics::terminal_report::print_backtest_summary;
use crate::trees::{
    evaluate_predictions, generate_bayesian_tree_dashboard,
    generate_online_gbdt_dashboard, generate_sylva_tree_tuning_dashboard,
    BayesianGbdtEquationModel, BayesianOnlineGbdtTrainer, BayesianTreeConfig,
    BayesianTreeOptimizer, BayesianTreeOptimizerConfig, ElasticNetConfig,
    ElasticNetEquationModel, ElasticNetTrainer, GbdtConfig, GbdtEquationModel,
    GbdtTrainer, OnlineGbdtAutoTuner, OnlineGbdtAutoTuningConfig, OnlineGbdtConfig,
    OnlineGbdtEquationModel, OnlineGbdtTrainer, PurgedCrossValidator,
    SylvaTreeAutoTuningConfig, SylvaTreeModelType, SylvaTreeOptimizer, TabularDataset,
};
use crate::ui::prompts::{parse_usize_list, prompt_leverage_for_analysis, prompt_timeframe_for_analysis};
use std::error::Error;
use std::io::{self, Write};
use std::sync::Arc;

pub fn run_tree_session(
    conn: &Connection,
    custom_mask_and_label: Option<(FeatureMask, &'static str)>,
    model_choice: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let is_conventional_online = model_choice == "4" || model_choice == "1" || model_choice.is_empty();
    let is_bayesian_online = model_choice == "5" || model_choice == "2";
    let is_sylva_auto = model_choice == "6" || model_choice == "3";
    let is_grid_auto = model_choice == "7" || model_choice == "4";
    let is_batch_grid = model_choice == "8" || model_choice == "5";
    let is_elastic_net = model_choice == "9" || model_choice == "6";

    let (feature_mask, feature_label) = match custom_mask_and_label {
        Some((mask, label)) => {
            println!("\n  ✂️ Modo GBDT Allianz Activado: Poda selectiva de 19 variables aplicada.");
            println!("     • Variables podadas (19): F1, F2, F3, F7, F8, F12, F15, F19, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30, F32");
            println!("     • Variables activas (13): F4, F5, F6, F9, F10, F11, F13, F14, F16, F17, F18, F20, F31");
            (Some(mask), label)
        }
        None => {
            if is_sylva_auto || is_conventional_online || is_bayesian_online || is_grid_auto || is_batch_grid {
                println!("\n  📊 Selección del Espacio de Características (Features):");
                println!("  [1] 🌐 Red Completa Cuantitativa (32 Variables: Micro + MCO + EMAs + Momentum + Markov + Ciclo Semanal + FracDiff) [Por defecto]");
                println!("  [2] 🎯 Conjunto Reducido de 30 Features de Mercado Puro (F1..F30, excluyendo F31 FracDiff y F32 Estado)");
                println!("  [3] 🕯️ Microestructura Pura Exclusiva (F1..F5: Spreads, Imbalances, Micro-Volatilidad)");
                print!("  👉 Selecciona el espacio de variables (1-3) [Por defecto '1']: ");
                io::stdout().flush()?;
                let mut feat_in = String::new();
                io::stdin().read_line(&mut feat_in)?;
                match feat_in.trim() {
                    "2" => (Some(FeatureMask::new_30_features()), "30-Features (F1..F30 Mercado Puro)"),
                    "3" => (Some(FeatureMask::new_microstructure_5()), "Microstructure-5 (F1..F5)"),
                    _ => (None, "Red Completa (32 Features)"),
                }
            } else {
                (None, "Red Completa (32 Features)")
            }
        }
    };

    let tf = prompt_timeframe_for_analysis();
    let leverage = prompt_leverage_for_analysis();
    let klines = get_all_klines_closed_only(conn, &tf)?;
    if klines.is_empty() {
        println!("  ⚠️ No hay velas cerradas cargadas para '{}'. Descarga datos primero con la opción [2].", tf);
        return Ok(());
    }

    let (is_start_idx, oos_start_idx) = get_dataset_partition_indices(&klines, 100);
    println!(
        "\n  Dataset: {} velas | Warm-up: 0..{} | In-Sample: {}..{} | Out-Of-Sample: {}..{}",
        klines.len(), is_start_idx, is_start_idx, oos_start_idx, oos_start_idx, klines.len()
    );

    let rolling_window = 100;
    let candidate_horizons: Vec<usize> = match tf.as_str() {
        "1m" => vec![3, 5, 8, 12],
        "5m" => vec![2, 3, 4, 6],
        "15m" => vec![2, 3, 4, 6],
        "1h" | "1H" => vec![1, 2, 3, 4, 6, 8],
        "4h" | "4H" => vec![2, 3, 6, 12],
        "1d" | "1D" => vec![2, 3, 5],
        _ => vec![1, 2, 3, 4],
    };

    println!("\n  🎯 Horizontes Causales Multi-Vela a Evaluar Automáticamente: {:?}", candidate_horizons);
    println!("  🔄 Precomputando indicadores causales e historiales tabulares In-Sample...");
    let cached = CachedIndicators::new(&klines);
    let cached_arc = Arc::new(cached);

    let mut datasets_by_h = std::collections::HashMap::new();
    for &h in &candidate_horizons {
        let ds = TabularDataset::extract_from_klines(
            &klines,
            is_start_idx,
            oos_start_idx,
            rolling_window,
            h,
            &cached_arc,
            feature_mask.as_ref(),
        )?;
        datasets_by_h.insert(h, ds);
    }

    let default_target_horizon = candidate_horizons[0];
    let is_dataset = datasets_by_h.get(&default_target_horizon).cloned().unwrap();

    let active_feat_count = feature_mask.as_ref().map(|m| m.active_count()).unwrap_or(TOTAL_FEATURES);
    println!("  ✅ Datasets In-Sample estructurados para {} horizontes causales con {} variables activas ({}).", candidate_horizons.len(), active_feat_count, feature_label);

    let threshold_mode = ThresholdMode::DynamicAtrRatio;
    println!("  🌊 Régimen de Umbrales: Dinámicos Adaptativos por Volatilidad (ATR5/ATR50 - 0% Look-Ahead Bias).");

    let capital_percent = 10.0;

    let config_nom = BacktestConfig {
        initial_capital: 10_000.0,
        leverage,
        position_size_pct: 0.10,
        fee_rate: 0.0005,
        max_holding_bars: 24,
        use_compound: false,
        atr_sl_multiplier: 0.0,
    };

    let config_pct = BacktestConfig {
        initial_capital: 10_000.0,
        leverage,
        position_size_pct: 0.10,
        fee_rate: 0.0005,
        max_holding_bars: 24,
        use_compound: true,
        atr_sl_multiplier: 0.0,
    };

    let sim_nom = BacktestSimulator::new(config_nom.clone());
    let sim_pct = BacktestSimulator::new(config_pct.clone());
    let tuner = ThresholdTuner::new(config_nom.clone());

    let model_title: String;

    if is_sylva_auto {
        let model_type = SylvaTreeModelType::OnlineConventional;
        println!("\n  🧬 AUTO-OPTIMIZACIÓN EVOLUTIVA SYLVA EVO");
        println!("  • Modelo: 🚀 Online GBDT Convencional (Streaming Hoeffding Trees con Olvido Exponencial)");

        println!("\n  ⚙️ SELECCIÓN DE MOTOR DE CÓMPUTO:");
        println!("  [1] 💻 CPU Multihilo Puro (32 Cores Rayon - 100% Determinista Nativo bit-a-bit) [Por defecto]");
        println!("  [2] 🎮 GPU Acelerada (AMD Radeon RX 7700 XT - Compute Shaders en VRAM)");
        print!("  👉 Elige el motor de ejecución (1-2) [Por defecto '1']: ");
        io::stdout().flush()?;
        let mut eng_in = String::new();
        io::stdin().read_line(&mut eng_in)?;
        let use_gpu = eng_in.trim() == "2";

        print!("  🔁 Número de Generaciones Evolutivas [Por defecto 8]: ");
        io::stdout().flush()?;
        let mut gen_in = String::new();
        io::stdin().read_line(&mut gen_in)?;
        let generations = gen_in.trim().parse::<usize>().unwrap_or(8);

        print!("  👥 Tamaño de Población por Generación [Por defecto 20]: ");
        io::stdout().flush()?;
        let mut pop_in = String::new();
        io::stdin().read_line(&mut pop_in)?;
        let population_size = pop_in.trim().parse::<usize>().unwrap_or(20);

        print!("  🎲 Candidatos Exploratorios Iniciales [Por defecto 15]: ");
        io::stdout().flush()?;
        let mut init_in = String::new();
        io::stdin().read_line(&mut init_in)?;
        let initial_exploratory_trials = init_in.trim().parse::<usize>().unwrap_or(15);

        let sylva_cfg = SylvaTreeAutoTuningConfig {
            model_type,
            population_size,
            generations,
            initial_exploratory_trials,
            mutation_rate: 0.30,
            rolling_window,
            candidate_horizons: candidate_horizons.clone(),
            seed: 987654321,
            use_gpu,
            threshold_mode,
        };

        let sylva_optimizer = SylvaTreeOptimizer::new_with_engine(
            sylva_cfg,
            config_nom.clone(),
            config_pct.clone(),
            Arc::clone(&cached_arc),
            use_gpu,
        );
        let sylva_res = sylva_optimizer.optimize(&klines, &datasets_by_h, is_start_idx, oos_start_idx, &tf, feature_mask.as_ref())?;

        let champ = &sylva_res.champion_trial;
        let model_name = match champ.config.model_type {
            SylvaTreeModelType::OnlineConventional => "Sylva-Evo Online GBDT",
            SylvaTreeModelType::BayesianOnline => "Sylva-Evo Bayesian Online Trees",
        };

        model_title = format!(
            "{} {} (Depth={}, Trees={}, Decay={:.4}, LR={:.3})",
            model_name, feature_label, champ.config.max_depth, champ.config.n_trees, champ.config.decay_factor, champ.config.learning_rate
        );

        print_backtest_summary(&format!("{} (Nominal Completo IS+OOS)", model_title), &tf, &champ.report_nom);
        print_backtest_summary(&format!("{} (Compuesto Completo IS+OOS)", model_title), &tf, &champ.report_pct);

        let _ = generate_sylva_tree_tuning_dashboard(&model_title, &tf, &klines[is_start_idx..], &sylva_res, feature_label);
        let _ = generate_dual_dashboard(&model_title, &tf, &klines[is_start_idx..], &champ.report_nom, &champ.report_pct, leverage, capital_percent);
    } else if is_conventional_online {
        println!("\n  ⚙️ Configuración de Online GBDT (Streaming Decision Trees):");
        print!("  🌲 Número de árboles en el ensamble online [Por defecto 30]: ");
        io::stdout().flush()?;
        let mut n_trees_in = String::new();
        io::stdin().read_line(&mut n_trees_in)?;
        let n_trees = n_trees_in.trim().parse::<usize>().unwrap_or(30);

        print!("  🌳 Profundidad máxima del árbol online (max_depth) [Por defecto 3]: ");
        io::stdout().flush()?;
        let mut depth_in = String::new();
        io::stdin().read_line(&mut depth_in)?;
        let max_depth = depth_in.trim().parse::<usize>().unwrap_or(3);

        print!("  📉 Factor de olvido exponencial (decay_factor) [Por defecto 0.995]: ");
        io::stdout().flush()?;
        let mut decay_in = String::new();
        io::stdin().read_line(&mut decay_in)?;
        let decay_factor = decay_in.trim().parse::<f32>().unwrap_or(0.995);

        print!("  ⚡ Tasa de aprendizaje (learning_rate) [Por defecto 0.03]: ");
        io::stdout().flush()?;
        let mut lr_in = String::new();
        io::stdin().read_line(&mut lr_in)?;
        let learning_rate = lr_in.trim().parse::<f32>().unwrap_or(0.03);

        let online_cfg = OnlineGbdtConfig {
            n_trees,
            max_depth,
            learning_rate,
            decay_factor,
            l2_reg: 1.0,
            l1_reg: 0.01,
            min_samples_split: 25,
            grace_period: 15,
            split_confidence: 0.05,
            tie_threshold: 0.05,
            colsample_bytree: 0.8,
            gamma: 0.0001,
        };

        println!("\n  🎯 Auto-Evaluando Horizontes Causales Multi-Vela ({:?})...", candidate_horizons);
        let mut best_h = candidate_horizons[0];
        let mut best_fitness = -1000.0f64;
        let mut best_initial_model = None;
        let mut best_tune_res = None;

        for &h in &candidate_horizons {
            let h_dataset = datasets_by_h.get(&h).unwrap();
            let trainer = OnlineGbdtTrainer::new(online_cfg.clone());
            if let Ok(m) = trainer.fit_stream(h_dataset, feature_mask.as_ref()) {
                let mut h_sim_cfg = config_nom.clone();
                h_sim_cfg.max_holding_bars = h;
                let h_tuner = ThresholdTuner::new(h_sim_cfg);
                let cached_clone = Arc::clone(&cached_arc);
                let tr = h_tuner.tune(
                    &klines,
                    is_start_idx,
                    oos_start_idx,
                    &[0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012, 0.0018, 0.0025, 0.0035],
                    &[-0.0001, -0.0002, -0.0003, -0.0005, -0.0008, -0.0012, -0.0018, -0.0025, -0.0035],
                    |thr_l, thr_s| {
                        let mut em = OnlineGbdtEquationModel::new(
                            m.clone(),
                            thr_l,
                            thr_s,
                            rolling_window,
                        ).with_cached_indicators(Arc::clone(&cached_clone))
                         .with_threshold_mode(threshold_mode);
                        if let Some(ref fmask) = feature_mask {
                            em = em.with_mask(fmask.clone());
                        }
                        em
                    },
                );
                println!("     • Horizonte H = {:>2} velas ➔ In-Sample Fitness: {:>6.2} (Thr: {:+.4}/{:+.4})", h, tr.best_fitness, tr.best_threshold_long, tr.best_threshold_short);
                if tr.best_fitness > best_fitness || best_initial_model.is_none() {
                    best_fitness = tr.best_fitness;
                    best_h = h;
                    best_initial_model = Some(m);
                    best_tune_res = Some(tr);
                }
            }
        }

        let initial_online_model = best_initial_model.unwrap();
        let tune_res = best_tune_res.unwrap();
        println!("  🏆 Horizonte Óptimo Seleccionado Automáticamente: H = {} velas (Holding acoplado: {} barras)", best_h, best_h);

        let mut sim_cfg_nom = config_nom.clone();
        sim_cfg_nom.max_holding_bars = best_h;
        let mut sim_cfg_pct = config_pct.clone();
        sim_cfg_pct.max_holding_bars = best_h;
        let sim_nom = BacktestSimulator::new(sim_cfg_nom);
        let sim_pct = BacktestSimulator::new(sim_cfg_pct);

        let mut model_nom = OnlineGbdtEquationModel::new(
            initial_online_model.clone(),
            tune_res.best_threshold_long,
            tune_res.best_threshold_short,
            rolling_window,
        ).with_cached_indicators(Arc::clone(&cached_arc))
         .with_threshold_mode(threshold_mode);
        let mut model_pct = OnlineGbdtEquationModel::new(
            initial_online_model.clone(),
            tune_res.best_threshold_long,
            tune_res.best_threshold_short,
            rolling_window,
        ).with_cached_indicators(Arc::clone(&cached_arc))
         .with_threshold_mode(threshold_mode);

        if let Some(ref fmask) = feature_mask {
            model_nom = model_nom.with_mask(fmask.clone());
            model_pct = model_pct.with_mask(fmask.clone());
        }

        println!("\n  🚀 Ejecutando Backtest Continuo con Aprendizaje Online Vela a Vela (0% Look-Ahead Bias)...");
        let report_nom = sim_nom.run_range(&mut model_nom, &klines, is_start_idx, klines.len());
        let report_pct = sim_pct.run_range(&mut model_pct, &klines, is_start_idx, klines.len());

        model_title = format!(
            "Online GBDT {} Model (Trees={}, Depth={}, Decay={:.3}, LR={:.2})",
            feature_label, online_cfg.n_trees, online_cfg.max_depth, online_cfg.decay_factor, online_cfg.learning_rate
        );

        println!("\n--- IMPORTANCIA DE CARACTERÍSTICAS ONLINE GBDT (100% Variables Cuantitativas) ---");
        let ranking = model_nom.model.get_feature_importance_ranking();
        for (rank_pos, (idx, name, gain_pct)) in ranking.iter().filter(|r| r.2 > 0.0 || feature_mask.is_none()).enumerate() {
            println!("  #{:<2} F{:<2} : {:<42} : {:5.2}% Ganancia", rank_pos + 1, idx + 1, name, gain_pct);
        }

        print_backtest_summary(&format!("{} (Nominal Completo IS+OOS)", model_title), &tf, &report_nom);
        print_backtest_summary(&format!("{} (Compuesto Completo IS+OOS)", model_title), &tf, &report_pct);

        let _ = generate_online_gbdt_dashboard(&model_title, &tf, &klines[is_start_idx..], &report_nom, &report_pct, &model_nom.model, leverage, capital_percent);
        let _ = generate_dual_dashboard(&model_title, &tf, &klines[is_start_idx..], &report_nom, &report_pct, leverage, capital_percent);
    } else if is_bayesian_online {
        println!("\n  🔬 Modalidad de Optimización Bayesiana de Árboles Online:");
        println!("  [1] 🚀 Optimización Bayesiana Autónoma con Procesos Gaussianos (GP Matérn 5/2 + Expected Improvement) [Por defecto]");
        println!("  [2] 🌲 Calibración Manual de Parámetros Bayesianos (NIG Conjugada & Incertidumbre)");
        print!("  👉 Selecciona método bayesiano (1-2) [Por defecto '1']: ");
        io::stdout().flush()?;
        let mut bayes_choice = String::new();
        io::stdin().read_line(&mut bayes_choice)?;

        let is_bo_auto = bayes_choice.trim() == "1" || bayes_choice.trim().is_empty();

        if is_bo_auto {
            let bo_config = BayesianTreeOptimizerConfig::default();
            let bo_optimizer = BayesianTreeOptimizer::new(bo_config, config_nom.clone(), config_pct.clone(), Arc::clone(&cached_arc));
            let bo_res = bo_optimizer.optimize(&klines, &is_dataset, is_start_idx, oos_start_idx, &tf, feature_mask.as_ref())?;

            let champion_cand = &bo_res.candidate_reports[0];
            model_title = format!(
                "Bayesian Online Trees {} (Trees={}, Depth={}, Decay={:.4}, Kappa={:.2})",
                feature_label, bo_res.champion_config.n_trees, bo_res.champion_config.max_depth, bo_res.champion_config.decay_factor, bo_res.champion_config.uncertainty_penalty_kappa
            );

            println!("\n--- IMPORTANCIA DE CARACTERÍSTICAS BAYESIANAS (100% Variables Cuantitativas) ---");
            let ranking = bo_res.champion_model.get_feature_importance_ranking();
            for (rank_pos, (idx, name, gain_pct)) in ranking.iter().filter(|r| r.2 > 0.0 || feature_mask.is_none()).enumerate() {
                println!("  #{:<2} F{:<2} : {:<42} : {:5.2}% Ganancia", rank_pos + 1, idx + 1, name, gain_pct);
            }

            print_backtest_summary(&format!("{} (Nominal Completo IS+OOS)", model_title), &tf, &champion_cand.report_nom);
            print_backtest_summary(&format!("{} (Compuesto Completo IS+OOS)", model_title), &tf, &champion_cand.report_pct);

            let _ = generate_bayesian_tree_dashboard(&model_title, &tf, &klines[is_start_idx..], &champion_cand.report_nom, &champion_cand.report_pct, &bo_res.champion_model, leverage, capital_percent);
            let _ = generate_dual_dashboard(&model_title, &tf, &klines[is_start_idx..], &champion_cand.report_nom, &champion_cand.report_pct, leverage, capital_percent);
        } else {
            print!("  🌲 Número de árboles bayesianos [Por defecto 25]: ");
            io::stdout().flush()?;
            let mut trees_in = String::new();
            io::stdin().read_line(&mut trees_in)?;
            let n_trees = trees_in.trim().parse::<usize>().unwrap_or(25);

            print!("  🌳 Profundidad máxima del árbol [Por defecto 3]: ");
            io::stdout().flush()?;
            let mut depth_in = String::new();
            io::stdin().read_line(&mut depth_in)?;
            let max_depth = depth_in.trim().parse::<usize>().unwrap_or(3);

            print!("  📉 Factor de olvido exponencial bayesiano (delta) [Por defecto 0.995]: ");
            io::stdout().flush()?;
            let mut decay_in = String::new();
            io::stdin().read_line(&mut decay_in)?;
            let decay_factor = decay_in.trim().parse::<f32>().unwrap_or(0.995);

            print!("  🛡️ Penalización por incertidumbre epistémica (kappa_risk) [Por defecto 0.50]: ");
            io::stdout().flush()?;
            let mut kappa_in = String::new();
            io::stdin().read_line(&mut kappa_in)?;
            let uncertainty_penalty_kappa = kappa_in.trim().parse::<f32>().unwrap_or(0.50);

            let b_cfg = BayesianTreeConfig {
                n_trees,
                max_depth,
                learning_rate: 0.03,
                decay_factor,
                prior_mean: 0.0,
                prior_precision: 1.0,
                prior_shape: 2.5,
                prior_scale: 1.0,
                min_samples_leaf: 20,
                uncertainty_penalty_kappa,
                use_thompson_sampling: false,
                confidence_level: 0.95,
                colsample_bytree: 0.8,
            };

            let trainer = BayesianOnlineGbdtTrainer::new(b_cfg.clone());
            let trained_model = trainer.fit_stream(&is_dataset, feature_mask.as_ref())?;

            let cached_clone = Arc::clone(&cached_arc);
            let tune_res = tuner.tune(
                &klines,
                is_start_idx,
                oos_start_idx,
                &[0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012, 0.0018, 0.0025, 0.0035],
                &[-0.0001, -0.0002, -0.0003, -0.0005, -0.0008, -0.0012, -0.0018, -0.0025, -0.0035],
                |thr_l, thr_s| {
                    let mut m = BayesianGbdtEquationModel::new(
                        trained_model.clone(),
                        thr_l,
                        thr_s,
                        rolling_window,
                    ).with_cached_indicators(Arc::clone(&cached_clone))
                     .with_threshold_mode(threshold_mode);
                    if let Some(ref fmask) = feature_mask {
                        m = m.with_mask(fmask.clone());
                    }
                    m
                },
            );

            let mut model_nom = BayesianGbdtEquationModel::new(
                trained_model.clone(),
                tune_res.best_threshold_long,
                tune_res.best_threshold_short,
                rolling_window,
            ).with_cached_indicators(Arc::clone(&cached_arc))
             .with_threshold_mode(threshold_mode);
            let mut model_pct = BayesianGbdtEquationModel::new(
                trained_model.clone(),
                tune_res.best_threshold_long,
                tune_res.best_threshold_short,
                rolling_window,
            ).with_cached_indicators(Arc::clone(&cached_arc))
             .with_threshold_mode(threshold_mode);

            if let Some(ref fmask) = feature_mask {
                model_nom = model_nom.with_mask(fmask.clone());
                model_pct = model_pct.with_mask(fmask.clone());
            }

            let report_nom = sim_nom.run_range(&mut model_nom, &klines, is_start_idx, klines.len());
            let report_pct = sim_pct.run_range(&mut model_pct, &klines, is_start_idx, klines.len());

            model_title = format!(
                "Bayesian Online Trees {} (Trees={}, Depth={}, Kappa={:.2})",
                feature_label, b_cfg.n_trees, b_cfg.max_depth, b_cfg.uncertainty_penalty_kappa
            );

            println!("\n--- IMPORTANCIA DE CARACTERÍSTICAS BAYESIANAS (100% Variables Cuantitativas) ---");
            let ranking = trained_model.get_feature_importance_ranking();
            for (rank_pos, (idx, name, gain_pct)) in ranking.iter().filter(|r| r.2 > 0.0 || feature_mask.is_none()).enumerate() {
                println!("  #{:<2} F{:<2} : {:<42} : {:5.2}% Ganancia", rank_pos + 1, idx + 1, name, gain_pct);
            }

            print_backtest_summary(&format!("{} (Nominal Completo IS+OOS)", model_title), &tf, &report_nom);
            print_backtest_summary(&format!("{} (Compuesto Completo IS+OOS)", model_title), &tf, &report_pct);

            let _ = generate_bayesian_tree_dashboard(&model_title, &tf, &klines[is_start_idx..], &report_nom, &report_pct, &trained_model, leverage, capital_percent);
            let _ = generate_dual_dashboard(&model_title, &tf, &klines[is_start_idx..], &report_nom, &report_pct, leverage, capital_percent);
        }
    } else if is_grid_auto {
        let auto_tuner = OnlineGbdtAutoTuner::new(OnlineGbdtAutoTuningConfig::default(), config_nom, config_pct, Arc::clone(&cached_arc));
        let tune_res = auto_tuner.auto_optimize(&klines, &is_dataset, is_start_idx, oos_start_idx, &tf, feature_mask.as_ref())?;

        let champion_cand = &tune_res.candidate_reports[0];
        model_title = format!(
            "Online GBDT {} Auto-Tuned (Depth={}, Trees={}, Decay={:.3})",
            feature_label, champion_cand.max_depth, champion_cand.n_trees, tune_res.champion_config.decay_factor
        );

        println!("\n--- IMPORTANCIA DE CARACTERÍSTICAS ONLINE GBDT AUTO-OPTIMIZADO (100% Variables Cuantitativas) ---");
        let ranking = tune_res.champion_model.get_feature_importance_ranking();
        for (rank_pos, (idx, name, gain_pct)) in ranking.iter().filter(|r| r.2 > 0.0 || feature_mask.is_none()).enumerate() {
            println!("  #{:<2} F{:<2} : {:<42} : {:5.2}% Ganancia", rank_pos + 1, idx + 1, name, gain_pct);
        }

        print_backtest_summary(&format!("{} (Nominal Completo IS+OOS)", model_title), &tf, &champion_cand.report_nom);
        print_backtest_summary(&format!("{} (Compuesto Completo IS+OOS)", model_title), &tf, &champion_cand.report_pct);

        let _ = generate_online_gbdt_dashboard(&model_title, &tf, &klines[is_start_idx..], &champion_cand.report_nom, &champion_cand.report_pct, &tune_res.champion_model, leverage, capital_percent);
        let _ = generate_dual_dashboard(&model_title, &tf, &klines[is_start_idx..], &champion_cand.report_nom, &champion_cand.report_pct, leverage, capital_percent);
    } else if is_batch_grid {
        println!("\n  ⚙️ Configuración Anti-Overfitting de GBDT Batch (Admite listas como '3,4,5' o valores únicos):");
        print!("  🌳 Profundidades máximas del árbol (max_depth) [Por defecto 3]: ");
        io::stdout().flush()?;
        let mut depth_in = String::new();
        io::stdin().read_line(&mut depth_in)?;
        let gbdt_depths = parse_usize_list(&depth_in, 3);

        print!("  🍃 Mínimo de muestras por hoja (min_samples_leaf) [Por defecto 20]: ");
        io::stdout().flush()?;
        let mut min_in = String::new();
        io::stdin().read_line(&mut min_in)?;
        let gbdt_min_samples = parse_usize_list(&min_in, 20);

        print!("  🌲 Número de árboles en el ensamble (n_trees) [Por defecto 40]: ");
        io::stdout().flush()?;
        let mut trees_in = String::new();
        io::stdin().read_line(&mut trees_in)?;
        let gbdt_n_trees_list = parse_usize_list(&trees_in, 40);

        let total_combinations = gbdt_depths.len() * gbdt_min_samples.len() * gbdt_n_trees_list.len();
        let cv = PurgedCrossValidator::new(5, 0.01);
        let splits = cv.split(is_dataset.len(), default_target_horizon);

        let colors = [
            "#10b981", "#3b82f6", "#8b5cf6", "#f59e0b", "#ec4899",
            "#06b6d4", "#f97316", "#14b8a6", "#6366f1", "#a855f7",
            "#84cc16", "#ef4444", "#eab308", "#0ea5e9", "#d946ef",
        ];

        println!(
            "\n  ⚙️ Ejecutando Purged K-Fold Cross-Validation en {} combinaciones ({} splits)...",
            total_combinations, splits.len()
        );

        let mut candidate_reports = Vec::new();
        let mut best_fitness = -1000.0f64;
        let mut best_config = GbdtConfig::default();
        let mut best_model = None;
        let mut color_idx = 0;

        for &depth in &gbdt_depths {
            for &min_leaf in &gbdt_min_samples {
                for &n_trees in &gbdt_n_trees_list {
                    let cfg = GbdtConfig {
                        n_trees,
                        max_depth: depth,
                        learning_rate: 0.03,
                        l2_reg: 1.0,
                        l1_reg: 0.01,
                        min_samples_leaf: min_leaf,
                        subsample: 0.8,
                        colsample_bytree: 0.8,
                        gamma: 0.0001,
                    };

                    let mut total_mse = 0.0f32;
                    let mut total_mda = 0.0f32;
                    let mut total_ic = 0.0f32;

                    for split in &splits {
                        let train_ds = is_dataset.subset(&split.train_indices);
                        let test_ds = is_dataset.subset(&split.test_indices);

                        let trainer = GbdtTrainer::new(cfg.clone());
                        let trained = trainer.fit(&train_ds, feature_mask.as_ref())?;

                        let test_preds = trained.predict_dataset(&test_ds);
                        let test_targets: Vec<f32> = test_ds.samples.iter().map(|s| s.target).collect();
                        let (mse, mda, ic) = evaluate_predictions(&test_preds, &test_targets);
                        total_mse += mse;
                        total_mda += mda;
                        total_ic += ic;
                    }

                    let n_sp = splits.len() as f32;
                    let cv_mse = total_mse / n_sp;
                    let cv_mda = total_mda / n_sp;
                    let cv_ic = total_ic / n_sp;

                    let final_trainer = GbdtTrainer::new(cfg.clone());
                    let final_trained = final_trainer.fit(&is_dataset, feature_mask.as_ref())?;

                    let cached_clone = Arc::clone(&cached_arc);
                    let tune_res = tuner.tune(
                        &klines,
                        is_start_idx,
                        oos_start_idx,
                        &[0.0001, 0.0003, 0.0005, 0.0008, 0.0012, 0.0018, 0.0025, 0.0035],
                        &[-0.0001, -0.0003, -0.0005, -0.0008, -0.0012, -0.0018, -0.0025, -0.0035],
                        |thr_l, thr_s| {
                            let mut m = GbdtEquationModel::new(
                                final_trained.clone(),
                                thr_l,
                                thr_s,
                                rolling_window,
                            ).with_cached_indicators(Arc::clone(&cached_clone));
                            if let Some(ref fmask) = feature_mask {
                                m = m.with_mask(fmask.clone());
                            }
                            m
                        },
                    );

                    let mut model_nom = GbdtEquationModel::new(
                        final_trained.clone(),
                        tune_res.best_threshold_long,
                        tune_res.best_threshold_short,
                        rolling_window,
                    ).with_cached_indicators(Arc::clone(&cached_arc));
                    let mut model_pct = GbdtEquationModel::new(
                        final_trained.clone(),
                        tune_res.best_threshold_long,
                        tune_res.best_threshold_short,
                        rolling_window,
                    ).with_cached_indicators(Arc::clone(&cached_arc));

                    if let Some(ref fmask) = feature_mask {
                        model_nom = model_nom.with_mask(fmask.clone());
                        model_pct = model_pct.with_mask(fmask.clone());
                    }

                    let report_nom = sim_nom.run_range(&mut model_nom, &klines, is_start_idx, klines.len());
                    let report_pct = sim_pct.run_range(&mut model_pct, &klines, is_start_idx, klines.len());

                    let cand_color = colors[color_idx % colors.len()].to_string();
                    color_idx += 1;

                    candidate_reports.push(GbdtGridCandidateReport {
                        max_depth: depth,
                        min_samples_leaf: min_leaf,
                        n_trees,
                        target_horizon: default_target_horizon,
                        cv_mse,
                        cv_mda,
                        cv_ic,
                        best_thr_long: tune_res.best_threshold_long,
                        best_thr_short: tune_res.best_threshold_short,
                        is_sylva_fitness: tune_res.best_fitness,
                        rank_slope: 0,
                        rank_cap_dd: 0,
                        rank_r2: 0,
                        rank_smoothness: 0,
                        weighted_avg_rank: 0.0,
                        sylva_rank_fitness: tune_res.best_fitness,
                        raw_slope_ratio: 0.0,
                        raw_cap_dd_ratio: 0.0,
                        raw_r2_score: 0.0,
                        raw_smoothness_score: 0.0,
                        report_nom,
                        report_pct,
                        color: cand_color,
                    });

                    if tune_res.best_fitness > best_fitness {
                        best_fitness = tune_res.best_fitness;
                        best_config = cfg;
                        best_model = Some(final_trained);
                    }
                }
            }
        }

        let trained_gbdt = best_model.unwrap();
        model_title = format!(
            "GBDT {} Purged CV (Depth={}, MinLeaf={}, Trees={})",
            feature_label, best_config.max_depth, best_config.min_samples_leaf, best_config.n_trees
        );

        let champion_cand = candidate_reports.iter().find(|c| c.max_depth == best_config.max_depth && c.n_trees == best_config.n_trees).unwrap();

        println!("\n--- IMPORTANCIA DE CARACTERÍSTICAS GBDT BATCH (100% Variables Cuantitativas) ---");
        let ranking = trained_gbdt.get_feature_importance_ranking();
        for (rank_pos, (idx, name, gain_pct)) in ranking.iter().filter(|r| r.2 > 0.0 || feature_mask.is_none()).enumerate() {
            println!("  #{:<2} F{:<2} : {:<42} : {:5.2}% Ganancia", rank_pos + 1, idx + 1, name, gain_pct);
        }

        print_backtest_summary(&format!("{} (Nominal Completo IS+OOS)", model_title), &tf, &champion_cand.report_nom);
        print_backtest_summary(&format!("{} (Compuesto Completo IS+OOS)", model_title), &tf, &champion_cand.report_pct);

        let summary = GbdtGridDashboardSummary {
            tf: tf.clone(),
            is_start_idx,
            oos_start_idx,
            initial_capital: config_nom.initial_capital,
            candidates: candidate_reports.clone(),
        };

        let _ = generate_gbdt_grid_dashboard(&summary, &klines);
        let _ = generate_dual_dashboard(&model_title, &tf, &klines[is_start_idx..], &champion_cand.report_nom, &champion_cand.report_pct, leverage, capital_percent);
    } else if is_elastic_net {
        println!("\n  ⚙️ Configuración de Elastic Net (L1 Lasso + L2 Ridge Coordinate Descent):");
        print!("  ⚖️ Proporción L1 / Lasso (l1_ratio entre 0.0=Ridge y 1.0=Lasso) [Por defecto 0.50]: ");
        io::stdout().flush()?;
        let mut l1_in = String::new();
        io::stdin().read_line(&mut l1_in)?;
        let l1_ratio = l1_in.trim().parse::<f32>().unwrap_or(0.50).clamp(0.0, 1.0);

        print!("  🔒 Factor de regularización global (lambda) [Por defecto 0.005]: ");
        io::stdout().flush()?;
        let mut lam_in = String::new();
        io::stdin().read_line(&mut lam_in)?;
        let lambda = lam_in.trim().parse::<f32>().unwrap_or(0.005).max(1e-6);

        let el_cfg = ElasticNetConfig {
            l1_ratio,
            lambda,
            max_iter: 1000,
            tol: 1e-6,
        };

        println!("\n  🔄 Ajustando Elastic Net vía Coordinate Descent en In-Sample...");
        let trainer = ElasticNetTrainer::new(el_cfg.clone());
        let el_model = trainer.fit(&is_dataset, feature_mask.as_ref())?;

        let cached_clone = Arc::clone(&cached_arc);
        let tune_res = tuner.tune(
            &klines,
            is_start_idx,
            oos_start_idx,
            &[0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.0012, 0.0018, 0.0025, 0.0035],
            &[-0.0001, -0.0002, -0.0003, -0.0005, -0.0008, -0.0012, -0.0018, -0.0025, -0.0035],
            |thr_l, thr_s| {
                let mut m = ElasticNetEquationModel::new(
                    el_model.clone(),
                    thr_l,
                    thr_s,
                    rolling_window,
                ).with_cached_indicators(Arc::clone(&cached_clone));
                if let Some(ref fmask) = feature_mask {
                    m = m.with_mask(fmask.clone());
                }
                m
            },
        );

        let mut model_nom = ElasticNetEquationModel::new(
            el_model.clone(),
            tune_res.best_threshold_long,
            tune_res.best_threshold_short,
            rolling_window,
        ).with_cached_indicators(Arc::clone(&cached_arc));
        let mut model_pct = ElasticNetEquationModel::new(
            el_model.clone(),
            tune_res.best_threshold_long,
            tune_res.best_threshold_short,
            rolling_window,
        ).with_cached_indicators(Arc::clone(&cached_arc));

        if let Some(ref fmask) = feature_mask {
            model_nom = model_nom.with_mask(fmask.clone());
            model_pct = model_pct.with_mask(fmask.clone());
        }

        let report_nom = sim_nom.run_range(&mut model_nom, &klines, is_start_idx, klines.len());
        let report_pct = sim_pct.run_range(&mut model_pct, &klines, is_start_idx, klines.len());

        model_title = format!(
            "Elastic Net {} (L1Ratio={:.2}, Lambda={:.4})",
            feature_label, el_cfg.l1_ratio, el_cfg.lambda
        );

        print_backtest_summary(&format!("{} (Nominal Completo IS+OOS)", model_title), &tf, &report_nom);
        print_backtest_summary(&format!("{} (Compuesto Completo IS+OOS)", model_title), &tf, &report_pct);

        let _ = generate_dual_dashboard(&model_title, &tf, &klines[is_start_idx..], &report_nom, &report_pct, leverage, capital_percent);
    }

    Ok(())
}
