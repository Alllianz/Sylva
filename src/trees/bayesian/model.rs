use crate::features::FeatureMask;
use crate::features::TOTAL_FEATURES;
use crate::trees::bayesian::tree::BayesianDecisionTree;
use crate::trees::bayesian::types::{BayesianPrediction, BayesianTreeConfig};
use crate::trees::dataset::TabularDataset;

/// Modelo de Boosting de Árboles con Inferencia Bayesiana Online y Cuantificación de Incertidumbre
#[derive(Clone, Debug)]
pub struct BayesianOnlineGbdtModel {
    pub base_score: f32,
    pub trees: Vec<BayesianDecisionTree>,
    pub config: BayesianTreeConfig,
    pub feature_importances: [f32; TOTAL_FEATURES],
    pub total_samples_seen: usize,
    pub running_mean_target: f32,
    pub rng_state: u64,
}

impl BayesianOnlineGbdtModel {
    pub fn new(config: BayesianTreeConfig, active_features: &[usize]) -> Self {
        let mut trees = Vec::with_capacity(config.n_trees);
        let mut rng = 987654321u64;

        for tree_idx in 0..config.n_trees {
            let mut tree_feats = active_features.to_vec();
            if config.colsample_bytree < 0.999 && tree_feats.len() > 2 {
                let col_size = ((tree_feats.len() as f32) * config.colsample_bytree).max(1.0) as usize;
                shuffle_features(&mut tree_feats, &mut rng);
                tree_feats.truncate(col_size);
            }
            trees.push(BayesianDecisionTree::new(tree_idx, tree_feats, &config));
        }

        Self {
            base_score: 0.0,
            trees,
            config,
            feature_importances: [0.0; TOTAL_FEATURES],
            total_samples_seen: 0,
            running_mean_target: 0.0,
            rng_state: 135792468,
        }
    }

    /// Inferencia bayesiana completa: media, varianza, bandas creíbles y probabilidad a posteriori
    #[inline(always)]
    pub fn predict(&self, features: &[f32; TOTAL_FEATURES]) -> BayesianPrediction {
        let mut mean_acc = self.base_score;
        let mut var_acc = 0.0f32;
        let lr = self.config.learning_rate;

        for tree in &self.trees {
            let (m, v) = tree.predict(features);
            mean_acc += lr * m;
            var_acc += lr * lr * v;
        }

        let std_dev = var_acc.sqrt().max(1e-5);
        let z_score = 1.96f32; // 95% intervalo de credibilidad
        let lower = mean_acc - z_score * std_dev;
        let upper = mean_acc + z_score * std_dev;

        // Probabilidad a posteriori P(Y > 0 | x) usando la CDF normal
        let prob_pos = normal_cdf(mean_acc / std_dev);

        // Alpha penalizado por incertidumbre: si la varianza es muy alta, reduce el alpha
        let kappa = self.config.uncertainty_penalty_kappa;
        let adjusted = if mean_acc > 0.0 {
            (mean_acc - kappa * std_dev).max(0.0)
        } else {
            (mean_acc + kappa * std_dev).min(0.0)
        };

        BayesianPrediction {
            mean: mean_acc,
            variance: var_acc,
            std_dev,
            lower_bound: lower,
            upper_bound: upper,
            prob_positive: prob_pos,
            adjusted_alpha: adjusted,
        }
    }

    /// Inferencia por lotes sobre un dataset tabular
    pub fn predict_dataset(&self, dataset: &TabularDataset) -> Vec<BayesianPrediction> {
        dataset
            .samples
            .iter()
            .map(|s| self.predict(&s.features))
            .collect()
    }

    /// Paso de actualización bayesiana online con una muestra (x_t, y_t)
    pub fn update_step(&mut self, features: &[f32; TOTAL_FEATURES], target: f32) -> BayesianPrediction {
        self.total_samples_seen += 1;

        let decay = self.config.decay_factor;
        self.running_mean_target = self.running_mean_target * decay + (1.0 - decay) * target;
        if self.total_samples_seen < 50 {
            self.base_score = self.running_mean_target;
        }

        let current_pred = self.predict(features);
        let mut residual = current_pred.mean - target;
        let lr = self.config.learning_rate;

        // Actualizar cada árbol bayesiano con el residuo
        for tree in &mut self.trees {
            let _ = tree.update(features, residual, &self.config);
            let (tree_mean, _) = tree.predict(features);
            residual -= lr * tree_mean;
        }

        if self.total_samples_seen % 25 == 0 {
            let mut importances = [0.0f32; TOTAL_FEATURES];
            for tree in &self.trees {
                tree.compute_feature_importance(&mut importances);
            }
            self.feature_importances = importances;
        }

        current_pred
    }

    /// Retorna el ranking de importancia de variables
    pub fn get_feature_importance_ranking(&self) -> Vec<(usize, String, f32)> {
        let names = crate::features::feature_names();
        let total_imp: f32 = self.feature_importances.iter().sum();
        let norm_factor = if total_imp > 0.0 { 100.0 / total_imp } else { 1.0 };

        let mut ranking: Vec<(usize, String, f32)> = self
            .feature_importances
            .iter()
            .enumerate()
            .map(|(idx, &imp)| {
                let name = names.get(idx).unwrap_or(&"Unknown").to_string();
                (idx, name, imp * norm_factor)
            })
            .collect();

        ranking.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        ranking
    }

    pub fn total_leaves(&self) -> usize {
        self.trees.iter().map(|t| t.n_leaves()).sum()
    }
}

/// Entrenador streaming para el modelo bayesiano
pub struct BayesianOnlineGbdtTrainer {
    pub config: BayesianTreeConfig,
}

impl BayesianOnlineGbdtTrainer {
    pub fn new(config: BayesianTreeConfig) -> Self {
        Self { config }
    }

    pub fn fit_stream(
        &self,
        dataset: &TabularDataset,
        mask: Option<&FeatureMask>,
    ) -> Result<BayesianOnlineGbdtModel, &'static str> {
        let n_samples = dataset.len();
        if n_samples < 50 {
            return Err("Muestras insuficientes para entrenar Bayesian Online GBDT");
        }

        let mut active_features = Vec::new();
        for i in 0..TOTAL_FEATURES {
            if let Some(m) = mask {
                if m.is_active(i) {
                    active_features.push(i);
                }
            } else {
                active_features.push(i);
            }
        }

        if active_features.is_empty() {
            return Err("No hay características activas en la máscara para Bayesian GBDT");
        }

        let mut model = BayesianOnlineGbdtModel::new(self.config.clone(), &active_features);

        let initial_n = 50.min(n_samples);
        let sum_first: f32 = dataset.samples[..initial_n].iter().map(|s| s.target).sum();
        model.base_score = sum_first / (initial_n as f32);
        model.running_mean_target = model.base_score;

        for sample in &dataset.samples {
            model.update_step(&sample.features, sample.target);
        }

        let mut importances = [0.0f32; TOTAL_FEATURES];
        for tree in &model.trees {
            tree.compute_feature_importance(&mut importances);
        }
        model.feature_importances = importances;

        Ok(model)
    }
}

/// Aproximación numérica rápida de la función de distribución acumulada normal \Phi(x)
#[inline(always)]
fn normal_cdf(x: f32) -> f32 {
    0.5 * (1.0 + (x / std::f32::consts::SQRT_2).tanh())
}

#[inline(always)]
fn fast_rand(state: &mut u64) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*state >> 33) as usize
}

fn shuffle_features(arr: &mut [usize], state: &mut u64) {
    let n = arr.len();
    for i in (1..n).rev() {
        let j = fast_rand(state) % (i + 1);
        arr.swap(i, j);
    }
}
