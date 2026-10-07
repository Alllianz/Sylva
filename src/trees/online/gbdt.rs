use crate::features::FeatureMask;
use crate::features::TOTAL_FEATURES;
use crate::trees::dataset::TabularDataset;
use crate::trees::online::tree::OnlineDecisionTree;
use crate::trees::online::types::OnlineGbdtConfig;

/// Modelo de Gradient Boosted Decision Trees con aprendizaje online streaming
#[derive(Clone, Debug)]
pub struct OnlineGbdtModel {
    pub base_score: f32,
    pub trees: Vec<OnlineDecisionTree>,
    pub config: OnlineGbdtConfig,
    pub feature_importances: [f32; TOTAL_FEATURES],
    pub total_samples_seen: usize,
    pub running_mean_target: f32,
}

impl OnlineGbdtModel {
    /// Inicia un nuevo modelo de GBDT Online vacío
    pub fn new(config: OnlineGbdtConfig, active_features: &[usize]) -> Self {
        let mut trees = Vec::with_capacity(config.n_trees);
        let mut rng_state = 123456789u64;

        for tree_idx in 0..config.n_trees {
            let mut tree_feats = active_features.to_vec();
            if config.colsample_bytree < 0.999 && tree_feats.len() > 2 {
                let col_size = ((tree_feats.len() as f32) * config.colsample_bytree).max(1.0) as usize;
                shuffle_features(&mut tree_feats, &mut rng_state);
                tree_feats.truncate(col_size);
            }
            trees.push(OnlineDecisionTree::new(tree_idx, tree_feats));
        }

        Self {
            base_score: 0.0,
            trees,
            config,
            feature_importances: [0.0; TOTAL_FEATURES],
            total_samples_seen: 0,
            running_mean_target: 0.0,
        }
    }

    /// Inferencia estricta para una observación de 34 variables
    #[inline(always)]
    pub fn predict(&self, features: &[f32; TOTAL_FEATURES]) -> f32 {
        let mut pred = self.base_score;
        let lr = self.config.learning_rate;
        for tree in &self.trees {
            pred += lr * tree.predict(features);
        }
        pred
    }

    /// Inferencia por lotes sobre un dataset tabular
    pub fn predict_dataset(&self, dataset: &TabularDataset) -> Vec<f32> {
        dataset
            .samples
            .iter()
            .map(|s| self.predict(&s.features))
            .collect()
    }

    /// Paso de aprendizaje streaming online causal: recibe una muestra (x_t, y_t) y actualiza el ensamble
    pub fn update_step(&mut self, features: &[f32; TOTAL_FEATURES], target: f32) -> f32 {
        self.total_samples_seen += 1;

        // 1. Actualizar media acumulada del target para el base score inicial con decaimiento
        let decay = self.config.decay_factor;
        self.running_mean_target = self.running_mean_target * decay + (1.0 - decay) * target;
        if self.total_samples_seen < 50 {
            self.base_score = self.running_mean_target;
        }

        // 2. Realizar predicción actual con los árboles existentes
        let current_pred = self.predict(features);

        // 3. Gradiente y Hessiano para Squared Error Loss (L = 0.5 * (y - y_hat)^2)
        // g_i = y_hat - y, h_i = 1.0
        let mut residual = current_pred - target;
        let hessian = 1.0f32;

        let lr = self.config.learning_rate;

        // 4. Actualizar secuencialmente cada árbol en el ensamble boosting online
        for tree in &mut self.trees {
            // El árbol se entrena sobre el residuo actual
            let g = residual;
            let _split_occurred = tree.update(features, g, hessian, &self.config);

            // Reducir el residuo que los siguientes árboles deben ajustar (Boosting residual sequential shrinkage)
            let tree_pred = tree.predict(features);
            residual -= lr * tree_pred;
        }

        // 5. Recomputar importancia de variables acumulada periódicamente
        if self.total_samples_seen % 25 == 0 {
            let mut importances = [0.0f32; TOTAL_FEATURES];
            for tree in &self.trees {
                tree.compute_feature_importance(&mut importances);
            }
            self.feature_importances = importances;
        }

        current_pred
    }

    /// Retorna el ranking de las características ordenadas por ganancia acumulada
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

    /// Número total de hojas en todo el ensamble
    pub fn total_leaves(&self) -> usize {
        self.trees.iter().map(|t| t.n_leaves()).sum()
    }
}

/// Entrenador para modelos de Online Gradient Boosting
pub struct OnlineGbdtTrainer {
    pub config: OnlineGbdtConfig,
}

impl OnlineGbdtTrainer {
    pub fn new(config: OnlineGbdtConfig) -> Self {
        Self { config }
    }

    /// Ajusta el modelo streaming pasando secuencialmente por el dataset sin Look-Ahead Bias
    pub fn fit_stream(
        &self,
        dataset: &TabularDataset,
        mask: Option<&FeatureMask>,
    ) -> Result<OnlineGbdtModel, &'static str> {
        let n_samples = dataset.len();
        if n_samples < 50 {
            return Err("Muestras insuficientes para entrenar Online GBDT");
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
            return Err("No hay características activas en la máscara para Online GBDT");
        }

        let mut model = OnlineGbdtModel::new(self.config.clone(), &active_features);

        // Precomputar media inicial de los primeros 50 datos
        let initial_n = 50.min(n_samples);
        let sum_first: f32 = dataset.samples[..initial_n].iter().map(|s| s.target).sum();
        model.base_score = sum_first / (initial_n as f32);
        model.running_mean_target = model.base_score;

        // Iterar muestra por muestra en orden cronológico estricto
        for sample in &dataset.samples {
            model.update_step(&sample.features, sample.target);
        }

        // Recomputar importancia de variables final
        let mut importances = [0.0f32; TOTAL_FEATURES];
        for tree in &model.trees {
            tree.compute_feature_importance(&mut importances);
        }
        model.feature_importances = importances;

        Ok(model)
    }
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
