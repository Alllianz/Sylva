use crate::features::FeatureMask;
use crate::features::TOTAL_FEATURES;
use crate::trees::batch::split::{calculate_leaf_weight, find_best_split};
use crate::trees::batch::tree::{DecisionTree, TreeNode};
use crate::trees::dataset::TabularDataset;

/// Configuración de hiperparámetros del Gradient Boosted Decision Tree (GBDT)
#[derive(Clone, Debug)]
pub struct GbdtConfig {
    pub n_trees: usize,
    pub max_depth: usize,
    pub learning_rate: f32, // eta
    pub l2_reg: f32,        // lambda
    pub l1_reg: f32,        // alpha
    pub min_samples_leaf: usize,
    pub subsample: f32,          // Bagging ratio (0.5..1.0)
    pub colsample_bytree: f32,   // Feature fraction (0.5..1.0)
    pub gamma: f32,              // Complexity penalty per split
}

impl Default for GbdtConfig {
    fn default() -> Self {
        Self {
            n_trees: 50,
            max_depth: 3,
            learning_rate: 0.03,
            l2_reg: 1.0,
            l1_reg: 0.01,
            min_samples_leaf: 20,
            subsample: 0.8,
            colsample_bytree: 0.7,
            gamma: 0.0001,
        }
    }
}

/// Modelo entrenado de Gradient Boosted Decision Trees
#[derive(Clone, Debug)]
pub struct GbdtModel {
    pub base_score: f32,
    pub trees: Vec<DecisionTree>,
    pub learning_rate: f32,
    pub feature_importances: [f32; TOTAL_FEATURES],
}

impl GbdtModel {
    /// Inferencia para una observación de 34 variables
    #[inline(always)]
    pub fn predict(&self, features: &[f32; TOTAL_FEATURES]) -> f32 {
        let mut pred = self.base_score;
        for tree in &self.trees {
            pred += self.learning_rate * tree.predict(features);
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

    /// Retorna el ranking de las características ordenadas por importancia (ganancia acumulada)
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
}

/// Entrenador del algoritmo Gradient Boosting
pub struct GbdtTrainer {
    pub config: GbdtConfig,
}

impl GbdtTrainer {
    pub fn new(config: GbdtConfig) -> Self {
        Self { config }
    }

    /// Ajusta el ensamble de árboles aditivos sobre el dataset
    pub fn fit(
        &self,
        dataset: &TabularDataset,
        mask: Option<&FeatureMask>,
    ) -> Result<GbdtModel, &'static str> {
        let n_samples = dataset.len();
        if n_samples < self.config.min_samples_leaf * 4 {
            return Err("Muestras insuficientes para entrenar GBDT");
        }

        // 1. Filtrar características activas según la máscara
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
            return Err("No hay características activas en la máscara para GBDT");
        }

        // 2. Base score inicial: media del target
        let sum_targets: f32 = dataset.samples.iter().map(|s| s.target).sum();
        let base_score = sum_targets / (n_samples as f32);

        // Vector de predicciones acumuladas
        let mut current_preds = vec![base_score; n_samples];
        let mut trees = Vec::with_capacity(self.config.n_trees);
        let mut feature_importances = [0.0f32; TOTAL_FEATURES];

        // Estado pseudoaleatorio determinista para reproducibilidad (sin dependencias externas)
        let mut rng_state: u64 = 987654321;

        for _tree_idx in 0..self.config.n_trees {
            // Calcular gradientes de primer y segundo orden (Squared Error Loss: L = 0.5 * (y - y_hat)^2)
            // g_i = y_hat - y, h_i = 1.0
            let mut gradients = Vec::with_capacity(n_samples);
            let mut hessians = Vec::with_capacity(n_samples);

            for (i, sample) in dataset.samples.iter().enumerate() {
                let g = current_preds[i] - sample.target;
                gradients.push(g);
                hessians.push(1.0f32);
            }

            // Sub-muestreo de filas (Bagging)
            let mut sample_indices: Vec<usize> = (0..n_samples).collect();
            if self.config.subsample < 0.999 {
                let sample_size = ((n_samples as f32) * self.config.subsample).max(self.config.min_samples_leaf as f32 * 2.0) as usize;
                shuffle_indices(&mut sample_indices, &mut rng_state);
                sample_indices.truncate(sample_size);
            }

            // Sub-muestreo de características (Colsample by tree)
            let mut tree_features = active_features.clone();
            if self.config.colsample_bytree < 0.999 && tree_features.len() > 2 {
                let col_size = ((tree_features.len() as f32) * self.config.colsample_bytree).max(1.0) as usize;
                shuffle_features(&mut tree_features, &mut rng_state);
                tree_features.truncate(col_size);
            }

            // Construir el árbol recursivamente
            let root = self.build_tree(
                &dataset.samples,
                &sample_indices,
                &gradients,
                &hessians,
                &tree_features,
                0,
            );

            // Acumular importancia de características del árbol
            root.compute_feature_importance(&mut feature_importances);

            // Actualizar predicciones actuales con shrinkage (learning rate)
            for (i, sample) in dataset.samples.iter().enumerate() {
                let update = root.predict(&sample.features);
                current_preds[i] += self.config.learning_rate * update;
            }

            trees.push(DecisionTree::new(root));
        }

        Ok(GbdtModel {
            base_score,
            trees,
            learning_rate: self.config.learning_rate,
            feature_importances,
        })
    }

    /// Ajusta el ensamble de árboles con detención temprana (Early Stopping) sobre un conjunto de validación purgado
    pub fn fit_with_early_stopping(
        &self,
        train_set: &TabularDataset,
        val_set: &TabularDataset,
        patience: usize,
        mask: Option<&FeatureMask>,
    ) -> Result<(GbdtModel, usize), &'static str> {
        let n_samples = train_set.len();
        let n_val = val_set.len();
        if n_samples < self.config.min_samples_leaf * 4 || n_val == 0 {
            return Err("Muestras insuficientes para entrenar GBDT con early stopping");
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

        let sum_targets: f32 = train_set.samples.iter().map(|s| s.target).sum();
        let base_score = sum_targets / (n_samples as f32);

        let mut current_preds = vec![base_score; n_samples];
        let mut val_preds = vec![base_score; n_val];

        let mut best_val_mse = f32::MAX;
        let mut best_tree_count = 0;
        let mut no_improvement_count = 0;

        let mut trees = Vec::with_capacity(self.config.n_trees);
        let mut feature_importances = [0.0f32; TOTAL_FEATURES];
        let mut rng_state: u64 = 987654321;

        for tree_idx in 0..self.config.n_trees {
            let mut gradients = Vec::with_capacity(n_samples);
            let mut hessians = Vec::with_capacity(n_samples);

            for (i, sample) in train_set.samples.iter().enumerate() {
                let g = current_preds[i] - sample.target;
                gradients.push(g);
                hessians.push(1.0f32);
            }

            let mut sample_indices: Vec<usize> = (0..n_samples).collect();
            if self.config.subsample < 0.999 {
                let sample_size = ((n_samples as f32) * self.config.subsample).max(self.config.min_samples_leaf as f32 * 2.0) as usize;
                shuffle_indices(&mut sample_indices, &mut rng_state);
                sample_indices.truncate(sample_size);
            }

            let mut tree_features = active_features.clone();
            if self.config.colsample_bytree < 0.999 && tree_features.len() > 2 {
                let col_size = ((tree_features.len() as f32) * self.config.colsample_bytree).max(1.0) as usize;
                shuffle_features(&mut tree_features, &mut rng_state);
                tree_features.truncate(col_size);
            }

            let root = self.build_tree(
                &train_set.samples,
                &sample_indices,
                &gradients,
                &hessians,
                &tree_features,
                0,
            );

            root.compute_feature_importance(&mut feature_importances);

            for (i, sample) in train_set.samples.iter().enumerate() {
                let update = root.predict(&sample.features);
                current_preds[i] += self.config.learning_rate * update;
            }

            // Calcular pérdida en el conjunto de validación
            let mut val_mse = 0.0f32;
            for (i, sample) in val_set.samples.iter().enumerate() {
                let update = root.predict(&sample.features);
                val_preds[i] += self.config.learning_rate * update;
                let diff = val_preds[i] - sample.target;
                val_mse += diff * diff;
            }
            val_mse /= n_val as f32;

            trees.push(DecisionTree::new(root));

            if val_mse < best_val_mse - 1e-7 {
                best_val_mse = val_mse;
                best_tree_count = tree_idx + 1;
                no_improvement_count = 0;
            } else {
                no_improvement_count += 1;
                if no_improvement_count >= patience {
                    break;
                }
            }
        }

        // Podar árboles sobrantes que solo agregan sobreajuste
        trees.truncate(best_tree_count.max(1));

        Ok((
            GbdtModel {
                base_score,
                trees,
                learning_rate: self.config.learning_rate,
                feature_importances,
            },
            best_tree_count,
        ))
    }

    fn build_tree(
        &self,
        samples: &[crate::trees::dataset::DatasetSample],
        sample_indices: &[usize],
        gradients: &[f32],
        hessians: &[f32],
        candidate_features: &[usize],
        depth: usize,
    ) -> TreeNode {
        // Criterios de parada
        if depth >= self.config.max_depth || sample_indices.len() < self.config.min_samples_leaf * 2 {
            let mut total_g = 0.0f32;
            let mut total_h = 0.0f32;
            for &idx in sample_indices {
                total_g += gradients[idx];
                total_h += hessians[idx];
            }
            let leaf_val = calculate_leaf_weight(total_g, total_h, self.config.l1_reg, self.config.l2_reg);
            return TreeNode::Leaf {
                value: leaf_val,
                weight: total_h,
                n_samples: sample_indices.len(),
            };
        }

        // Buscar mejor división
        let best_split = find_best_split(
            samples,
            sample_indices,
            gradients,
            hessians,
            candidate_features,
            self.config.min_samples_leaf,
            self.config.l1_reg,
            self.config.l2_reg,
            self.config.gamma,
        );

        match best_split {
            Some(split) => {
                let left_node = self.build_tree(
                    samples,
                    &split.left_indices,
                    gradients,
                    hessians,
                    candidate_features,
                    depth + 1,
                );
                let right_node = self.build_tree(
                    samples,
                    &split.right_indices,
                    gradients,
                    hessians,
                    candidate_features,
                    depth + 1,
                );

                TreeNode::Internal {
                    feature_idx: split.feature_idx,
                    threshold: split.threshold,
                    gain: split.gain,
                    left: Box::new(left_node),
                    right: Box::new(right_node),
                }
            }
            None => {
                let mut total_g = 0.0f32;
                let mut total_h = 0.0f32;
                for &idx in sample_indices {
                    total_g += gradients[idx];
                    total_h += hessians[idx];
                }
                let leaf_val = calculate_leaf_weight(total_g, total_h, self.config.l1_reg, self.config.l2_reg);
                TreeNode::Leaf {
                    value: leaf_val,
                    weight: total_h,
                    n_samples: sample_indices.len(),
                }
            }
        }
    }
}

#[inline(always)]
fn fast_rand(state: &mut u64) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*state >> 33) as usize
}

fn shuffle_indices(arr: &mut [usize], state: &mut u64) {
    let n = arr.len();
    for i in (1..n).rev() {
        let j = fast_rand(state) % (i + 1);
        arr.swap(i, j);
    }
}

fn shuffle_features(arr: &mut [usize], state: &mut u64) {
    let n = arr.len();
    for i in (1..n).rev() {
        let j = fast_rand(state) % (i + 1);
        arr.swap(i, j);
    }
}
