use crate::data::db::Kline;
use crate::features::cached::CachedIndicators;
use crate::features::mask::FeatureMask;
use crate::features::names::TOTAL_FEATURES;
use crate::features::normalizer::FeatureNormalizer;

/// Representa una muestra individual del dataset tabular
#[derive(Clone, Debug)]
pub struct DatasetSample {
    pub features: [f32; TOTAL_FEATURES],
    pub target: f32, // Retorno forward: (Close_{t+h} - Close_t) / Close_t
    pub timestamp: i64,
    pub index: usize,
}

/// Contenedor de datos tabulares para entrenamiento y validación
#[derive(Clone, Debug)]
pub struct TabularDataset {
    pub samples: Vec<DatasetSample>,
    pub feature_names: Vec<String>,
}

impl TabularDataset {
    pub fn new(samples: Vec<DatasetSample>) -> Self {
        let feature_names = crate::features::feature_names()
            .iter()
            .map(|&s| s.to_string())
            .collect();
        Self {
            samples,
            feature_names,
        }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Extrae un dataset tabular estrictamente causal a partir del rango cronológico [start_idx..end_idx]
    pub fn extract_from_klines(
        klines: &[Kline],
        start_idx: usize,
        end_idx: usize,
        rolling_window: usize,
        target_horizon: usize,
        cached: &CachedIndicators,
        mask: Option<&FeatureMask>,
    ) -> Result<Self, &'static str> {
        let n_klines = klines.len();
        if n_klines < rolling_window + target_horizon + 10 {
            return Err("Historial de velas insuficiente para extraer dataset tabular");
        }

        let normalizer = FeatureNormalizer::new(rolling_window);
        let normalizer = if let Some(m) = mask {
            normalizer.with_mask(m.clone())
        } else {
            normalizer
        };

        let mut samples = Vec::with_capacity(end_idx.saturating_sub(start_idx));
        let effective_start = start_idx.max(rolling_window);
        let effective_end = end_idx.min(n_klines.saturating_sub(target_horizon));

        if effective_start >= effective_end {
            return Err("Rango temporal inválido para extracción tabular");
        }

        for t in effective_start..effective_end {
            let curr_p = klines[t].close;
            let target_p = klines[t + target_horizon].close;
            if curr_p <= 0.0 || target_p <= 0.0 {
                continue;
            }

            // Retorno porcentual objetivo forward estrictamente causal
            let target = ((target_p - curr_p) / curr_p) as f32;

            // Extraer features normalizadas con Z-score rodante [t-W+1 ..= t]
            let feats = normalizer.normalize(klines, t, cached);

            samples.push(DatasetSample {
                features: feats,
                target,
                timestamp: klines[t].timestamp,
                index: t,
            });
        }

        let feature_names = crate::features::feature_names()
            .iter()
            .map(|&s| s.to_string())
            .collect();

        Ok(Self {
            samples,
            feature_names,
        })
    }

    /// Retorna una partición del dataset por índices específicos
    pub fn subset(&self, indices: &[usize]) -> Self {
        let mut sub = Vec::with_capacity(indices.len());
        for &idx in indices {
            if idx < self.samples.len() {
                sub.push(self.samples[idx].clone());
            }
        }
        Self::new(sub)
    }
}
