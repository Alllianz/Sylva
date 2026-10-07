use crate::features::FeatureMask;
use crate::features::TOTAL_FEATURES;
use crate::trees::batch::GbdtConfig;
use crate::trees::dataset::TabularDataset;
use crate::trees::gpu::context::GpuContext;
use crate::trees::gpu::flat_tree::FlatEnsemble;
use crate::trees::gpu::inference_pipeline::GpuInferencePipeline;
use crate::trees::gpu::threshold_pipeline::{GpuThresholdPipeline, ThresholdPairInput};
use crate::trees::gpu::trainer::GpuTreeTrainer;
use crate::trees::gpu::types::GpuThresholdResult;
use crate::trees::online::gbdt::OnlineGbdtModel;
use std::error::Error;

/// Motor Cuantitativo Unificado de Alto Rendimiento en GPU (WebGPU / AMD Radeon RX 7700 XT)
pub struct GpuTreeEngine {
    context: GpuContext,
    inference_pipeline: GpuInferencePipeline,
    threshold_pipeline: GpuThresholdPipeline,
    trainer: GpuTreeTrainer,
}

impl GpuTreeEngine {
    /// Inicializa el motor GPU conectándose al adaptador gráfico AMD RDNA 3
    pub fn new() -> Result<Self, Box<dyn Error + Send + Sync>> {
        let context = GpuContext::new()?;
        let inference_pipeline = GpuInferencePipeline::new(context.clone())?;
        let threshold_pipeline = GpuThresholdPipeline::new(context.clone())?;
        let trainer = GpuTreeTrainer::new(context.clone())?;

        Ok(Self {
            context,
            inference_pipeline,
            threshold_pipeline,
            trainer,
        })
    }

    /// Nombre y backend del hardware GPU activo (ej: AMD Radeon RX 7700 XT (DirectX 12 / Vulkan))
    pub fn hardware_info(&self) -> String {
        self.context.info_string()
    }

    /// Predice un TabularDataset completo de forma masiva en GPU
    pub fn predict_dataset(
        &self,
        model: &OnlineGbdtModel,
        dataset: &TabularDataset,
    ) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
        let n_samples = dataset.samples.len();
        if n_samples == 0 {
            return Ok(Vec::new());
        }

        let flat_ensemble = FlatEnsemble::from_online_gbdt(model);

        // Aplanar características contiguas [M * 32]
        let mut flat_features = Vec::with_capacity(n_samples * TOTAL_FEATURES);
        for s in &dataset.samples {
            flat_features.extend_from_slice(&s.features);
        }

        self.inference_pipeline
            .run_inference(&flat_ensemble, &flat_features, n_samples)
    }

    /// Predice un slice de vectores de características en GPU
    pub fn predict_features(
        &self,
        model: &OnlineGbdtModel,
        features: &[[f32; TOTAL_FEATURES]],
    ) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
        let n_samples = features.len();
        if n_samples == 0 {
            return Ok(Vec::new());
        }

        let flat_ensemble = FlatEnsemble::from_online_gbdt(model);

        let mut flat_features = Vec::with_capacity(n_samples * TOTAL_FEATURES);
        for feat in features {
            flat_features.extend_from_slice(feat);
        }

        self.inference_pipeline
            .run_inference(&flat_ensemble, &flat_features, n_samples)
    }

    /// Realiza un barrido masivo de parejas de umbrales en paralelo sobre la GPU
    pub fn sweep_thresholds(
        &self,
        pairs: &[ThresholdPairInput],
        predictions: &[f32],
        bar_returns: &[f32],
    ) -> Result<Vec<GpuThresholdResult>, Box<dyn Error + Send + Sync>> {
        self.threshold_pipeline
            .sweep_thresholds(pairs, predictions, bar_returns)
    }

    /// Entrena un ensamble GBDT completo nativamente en la VRAM de la GPU
    pub fn train_gbdt(
        &self,
        dataset: &TabularDataset,
        config: &GbdtConfig,
        mask: Option<&FeatureMask>,
    ) -> Result<FlatEnsemble, Box<dyn Error + Send + Sync>> {
        self.trainer.train_gbdt(dataset, config, mask)
    }

    /// Inferencia vectorial directa para un FlatEnsemble ya residente en GPU
    pub fn predict_flat(
        &self,
        ensemble: &FlatEnsemble,
        dataset: &TabularDataset,
    ) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
        let n_samples = dataset.samples.len();
        if n_samples == 0 {
            return Ok(Vec::new());
        }

        let mut flat_features = Vec::with_capacity(n_samples * TOTAL_FEATURES);
        for s in &dataset.samples {
            flat_features.extend_from_slice(&s.features);
        }

        self.inference_pipeline
            .run_inference(ensemble, &flat_features, n_samples)
    }
}

