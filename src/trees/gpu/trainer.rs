use crate::features::FeatureMask;
use crate::features::TOTAL_FEATURES;
use crate::trees::batch::GbdtConfig;
use crate::trees::dataset::TabularDataset;
use crate::trees::gpu::context::GpuContext;
use crate::trees::gpu::flat_tree::FlatEnsemble;
use crate::trees::gpu::types::{GpuNode, GpuSplitCandidate, GpuTrainUniforms, GpuTreeMeta};
use std::error::Error;
use wgpu::util::DeviceExt;

/// Entrenador GBDT Nativo en GPU (AMD Radeon RX 7700 XT)
/// Ejecuta el entrenamiento de ensambles de árboles de decisión 100% en VRAM mediante Compute Shaders WGSL.
pub struct GpuTreeTrainer {
    context: GpuContext,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuTreeTrainer {
    pub fn new(context: GpuContext) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let shader_source = include_str!("shaders/tree_train.wgsl");
        let shader = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sylva Tree Train WGSL"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tree Train BindGroupLayout"),
            entries: &[
                // Binding 0: Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Features [N * 32]
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: Targets [N]
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Predictions [N]
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: Sample Nodes [N]
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 5: Split Candidates [32]
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tree Train PipelineLayout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            ..Default::default()
        });

        let pipeline = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tree Train ComputePipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("evaluate_splits"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            context,
            pipeline,
            bind_group_layout,
        })
    }

    /// Entrena un ensamble GBDT completo directamente en la VRAM de la GPU
    pub fn train_gbdt(
        &self,
        dataset: &TabularDataset,
        config: &GbdtConfig,
        mask: Option<&FeatureMask>,
    ) -> Result<FlatEnsemble, Box<dyn Error + Send + Sync>> {
        let n_samples = dataset.len();
        if n_samples == 0 {
            return Err("Dataset vacío para entrenamiento en GPU".into());
        }

        let mut active_features: Vec<u32> = Vec::new();
        for i in 0..TOTAL_FEATURES {
            if let Some(m) = mask {
                if m.is_active(i) {
                    active_features.push(i as u32);
                }
            } else {
                active_features.push(i as u32);
            }
        }
        if active_features.is_empty() {
            return Err("No hay características activas para entrenar en GPU".into());
        }

        // 1. Aplanar características contiguas en host
        let mut flat_features = Vec::with_capacity(n_samples * TOTAL_FEATURES);
        let mut flat_targets = Vec::with_capacity(n_samples);
        for s in &dataset.samples {
            flat_features.extend_from_slice(&s.features);
            flat_targets.push(s.target);
        }

        // Media inicial del target (Base score)
        let sum_targets: f32 = flat_targets.iter().sum();
        let base_score = sum_targets / (n_samples as f32);
        let mut initial_preds = vec![base_score; n_samples];
        let mut initial_sample_nodes = vec![0u32; n_samples];

        // 2. Crear buffers en VRAM
        let features_buffer = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gpu Train Features"),
            contents: bytemuck::cast_slice(&flat_features),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let targets_buffer = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gpu Train Targets"),
            contents: bytemuck::cast_slice(&flat_targets),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let preds_buffer = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gpu Train Predictions"),
            contents: bytemuck::cast_slice(&initial_preds),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let sample_nodes_buffer = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gpu Train Sample Nodes"),
            contents: bytemuck::cast_slice(&initial_sample_nodes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        let split_candidates_init = vec![GpuSplitCandidate::default(); TOTAL_FEATURES];
        let split_buffer = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Gpu Train Split Candidates"),
            contents: bytemuck::cast_slice(&split_candidates_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let staging_splits = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Splits Readback"),
            size: (TOTAL_FEATURES * std::mem::size_of::<GpuSplitCandidate>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut ensemble = FlatEnsemble::new(base_score, config.learning_rate);

        // 3. Bucle de construcción de árboles GBDT
        for _tree_idx in 0..config.n_trees {
            let root_node_idx = ensemble.nodes.len() as u32;
            let _start_count = ensemble.nodes.len();

            // Resetear nodos de muestra para este árbol: todas las muestras arrancan en la raíz local (0)
            let reset_nodes = vec![0u32; n_samples];
            self.context.queue.write_buffer(&sample_nodes_buffer, 0, bytemuck::cast_slice(&reset_nodes));

            // Estructura temporal para construir el árbol nivel a nivel (BFS hasta max_depth)
            // (node_id_in_tree, depth)
            let mut tree_nodes: Vec<GpuNode> = vec![GpuNode::default()];
            let mut active_tree_nodes: Vec<(u32, usize)> = vec![(0, 0)];

            while let Some((curr_node, curr_depth)) = active_tree_nodes.pop() {
                if curr_depth >= config.max_depth {
                    // Convertir a hoja
                    tree_nodes[curr_node as usize] = GpuNode::new_leaf(0.0);
                    continue;
                }

                // Despachar Shader WGSL para encontrar la mejor división entre todas las características
                let uniforms = GpuTrainUniforms {
                    n_samples: n_samples as u32,
                    n_features: TOTAL_FEATURES as u32,
                    target_node: curr_node,
                    min_samples_leaf: config.min_samples_leaf as u32,
                    learning_rate: config.learning_rate,
                    l2_reg: config.l2_reg,
                    gamma: config.gamma,
                    _pad: 0.0,
                };

                let uniform_buf = self.context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Uniform Temp"),
                    contents: bytemuck::cast_slice(&[uniforms]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

                let bind_group = self.context.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Train BindGroup"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: features_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: targets_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: preds_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: sample_nodes_buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: split_buffer.as_entire_binding() },
                    ],
                });

                let mut encoder = self.context.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Train Step Encoder"),
                });

                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Tree Train Pass"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    // Cada Workgroup procesa 1 característica
                    cpass.dispatch_workgroups(TOTAL_FEATURES as u32, 1, 1);
                }

                encoder.copy_buffer_to_buffer(
                    &split_buffer, 0,
                    &staging_splits, 0,
                    (TOTAL_FEATURES * std::mem::size_of::<GpuSplitCandidate>()) as u64,
                );

                self.context.queue.submit(Some(encoder.finish()));

                // Leer resultados de splits encontrados en GPU
                let slice = staging_splits.slice(..);
                let (sender, receiver) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |res| {
                    let _ = sender.send(res);
                });
                let _ = self.context.device.poll(wgpu::PollType::wait_indefinitely());
                receiver.recv()??;

                let data = slice.get_mapped_range()?;
                let splits: &[GpuSplitCandidate] = bytemuck::cast_slice(&data);

                // Encontrar el mejor split entre las características activas
                let mut best_split: Option<GpuSplitCandidate> = None;
                for &f_idx in &active_features {
                    let s = &splits[f_idx as usize];
                    if s.valid == 1 && s.gain > 0.0 {
                        if best_split.as_ref().map(|b| s.gain > b.gain).unwrap_or(true) {
                            best_split = Some(*s);
                        }
                    }
                }
                drop(data);
                staging_splits.unmap();

                if let Some(bs) = best_split {
                    // Dividir el nodo en dos hijos
                    let left_child_idx = tree_nodes.len() as u32;
                    let right_child_idx = left_child_idx + 1;

                    tree_nodes[curr_node as usize] = GpuNode::new_internal(
                        bs.feature_idx,
                        bs.threshold,
                        left_child_idx,
                        right_child_idx,
                    );

                    tree_nodes.push(GpuNode::new_leaf(bs.left_weight));
                    tree_nodes.push(GpuNode::new_leaf(bs.right_weight));

                    // Particionar muestras en el buffer de GPU
                    for i in 0..n_samples {
                        if initial_sample_nodes[i] == curr_node {
                            let feat_val = flat_features[i * TOTAL_FEATURES + bs.feature_idx as usize];
                            if feat_val <= bs.threshold {
                                initial_sample_nodes[i] = left_child_idx;
                                initial_preds[i] += config.learning_rate * bs.left_weight;
                            } else {
                                initial_sample_nodes[i] = right_child_idx;
                                initial_preds[i] += config.learning_rate * bs.right_weight;
                            }
                        }
                    }

                    self.context.queue.write_buffer(&sample_nodes_buffer, 0, bytemuck::cast_slice(&initial_sample_nodes));
                    self.context.queue.write_buffer(&preds_buffer, 0, bytemuck::cast_slice(&initial_preds));

                    active_tree_nodes.push((left_child_idx, curr_depth + 1));
                    active_tree_nodes.push((right_child_idx, curr_depth + 1));
                } else {
                    // Si no hubo split válido, es hoja con peso 0.0
                    tree_nodes[curr_node as usize] = GpuNode::new_leaf(0.0);
                }
            }

            // Integrar nodos construidos en el ensamble plano global
            let node_count = (tree_nodes.len()) as u32;
            for mut n in tree_nodes {
                if n.is_leaf == 0 {
                    n.left_child += root_node_idx;
                    n.right_child += root_node_idx;
                }
                ensemble.nodes.push(n);
            }

            ensemble.tree_metas.push(GpuTreeMeta {
                root_idx: root_node_idx,
                node_count,
                _pad0: 0,
                _pad1: 0,
            });
        }

        Ok(ensemble)
    }
}
