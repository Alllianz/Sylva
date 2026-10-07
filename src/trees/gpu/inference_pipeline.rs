use crate::trees::gpu::context::GpuContext;
use crate::trees::gpu::flat_tree::FlatEnsemble;
use crate::trees::gpu::types::GpuInferenceUniforms;
use std::error::Error;
use wgpu::util::DeviceExt;

/// Pipeline de Cómputo GPU para Inferencia Vectorial Masiva de Árboles
pub struct GpuInferencePipeline {
    context: GpuContext,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuInferencePipeline {
    /// Inicializa y compila el Compute Shader WGSL en la GPU AMD
    pub fn new(context: GpuContext) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let shader_source = include_str!("shaders/tree_infer.wgsl");
        let shader = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sylva Tree Infer WGSL"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tree Infer BindGroupLayout"),
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
                // Binding 1: Features [M * 32]
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
                // Binding 2: Nodes
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
                // Binding 3: Tree Metas
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
                // Binding 4: Predictions [M]
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
            label: Some("Tree Infer PipelineLayout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            ..Default::default()
        });

        let pipeline = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tree Infer ComputePipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            context,
            pipeline,
            bind_group_layout,
        })
    }

    /// Ejecuta la inferencia de todos los árboles del ensamble sobre `n_samples` en la GPU
    pub fn run_inference(
        &self,
        ensemble: &FlatEnsemble,
        features_flat: &[f32],
        n_samples: usize,
    ) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
        if n_samples == 0 || ensemble.nodes.is_empty() {
            return Ok(vec![ensemble.base_score; n_samples]);
        }

        let device = &self.context.device;
        let queue = &self.context.queue;

        // 1. Buffer de Uniformes
        let uniforms = GpuInferenceUniforms {
            n_samples: n_samples as u32,
            n_trees: ensemble.tree_metas.len() as u32,
            base_score: ensemble.base_score,
            learning_rate: ensemble.learning_rate,
        };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Inference Uniforms Buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 2. Buffer de Características (Features)
        let features_bytes: &[u8] = bytemuck::cast_slice(features_flat);
        let features_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Features Storage Buffer"),
            contents: features_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 3. Buffer de Nodos
        let nodes_bytes: &[u8] = bytemuck::cast_slice(&ensemble.nodes);
        let nodes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Nodes Storage Buffer"),
            contents: nodes_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 4. Buffer de Metadatos de Árboles
        let tree_metas_bytes: &[u8] = bytemuck::cast_slice(&ensemble.tree_metas);
        let tree_metas_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tree Metas Storage Buffer"),
            contents: tree_metas_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        // 5. Buffer de Salida de Predicciones y Staging para lectura
        let out_bytes_size = (n_samples * std::mem::size_of::<f32>()) as u64;
        let predictions_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Predictions Output Buffer"),
            size: out_bytes_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Readback Buffer"),
            size: out_bytes_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 6. Bind Group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Inference BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: features_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: nodes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tree_metas_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: predictions_buf.as_entire_binding(),
                },
            ],
        });

        // 7. Dispatch de Compute Pass
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tree Infer Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tree Infer Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroups_x = ((n_samples as u32) + 63) / 64;
            compute_pass.dispatch_workgroups(workgroups_x, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&predictions_buf, 0, &staging_buf, 0, out_bytes_size);
        queue.submit(Some(encoder.finish()));

        // 8. Mapear y leer de regreso a la RAM
        let buffer_slice = staging_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        receiver.recv()??;

        let data = buffer_slice.get_mapped_range()?;
        let result_f32: &[f32] = bytemuck::cast_slice(&data);
        let output = result_f32.to_vec();

        drop(data);
        staging_buf.unmap();

        Ok(output)
    }
}
