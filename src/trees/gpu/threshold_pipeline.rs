use crate::trees::gpu::context::GpuContext;
use crate::trees::gpu::types::GpuThresholdResult;
use bytemuck::{Pod, Zeroable};
use std::error::Error;
use wgpu::util::DeviceExt;

/// Entrada de par de umbrales para el barrido en GPU (16 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct ThresholdPairInput {
    pub thr_long: f32,
    pub thr_short: f32,
    pub fee_rate: f32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
struct SweepUniforms {
    pub n_samples: u32,
    pub n_pairs: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

/// Pipeline de Cómputo GPU para Barrido Masivo Paralelo de Umbrales
pub struct GpuThresholdPipeline {
    context: GpuContext,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl GpuThresholdPipeline {
    pub fn new(context: GpuContext) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let shader_source = include_str!("shaders/threshold_sweep.wgsl");
        let shader = context.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Threshold Sweep WGSL"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Threshold Sweep BindGroupLayout"),
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
                // Binding 1: Pairs
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
                // Binding 2: Predictions [M]
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
                // Binding 3: Bar Returns [M]
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
                // Binding 4: Results [P]
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
            label: Some("Threshold Sweep PipelineLayout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            ..Default::default()
        });

        let pipeline = context.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Threshold Sweep ComputePipeline"),
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

    /// Ejecuta el barrido de cientos de parejas de umbrales sobre predicciones y retornos en GPU
    pub fn sweep_thresholds(
        &self,
        pairs: &[ThresholdPairInput],
        predictions: &[f32],
        bar_returns: &[f32],
    ) -> Result<Vec<GpuThresholdResult>, Box<dyn Error + Send + Sync>> {
        let n_pairs = pairs.len();
        let n_samples = predictions.len();

        if n_pairs == 0 || n_samples == 0 {
            return Ok(Vec::new());
        }

        let device = &self.context.device;
        let queue = &self.context.queue;

        let uniforms = SweepUniforms {
            n_samples: n_samples as u32,
            n_pairs: n_pairs as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sweep Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let pairs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Pairs Buffer"),
            contents: bytemuck::cast_slice(pairs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let preds_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Predictions Buffer"),
            contents: bytemuck::cast_slice(predictions),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let rets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bar Returns Buffer"),
            contents: bytemuck::cast_slice(bar_returns),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let out_bytes_size = (n_pairs * std::mem::size_of::<GpuThresholdResult>()) as u64;
        let results_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sweep Results Storage"),
            size: out_bytes_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sweep Staging Buffer"),
            size: out_bytes_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sweep BindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pairs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: preds_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: rets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: results_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Threshold Sweep Command Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sweep Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroups_x = ((n_pairs as u32) + 63) / 64;
            compute_pass.dispatch_workgroups(workgroups_x, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&results_buf, 0, &staging_buf, 0, out_bytes_size);
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });

        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        receiver.recv()??;

        let data = buffer_slice.get_mapped_range()?;
        let sweep_res: &[GpuThresholdResult] = bytemuck::cast_slice(&data);
        let output = sweep_res.to_vec();

        drop(data);
        staging_buf.unmap();

        Ok(output)
    }
}
