use std::error::Error;
use std::sync::Arc;

/// Contexto de inicialización de la GPU (Adaptador AMD, Device y Command Queue)
#[derive(Clone)]
pub struct GpuContext {
    pub instance: Arc<wgpu::Instance>,
    pub adapter: Arc<wgpu::Adapter>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub device_name: String,
    pub backend_name: String,
}

impl GpuContext {
    /// Inicializa la GPU solicitando explícitamente alto rendimiento (AMD Radeon RX 7700 XT)
    pub fn new() -> Result<Self, Box<dyn Error + Send + Sync>> {
        let instance = wgpu::Instance::default();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|e| format!("Error solicitando adaptador GPU: {:?}", e))?;

        let info = adapter.get_info();
        let device_name = info.name.clone();
        let backend_name = format!("{:?}", info.backend);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Sylva GPU Engine"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            },
        ))
        .map_err(|e| format!("Error creando dispositivo GPU: {:?}", e))?;

        Ok(Self {
            instance: Arc::new(instance),
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
            device_name,
            backend_name,
        })
    }

    /// Retorna una descripción formateada del hardware gráfico activo
    pub fn info_string(&self) -> String {
        format!("{} ({})", self.device_name, self.backend_name)
    }
}
