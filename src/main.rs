use cgmath::prelude::*;
use egui::{
    Slider,
    Ui,
};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
use wgpu::util::DeviceExt;

mod texture;
mod mesh;
mod camera;
mod gui;

use camera::Camera;
use mesh::Vertex;
use gui::{Gui, GuiEvent};

const ROTATION_SPEED: f32 = 2.0 * std::f32::consts::PI / 60.0;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
}

impl mesh::Vertex for InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ]
        }
    }
}

struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        let model = cgmath::Matrix4::from_translation(self.position) * cgmath::Matrix4::from(self.rotation);
        InstanceRaw {
            model: model.into(),
            normal: cgmath::Matrix3::from(self.rotation).into(), 
        }
    }
}

fn quat_mul(q: cgmath::Quaternion<f32>, r: cgmath::Quaternion<f32>) -> cgmath::Quaternion<f32> {
    let w = r.s * q.s - r.v.x * q.v.x - r.v.y * q.v.y - r.v.z * q.v.z;
    let xi = r.s * q.v.x + r.v.x * q.s - r.v.y * q.v.z + r.v.z * q.v.y;
    let yj = r.s * q.v.y + r.v.x * q.v.z + r.v.y * q.s - r.v.z * q.v.x;
    let zk = r.s * q.v.z - r.v.x * q.v.y + r.v.y * q.v.x + r.v.z * q.s;

    cgmath::Quaternion::new(w, xi, yj, zk)
}

// #[repr(C)]
// #[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
// struct LightUniform {
//     position: [f32; 3],
//     // Uniforms require 16 byte spacing, so we need padding here
//     _padding: u32,
//     color: [f32; 3],
// }

fn create_render_pipeline(
    label: &str,
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(&shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: vertex_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    alpha: wgpu::BlendComponent::REPLACE,
                    color: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            }],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            //cull_mode: None,
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
    })
}

struct WirePass {
    clear_color: wgpu::Color,
    radius: f32,
    iterations: u32,
    mesh: mesh::Mesh,
    instances: Vec<Instance>,
    instance_buffer: wgpu::Buffer,
    render_pipeline: wgpu::RenderPipeline,
}

impl WirePass {
    fn new(
        device: &wgpu::Device, 
        config: &wgpu::SurfaceConfiguration,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let clear_color = wgpu::Color::BLACK;

        let radius = 1.0;
        let iterations = 2;

        let mesh = mesh::Mesh::icosphere(
            &device,
            radius,
            iterations,
            false,
        ).unwrap();

        const NUM_INSTANCES_PER_DIM: u32 = 10;
        const SPACE_BETWEEN: f32 = 3.0;
        let instances = (0..NUM_INSTANCES_PER_DIM).flat_map(|z| {
            (0..NUM_INSTANCES_PER_DIM).map(move |x| {
                let x = SPACE_BETWEEN * (x as f32 - NUM_INSTANCES_PER_DIM as f32 / 2.0);
                let z = SPACE_BETWEEN * (z as f32 - NUM_INSTANCES_PER_DIM as f32 / 2.0);

                let position = cgmath::Vector3 { x, y: 0.0, z };

                let rotation = if position.is_zero() {
                    cgmath::Quaternion::from_axis_angle(cgmath::Vector3::unit_z(), cgmath::Deg(0.0))
                } else {
                    cgmath::Quaternion::from_axis_angle(position.normalize(), cgmath::Deg(45.0))
                };

                Instance {
                    position,
                    rotation,
                }
            })
        }).collect::<Vec<_>>();

        let instance_buffer = {
            let instance_data = instances.iter()
                .map(Instance::to_raw)
                .collect::<Vec<_>>();
            
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Wire Instance Buffer"),
                contents: bytemuck::cast_slice(&instance_data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            })
        };

        let render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Wire Pipeline Layout"),
                bind_group_layouts: &[
                    //&texture_bind_group_layout,
                    &camera_bind_group_layout,
                    // &light_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Wire Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("wire.wgsl").into()),
            };
            create_render_pipeline(
                "Wire Render Pipeline",
                &device,
                &layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                // &[model::ModelVertex::desc(), InstanceRaw::desc()],
                &[mesh::MeshVertex::desc(), InstanceRaw::desc()],
                shader,
            )
        };

        Self {
            clear_color,
            radius,
            iterations,
            mesh,
            instances,
            instance_buffer,
            render_pipeline,
        }
    }

    fn remesh(
        &mut self, 
        device: &wgpu::Device,
    )
    {
        self.mesh = mesh::Mesh::icosphere(
            &device,
            self.radius,
            self.iterations,
            false
        ).unwrap();
    }

    fn update(&mut self, _dt: std::time::Duration, queue: &mut wgpu::Queue) {
        // Update the instances.
        for instance in &mut self.instances {
            let amount = cgmath::Quaternion::from_angle_y(cgmath::Rad(ROTATION_SPEED));
            let current = instance.rotation;
            instance.rotation = quat_mul(amount, current);
        }
        let instance_data = self.instances.iter()
            .map(Instance::to_raw)
            .collect::<Vec<_>>();
        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instance_data));
    }

    fn render(
        &self, 
        view: &wgpu::TextureView, 
        encoder: &mut wgpu::CommandEncoder,
        depth_texture: &texture::Texture,
        camera_bind_group: &wgpu::BindGroup,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Wire Render Pass"),
            color_attachments: &[
                wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: true,
                    },
                }
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        });
        
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));

        use crate::mesh::DrawMesh;
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.draw_mesh_instanced(
            &self.mesh,
            0..self.instances.len() as u32,
            Some(vec![&camera_bind_group]),
            // &self.light_bind_group,
        );
    }
}

struct DisplacePass {
    texture: texture::Texture,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    mesh: mesh::Mesh,
    render_pipeline: wgpu::RenderPipeline,
}

impl DisplacePass {
    fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let texture = texture::Texture::create_render_texture(device, config, "Displace Texture");

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Displace Pass Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    count: None,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    visibility: wgpu::ShaderStages::FRAGMENT,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    count: None,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    visibility: wgpu::ShaderStages::FRAGMENT,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Displace Bind Group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
            ],
        });

        let mesh = mesh::Mesh::quad(
            &device,
            2.0,
            2.0,
            true,
        ).unwrap();

        let render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Displace Pipeline Layout"),
                bind_group_layouts: &[
                    &layout,
                ],
                push_constant_ranges: &[],
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Displace Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("displace.wgsl").into()),
            };
            create_render_pipeline(
                "Displace Render Pipeline",
                &device,
                &layout,
                config.format,
                None,
                &[mesh::MeshVertex::desc()],
                shader,
            )
        };

        Self {
            texture,
            layout,
            bind_group,
            mesh,
            render_pipeline,
        }
    }

    fn resize(&mut self, device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) {
        self.texture = texture::Texture::create_render_texture(device, config, "Displace Texture");
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Displace Bind Group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.texture.sampler),
                },
            ],
        });
    }

    fn render(
        &self, 
        view: &wgpu::TextureView, 
        encoder: &mut wgpu::CommandEncoder
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Displace Render Pass"),
            color_attachments: &[
                wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                }
            ],
            depth_stencil_attachment: None,
        });

        use crate::mesh::DrawMesh;
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.draw_mesh(
            &self.mesh,
            Some(vec![&self.bind_group]),
        );
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    depth_texture: texture::Texture,
    depth_bind_group: wgpu::BindGroup,
    camera: Camera,
    // light_uniform: LightUniform,
    // light_buffer: wgpu::Buffer,
    // light_bind_group: wgpu::BindGroup,
    // light_render_pipeline: wgpu::RenderPipeline,
    wire_pass: WirePass,
    displace_pass: DisplacePass,
    mouse_pressed: bool,
}

impl epi::App for State {
    fn name(&self) -> &str {
        "easy"
    }

    fn update(&mut self, ctx: &egui::CtxRef, _frame: &epi::Frame) {
        egui::Window::new("easy")
            //.frame(egui::containers::Frame::dark_canvas(&ctx.style()))
            .show(ctx, |ui| self.ui(ui));
    }
}

impl State {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                label: None,
            },
            None, 
        ).await.unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&device, &config);

        let depth_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Depth Bind Group Layout"),
            entries: &[
                // Diffuse
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        //sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        sample_type: wgpu::TextureSampleType::Depth,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    //ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // // Normal
                // wgpu::BindGroupLayoutEntry {
                //     binding: 2,
                //     visibility: wgpu::ShaderStages::FRAGMENT,
                //     ty: wgpu::BindingType::Texture {
                //         multisampled: false,
                //         view_dimension: wgpu::TextureViewDimension::D2,
                //         sample_type: wgpu::TextureSampleType::Float { filterable: true },
                //     },
                //     count: None,
                // },
                // wgpu::BindGroupLayoutEntry {
                //     binding: 3,
                //     visibility: wgpu::ShaderStages::FRAGMENT,
                //     ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                //     count: None,
                // },
            ],
        });

        let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth_texture"); 

        let depth_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                label: Some("Depth Bind Group"),
                layout: &depth_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&depth_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&depth_texture.sampler),
                    }
                ],
            }
        );  

        let camera = Camera::new(
            &device,
            (0.0, 5.0, 10.0),
            cgmath::Deg(-90.0), cgmath::Deg(-20.0),
            config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0,
        );

        // let light_uniform = LightUniform {
        //     position: [2.0, 2.0, 2.0],
        //     _padding: 0,
        //     color: [1.0, 1.0, 1.0],
        // };

        // let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some("Light VB"),
        //     contents: bytemuck::cast_slice(&[light_uniform]),
        //     usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        // });

        // let light_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        //     label:None,
        //     entries: &[wgpu::BindGroupLayoutEntry {
        //         binding: 0,
        //         visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        //         ty: wgpu::BindingType::Buffer {
        //             ty: wgpu::BufferBindingType::Uniform,
        //             has_dynamic_offset: false,
        //             min_binding_size: None,
        //         },
        //         count: None,
        //     }],
        // });
        // let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        //     label: None,
        //     layout: &light_bind_group_layout,
        //     entries: &[wgpu::BindGroupEntry {
        //         binding: 0,
        //         resource: light_buffer.as_entire_binding(),
        //     }],
        // });

        let wire_pass = WirePass::new(
            &device, 
            &config,
            &camera.layout
        );

        let displace_pass = DisplacePass::new(
            &device,
            &config,
        );

        // let light_render_pipeline = {
        //     let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        //         label: Some("Light Pipeline Layout"),
        //         bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
        //         push_constant_ranges: &[],
        //     });
        //     let shader = wgpu::ShaderModuleDescriptor {
        //         label: Some("Light Shader"),
        //         source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
        //     };
        //     create_render_pipeline(
        //         &device,
        //         &layout,
        //         config.format,
        //         Some(texture::Texture::DEPTH_FORMAT),
        //         &[model::ModelVertex::desc()],
        //         shader,
        //     )
        // };

        Self {
            surface,
            device,
            queue,
            config,
            size,
            depth_texture,
            depth_bind_group,
            camera,
            // light_uniform,
            // light_buffer,
            // light_bind_group,
            // light_render_pipeline,
            wire_pass,
            displace_pass,
            mouse_pressed: false,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            self.camera.projection.resize(new_size.width, new_size.height);

            self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        
            self.displace_pass.resize(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &DeviceEvent) -> bool {
        match event {
            DeviceEvent::Key(
                KeyboardInput {
                    virtual_keycode: Some(key),
                    state,
                    ..
                }
            ) => {
                match key {
                    // VirtualKeyCode::Numpad0 | VirtualKeyCode::Key0 => {
                    //     self.wire_pass.remesh(&self.device, 1.0, 0);
                    //     true
                    // }
                    // VirtualKeyCode::Numpad1 | VirtualKeyCode::Key1 => {
                    //     self.wire_pass.remesh(&self.device, 1.0, 1);
                    //     true
                    // }
                    // VirtualKeyCode::Numpad2 | VirtualKeyCode::Key2 => {
                    //     self.wire_pass.remesh(&self.device, 1.0, 2);
                    //     true
                    // }
                    // VirtualKeyCode::Numpad3 | VirtualKeyCode::Key3 => {
                    //     self.wire_pass.remesh(&self.device, 1.0, 3);
                    //     true
                    // }
                    _ => self.camera.controller.process_keyboard(*key, *state),
                }
            },
            DeviceEvent::MouseWheel {
                delta,
                ..
            } => {
                self.camera.controller.process_scroll(delta);
                true
            }
            DeviceEvent::Button {
                button: 1, // Left
                state,
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            DeviceEvent::MouseMotion { 
                delta
            } => {
                if self.mouse_pressed {
                    self.camera.controller.process_mouse(delta.0, delta.1);
                }
                true
            }
            _ => false,
        }
    }

    fn ui(&mut self, ui: &mut Ui) {
        ui.label(format!("Mouse pressed?: {}", self.mouse_pressed));
        if ui.add(Slider::new(&mut self.wire_pass.radius, 0.0..=5.0).text("radius")).changed() {
            self.wire_pass.remesh(&self.device);
        }
    }

    fn update(&mut self, dt: std::time::Duration) {
        self.camera.update(dt, &mut self.queue);
    
        // Update the light.
        // let prev_pos: cgmath::Vector3<_> = self.light_uniform.position.into();
        // self.light_uniform.position = (cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(60.0 * dt.as_secs_f32()))
        //     * prev_pos).into();
        // self.queue.write_buffer(&self.light_buffer, 0, bytemuck::cast_slice(&[self.light_uniform]));

        self.wire_pass.update(dt, &mut self.queue);
    }

    fn render(
        &mut self, 
        output_view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), wgpu::SurfaceError> {
        self.wire_pass.render(
            //&view,
            &self.displace_pass.texture.view,
            encoder,
            &self.depth_texture,
            &self.camera.bind_group,
        );

        self.displace_pass.render(
            &output_view,
            encoder,
        );

        {
            // use crate::model::DrawLight;
            // render_pass.set_pipeline(&self.light_render_pipeline);
            // render_pass.draw_light_model(
            //     &self.obj_model,
            //     &self.camera_bind_group,
            //     &self.light_bind_group,
            // );
        }

        Ok(())
    }
}

fn main() {
    env_logger::init();
    //let event_loop = EventLoop::new();
    let event_loop = EventLoop::with_user_event();
    let window = WindowBuilder::new()
        .build(&event_loop)
        .unwrap();

    // Wait for State::new to finish...
    let mut state = pollster::block_on(State::new(&window));
    let mut last_render_time = std::time::Instant::now();

    // Set up gui.
    let mut gui = Gui::new(
        &window, 
        &event_loop,
        &state.device, 
        &state.config,
    );

    // Call epi setup once.
    gui.setup(&mut state);

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::DeviceEvent {
                ref event,
                ..
            } => {
                match event {
                    DeviceEvent::Key(..) => {
                        if !gui.using_keyboard {
                            state.input(event);
                        }
                    }
                    DeviceEvent::Button{..} 
                    | DeviceEvent::MouseMotion{..}
                    | DeviceEvent::MouseWheel{..} => {
                        if !gui.using_pointer {
                            state.input(event);
                        }
                    }
                    _ => {
                        state.input(event);
                    }
                }
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                if !gui.window_event(&event) {
                    match event {
                        WindowEvent::CloseRequested 
                        | WindowEvent::KeyboardInput {
                            input: KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                            ..
                        } => *control_flow = ControlFlow::Exit,
                        WindowEvent::Resized(physical_size) => {
                            state.resize(*physical_size);
                        }
                        WindowEvent::ScaleFactorChanged { new_inner_size,.. } => {
                            // new_inner_size is &&mut so we have to dereference 2x
                            state.resize(**new_inner_size);
                        }
                        _ => {}
                    }
                }
            }
            Event::RedrawRequested(_) => {
                let now = std::time::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;

                gui.update(&window, &mut state);
                state.update(dt);

                let output = state.surface.get_current_texture().unwrap();
                let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

                match state.render(&view, &mut encoder) {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // Quit if the system is out of memory
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }

                gui.render(&state.device, &state.queue, &view, &mut encoder);

                state.queue.submit(std::iter::once(encoder.finish()));

                output.present();
            }
            Event::MainEventsCleared | Event::UserEvent(GuiEvent::RequestRedraw) => {
                // Manually request a redraw
                window.request_redraw();
            }
            _ => {}
        }
    });
}