use egui::{
    ClippedMesh,
    CtxRef,
};
use egui_wgpu_backend::{
    RenderPass, 
    ScreenDescriptor,
};
use egui_winit::State;
use epi::Frame;
use winit::{window::Window, event_loop::EventLoop, event::WindowEvent};

pub enum GuiEvent {
    RequestRedraw,
}

struct RepaintSignal(std::sync::Mutex<winit::event_loop::EventLoopProxy<GuiEvent>>);

impl epi::backend::RepaintSignal for RepaintSignal {
    fn request_repaint(&self) {
        self.0.lock().unwrap().send_event(GuiEvent::RequestRedraw).ok();
    }
}

pub struct Gui {
    context: CtxRef,
    state: State,
    render_pass: RenderPass,
    frame: Frame,
    screen_descriptor: ScreenDescriptor,
    paint_jobs: Option<Vec<ClippedMesh>>,
    pub using_pointer: bool,
    pub using_keyboard: bool,
}

impl Gui {
    pub fn new(
        window: &Window,
        event_loop: &EventLoop<GuiEvent>,
        device: &wgpu::Device, 
        config: &wgpu::SurfaceConfiguration,

    ) -> Self {
        let context = egui::CtxRef::default();
        let state = egui_winit::State::new(&window);
        let render_pass = egui_wgpu_backend::RenderPass::new(&device, config.format, 1);
    
        let repaint_signal = std::sync::Arc::new(RepaintSignal(std::sync::Mutex::new(
            event_loop.create_proxy(),
        )));
        let frame = epi::Frame(std::sync::Arc::new(std::sync::Mutex::new(
            epi::backend::FrameData {
                info: epi::IntegrationInfo {
                    name: "egui_frame_data",
                    web_info: None,
                    cpu_usage: None,
                    native_pixels_per_point: Some(window.scale_factor() as _),
                    prefer_dark_mode: None,
                },
                output: epi::backend::AppOutput::default(),
                repaint_signal: repaint_signal.clone(),
            },
        )));

        let screen_descriptor = ScreenDescriptor {
            physical_width: config.width,
            physical_height: config.height,
            scale_factor: window.scale_factor() as f32,
        };

        Self {
            context,
            state,
            render_pass,
            frame,
            screen_descriptor,
            paint_jobs: None,
            using_pointer: false,
            using_keyboard: false,
        }
    }

    pub fn setup(
        &mut self,
        app: &mut dyn epi::App,
    ) {
        app.setup(&self.context, &self.frame, None);
    }

    pub fn window_event(
        &mut self, 
        event: &WindowEvent
    ) -> bool {
        self.state.on_event(&self.context, &event)
    }

    pub fn update(
        &mut self, 
        window: &Window, 
        app: &mut dyn epi::App,
    ) {
        let frame_start = std::time::Instant::now();
        let raw_input = self.state.take_egui_input(window);
        let (output, shapes) = self.context.run(raw_input, |ctx| {
            // Draw the demo application.
            app.update(ctx, &self.frame);
        });

        self.state.handle_output(&window, &self.context, output);

        self.using_pointer = self.context.wants_pointer_input();
        self.using_keyboard = self.context.wants_keyboard_input();

        let paint_jobs = self.context.tessellate(shapes);
        self.paint_jobs = Some(paint_jobs);

        let frame_time = (std::time::Instant::now() - frame_start).as_secs_f64() as f32;
        self.frame.lock().info.cpu_usage = Some(frame_time);
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device, 
        queue: &wgpu::Queue,
        view: &wgpu::TextureView, 
        encoder: &mut wgpu::CommandEncoder,
    ) {
        match &self.paint_jobs {
            Some(paint_jobs) => {
                //self.render_pass.extract_frame_data(&self.frame);

                // Upload all resources for the GPU.
                self.render_pass.update_texture(&device, &queue, &self.context.font_image());
                self.render_pass.update_user_textures(&device, &queue);
                self.render_pass.update_buffers(&device, &queue, &paint_jobs, &self.screen_descriptor);
        
                // Record all render passes.
                self.render_pass
                    .execute(
                        encoder,
                        &view,
                        &paint_jobs,
                        &self.screen_descriptor,
                        None,
                    )
                    .unwrap();
            },
            None => {
                println!("No paint jobs, aborting render!");
            }
        };
    }
}
