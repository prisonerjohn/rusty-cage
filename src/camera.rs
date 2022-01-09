use cgmath::*;
use winit::event::*;
use winit::dpi::PhysicalPosition;
use std::time::Duration;
use std::f32::consts::FRAC_PI_2;
use wgpu::util::DeviceExt;

#[rustfmt::skip]
const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;

#[derive(Debug)]
pub struct Eye {
    pub position: Point3<f32>,
    yaw: Rad<f32>,
    pitch: Rad<f32>,
}

impl Eye {
    fn new<
        V: Into<Point3<f32>>,
        Y: Into<Rad<f32>>,
        P: Into<Rad<f32>>,
    >(
        position: V,
        yaw: Y,
        pitch: P,
    ) -> Self {
        Self {
            position: position.into(),
            yaw: yaw.into(),
            pitch: pitch.into(),
        }
    }

    fn calc_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_to_rh(
            self.position,
            Vector3::new(
                self.yaw.0.cos(),
                self.pitch.0.sin(),
                self.yaw.0.sin(),
            ).normalize(),
            Vector3::unit_y(),
        )
    }
}

pub struct Projection {
    aspect: f32,
    fov_y: Rad<f32>,
    z_near: f32,
    z_far: f32,
}

impl Projection {
    fn new<F: Into<Rad<f32>>>(
        width: u32,
        height: u32,
        fov_y: F,
        z_near: f32,
        z_far: f32,
    ) -> Self {
        Self {
            aspect: width as f32 / height as f32,
            fov_y: fov_y.into(),
            z_near,
            z_far,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    fn calc_matrix(&self) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX * perspective(self.fov_y, self.aspect, self.z_near, self.z_far)
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
    view_pos: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

impl Uniform {
    fn new() -> Self {
        Self {
            view_pos: [0.0; 4],
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, eye: &Eye, projection: &Projection) {
        self.view_pos = eye.position.to_homogeneous().into();
        self.view_proj = (projection.calc_matrix() * eye.calc_matrix()).into();
    }
}

#[derive(Debug)]
pub struct Controller {
    move_left: f32,
    move_right: f32,
    move_forward: f32,
    move_backward: f32,
    move_up: f32,
    move_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    scroll: f32,
    speed: f32,
    sensitivity: f32,
}

impl Controller {
    fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            move_left: 0.0,
            move_right: 0.0,
            move_forward: 0.0,
            move_backward: 0.0,
            move_up: 0.0,
            move_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            speed,
            sensitivity,
        }
    }

    pub fn process_keyboard(&mut self, key: VirtualKeyCode, state: ElementState) -> bool {
        let move_amount = if state == ElementState::Pressed { 1.0 } else { 0.0 };
        match key {
            VirtualKeyCode::W | VirtualKeyCode::Up => {
                self.move_forward = move_amount;
                true
            }
            VirtualKeyCode::S | VirtualKeyCode::Down => {
                self.move_backward = move_amount;
                true
            }
            VirtualKeyCode::A | VirtualKeyCode::Left => {
                self.move_left = move_amount;
                true
            }
            VirtualKeyCode::D | VirtualKeyCode::Right => {
                self.move_right = move_amount;
                true
            }
            VirtualKeyCode::E => {
                self.move_up = move_amount;
                true
            }
            VirtualKeyCode::Q => {
                self.move_down = move_amount;
                true
            }
            _ => false,
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx as f32;
        self.rotate_vertical = mouse_dy as f32;
    }

    pub fn process_scroll(&mut self, delta: &MouseScrollDelta) {
        self.scroll = -match delta {
            // Assume a line ~ 100 px
            MouseScrollDelta::LineDelta(_, scroll) => scroll * 100.0,
            MouseScrollDelta::PixelDelta(PhysicalPosition {
                y: scroll,
                ..
            }) => *scroll as f32,
        };
    }

    fn update_eye(&mut self, eye: &mut Eye, dt: Duration) {
        let dt = dt.as_secs_f32();

        // Move forward/backward and left/right
        let (yaw_sin, yaw_cos) = eye.yaw.0.sin_cos();
        let forward = Vector3::new(yaw_cos, 0.0, yaw_sin).normalize();
        let right = Vector3::new(-yaw_sin, 0.0, yaw_cos).normalize();
        eye.position += forward * (self.move_forward - self.move_backward) * self.speed * dt;
        eye.position += right * (self.move_right - self.move_left) * self.speed * dt;

        // Move in/out
        let (pitch_sin, pitch_cos) = eye.pitch.0.sin_cos();
        let scrollward = Vector3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize();
        eye.position += scrollward * self.scroll * self.speed * self.sensitivity * dt;
        self.scroll = 0.0;

        // Move up/down
        eye.position.y += (self.move_up - self.move_down) * self.speed * dt;

        // Rotate
        eye.yaw += Rad(self.rotate_horizontal) * self.sensitivity * dt;
        eye.pitch += Rad(-self.rotate_vertical) * self.sensitivity * dt;

        // Reset rotate values
        // If process_mouse is not called every frame, these values will not get set 
        // to zero, and the camera will rotate when moving in a non cardinal direction
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        // Contain the camera angle
        if eye.pitch < -Rad(SAFE_FRAC_PI_2) {
            eye.pitch = -Rad(SAFE_FRAC_PI_2);
        } else if eye.pitch > Rad(SAFE_FRAC_PI_2) {
            eye.pitch = Rad(SAFE_FRAC_PI_2);
        }
    }
}

pub struct Camera {
    pub eye: Eye,
    pub projection: Projection,
    pub controller: Controller,
    uniform: Uniform,
    buffer: wgpu::Buffer,
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl Camera {
    pub fn new<
        V: Into<Point3<f32>>,
        Y: Into<Rad<f32>>,
        P: Into<Rad<f32>>,
        F: Into<Rad<f32>>
    >(
        device: &wgpu::Device,
        position: V,
        yaw: Y,
        pitch: P,
        proj_width: u32,
        proj_height: u32,
        fov_y: F,
        z_near: f32,
        z_far: f32,
    ) -> Self {
        let eye = Eye::new(position, yaw, pitch);
        let projection = Projection::new(proj_width, proj_height, fov_y, z_near, z_far);
        
        let controller = Controller::new(4.0, 0.4);

        let mut uniform = Uniform::new();
        uniform.update_view_proj(&eye, &projection);

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            eye,
            projection,
            controller,
            uniform,
            buffer,
            layout,
            bind_group,
        }
    }

    pub fn update(&mut self, dt: std::time::Duration, queue: &mut wgpu::Queue) {
        self.controller.update_eye(&mut self.eye, dt);
        self.uniform.update_view_proj(&self.eye, &self.projection);
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[self.uniform]));
    }
}
