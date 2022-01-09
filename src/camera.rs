use cgmath::*;
use winit::event::*;
use winit::dpi::PhysicalPosition;
use std::time::Duration;
use std::f32::consts::FRAC_PI_2;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;

#[derive(Debug)]
pub struct Camera {
    pub position: Point3<f32>,
    yaw: Rad<f32>,
    pitch: Rad<f32>,
}

impl Camera {
    pub fn new<
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

    pub fn calc_matrix(&self) -> Matrix4<f32> {
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
    pub fn new<F: Into<Rad<f32>>>(
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

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        OPENGL_TO_WGPU_MATRIX * perspective(self.fov_y, self.aspect, self.z_near, self.z_far)
    }
}

#[derive(Debug)]
pub struct CameraController {
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

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
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

    pub fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
        let dt = dt.as_secs_f32();

        // Move forward/backward and left/right
        let (yaw_sin, yaw_cos) = camera.yaw.0.sin_cos();
        let forward = Vector3::new(yaw_cos, 0.0, yaw_sin).normalize();
        let right = Vector3::new(-yaw_sin, 0.0, yaw_cos).normalize();
        camera.position += forward * (self.move_forward - self.move_backward) * self.speed * dt;
        camera.position += right * (self.move_right - self.move_left) * self.speed * dt;

        // Move in/out
        let (pitch_sin, pitch_cos) = camera.pitch.0.sin_cos();
        let scrollward = Vector3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize();
        camera.position += scrollward * self.scroll * self.speed * self.sensitivity * dt;
        self.scroll = 0.0;

        // Move up/down
        camera.position.y += (self.move_up - self.move_down) * self.speed * dt;

        // Rotate
        camera.yaw += Rad(self.rotate_horizontal) * self.sensitivity * dt;
        camera.pitch += Rad(-self.rotate_vertical) * self.sensitivity * dt;

        // Reset rotate values
        // If process_mouse is not called every frame, these values will not get set 
        // to zero, and the camera will rotate when moving in a non cardinal direction
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        // Contain the camera angle
        if camera.pitch < -Rad(SAFE_FRAC_PI_2) {
            camera.pitch = -Rad(SAFE_FRAC_PI_2);
        } else if camera.pitch > Rad(SAFE_FRAC_PI_2) {
            camera.pitch = Rad(SAFE_FRAC_PI_2);
        }
    }
}
