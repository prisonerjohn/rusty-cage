// Vertex shader

// struct Camera {
//     view_pos: vec4<f32>;
//     view_proj: mat4x4<f32>;
// };
// [[group(0), binding(0)]]
// var<uniform> camera: Camera;

// struct Light {
//     position: vec3<f32>;
//     color: vec3<f32>;
// };
// [[group(1), binding(0)]]
// var<uniform> light: Light;

struct VertexInput {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] tex_coords: vec2<f32>;
};

struct VertexOutput {
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] tex_coords: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    let scale = 0.25;
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.position, 1.0);
    out.tex_coords = model.tex_coords;
    return out;
}

// Fragment shader

// [[group(0), binding(0)]]
// var t_shadow: texture_depth_2d;
// [[group(0), binding(1)]]
// var s_shadow: sampler_comparison;

[[group(0), binding(0)]]
var t_diffuse: texture_2d<f32>;
[[group(0), binding(1)]]
var s_diffuse: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    // let near = 0.1;
    // let far = 100.0;
    // let depth = textureSampleCompare(t_shadow, s_shadow, in.tex_coords, in.clip_position.w);
    // //let r = (2.0 * near) / (far + near - depth * (far - near));
    // //return vec4<f32>(vec3<f32>(r), 1.0);
    // if (depth > 0.0) {
    //     return vec4<f32>(vec3<f32>(1.0), 1.0);
    // } else {
    //     return vec4<f32>(vec3<f32>(0.0), 1.0);
    // }

    let displacement = 0.1;
    let dispR = 1.0 + displacement;
    let dispB = 1.0 - displacement;
    let uvR = ((in.tex_coords - vec2<f32>(0.5, 0.5)) * dispR) + vec2<f32>(0.5, 0.5);
    let uvG = in.tex_coords;
    let uvB = ((in.tex_coords - vec2<f32>(0.5, 0.5)) * dispB) + vec2<f32>(0.5, 0.5);
    let colR = textureSample(t_diffuse, s_diffuse, uvR).r;
    let colG = textureSample(t_diffuse, s_diffuse, uvG).g;
    let colB = textureSample(t_diffuse, s_diffuse, uvB).b;
    return vec4<f32>(colR, colG, colB, 1.0);

    //let texCol = textureSample(t_diffuse, s_diffuse, in.tex_coords);
    //return texCol * vec4<f32>(1.0, 0.0, 0.0, 1.0);
    //return vec4<f32>(in.tex_coords, 0.0, 1.0);
}
