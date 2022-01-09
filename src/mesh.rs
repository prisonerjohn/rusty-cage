use anyhow::*;
use cgmath::num_traits::Float;
use cgmath::{InnerSpace, BaseNum, BaseFloat};
use std::ops::Range;
use std::mem;
use std::f32::consts;
use wgpu::util::DeviceExt;

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
    normal: [f32; 3],
    tangent: [f32; 3],
    bitangent: [f32; 3],
}

impl Vertex for MeshVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<MeshVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 11]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[derive(Copy, Clone)]
struct ComputedVectors {
    tangent: cgmath::Vector3<f32>,
    bitangent: cgmath::Vector3<f32>,
}

fn calculate_tangents_bitangents(
    positions: & Vec<cgmath::Vector3<f32>>,
    tex_coords: & Vec<cgmath::Vector2<f32>>,
    indices: & Vec<u32>,
) -> Vec<ComputedVectors> {
    // Calculate tangents and bitangents using triangles.
    let mut computed_vectors: Vec<ComputedVectors> = vec![ComputedVectors {
        tangent: cgmath::Vector3::new(0.0, 0.0, 0.0),
        bitangent: cgmath::Vector3::new(0.0, 0.0, 0.0),
    }; positions.len()];
    let mut triangles_included = (0..positions.len()).collect::<Vec<_>>();
    for c in indices.chunks(3) {
        let i0 = c[0] as usize;
        let i1 = c[1] as usize;
        let i2 = c[2] as usize;

        // Calculate the edges of the triangle.
        let delta_pos1 = positions[i1] - positions[i0];
        let delta_pos2 = positions[i2] - positions[i0];

        // This will give us a direction to calculate the tangent/bitangent.
        let delta_uv1 = tex_coords[i1] - tex_coords[i0];
        let delta_uv2 = tex_coords[i2] - tex_coords[i0];

        // Solve the following system of equations to get the tangent/bitangent.
        //    delta_pos1 = delta_uv1.x * T + delta_uv1.y * B
        //    delta_pos2 = delta_uv2.x * T + delta_uv2.y * B
        let r = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
        let tangent = (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
        let bitangent = (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * r;

        // Use the same tangent/bitangent for each vertex in the triangle.
        computed_vectors[i0].tangent += tangent;
        computed_vectors[i1].tangent += tangent;
        computed_vectors[i2].tangent += tangent;

        computed_vectors[i0].bitangent += bitangent;
        computed_vectors[i1].bitangent += bitangent;
        computed_vectors[i2].bitangent += bitangent;

        // Accumulate usage to average the tangents/bitangents.
        triangles_included[i0] += 1;
        triangles_included[i1] += 1;
        triangles_included[i2] += 1;
    }

    // Average the tangents/bitangents
    for (i, n) in triangles_included.into_iter().enumerate() {
        let denom = 1.0 / n as f32;
        computed_vectors[i].tangent = (computed_vectors[i].tangent * denom).normalize();
        computed_vectors[i].bitangent = (computed_vectors[i].bitangent * denom).normalize();
    }

    computed_vectors
}

pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: Option<wgpu::Buffer>,
    pub num_elements: u32,
}

impl Mesh {
    pub fn quad(
        device: &wgpu::Device,
        width: f32,
        height: f32,
        use_indices: bool,
    ) -> Result<Self> {
        let mut positions = Vec::new();
        positions.push(cgmath::Vector3::new(width * -0.5, height * -0.5, 0.0));
        positions.push(cgmath::Vector3::new(width *  0.5, height * -0.5, 0.0));
        positions.push(cgmath::Vector3::new(width *  0.5, height *  0.5, 0.0));
        positions.push(cgmath::Vector3::new(width * -0.5, height *  0.5, 0.0));

        let mut tex_coords = Vec::new();
        tex_coords.push(cgmath::Vector2::new(0.0, 0.0));
        tex_coords.push(cgmath::Vector2::new(1.0, 0.0));
        tex_coords.push(cgmath::Vector2::new(1.0, 1.0));
        tex_coords.push(cgmath::Vector2::new(0.0, 1.0));

        const NORMAL: cgmath::Vector3<f32> = cgmath::Vector3::new(0.0, 0.0, 1.0);

        let indices: Vec<u32> = vec![0, 1, 2, 2, 3, 0];

        let computed_vectors = calculate_tangents_bitangents(
            &positions,
            &tex_coords, 
            &indices
        );

        if use_indices {
            let mut vertices = Vec::new();
            for i in 0..positions.len() {
                vertices.push(MeshVertex {
                    position: positions[i].into(),
                    tex_coords: tex_coords[i].into(),
                    normal: NORMAL.into(),
                    tangent: computed_vectors[i].tangent.into(),
                    bitangent: computed_vectors[i].bitangent.into(),
                });
            }

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            let num_elements  = indices.len() as u32;

            Ok(Self {
                vertex_buffer,
                index_buffer: Some(index_buffer),
                num_elements,
            })
        } else {
            let mut vertices = Vec::new();
            for idx in 0..indices.len() {
                let i = indices[idx] as usize;
                vertices.push(MeshVertex {
                    position: positions[i].into(),
                    tex_coords: tex_coords[i].into(),
                    normal: NORMAL.into(),
                    tangent: computed_vectors[i].tangent.into(),
                    bitangent: computed_vectors[i].bitangent.into(),
                });
            }

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let num_elements  = vertices.len() as u32;

            Ok(Self {
                vertex_buffer,
                index_buffer: None,
                num_elements,
            })
        }
    }

    // pub fn pentagon(
    //     device: &wgpu::Device,
    // ) -> Result<Self> {
    //     const VERTICES: &[MeshVertex] = &[
    //         MeshVertex {
    //             position: [-0.0868241, -0.49240386, 0.0],
    //             tex_coords: [1.0 - 0.4131759, 1.0 - 0.00759614],
    //         }, // A
    //         MeshVertex {
    //             position: [-0.49513406, -0.06958647, 0.0],
    //             tex_coords: [1.0 - 0.0048659444, 1.0 - 0.43041354],
    //         }, // B
    //         MeshVertex {
    //             position: [-0.21918549, 0.44939706, 0.0],
    //             tex_coords: [1.0 - 0.28081453, 1.0 - 0.949397],
    //         }, // C
    //         MeshVertex {
    //             position: [0.35966998, 0.3473291, 0.0],
    //             tex_coords: [1.0 - 0.85967, 1.0 - 0.84732914],
    //         }, // D
    //         MeshVertex {
    //             position: [0.44147372, -0.2347359, 0.0],
    //             tex_coords: [1.0 - 0.9414737, 1.0 - 0.2652641],
    //         }, // E
    //     ];
        
    //     const INDICES: &[u32] = &[0, 1, 4, 1, 2, 4, 2, 3, 4, /* padding */ 0];

    //     let name = "Pentagon".to_string();
    //     let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //         label: Some("Vertex Buffer"),
    //         contents: bytemuck::cast_slice(VERTICES),
    //         usage: wgpu::BufferUsages::VERTEX,
    //     });
    //     let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //         label: Some("Index Buffer"),
    //         contents: bytemuck::cast_slice(INDICES),
    //         usage: wgpu::BufferUsages::INDEX,
    //     });
    //     let num_elements  = INDICES.len() as u32;

    //     Ok(Self {
    //         name,
    //         vertex_buffer,
    //         index_buffer,
    //         num_elements,
    //     })
    // }

    pub fn icosphere(
        device: &wgpu::Device,
        radius: f32,
        iterations: u32,
        use_indices: bool,
    ) -> Result<Self> {
        // Generate icosahedron.
        let sqrt5 = 5.0f32.sqrt();
        let phi = (1.0 + sqrt5) * 0.5;
        let inv_norm = 1.0 / (phi * phi + 1.0).sqrt();

        let mut positions = Vec::new();
        positions.push(inv_norm * cgmath::Vector3::new(-1.0,  phi,  0.0));
        positions.push(inv_norm * cgmath::Vector3::new( 1.0,  phi,  0.0));
        positions.push(inv_norm * cgmath::Vector3::new( 0.0,  1.0, -phi));
        positions.push(inv_norm * cgmath::Vector3::new( 0.0,  1.0,  phi));
        positions.push(inv_norm * cgmath::Vector3::new(-phi,  0.0, -1.0));
        positions.push(inv_norm * cgmath::Vector3::new(-phi,  0.0,  1.0));
        positions.push(inv_norm * cgmath::Vector3::new( phi,  0.0, -1.0));
        positions.push(inv_norm * cgmath::Vector3::new( phi,  0.0,  1.0));
        positions.push(inv_norm * cgmath::Vector3::new( 0.0, -1.0, -phi));
        positions.push(inv_norm * cgmath::Vector3::new( 0.0, -1.0,  phi));
        positions.push(inv_norm * cgmath::Vector3::new(-1.0, -phi,  0.0));
        positions.push(inv_norm * cgmath::Vector3::new( 1.0, -phi,  0.0));

        let mut indices: Vec<u32> = Vec::new();
        indices.push( 0); indices.push( 1); indices.push( 2);
        indices.push( 0); indices.push( 3); indices.push( 1);
		indices.push( 0); indices.push( 4); indices.push( 5);
		indices.push( 1); indices.push( 7); indices.push( 6);
		indices.push( 1); indices.push( 6); indices.push( 2);
		indices.push( 1); indices.push( 3); indices.push( 7);
		indices.push( 0); indices.push( 2); indices.push( 4);
		indices.push( 0); indices.push( 5); indices.push( 3);
		indices.push( 2); indices.push( 6); indices.push( 8);
		indices.push( 2); indices.push( 8); indices.push( 4);
		indices.push( 3); indices.push( 5); indices.push( 9);
		indices.push( 3); indices.push( 9); indices.push( 7);
		indices.push(11); indices.push( 6); indices.push( 7);
		indices.push(10); indices.push( 5); indices.push( 4);
		indices.push(10); indices.push( 4); indices.push( 8);
		indices.push(10); indices.push( 9); indices.push( 5);
		indices.push(11); indices.push( 8); indices.push( 6);
		indices.push(11); indices.push( 7); indices.push( 9);
		indices.push(10); indices.push( 8); indices.push(11);
		indices.push(10); indices.push(11); indices.push( 9);

        // Tessellate.
        let mut size = indices.len();
        for _it in 0..iterations {
            size *= 4;
            let mut new_indices: Vec<u32> = Vec::new();
            for i in 0..(size / 12) {
                let i1 = indices[i * 3 + 0];
                let i2 = indices[i * 3 + 1];
                let i3 = indices[i * 3 + 2];
                let i1_2 = positions.len() as u32;
                let i2_3 = i1_2 + 1;
                let i1_3 = i1_2 + 2;
                let v1 = positions[i1 as usize];
                let v2 = positions[i2 as usize];
                let v3 = positions[i3 as usize];
                
                // Make a vertex at the center of each edge and project it onto the sphere.
                positions.push((v1 + v2).normalize());
                positions.push((v2 + v3).normalize());
                positions.push((v1 + v3).normalize());

                // Recreate indices
                new_indices.push(i1);
                new_indices.push(i1_2);
                new_indices.push(i1_3);
                new_indices.push(i2);
                new_indices.push(i2_3);
                new_indices.push(i1_2);
                new_indices.push(i3);
                new_indices.push(i1_3);
                new_indices.push(i2_3);
                new_indices.push(i1_2);
                new_indices.push(i2_3);
                new_indices.push(i1_3);
            }
            mem::swap(&mut indices, &mut new_indices);
        }

        // Generate tex coords.
        let mut tex_coords = Vec::new();
        for pos in positions.iter() {
            let r0 = (pos.x * pos.x + pos.z * pos.z).sqrt();
            let alpha = pos.z.atan2(pos.x);
            let u = alpha / (consts::PI * 2.0) + 0.5;
            let v = pos.y.atan2(r0) / consts::PI + 0.5;
            tex_coords.push(cgmath::Vector2::new(1.0 - u, 1.0 - v));
        }

        let mut indices_to_split = Vec::new();
        for c in indices.chunks(3) {
            let t0 = tex_coords[c[0] as usize];
            let t1 = tex_coords[c[1] as usize];
            let t2 = tex_coords[c[2] as usize];

            if (t2.x - t0.x).abs() > 0.5 {
                if t0.x < 0.5 {
                    indices_to_split.push(c[0]);
                } else {
                    indices_to_split.push(c[2]);
                }
            }
            if (t1.x - t0.x).abs() > 0.5 {
                if t0.x < 0.5 {
                    indices_to_split.push(c[0]);
                } else {
                    indices_to_split.push(c[1]);
                }
            }
            if (t2.x - t1.x).abs() > 0.5 {
                if t1.x < 0.5 {
                    indices_to_split.push(c[1]);
                } else {
                    indices_to_split.push(c[2]);
                }
            }
        }

        // Split verts.
        for idx in indices_to_split.iter() {
            let i: usize = idx.clone().try_into().unwrap();
            let position = positions[i];
            let tex_coord = tex_coords[i] + cgmath::Vector2::new(1.0, 0.0);
            positions.push(position);
            tex_coords.push(tex_coord);
            let new_index = (positions.len() - 1) as u32;
            for j in 0..indices.len() {
                if i == indices[j] as usize {
                    let ndx1 = indices[(j + 1) % 3 + (j / 3) * 3] as usize;
                    let ndx2 = indices[(j + 2) % 3 + (j / 3) * 3] as usize;
                    if tex_coords[ndx1].x > 0.5 || tex_coords[ndx2].x > 0.5 {
                        indices[j] = new_index;
                    }
                }
            }
        }

        // Flip faces.
        // for i in 0..(indices.len() / 3) {
        //     indices.swap(i * 3 + 1, i * 3 + 2);
        // }

        // Clone unit positions for normals.
        let normals = positions.clone();
 
        // Scale positions by radius.
        for i in 0..positions.len() {
            positions[i] *= radius;
        }

        let computed_vectors = calculate_tangents_bitangents(
            &positions,
            &tex_coords,
            &indices,
        );

        if use_indices {
            let mut vertices = Vec::new();
            for i in 0..positions.len() {
                vertices.push(MeshVertex {
                    position: positions[i].into(),
                    tex_coords: tex_coords[i].into(),
                    normal: normals[i].into(),
                    tangent: computed_vectors[i].tangent.into(),
                    bitangent: computed_vectors[i].bitangent.into(),
                });
            }

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            let num_elements  = indices.len() as u32;

            Ok(Self {
                vertex_buffer,
                index_buffer: Some(index_buffer),
                num_elements,
            })
        } else {
            let mut vertices = Vec::new();
            for idx in 0..indices.len() {
                let i = indices[idx] as usize;
                vertices.push(MeshVertex {
                    position: positions[i].into(),
                    tex_coords: tex_coords[i].into(),
                    normal: normals[i].into(),
                    tangent: computed_vectors[i].tangent.into(),
                    bitangent: computed_vectors[i].bitangent.into(),
                });
            }

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let num_elements  = vertices.len() as u32;

            Ok(Self {
                vertex_buffer,
                index_buffer: None,
                num_elements,
            })
        }
    }
}

pub trait DrawMesh<'a> {
    fn draw_mesh(
        &mut self, 
        mesh: &'a Mesh,
        // material: &'a Material, 
        bind_groups: Option<Vec<&'a wgpu::BindGroup>>,
        // camera_bind_group: &'a wgpu::BindGroup,
        // light_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        // material: &'a Material,
        instances: Range<u32>,
        bind_groups: Option<Vec<&'a wgpu::BindGroup>>,
        //camera_bind_group: &'a wgpu::BindGroup,
        // light_bind_group: &'a wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawMesh<'b> for wgpu::RenderPass<'a>
where 
    'b: 'a,
{
    fn draw_mesh(
        &mut self, 
        mesh: &'b Mesh, 
        // material: &'b Material, 
        bind_groups: Option<Vec<&'a wgpu::BindGroup>>,
        // camera_bind_group: &'b wgpu::BindGroup,
        // light_bind_group: &'b wgpu::BindGroup,
    ) {
        // self.draw_mesh_instanced(mesh, material, 0..1, camera_bind_group, light_bind_group);
        self.draw_mesh_instanced(mesh, 0..1, bind_groups);
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        // material: &'b Material,
        instances: Range<u32>,
        bind_groups: Option<Vec<&'a wgpu::BindGroup>>,
        // camera_bind_group: &'b wgpu::BindGroup,
        // light_bind_group: &'b wgpu::BindGroup,
    ) {
        if bind_groups.is_some() {
            for (index, bind_group) in bind_groups.unwrap().iter().enumerate() {
                self.set_bind_group(index as u32, *bind_group, &[]);
            }
        }
        // self.set_bind_group(0, camera_bind_group, &[]);
        // self.set_bind_group(0, &material.bind_group, &[]);
        // self.set_bind_group(1, camera_bind_group, &[]);
        // self.set_bind_group(2, light_bind_group, &[]);
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        match &mesh.index_buffer {
            Some(index_buffer) => {
                self.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);          
                self.draw_indexed(0..mesh.num_elements, 0, instances);
            },
            None => {
                self.draw(0..mesh.num_elements, instances);
            },
        }
    }
}
