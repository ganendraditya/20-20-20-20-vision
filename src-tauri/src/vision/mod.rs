use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct Landmark3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

pub struct FaceMeshEngine {
    session: Session,
}

impl FaceMeshEngine {
    /// Initialize the ONNX FaceMesh session
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| format!("Failed to create ONNX session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Failed to set optimization level: {}", e))?
            .with_intra_threads(2)
            .map_err(|e| format!("Failed to set intra threads: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load ONNX model: {}", e))?;

        Ok(Self { session })
    }

    /// Preprocess an RGB image buffer (width x height) into [1, 192, 192, 3] normalized float tensor
    pub fn preprocess(&self, rgb_data: &[u8], width: usize, height: usize) -> Array4<f32> {
        let mut input_tensor = Array4::<f32>::zeros((1, 192, 192, 3));
        
        // Simple bilinear or nearest neighbor scaling to 192x192
        let x_ratio = width as f32 / 192.0;
        let y_ratio = height as f32 / 192.0;

        for y in 0..192 {
            for x in 0..192 {
                let src_x = (x as f32 * x_ratio).min((width - 1) as f32) as usize;
                let src_y = (y as f32 * y_ratio).min((height - 1) as f32) as usize;
                let src_idx = (src_y * width + src_x) * 3;

                if src_idx + 2 < rgb_data.len() {
                    // Normalize [0..255] to [0.0..1.0] (or [-1.0..1.0] depending on model spec)
                    input_tensor[[0, y, x, 0]] = rgb_data[src_idx] as f32 / 255.0;
                    input_tensor[[0, y, x, 1]] = rgb_data[src_idx + 1] as f32 / 255.0;
                    input_tensor[[0, y, x, 2]] = rgb_data[src_idx + 2] as f32 / 255.0;
                }
            }
        }

        input_tensor
    }

    /// Run inference and extract 468 3D landmarks
    pub fn infer(&mut self, input_tensor: Array4<f32>) -> Result<Vec<Landmark3D>, String> {
        // Create an ONNX Value tensor from ndarray
        let tensor_value = ort::value::Tensor::from_array(input_tensor)
            .map_err(|e| format!("Failed to create tensor value: {}", e))?;
        
        let inputs = ort::inputs![tensor_value];
        let outputs = self.session.run(inputs).map_err(|e| format!("Inference failed: {}", e))?;

        // FaceMesh ONNX model returns [1, 1, 1, 1404] (468 points * 3 coordinates)
        let (_shape, slice) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract output tensor: {}", e))?;

        let mut landmarks = Vec::with_capacity(468);

        for i in 0..468 {
            let offset = i * 3;
            if offset + 2 < slice.len() {
                landmarks.push(Landmark3D {
                    x: slice[offset] / 192.0,      // Normalized [0.0..1.0]
                    y: slice[offset + 1] / 192.0,  // Normalized [0.0..1.0]
                    z: slice[offset + 2] / 192.0,  // Relative depth
                });
            }
        }

        Ok(landmarks)
    }
}
