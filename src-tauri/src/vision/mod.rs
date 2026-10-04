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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceBoundingBox {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub confidence: f32,
}

impl FaceBoundingBox {
    pub fn area(&self) -> f32 {
        let width = (self.xmax - self.xmin).max(0.0);
        let height = (self.ymax - self.ymin).max(0.0);
        width * height
    }
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

    /// Preprocess an RGB image buffer (width x height) into [1, 192, 192, 3] normalized float tensor.
    /// If a dominant face bounding box is provided (in [0..1] normalized full-frame coordinates),
    /// crops with padding around that primary face to isolate the user from background subjects.
    /// Otherwise, falls back to the center square crop.
    pub fn preprocess(&self, rgb_data: &[u8], width: usize, height: usize, dominant_box: Option<&FaceBoundingBox>) -> Array4<f32> {
        let mut input_tensor = Array4::<f32>::zeros((1, 192, 192, 3));
        
        if width == 0 || height == 0 {
            return input_tensor;
        }

        let (crop_x, crop_y, side) = match dominant_box {
            Some(bbox) => {
                // Convert normalized box to pixel coordinates in full frame
                let bx1 = bbox.xmin * width as f32;
                let by1 = bbox.ymin * height as f32;
                let bx2 = bbox.xmax * width as f32;
                let by2 = bbox.ymax * height as f32;

                let bw = (bx2 - bx1).max(10.0);
                let bh = (by2 - by1).max(10.0);
                let cx = (bx1 + bx2) / 2.0;
                let cy = (by1 + by2) / 2.0;

                // Expand by 25% margin to preserve full forehead, jawline, and ear landmarks
                let raw_side = bw.max(bh) * 1.5;
                let max_side = (width.min(height) as f32).min(raw_side);

                // Clamp top-left origin within frame boundaries
                let x0 = (cx - max_side / 2.0).clamp(0.0, (width as f32 - max_side).max(0.0)) as usize;
                let y0 = (cy - max_side / 2.0).clamp(0.0, (height as f32 - max_side).max(0.0)) as usize;
                let s = (max_side as usize).min(width.saturating_sub(x0)).min(height.saturating_sub(y0)).max(1);

                (x0, y0, s)
            }
            None => {
                // Default center square crop
                let s = width.min(height);
                let x0 = (width - s) / 2;
                let y0 = (height - s) / 2;
                (x0, y0, s)
            }
        };

        let scale = side as f32 / 192.0;

        for y in 0..192 {
            for x in 0..192 {
                let src_x = crop_x + (x as f32 * scale).min((side - 1) as f32) as usize;
                let src_y = crop_y + (y as f32 * scale).min((side - 1) as f32) as usize;
                let src_idx = (src_y * width + src_x) * 3;

                if src_idx + 2 < rgb_data.len() {
                    // Normalize [0..255] to [0.0..1.0]
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

        if slice.len() < 468 * 3 {
            return Err(format!(
                "Output tensor size mismatch: expected at least 1404 elements, got {}",
                slice.len()
            ));
        }

        let mut landmarks = Vec::with_capacity(468);

        for i in 0..468 {
            let offset = i * 3;
            landmarks.push(Landmark3D {
                x: slice[offset] / 192.0,      // Normalized [0.0..1.0]
                y: slice[offset + 1] / 192.0,  // Normalized [0.0..1.0]
                z: slice[offset + 2] / 192.0,  // Relative depth
            });
        }

        Ok(landmarks)
    }
}

pub struct FaceDetectorEngine {
    session: Session,
}

impl FaceDetectorEngine {
    /// Initialize the ONNX UltraFace detector session
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| format!("Failed to create FaceDetector session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Failed to set FaceDetector optimization level: {}", e))?
            .with_intra_threads(1)
            .map_err(|e| format!("Failed to set FaceDetector intra threads: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load FaceDetector model: {}", e))?;

        Ok(Self { session })
    }

    /// Preprocess an RGB image buffer (width x height) into [1, 3, 240, 320] normalized float tensor (NCHW).
    /// Resizes the camera frame to 320x240 and normalizes with (pixel - 127) / 128.
    pub fn preprocess(&self, rgb_data: &[u8], width: usize, height: usize) -> Array4<f32> {
        let mut input_tensor = Array4::<f32>::zeros((1, 3, 240, 320));

        if width == 0 || height == 0 {
            return input_tensor;
        }

        let scale_x = width as f32 / 320.0;
        let scale_y = height as f32 / 240.0;

        for y in 0..240 {
            for x in 0..320 {
                let src_x = (x as f32 * scale_x).min((width - 1) as f32) as usize;
                let src_y = (y as f32 * scale_y).min((height - 1) as f32) as usize;
                let src_idx = (src_y * width + src_x) * 3;

                if src_idx + 2 < rgb_data.len() {
                    // Normalize [0..255] with (p - 127.0) / 128.0
                    input_tensor[[0, 0, y, x]] = (rgb_data[src_idx] as f32 - 127.0) / 128.0;
                    input_tensor[[0, 1, y, x]] = (rgb_data[src_idx + 1] as f32 - 127.0) / 128.0;
                    input_tensor[[0, 2, y, x]] = (rgb_data[src_idx + 2] as f32 - 127.0) / 128.0;
                }
            }
        }

        input_tensor
    }

    /// Run FaceDetector inference. Returns true if any face candidate has confidence >= 0.70.
    pub fn detect_face(&mut self, input_tensor: Array4<f32>) -> Result<bool, String> {
        let (has_face, _) = self.detect_faces_and_primary_box(input_tensor)?;
        Ok(has_face)
    }

    /// Run FaceDetector inference with Multi-Face Disambiguation.
    /// Extracts all candidate faces meeting confidence >= 0.70, calculates their bounding box areas,
    /// and selects the primary user with the largest area (closest to the screen).
    ///
    /// Returns: (has_face: bool, dominant_bbox: Option<FaceBoundingBox>)
    pub fn detect_faces_and_primary_box(
        &mut self,
        input_tensor: Array4<f32>,
    ) -> Result<(bool, Option<FaceBoundingBox>), String> {
        let tensor_value = ort::value::Tensor::from_array(input_tensor)
            .map_err(|e| format!("Failed to create FaceDetector tensor value: {}", e))?;

        let inputs = ort::inputs![tensor_value];
        let outputs = self.session.run(inputs).map_err(|e| format!("FaceDetector inference failed: {}", e))?;

        // UltraFace outputs[0] is `scores` of shape [1, 4420, 2]
        let (_s_shape, scores) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract FaceDetector scores tensor: {}", e))?;

        // UltraFace outputs[1] is `boxes` of shape [1, 4420, 4] where box is [xmin, ymin, xmax, ymax] normalized [0..1]
        let (_b_shape, boxes) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract FaceDetector boxes tensor: {}", e))?;

        let mut largest_box: Option<FaceBoundingBox> = None;
        let mut max_area = 0.0f32;
        let mut has_face = false;

        for (score_chunk, box_chunk) in scores.chunks_exact(2).zip(boxes.chunks_exact(4)) {
            let conf = score_chunk[1];
            if conf >= 0.70 {
                has_face = true;
                let candidate = FaceBoundingBox {
                    xmin: box_chunk[0].clamp(0.0, 1.0),
                    ymin: box_chunk[1].clamp(0.0, 1.0),
                    xmax: box_chunk[2].clamp(0.0, 1.0),
                    ymax: box_chunk[3].clamp(0.0, 1.0),
                    confidence: conf,
                };

                let area = candidate.area();
                if area > max_area {
                    max_area = area;
                    largest_box = Some(candidate);
                }
            }
        }

        Ok((has_face, largest_box))
    }
}
