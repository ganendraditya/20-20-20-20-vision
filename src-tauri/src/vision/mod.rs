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

    /// Calculate Intersection-over-Union (IoU) between two bounding boxes
    pub fn iou(&self, other: &FaceBoundingBox) -> f32 {
        let inter_xmin = self.xmin.max(other.xmin);
        let inter_ymin = self.ymin.max(other.ymin);
        let inter_xmax = self.xmax.min(other.xmax);
        let inter_ymax = self.ymax.min(other.ymax);

        let inter_w = (inter_xmax - inter_xmin).max(0.0);
        let inter_h = (inter_ymax - inter_ymin).max(0.0);
        let inter_area = inter_w * inter_h;

        if inter_area <= 0.0 {
            return 0.0;
        }

        let area_a = self.area();
        let area_b = other.area();
        let union_area = area_a + area_b - inter_area;

        if union_area <= 0.0 {
            0.0
        } else {
            inter_area / union_area
        }
    }

    /// Calculate Normalized Center Distance between two bounding boxes
    pub fn center_distance_sq(&self, other: &FaceBoundingBox) -> f32 {
        let cx1 = (self.xmin + self.xmax) * 0.5;
        let cy1 = (self.ymin + self.ymax) * 0.5;
        let cx2 = (other.xmin + other.xmax) * 0.5;
        let cy2 = (other.ymin + other.ymax) * 0.5;
        (cx1 - cx2).powi(2) + (cy1 - cy2).powi(2)
    }

    /// Single source of truth for computing square crop region (x0, y0, side) with margin
    pub fn compute_crop_region(bbox: Option<&FaceBoundingBox>, width: usize, height: usize) -> (usize, usize, usize) {
        match bbox {
            Some(b) => {
                let bx1 = b.xmin * width as f32;
                let by1 = b.ymin * height as f32;
                let bx2 = b.xmax * width as f32;
                let by2 = b.ymax * height as f32;

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
                let s = width.min(height);
                let x0 = (width - s) / 2;
                let y0 = (height - s) / 2;
                (x0, y0, s)
            }
        }
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

        let (crop_x, crop_y, side) = FaceBoundingBox::compute_crop_region(dominant_box, width, height);
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

/// Confidence threshold for un-tracked initial detection (prevents phantom triggers)
pub const CONF_THRESHOLD_UNTRACKED: f32 = 0.70;

/// Relaxed confidence threshold for active tracked sessions (supports far-field & RoI recall)
pub const CONF_THRESHOLD_TRACKED: f32 = 0.45;

/// Normalized bounding box area threshold below which Adaptive RoI Zoom is triggered (Issue #63)
pub const FAR_FIELD_AREA_THRESHOLD: f32 = 0.015;

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
        self.preprocess_crop(rgb_data, width, height, 0, 0, width, height)
    }

    /// Preprocess a sub-region (crop_x, crop_y, crop_w, crop_h) of an RGB image buffer into [1, 3, 240, 320] tensor.
    pub fn preprocess_crop(
        &self,
        rgb_data: &[u8],
        full_width: usize,
        full_height: usize,
        crop_x: usize,
        crop_y: usize,
        crop_w: usize,
        crop_h: usize,
    ) -> Array4<f32> {
        let mut input_tensor = Array4::<f32>::zeros((1, 3, 240, 320));

        if full_width == 0 || full_height == 0 || crop_w == 0 || crop_h == 0 {
            return input_tensor;
        }

        let scale_x = crop_w as f32 / 320.0;
        let scale_y = crop_h as f32 / 240.0;

        for y in 0..240 {
            for x in 0..320 {
                let src_x = (crop_x + (x as f32 * scale_x) as usize).min(full_width - 1);
                let src_y = (crop_y + (y as f32 * scale_y) as usize).min(full_height - 1);
                let src_idx = (src_y * full_width + src_x) * 3;

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
        let (has_face, _, primary_box) = self.detect_faces_and_track(input_tensor, None)?;
        Ok((has_face, primary_box))
    }

    /// Run FaceDetector with Adaptive RoI Zoom (Issue #63).
    /// If tracker holds an established lock on a small far-field face (area < FAR_FIELD_THRESHOLD),
    /// crops an aspect-ratio-preserving 4:3 sub-window around the last known position.
    /// This preserves pixel resolution for small/distant faces, boosting far-field recall.
    /// If RoI detection succeeds, remaps the coordinates back to full-frame space.
    /// If RoI detection fails or no face is currently locked, falls back to full-frame detection.
    pub fn detect_with_adaptive_roi(
        &mut self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
        mut tracker: Option<&mut FaceTracker>,
    ) -> Result<(bool, Option<FaceBoundingBox>), String> {
        // Check if tracker has a locked face that is small (far-field / leaning back)
        let far_field_lock = tracker.as_ref().and_then(|t| t.current_lock()).filter(|b| b.area() < FAR_FIELD_AREA_THRESHOLD);

        if let Some(tracked) = far_field_lock {
            let cx = (tracked.xmin + tracked.xmax) * 0.5 * width as f32;
            let cy = (tracked.ymin + tracked.ymax) * 0.5 * height as f32;
            let face_w_px = (tracked.xmax - tracked.xmin).max(0.01) * width as f32;

            // Expand by ~4.5x to preserve anatomical context
            let min_w = (width as f32 * 0.35).min(width as f32).max(32.0);
            let mut roi_w_px = (face_w_px * 4.5).clamp(min_w, width as f32);
            let mut roi_h_px = (roi_w_px * 0.75).round();
            if roi_h_px > height as f32 {
                roi_h_px = height as f32;
                roi_w_px = (roi_h_px * (4.0 / 3.0)).round().min(width as f32);
            }

            let x0 = (cx - roi_w_px * 0.5).clamp(0.0, (width as f32 - roi_w_px).max(0.0)) as usize;
            let y0 = (cy - roi_h_px * 0.5).clamp(0.0, (height as f32 - roi_h_px).max(0.0)) as usize;
            let rw = (roi_w_px as usize).min(width.saturating_sub(x0));
            let rh = (roi_h_px as usize).min(height.saturating_sub(y0));

            if rw > 10 && rh > 10 {
                let roi_tensor = self.preprocess_crop(rgb_data, width, height, x0, y0, rw, rh);
                // Note: RoI inference uses relaxed tracked threshold CONF_THRESHOLD_TRACKED (0.45)
                // because an active user session lock was already confirmed
                let (det_roi, candidates_roi, _) = self.detect_faces_and_track_with_threshold(roi_tensor, CONF_THRESHOLD_TRACKED, None)?;

                if det_roi && !candidates_roi.is_empty() {
                    let roi_norm_x = x0 as f32 / width as f32;
                    let roi_norm_y = y0 as f32 / height as f32;
                    let roi_norm_w = rw as f32 / width as f32;
                    let roi_norm_h = rh as f32 / height as f32;

                    let remapped_candidates: Vec<FaceBoundingBox> = candidates_roi
                        .into_iter()
                        .map(|b| {
                            let xmin = (roi_norm_x + b.xmin * roi_norm_w).clamp(0.0, 1.0);
                            let xmax = (roi_norm_x + b.xmax * roi_norm_w).clamp(0.0, 1.0);
                            let ymin = (roi_norm_y + b.ymin * roi_norm_h).clamp(0.0, 1.0);
                            let ymax = (roi_norm_y + b.ymax * roi_norm_h).clamp(0.0, 1.0);
                            FaceBoundingBox {
                                xmin: xmin.min(xmax),
                                xmax: xmin.max(xmax),
                                ymin: ymin.min(ymax),
                                ymax: ymin.max(ymax),
                                confidence: b.confidence,
                            }
                        })
                        .filter(|b| (b.xmax - b.xmin) > 1e-4 && (b.ymax - b.ymin) > 1e-4)
                        .collect();

                    if !remapped_candidates.is_empty() {
                        let primary = if let Some(tr) = tracker.as_deref_mut() {
                            tr.update(&remapped_candidates)
                        } else {
                            remapped_candidates.into_iter().max_by(|a, b| a.area().total_cmp(&b.area()))
                        };

                        if let Some(target) = primary {
                            return Ok((true, Some(target)));
                        } else {
                            // RoI candidates were evaluated by tracker and did not match
                            return Ok((false, None));
                        }
                    }
                }
            }
        }

        // Fallback or full-frame execution:
        // Only reached if RoI was empty (user moved out of RoI window) or during initial un-tracked search
        let full_tensor = self.preprocess(rgb_data, width, height);
        let (has_face, _, primary_face) = self.detect_faces_and_track(full_tensor, tracker)?;
        Ok((has_face, primary_face))
    }

    /// Run FaceDetector inference and select primary user with optional Sticky Tracking.
    /// Supports Adaptive Far-Field Confidence Hysteresis:
    /// - Initial / un-tracked detection requires high confidence (>= 0.70) to prevent phantom triggers.
    /// - Once an active user is locked, spatial continuity matching accepts far-field candidates (>= 0.45)
    ///   allowing the user to lean back or stretch (Tier 3: 80-120cm) without dropping presence or blinks.
    /// Returns: (has_face: bool, all_faces: Vec<FaceBoundingBox>, selected_primary: Option<FaceBoundingBox>)
    pub fn detect_faces_and_track(
        &mut self,
        input_tensor: Array4<f32>,
        tracker: Option<&mut FaceTracker>,
    ) -> Result<(bool, Vec<FaceBoundingBox>, Option<FaceBoundingBox>), String> {
        let is_tracked = tracker.as_ref().map(|t| t.current_lock().is_some()).unwrap_or(false);
        let min_conf = if is_tracked { CONF_THRESHOLD_TRACKED } else { CONF_THRESHOLD_UNTRACKED };
        self.detect_faces_and_track_with_threshold(input_tensor, min_conf, tracker)
    }

    /// Internal inference executor with configurable confidence threshold
    pub fn detect_faces_and_track_with_threshold(
        &mut self,
        input_tensor: Array4<f32>,
        min_conf: f32,
        tracker: Option<&mut FaceTracker>,
    ) -> Result<(bool, Vec<FaceBoundingBox>, Option<FaceBoundingBox>), String> {
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

        let mut candidates = Vec::new();

        for (score_chunk, box_chunk) in scores.chunks_exact(2).zip(boxes.chunks_exact(4)) {
            let conf = score_chunk[1];
            if conf >= min_conf {
                candidates.push(FaceBoundingBox {
                    xmin: box_chunk[0].clamp(0.0, 1.0),
                    ymin: box_chunk[1].clamp(0.0, 1.0),
                    xmax: box_chunk[2].clamp(0.0, 1.0),
                    ymax: box_chunk[3].clamp(0.0, 1.0),
                    confidence: conf,
                });
            }
        }

        // Perform Non-Maximum Suppression (NMS) to collapse overlapping raw anchor boxes of the same face
        let suppressed_faces = nms_filter(&candidates, 0.40);
        let has_face = !suppressed_faces.is_empty();

        let primary_face = if let Some(tr) = tracker {
            tr.update(&suppressed_faces)
        } else {
            // Default stateless greedy: largest bounding box
            suppressed_faces.iter().copied().max_by(|a, b| a.area().total_cmp(&b.area()))
        };

        Ok((has_face, suppressed_faces, primary_face))
    }
}

/// Simple Non-Maximum Suppression (NMS) for candidate face boxes
fn nms_filter(boxes: &[FaceBoundingBox], iou_threshold: f32) -> Vec<FaceBoundingBox> {
    let mut sorted = boxes.to_vec();
    sorted.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));

    let mut selected: Vec<FaceBoundingBox> = Vec::new();

    for b in sorted {
        let overlaps = selected.iter().any(|s| s.iou(&b) > iou_threshold);
        if !overlaps {
            selected.push(b);
        }
    }

    selected
}

/// State-preserving Face Tracker with Spatial Continuity and Hysteresis.
/// Prevents jittering/target flickering in multi-person environments (2, 3, 5+ subjects)
/// and resists momentary hijacking by passersby.
pub struct FaceTracker {
    last_tracked_box: Option<FaceBoundingBox>,
    lost_frames: usize,
    max_lost_frames: usize,
    iou_match_threshold: f32,
    size_hijack_margin: f32,
}

impl Default for FaceTracker {
    fn default() -> Self {
        Self::new(22, 0.30, 1.35) // ~1.5s tolerance at 15 FPS
    }
}

impl FaceTracker {
    pub fn new(max_lost_frames: usize, iou_match_threshold: f32, size_hijack_margin: f32) -> Self {
        Self {
            last_tracked_box: None,
            lost_frames: 0,
            max_lost_frames,
            iou_match_threshold,
            size_hijack_margin,
        }
    }

    /// Reset tracker state
    pub fn reset(&mut self) {
        self.last_tracked_box = None;
        self.lost_frames = 0;
    }

    /// Current locked box if any
    pub fn current_lock(&self) -> Option<FaceBoundingBox> {
        self.last_tracked_box
    }

    /// Update tracker with current frame candidate faces and resolve primary target
    pub fn update(&mut self, candidates: &[FaceBoundingBox]) -> Option<FaceBoundingBox> {
        if candidates.is_empty() {
            self.lost_frames += 1;
            if self.lost_frames > self.max_lost_frames {
                self.last_tracked_box = None;
                None
            } else {
                // Graceful retention: preserve last known target during brief frame drop / momentary flicker
                self.last_tracked_box
            }
        } else {
            match self.last_tracked_box {
            None => {
                // Initial target lock: pick the largest face in frame (primary subject closest to screen).
                // If areas are virtually identical within 1% float noise, pick the one with higher confidence.
                let best = candidates
                    .iter()
                    .max_by(|a, b| {
                        let area_a = a.area();
                        let area_b = b.area();
                        if (area_a - area_b).abs() / (area_a.max(area_b).max(1e-5)) < 0.01 {
                            a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal)
                        } else {
                            area_a.partial_cmp(&area_b).unwrap_or(std::cmp::Ordering::Equal)
                        }
                    })
                    .copied();

                self.last_tracked_box = best;
                self.lost_frames = 0;
                best
            }
            Some(tracked) => {
                // Step 1: Find best spatial continuity match among candidates using IoU
                let mut best_match: Option<(usize, f32)> = None;
                for (idx, cand) in candidates.iter().enumerate() {
                    let score = cand.iou(&tracked);
                    if score >= self.iou_match_threshold {
                        if let Some((_, best_score)) = best_match {
                            if score > best_score {
                                best_match = Some((idx, score));
                            }
                        } else {
                            best_match = Some((idx, score));
                        }
                    }
                }

                // If IoU is low (e.g. abrupt movement), fallback to Euclidean center distance
                if best_match.is_none() {
                    let mut best_dist_idx = None;
                    let mut min_dist = 0.04f32; // ~20% frame distance limit
                    for (idx, cand) in candidates.iter().enumerate() {
                        let d = cand.center_distance_sq(&tracked);
                        if d < min_dist {
                            min_dist = d;
                            best_dist_idx = Some(idx);
                        }
                    }
                    if let Some(idx) = best_dist_idx {
                        best_match = Some((idx, 0.0));
                    }
                }

                if let Some((match_idx, _)) = best_match {
                    let matched_cand = candidates[match_idx];

                    // Proximity guard radius: sqrt(0.09) ≈ 0.30 normalized units (~30% of frame dimensions)
                    // Distant bystanders far across the room must NEVER steal an active session lock!
                    const INTRUDER_PROXIMITY_THRESHOLD_SQ: f32 = 0.09;
                    let max_intruder = candidates
                        .iter()
                        .enumerate()
                        .filter(|(idx, cand)| *idx != match_idx && cand.center_distance_sq(&tracked) <= INTRUDER_PROXIMITY_THRESHOLD_SQ)
                        .max_by(|(_, a), (_, b)| a.area().total_cmp(&b.area()))
                        .map(|(_, b)| b);

                    // If intruding person is massively larger (e.g. someone took over the screen directly),
                    // allow legitimate switch; otherwise stick faithfully to active user
                    let final_target = if let Some(intruder) = max_intruder {
                        if intruder.area() > matched_cand.area() * self.size_hijack_margin {
                            *intruder
                        } else {
                            matched_cand
                        }
                    } else {
                        matched_cand
                    };

                    self.last_tracked_box = Some(final_target);
                    self.lost_frames = 0;
                    Some(final_target)
                } else {
                    // Active user temporarily occluded or not matched in this frame
                    self.lost_frames += 1;
                    if self.lost_frames > self.max_lost_frames {
                        // User has permanently left or timed out -> hand-off to largest remaining face
                        let new_best = candidates
                            .iter()
                            .max_by(|a, b| a.area().partial_cmp(&b.area()).unwrap_or(std::cmp::Ordering::Equal))
                            .copied();

                        self.last_tracked_box = new_best;
                        self.lost_frames = 0;
                        new_best
                    } else {
                        // Tolerate brief occlusion or frame drop: maintain lock on last known position
                        Some(tracked)
                    }
                }
            }
        }
    }
}
}
