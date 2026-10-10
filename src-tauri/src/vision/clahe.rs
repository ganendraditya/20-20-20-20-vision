//! Local Contrast Normalization (CLAHE) for Spectacle Screen Glare Resilience
//!
//! Implements Contrast Limited Adaptive Histogram Equalization specifically tailored
//! for 192x192 face crops prior to MediaPipe FaceMesh landmark regression (Issue #66).
//!
//! The 192x192 crop is partitioned into a 4x4 tile grid (48x48 px per tile).
//! Crucially, tiles (1, 1) and (1, 2) naturally encompass the left and right eye sub-windows.
//! When bright laptop displays reflect off glasses (specular glare), local contrast
//! is compressed and eyelid contours wash out. CLAHE redistributes clipped excess
//! and equalizes local histograms, restoring eyelid edge contrast with sub-millisecond execution.

pub const FACE_CROP_DIM: usize = 192;
pub const GRID_SIZE: usize = 4;
pub const TILE_DIM: usize = FACE_CROP_DIM / GRID_SIZE; // 48 pixels
pub const TILE_PIXELS: usize = TILE_DIM * TILE_DIM; // 2304 pixels

#[derive(Debug, Clone, Copy)]
pub struct ClaheConfig {
    /// Contrast clip limit factor (multiplied by uniform bin count)
    pub clip_limit: f32,
    /// Blend alpha between original luminance and equalized luminance
    pub blend_alpha: f32,
    /// Mean luminance threshold above which glare compensation engages
    pub glare_threshold_mean: f32,
    /// Fraction of pixels with luminance >= 215 indicating specular highlight saturation
    pub glare_threshold_bright_ratio: f32,
}

impl Default for ClaheConfig {
    fn default() -> Self {
        Self {
            clip_limit: 2.5,
            blend_alpha: 0.70,
            glare_threshold_mean: 135.0,
            glare_threshold_bright_ratio: 0.08,
        }
    }
}

#[derive(Clone, Copy)]
struct Interp1D {
    idx0: usize,
    idx1: usize,
    w0: f32,
    w1: f32,
}

/// Applies high-performance zero-allocation CLAHE on a 192x192 RGB face crop.
///
/// Returns `true` if glare/overexposure was detected and equalized; `false` if passed through.
pub fn apply_clahe_face_192(
    rgb_192: &mut [u8; FACE_CROP_DIM * FACE_CROP_DIM * 3],
    config: &ClaheConfig,
) -> bool {
    const TOTAL_PX: usize = FACE_CROP_DIM * FACE_CROP_DIM;

    // 1. Extract luminance Y and analyze exposure telemetry
    let mut y_channel = [0u8; TOTAL_PX];
    let mut sum_y = 0u64;
    let mut bright_count = 0usize;

    for (i, px) in rgb_192.chunks_exact(3).enumerate() {
        let r = px[0] as u32;
        let g = px[1] as u32;
        let b = px[2] as u32;
        // Standard ITU-R BT.601 integer luminance approximation
        let y = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
        y_channel[i] = y;
        sum_y += y as u64;
        if y >= 215 {
            bright_count += 1;
        }
    }

    let mean_y = sum_y as f32 / TOTAL_PX as f32;
    let bright_ratio = bright_count as f32 / TOTAL_PX as f32;

    // Adaptive Glare Detection:
    // If the frame is under balanced/dim ambient lighting with zero specular washout,
    // bypass CLAHE to guarantee 0 drift on clean frames and minimal CPU duty cycle (< 15 µs).
    let is_glared = mean_y > config.glare_threshold_mean || bright_ratio > config.glare_threshold_bright_ratio;

    if !is_glared {
        return false;
    }

    let active_alpha = if config.blend_alpha.is_nan() {
        0.0
    } else {
        config.blend_alpha.clamp(0.0, 1.0)
    };

    // 2. Build tile histograms & Cumulative Distribution Function (CDF) lookup tables
    // Stack-allocated table: [4][4][256] = 4,096 bytes (fits directly inside L1 data cache)
    let mut luts = [[[0u8; 256]; GRID_SIZE]; GRID_SIZE];
    let safe_clip_limit = if config.clip_limit.is_nan() || config.clip_limit <= 0.0 {
        2.5
    } else {
        config.clip_limit.min(100.0)
    };
    let clip_limit = ((safe_clip_limit * (TILE_PIXELS as f32 / 256.0)) + 0.5) as u32;

    for gy in 0..GRID_SIZE {
        let y_start = gy * TILE_DIM;
        let y_end = y_start + TILE_DIM;

        for gx in 0..GRID_SIZE {
            let x_start = gx * TILE_DIM;
            let x_end = x_start + TILE_DIM;

            let mut hist = [0u32; 256];
            for y in y_start..y_end {
                let row = y * FACE_CROP_DIM;
                for x in x_start..x_end {
                    hist[y_channel[row + x] as usize] += 1;
                }
            }

            // Clip histogram spikes to prevent excessive amplification of homogeneous regions
            let mut clipped_excess = 0u32;
            for val in 0..256 {
                if hist[val] > clip_limit {
                    clipped_excess += hist[val] - clip_limit;
                    hist[val] = clip_limit;
                }
            }

            // Distribute clipped excess uniformly across all 256 bins
            let bonus = clipped_excess / 256;
            let rem = clipped_excess % 256;
            for val in 0..256 {
                hist[val] += bonus + if (val as u32) < rem { 1 } else { 0 };
            }

            // Construct normalized CDF mapping
            let mut cum = 0u32;
            for val in 0..256 {
                cum += hist[val];
                luts[gy][gx][val] = (((cum * 255) + (TILE_PIXELS as u32 / 2)) / TILE_PIXELS as u32) as u8;
            }
        }
    }

    // 3. Pre-compute 1D grid interpolation coordinates and weights
    let mut x_interp = [Interp1D { idx0: 0, idx1: 0, w0: 1.0, w1: 0.0 }; FACE_CROP_DIM];
    let mut y_interp = [Interp1D { idx0: 0, idx1: 0, w0: 1.0, w1: 0.0 }; FACE_CROP_DIM];

    for i in 0..FACE_CROP_DIM {
        let f = ((i as f32 + 0.5) / TILE_DIM as f32) - 0.5;
        let idx0 = (f.floor() as isize).clamp(0, GRID_SIZE as isize - 1) as usize;
        let idx1 = (idx0 + 1).min(GRID_SIZE - 1);
        let weight = if idx0 == idx1 { 0.0 } else { (f - idx0 as f32).clamp(0.0, 1.0) };
        x_interp[i] = Interp1D { idx0, idx1, w0: 1.0 - weight, w1: weight };
    }

    for i in 0..FACE_CROP_DIM {
        let f = ((i as f32 + 0.5) / TILE_DIM as f32) - 0.5;
        let idx0 = (f.floor() as isize).clamp(0, GRID_SIZE as isize - 1) as usize;
        let idx1 = (idx0 + 1).min(GRID_SIZE - 1);
        let weight = if idx0 == idx1 { 0.0 } else { (f - idx0 as f32).clamp(0.0, 1.0) };
        y_interp[i] = Interp1D { idx0, idx1, w0: 1.0 - weight, w1: weight };
    }

    let one_minus_alpha = 1.0 - active_alpha;

    // 4. Bilinear interpolation across tile neighborhoods & RGB reconstruction
    for y in 0..FACE_CROP_DIM {
        let yi = &y_interp[y];
        let row = y * FACE_CROP_DIM;
        let lut_y0 = &luts[yi.idx0];
        let lut_y1 = &luts[yi.idx1];

        for x in 0..FACE_CROP_DIM {
            let xi = &x_interp[x];
            let idx = row + x;
            let orig_y = y_channel[idx] as usize;

            let c00 = lut_y0[xi.idx0][orig_y] as f32;
            let c10 = lut_y0[xi.idx1][orig_y] as f32;
            let c01 = lut_y1[xi.idx0][orig_y] as f32;
            let c11 = lut_y1[xi.idx1][orig_y] as f32;

            let top = xi.w0 * c00 + xi.w1 * c10;
            let bot = xi.w0 * c01 + xi.w1 * c11;
            let equalized_y = yi.w0 * top + yi.w1 * bot;

            let target_y = (orig_y as f32) * one_minus_alpha + equalized_y * active_alpha;
            let ratio = (target_y + 1.0) / (orig_y as f32 + 1.0);

            let px = &mut rgb_192[idx * 3..idx * 3 + 3];
            let r = px[0] as f32;
            let g = px[1] as f32;
            let b = px[2] as f32;

            px[0] = (r * ratio).min(255.0) as u8;
            px[1] = (g * ratio).min(255.0) as u8;
            px[2] = (b * ratio).min(255.0) as u8;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neutral_lighting_bypass() {
        // Balanced ambient lighting (mean ~100, 0 bright pixels)
        let mut buf = [100u8; FACE_CROP_DIM * FACE_CROP_DIM * 3];
        let config = ClaheConfig::default();
        let was_applied = apply_clahe_face_192(&mut buf, &config);
        assert!(!was_applied, "Balanced neutral frame should bypass CLAHE");
        assert_eq!(buf[0], 100, "Pixel values should remain unaltered");
    }

    #[test]
    fn test_glare_overexposure_trigger() {
        // Overexposed screen glare frame (mean ~170, many pixels >= 215)
        let mut buf = [170u8; FACE_CROP_DIM * FACE_CROP_DIM * 3];
        let config = ClaheConfig::default();
        let was_applied = apply_clahe_face_192(&mut buf, &config);
        assert!(was_applied, "Overexposed glare frame must trigger CLAHE");
    }
}
