//! Decoding `.alp` geometry WORDS, for callers that never see a file.
//!
//! `read_alf` is the whole-file entry point. This is the other half of the same
//! knowledge, for a boundary that receives a bare `&[u16]` and a codec — which
//! is exactly the shape a host WASM module works in, and exactly the
//! shape that grew a second, drifting implementation when the format package had
//! nothing to offer it.
//!
//! What lives here is only what the FORMAT decides: which word is a marker, how
//! a component becomes a coordinate, and when a point has no position. Anything
//! about a specific host — a JS tag byte, an f32 render buffer, whether an absent
//! point is skipped or fatal — belongs to that host, not to this.

/// The reserved word marking a point with no laser return (`spec/format.md` §5.2).
///
/// Free to reserve because the encoder derives its step as `largest / 32767`, so
/// the grid is symmetric and `-32768` is unreachable for a real coordinate.
pub const INT16_NO_RETURN: i16 = -32768;

/// How the 16-bit words in a `.alp` payload are read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeomCodec {
    /// binary16 (IEEE half). Wire versions 1 and 2.
    Fp16,
    /// int16 fixed-point, `value * scale + offset[axis]`. Wire version 3.
    Int16,
}

/// A validated decoder for one frame's geometry words.
#[derive(Clone, Copy, Debug)]
pub struct GeometryDecoder {
    codec: GeomCodec,
    scale: f64,
    offset: [f64; 3],
}

impl GeometryDecoder {
    /// binary16, which needs no parameters.
    pub fn fp16() -> Self {
        GeometryDecoder {
            codec: GeomCodec::Fp16,
            scale: 1.0,
            offset: [0.0; 3],
        }
    }

    /// Fixed-point, rejecting parameters that cannot decode.
    ///
    /// A non-finite or non-positive scale would turn every stored integer into
    /// garbage or collapse the cloud to a point, and BOTH would render as a
    /// cloud rather than as an error — so they are refused here, once, rather
    /// than discovered per point.
    pub fn int16(scale: f64, offset: [f64; 3]) -> Result<Self, String> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!(
                "fixed-point geometry needs a positive finite scale, got {scale}"
            ));
        }
        if offset.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "fixed-point geometry needs finite offsets, got {offset:?}"
            ));
        }
        Ok(GeometryDecoder {
            codec: GeomCodec::Int16,
            scale,
            offset,
        })
    }

    pub fn codec(&self) -> GeomCodec {
        self.codec
    }

    pub fn is_fixed_point(&self) -> bool {
        self.codec == GeomCodec::Int16
    }

    /// Decode one component. `axis` is 0/1/2 and selects the fixed-point offset.
    #[inline]
    pub fn component(&self, word: u16, axis: usize) -> f64 {
        match self.codec {
            // NON-FINITE IS CONTENT here: a NaN in binary16 is the invalid-return
            // marker, and it is returned rather than rejected so the caller can
            // keep it IN PLACE and stay index-correlated with the other channels.
            GeomCodec::Fp16 => half::f16::from_bits(word).to_f64(),
            // The buffer is unsigned; the values are not. The cast IS the
            // reinterpretation.
            GeomCodec::Int16 => (word as i16) as f64 * self.scale + self.offset[axis],
        }
    }

    /// True when the point at `index` is an int16 invalid return.
    ///
    /// Whole-point, never per-component: the marker means "no return on this
    /// ray", and half a coordinate has no interpretation.
    #[inline]
    pub fn is_no_return(&self, words: &[u16], index: usize) -> bool {
        self.is_fixed_point()
            && words[index] as i16 == INT16_NO_RETURN
            && words[index + 1] as i16 == INT16_NO_RETURN
            && words[index + 2] as i16 == INT16_NO_RETURN
    }

    /// Decode one point, or `None` when it has no position to report.
    ///
    /// The marker is checked FIRST and that ordering is the point:
    /// `-32768 * scale + offset` is finite and entirely plausible, so leaving it
    /// to a finiteness test would let it through as a coordinate at the far edge
    /// of the grid — silently wrong data rather than an absent measurement.
    #[inline]
    pub fn point(&self, words: &[u16], index: usize) -> Option<[f64; 3]> {
        if self.is_no_return(words, index) {
            return None;
        }
        let point = [
            self.component(words[index], 0),
            self.component(words[index + 1], 1),
            self.component(words[index + 2], 2),
        ];
        if point.iter().all(|value| value.is_finite()) {
            Some(point)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_marker_is_recognised_before_the_arithmetic() {
        let decoder = GeometryDecoder::int16(0.01, [0.0; 3]).unwrap();
        let words = [0x8000u16, 0x8000, 0x8000];
        assert!(decoder.is_no_return(&words, 0));
        assert_eq!(decoder.point(&words, 0), None);
        // The bug this prevents: reconstructing it yields a plausible position.
        assert_eq!(decoder.component(0x8000, 0), -327.68);
    }

    #[test]
    fn a_partial_marker_is_not_an_invalid_return() {
        let decoder = GeometryDecoder::int16(0.01, [0.0; 3]).unwrap();
        let words = [0x8000u16, 0, 0];
        assert!(!decoder.is_no_return(&words, 0));
        // It decodes rather than vanishing; the FILE reader is what refuses it,
        // because only a file can be corrupt — a word cannot.
        assert!(decoder.point(&words, 0).is_some());
    }

    #[test]
    fn binary16_keeps_its_non_finite_content() {
        let decoder = GeometryDecoder::fp16();
        // 0x7e00 is NaN, the binary16 spelling of "no return".
        assert!(decoder.component(0x7e00, 0).is_nan());
        assert_eq!(decoder.point(&[0x7e00, 0x7e00, 0x7e00], 0), None);
    }

    #[test]
    fn fixed_point_parameters_are_validated_once() {
        assert!(GeometryDecoder::int16(0.0, [0.0; 3]).is_err());
        assert!(GeometryDecoder::int16(-1.0, [0.0; 3]).is_err());
        assert!(GeometryDecoder::int16(f64::NAN, [0.0; 3]).is_err());
        assert!(GeometryDecoder::int16(1.0, [f64::INFINITY, 0.0, 0.0]).is_err());
        assert!(GeometryDecoder::int16(1.0, [0.0; 3]).is_ok());
    }
}
