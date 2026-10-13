//! Import-only reader for PaintShop Pro `.pspimage` (and `.psp`, `.pspframe`, `.psptube`,
//! `.pspwarp`) native layered format.
//!
//! Format reference: Corel's published PSP file format specification, versions 3–6 (PSP 7
//! through X9), plus community documentation from the GIMP-PSP plug-in and libpsp.
//!
//! Supported: layers (raster, mask, adjustment), groups, blend modes, opacity, visibility,
//! names, document size and colour depth (8-bit RGB/RGBA).
//!
//! Not supported (import-only; rendered as cached raster pixels or silently dropped):
//! vector/text/warp layers (kept as their cached raster), tubes, selections, spot channels.

#![forbid(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unimplemented,
    clippy::todo,
    clippy::unreachable
)]

use std::io::{Cursor, Read};

use photocraft_color::{BlendMode, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Group, Layer, LayerContent};
use photocraft_geom::{Rect, Size};
use photocraft_raster::Surface;

use crate::{ImportResult, IoError};

// ── magic / detection ─────────────────────────────────────────────────────────

/// The leading bytes that identify a PaintShop Pro image file.
pub(crate) const SHORT_MAGIC: &[u8] = b"Paint Shop Pro Image File\n";

pub(crate) fn is_psp(bytes: &[u8]) -> bool {
    bytes.starts_with(SHORT_MAGIC)
}

// ── block type constants ──────────────────────────────────────────────────────

const BLK_IMAGE: u16 = 0x01;
const BLK_LAYER: u16 = 0x05;
const BLK_CHANNEL: u16 = 0x06;
const BLK_ADJUSTMENT: u16 = 0x0c;
const BLK_GROUP_LAYER: u16 = 0x0d;

// ── compression constants ─────────────────────────────────────────────────────

const COMP_NONE: u8 = 0;
const COMP_RLE: u8 = 1;
const COMP_LZ77: u8 = 2;

// ── layer type constants ───────────────────────────────────────────────────────

const LAYER_RASTER: u8 = 0;
const LAYER_VECTOR: u8 = 2;
const LAYER_GROUP: u8 = 4;

// ── blend mode mapping ────────────────────────────────────────────────────────

fn psp_blend(mode: u8) -> BlendMode {
    match mode {
        0 => BlendMode::Normal,
        1 => BlendMode::Darken,
        2 => BlendMode::Lighten,
        3 => BlendMode::Hue,
        4 => BlendMode::Saturation,
        5 => BlendMode::Color,
        6 => BlendMode::Luminosity,
        7 => BlendMode::Multiply,
        8 => BlendMode::Screen,
        9 => BlendMode::Dissolve,
        10 => BlendMode::Overlay,
        11 => BlendMode::HardLight,
        12 => BlendMode::SoftLight,
        13 => BlendMode::Difference,
        14 => BlendMode::Exclusion,
        15 => BlendMode::ColorDodge,
        16 => BlendMode::ColorBurn,
        17 => BlendMode::LinearDodge,
        18 => BlendMode::LinearBurn,
        _ => BlendMode::Normal,
    }
}

// ── low-level readers ─────────────────────────────────────────────────────────

fn err(msg: impl Into<String>) -> IoError {
    IoError::Psp(msg.into())
}

fn read_u8(c: &mut Cursor<&[u8]>) -> Result<u8, IoError> {
    let mut b = [0u8];
    c.read_exact(&mut b).map_err(|e| err(format!("truncated u8: {e}")))?;
    Ok(b[0])
}

fn read_u16le(c: &mut Cursor<&[u8]>) -> Result<u16, IoError> {
    let mut b = [0u8; 2];
    c.read_exact(&mut b).map_err(|e| err(format!("truncated u16: {e}")))?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32le(c: &mut Cursor<&[u8]>) -> Result<u32, IoError> {
    let mut b = [0u8; 4];
    c.read_exact(&mut b).map_err(|e| err(format!("truncated u32: {e}")))?;
    Ok(u32::from_le_bytes(b))
}

fn read_i32le(c: &mut Cursor<&[u8]>) -> Result<i32, IoError> {
    let mut b = [0u8; 4];
    c.read_exact(&mut b).map_err(|e| err(format!("truncated i32: {e}")))?;
    Ok(i32::from_le_bytes(b))
}

fn read_bytes_vec(c: &mut Cursor<&[u8]>, n: usize) -> Result<Vec<u8>, IoError> {
    const MAX_READ: usize = 256 * 1024 * 1024;
    if n > MAX_READ {
        return Err(err(format!("read request {n} bytes exceeds limit")));
    }
    let mut buf = vec![0u8; n];
    c.read_exact(&mut buf).map_err(|e| err(format!("truncated read of {n} bytes: {e}")))?;
    Ok(buf)
}

fn read_pascal_string(c: &mut Cursor<&[u8]>) -> Result<String, IoError> {
    let len = read_u16le(c)? as usize;
    if len == 0 {
        return Ok(String::new());
    }
    let raw = read_bytes_vec(c, len)?;
    Ok(String::from_utf8_lossy(&raw).into_owned())
}

// ── decompression ─────────────────────────────────────────────────────────────

fn decompress_rle(src: &[u8], expected: usize) -> Result<Vec<u8>, IoError> {
    let cap = expected.min(64 * 1024 * 1024);
    let mut out: Vec<u8> = Vec::with_capacity(cap);
    let mut i = 0;
    while i < src.len() {
        let count = src[i] as usize;
        i += 1;
        if count < 128 {
            let n = count + 1;
            let end = i.checked_add(n).ok_or_else(|| err("RLE arithmetic overflow"))?;
            if end > src.len() {
                return Err(err("RLE literal run extends past data"));
            }
            out.extend_from_slice(&src[i..end]);
            i = end;
        } else {
            if i >= src.len() {
                return Err(err("RLE truncated repeat byte"));
            }
            let byte = src[i];
            i += 1;
            let repeats = count - 128 + 2;
            out.extend(std::iter::repeat(byte).take(repeats));
        }
        if out.len() > expected.saturating_add(1024) {
            return Err(err("RLE output exceeds expected size"));
        }
    }
    Ok(out)
}

fn decompress_lz77(src: &[u8], expected: usize) -> Result<Vec<u8>, IoError> {
    use flate2::read::ZlibDecoder;
    let mut decoder = ZlibDecoder::new(src);
    let cap = expected.min(64 * 1024 * 1024);
    let mut out = Vec::with_capacity(cap);
    decoder.read_to_end(&mut out).map_err(|e| err(format!("zlib decompress failed: {e}")))?;
    Ok(out)
}

fn decompress(comp: u8, src: &[u8], expected: usize) -> Result<Vec<u8>, IoError> {
    match comp {
        COMP_NONE => Ok(src.to_vec()),
        COMP_RLE => decompress_rle(src, expected),
        COMP_LZ77 => decompress_lz77(src, expected),
        other => Err(err(format!("unknown compression type {other}"))),
    }
}

// ── block header ─────────────────────────────────────────────────────────────

struct BlockHeader {
    block_type: u16,
    /// Length of the initial data chunk immediately following this header.
    chunk_length: u32,
    /// Total block size (header + chunk + sub-blocks), used to advance past this block.
    total_length: u32,
}

/// Attempt to read a block header. Returns `None` at EOF or on padding.
fn read_block_header(c: &mut Cursor<&[u8]>, data: &[u8]) -> Option<(u64, BlockHeader)> {
    let start = c.position() as usize;
    if start + 4 > data.len() {
        return None;
    }
    let pos_before = c.position();

    // PSP v4+ blocks start with "~BK\0" (4 bytes); v3 starts with the 2-byte block type directly.
    let mut marker = [0u8; 4];
    c.read_exact(&mut marker).ok()?;
    if &marker == b"~BK\0" {
        let block_type = read_u16le(c).ok()?;
        let chunk_length = read_u32le(c).ok()?;
        let total_length = read_u32le(c).ok()?;
        Some((pos_before, BlockHeader { block_type, chunk_length, total_length }))
    } else {
        // Rewind and try v3: 2-byte type, 4-byte chunk, 4-byte total
        c.set_position(pos_before);
        let block_type = read_u16le(c).ok()?;
        let chunk_length = read_u32le(c).ok()?;
        let total_length = read_u32le(c).ok()?;
        Some((pos_before, BlockHeader { block_type, chunk_length, total_length }))
    }
}

// ── IMAGE block ───────────────────────────────────────────────────────────────

struct ImageHeader {
    width: u32,
    height: u32,
    color_mode: u8,
}

const COLOR_MODE_RGB: u8 = 4;
const COLOR_MODE_GRAY: u8 = 1;

fn parse_image_header(chunk: &[u8]) -> Result<ImageHeader, IoError> {
    if chunk.len() < 10 {
        return Err(err("IMAGE block too small"));
    }
    let width = u32::from_le_bytes(chunk[0..4].try_into().map_err(|_| err("IMAGE width"))?);
    let height = u32::from_le_bytes(chunk[4..8].try_into().map_err(|_| err("IMAGE height"))?);
    // Byte 14 = compression, 15 = bit depth, 18 = color mode (offsets from start of chunk)
    let color_mode = if chunk.len() > 18 { chunk[18] } else { COLOR_MODE_RGB };
    Ok(ImageHeader { width, height, color_mode })
}

// ── LAYER block ───────────────────────────────────────────────────────────────

struct LayerInfo {
    name: String,
    layer_type: u8,
    opacity: u8,
    blend_mode: u8,
    visible: bool,
    /// Position and size of this layer on the canvas (may be negative / larger than document).
    rect: Rect,
    channel_compression: u8,
}

fn parse_layer_info(chunk: &[u8]) -> Result<LayerInfo, IoError> {
    let mut c = Cursor::new(chunk);
    let name = read_pascal_string(&mut c)?;
    let layer_type = read_u8(&mut c)?;
    let x1 = read_i32le(&mut c)?;
    let y1 = read_i32le(&mut c)?;
    let x2 = read_i32le(&mut c)?;
    let y2 = read_i32le(&mut c)?;
    let opacity = read_u8(&mut c)?;
    let blend_mode = read_u8(&mut c)?;
    let flags = read_u8(&mut c)?;
    // Bit 0 of flags is "hidden" in some versions; others use inverted. We treat 0 = visible.
    let visible = (flags & 0x01) == 0;
    let channel_compression = read_u8(&mut c).unwrap_or(COMP_RLE);

    let w = x2.saturating_sub(x1).max(0);
    let h = y2.saturating_sub(y1).max(0);
    let rect = Rect::new(x1, y1, w, h);

    Ok(LayerInfo { name, layer_type, opacity, blend_mode, visible, rect, channel_compression })
}

// ── CHANNEL block ─────────────────────────────────────────────────────────────

const CHAN_COMPOSITE: u16 = 0;
const CHAN_RED: u16 = 1;
const CHAN_GREEN: u16 = 2;
const CHAN_BLUE: u16 = 3;
const CHAN_ALPHA: u16 = 4;

struct ChannelData {
    channel_type: u16,
    pixels: Vec<u8>,
}

fn parse_channel(chunk: &[u8]) -> Result<ChannelData, IoError> {
    let mut c = Cursor::new(chunk);
    let compressed_len = read_u32le(&mut c)? as usize;
    let uncompressed_len = read_u32le(&mut c)? as usize;
    let compression = read_u8(&mut c)?;
    let _bitmap_type = read_u8(&mut c)?;
    let channel_type = read_u16le(&mut c)?;

    let pos = c.position() as usize;
    let src = chunk.get(pos..pos.checked_add(compressed_len).ok_or_else(|| err("channel length overflow"))?).ok_or_else(|| err("channel data extends past block"))?;
    let pixels = decompress(compression, src, uncompressed_len)?;

    Ok(ChannelData { channel_type, pixels })
}

// ── assemble RGBA from planar channels ────────────────────────────────────────

fn channels_to_rgba8(channels: &[ChannelData], w: u32, h: u32) -> Vec<u8> {
    let n = (w as usize).saturating_mul(h as usize);
    let mut rgba = vec![255u8; n * 4];

    let find = |ct: u16| channels.iter().find(|c| c.channel_type == ct).map(|c| c.pixels.as_slice());

    if let Some(comp) = find(CHAN_COMPOSITE) {
        if comp.len() >= n * 4 {
            rgba.copy_from_slice(&comp[..n * 4]);
            return rgba;
        }
    }
    if let Some(r) = find(CHAN_RED) {
        for (i, &v) in r.iter().take(n).enumerate() {
            rgba[i * 4] = v;
        }
    }
    if let Some(g) = find(CHAN_GREEN) {
        for (i, &v) in g.iter().take(n).enumerate() {
            rgba[i * 4 + 1] = v;
        }
    }
    if let Some(b) = find(CHAN_BLUE) {
        for (i, &v) in b.iter().take(n).enumerate() {
            rgba[i * 4 + 2] = v;
        }
    }
    if let Some(a) = find(CHAN_ALPHA) {
        for (i, &v) in a.iter().take(n).enumerate() {
            rgba[i * 4 + 3] = v;
        }
    }
    rgba
}

// ── pending layer state ───────────────────────────────────────────────────────

struct PendingLayer {
    info: LayerInfo,
    channels: Vec<ChannelData>,
}

impl PendingLayer {
    fn build(self, warnings: &mut Vec<String>) -> Option<Layer> {
        let info = self.info;
        let channels = self.channels;

        if info.layer_type == LAYER_GROUP {
            let mut l = Layer::group(info.name, Vec::new());
            l.opacity = info.opacity as f32 / 255.0;
            l.blend = psp_blend(info.blend_mode);
            l.visible = info.visible;
            return Some(l);
        }

        if info.layer_type == LAYER_VECTOR {
            warnings.push(format!("PSP vector layer '{}' imported as rasterised pixels", info.name));
        }

        let w = info.rect.width().max(0) as u32;
        let h = info.rect.height().max(0) as u32;

        if w == 0 || h == 0 {
            // Empty layer — still add it so the layer list is complete
            let surf = Surface::new(PixelFormat::RGBA8);
            let mut l = Layer::new(info.name, LayerContent::Raster(surf));
            l.opacity = info.opacity as f32 / 255.0;
            l.blend = psp_blend(info.blend_mode);
            l.visible = info.visible;
            return Some(l);
        }

        let rgba = channels_to_rgba8(&channels, w, h);
        let expected = (w as usize).saturating_mul(h as usize).saturating_mul(4);
        if rgba.len() < expected {
            warnings.push(format!("PSP layer '{}': pixel data truncated; layer may appear incomplete", info.name));
        }

        // Place the surface at the layer's canvas position via the Rect offset.
        let rect = Rect::new(info.rect.x0, info.rect.y0, w as i32, h as i32);
        let surf = Surface::from_interleaved(PixelFormat::RGBA8, rect, &rgba);

        let mut l = Layer::new(info.name, LayerContent::Raster(surf));
        l.opacity = info.opacity as f32 / 255.0;
        l.blend = psp_blend(info.blend_mode);
        l.visible = info.visible;
        Some(l)
    }
}

// ── main import ───────────────────────────────────────────────────────────────

/// Import a PaintShop Pro `.pspimage` / `.psp` file.
pub(crate) fn import(name: &str, bytes: &[u8], _ctl: &photocraft_raster::Interrupt) -> Result<ImportResult, IoError> {
    if bytes.len() < 36 {
        return Err(err("file too small to be a PaintShop Pro image"));
    }
    if !is_psp(bytes) {
        return Err(err("not a PaintShop Pro image file (wrong magic)"));
    }

    // Skip the 32-byte file identifier, then the 2+2 byte major/minor version.
    let mut c = Cursor::new(bytes);
    c.set_position(32);
    let _version_major = read_u16le(&mut c)?;
    let _version_minor = read_u16le(&mut c)?;

    let mut doc_width: u32 = 0;
    let mut doc_height: u32 = 0;
    let mut warnings: Vec<String> = Vec::new();
    let mut layers: Vec<Layer> = Vec::new();
    let mut pending: Option<PendingLayer> = None;

    const MAX_LAYERS: usize = 1024;

    loop {
        if c.position() as usize >= bytes.len() {
            break;
        }
        let (pos_before, header) = match read_block_header(&mut c, bytes) {
            Some(h) => h,
            None => break,
        };

        let header_size = (c.position() - pos_before) as usize;
        let chunk_start = c.position() as usize;
        let chunk_len = header.chunk_length as usize;
        let chunk_end = chunk_start.saturating_add(chunk_len).min(bytes.len());
        let chunk = bytes.get(chunk_start..chunk_end).unwrap_or(&[]);

        // Advance cursor to total_end for the next iteration.
        let total_end = pos_before as usize + header_size + header.total_length as usize;
        let total_end = total_end.min(bytes.len());

        match header.block_type {
            BLK_IMAGE => {
                if let Ok(ih) = parse_image_header(chunk) {
                    doc_width = ih.width;
                    doc_height = ih.height;
                    if ih.color_mode != COLOR_MODE_RGB && ih.color_mode != COLOR_MODE_GRAY {
                        warnings.push(format!(
                            "PSP colour mode {} is not RGB or Grayscale; colours may not be accurate",
                            ih.color_mode
                        ));
                    }
                }
            }

            BLK_LAYER | BLK_ADJUSTMENT | BLK_GROUP_LAYER => {
                // Flush the previous layer before starting the next one.
                if let Some(p) = pending.take() {
                    if layers.len() < MAX_LAYERS {
                        if let Some(l) = p.build(&mut warnings) {
                            layers.push(l);
                        }
                    }
                }
                if let Ok(info) = parse_layer_info(chunk) {
                    pending = Some(PendingLayer { info, channels: Vec::new() });
                }
            }

            BLK_CHANNEL => {
                if let Some(ref mut p) = pending {
                    if let Ok(ch) = parse_channel(chunk) {
                        p.channels.push(ch);
                    }
                }
            }

            _ => { /* unknown or intentionally skipped block */ }
        }

        c.set_position(total_end as u64);
    }

    // Flush last pending layer.
    if let Some(p) = pending.take() {
        if layers.len() < MAX_LAYERS {
            if let Some(l) = p.build(&mut warnings) {
                layers.push(l);
            }
        }
    }

    if doc_width == 0 || doc_height == 0 {
        return Err(err("PSP IMAGE block not found or document has zero dimensions"));
    }
    if doc_width > 32768 || doc_height > 32768 {
        return Err(err(format!("PSP document {doc_width}×{doc_height} exceeds maximum 32768×32768")));
    }

    let size = Size::new(doc_width, doc_height);
    let mut doc = Document::new(name, size, ColorMode::Rgb, SampleType::U8);

    // PSP stores layers bottom-to-top; the document model uses top-to-bottom.
    layers.reverse();
    for l in layers {
        doc.layers.push(l);
    }

    warnings.push(
        "PaintShop Pro files are import-only; vector, text and tube layers are rasterised".into(),
    );

    Ok(ImportResult { document: doc, warnings, source_read_only: true, preview_only: false })
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_raster::Interrupt;

    fn make_magic() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(SHORT_MAGIC);
        // pad to 32 bytes
        v.resize(32, 0);
        // version major/minor
        v.extend_from_slice(&1u16.to_le_bytes());
        v.extend_from_slice(&0u16.to_le_bytes());
        v
    }

    #[test]
    fn detects_psp_magic() {
        let bytes = make_magic();
        assert!(is_psp(&bytes));
    }

    #[test]
    fn rejects_wrong_magic() {
        let bytes = b"PNG\x89..........";
        assert!(!is_psp(bytes));
    }

    #[test]
    fn import_rejects_non_psp() {
        let r = import("t.pspimage", b"PNG\x89...", &Interrupt::NONE);
        assert!(r.is_err());
    }

    #[test]
    fn import_rejects_too_small() {
        let r = import("t.pspimage", b"Paint Shop", &Interrupt::NONE);
        assert!(r.is_err());
    }

    #[test]
    fn rle_literal_run() {
        let src = [0x02u8, 0xAA, 0xBB, 0xCC];
        let out = decompress_rle(&src, 3).unwrap();
        assert_eq!(out, [0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn rle_repeat_run() {
        // 0x82 = 130 → repeats = 130-128+2 = 4
        let src = [0x82u8, 0xFF];
        let out = decompress_rle(&src, 4).unwrap();
        assert_eq!(out, [0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn rle_truncated_repeat_errors() {
        let src = [0x82u8]; // repeat byte missing
        assert!(decompress_rle(&src, 2).is_err());
    }

    #[test]
    fn rle_literal_past_end_errors() {
        let src = [0x02u8, 0xAA]; // says 3 bytes follow, only 1 present
        assert!(decompress_rle(&src, 3).is_err());
    }

    #[test]
    fn blend_unknown_fallback() {
        assert_eq!(psp_blend(200), BlendMode::Normal);
    }

    #[test]
    fn blend_known_modes() {
        assert_eq!(psp_blend(7), BlendMode::Multiply);
        assert_eq!(psp_blend(8), BlendMode::Screen);
        assert_eq!(psp_blend(10), BlendMode::Overlay);
    }

    #[test]
    fn channels_to_rgba8_planar() {
        let r = ChannelData { channel_type: CHAN_RED, pixels: vec![0xAA, 0xBB] };
        let g = ChannelData { channel_type: CHAN_GREEN, pixels: vec![0x11, 0x22] };
        let b = ChannelData { channel_type: CHAN_BLUE, pixels: vec![0x33, 0x44] };
        let channels = vec![r, g, b];
        let out = channels_to_rgba8(&channels, 2, 1);
        assert_eq!(out[0..4], [0xAA, 0x11, 0x33, 0xFF]); // alpha defaults to 255
        assert_eq!(out[4..8], [0xBB, 0x22, 0x44, 0xFF]);
    }
}
