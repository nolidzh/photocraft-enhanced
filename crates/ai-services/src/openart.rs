//! OpenArt.ai image-generation adapter.
//!
//! API reference: https://openart.ai/api (v2 REST endpoint).
//! All network calls are gated on `cfg(not(target_arch = "wasm32"))`.

use serde::Deserialize;

use crate::credentials::{ServiceCredPrefs, resolve_api_key};
use crate::types::{AiError, AiGenerateRequest, AiGenerateResult, AiGeneratedImage, AiService, AiServiceKind};

// ── constants ─────────────────────────────────────────────────────────────────

const API_BASE: &str = "https://api.openart.ai/v2";
const TIMEOUT_SECS: u64 = 120;

// ── adapter struct ────────────────────────────────────────────────────────────

/// OpenArt.ai service adapter. Constructed once per session with the current credentials.
pub struct OpenArtService {
    prefs: ServiceCredPrefs,
}

impl OpenArtService {
    pub fn new(prefs: ServiceCredPrefs) -> Self {
        Self { prefs }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn api_key(&self) -> Result<String, AiError> {
        resolve_api_key(AiServiceKind::OpenArt, &self.prefs)
            .map(|(k, _)| k)
            .ok_or(AiError::NoApiKey {
                service: "OpenArt.ai",
                env_var: AiServiceKind::OpenArt.env_var(),
            })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn agent() -> ureq::Agent {
        ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(TIMEOUT_SECS))
            .build()
    }
}

// ── AiService impl ────────────────────────────────────────────────────────────

impl AiService for OpenArtService {
    fn kind(&self) -> AiServiceKind {
        AiServiceKind::OpenArt
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn generate(&self, req: &AiGenerateRequest) -> Result<AiGenerateResult, AiError> {
        let key = self.api_key()?;

        // Build the request body.  OpenArt's /images/generations endpoint accepts:
        // { prompt, negative_prompt?, width, height, n, model?, guidance_scale?,
        //   init_image_base64?, strength? }
        let body = build_generate_body(req)?;

        let agent = Self::agent();
        let resp = agent
            .post(&format!("{API_BASE}/images/generations"))
            .set("Authorization", &format!("Bearer {key}"))
            .set("Content-Type", "application/json")
            .set("User-Agent", "PhotoCraft-dev/1.0")
            .send_json(&body)
            .map_err(|e| network_err(e))?;

        let status = resp.status();
        if status != 200 {
            let msg = resp.into_string().unwrap_or_else(|_| "unknown error".into());
            return Err(AiError::ApiError { service: "OpenArt.ai", status, message: msg });
        }

        let raw: serde_json::Value = resp.into_json()
            .map_err(|e| AiError::Parse { service: "OpenArt.ai", message: e.to_string() })?;

        parse_generate_response(raw)
    }

    #[cfg(target_arch = "wasm32")]
    fn generate(&self, _req: &AiGenerateRequest) -> Result<AiGenerateResult, AiError> {
        Err(AiError::Other { service: "OpenArt.ai", message: "AI generation is not available in the browser build".into() })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn list_models(&self) -> Result<Vec<(String, String)>, AiError> {
        let key = self.api_key()?;
        let agent = Self::agent();
        let resp = agent
            .get(&format!("{API_BASE}/models"))
            .set("Authorization", &format!("Bearer {key}"))
            .set("User-Agent", "PhotoCraft-dev/1.0")
            .call()
            .map_err(|e| network_err(e))?;

        let status = resp.status();
        if status != 200 {
            let msg = resp.into_string().unwrap_or_else(|_| "unknown error".into());
            return Err(AiError::ApiError { service: "OpenArt.ai", status, message: msg });
        }

        let raw: serde_json::Value = resp.into_json()
            .map_err(|e| AiError::Parse { service: "OpenArt.ai", message: e.to_string() })?;

        parse_models_response(raw)
    }

    #[cfg(target_arch = "wasm32")]
    fn list_models(&self) -> Result<Vec<(String, String)>, AiError> {
        Ok(Vec::new())
    }
}

// ── request / response helpers ────────────────────────────────────────────────

fn build_generate_body(req: &AiGenerateRequest) -> Result<serde_json::Value, AiError> {
    use serde_json::json;

    let mut body = json!({
        "prompt": req.prompt,
        "width":  req.width,
        "height": req.height,
        "n":      req.count.min(4).max(1),
        "guidance_scale": (req.guidance * 10.0) as u32,  // OpenArt uses 1–10 scale
    });

    if let Some(neg) = &req.negative_prompt {
        body["negative_prompt"] = json!(neg);
    }
    if !req.model.is_empty() {
        body["model"] = json!(req.model);
    }
    if let Some(src) = &req.source_image {
        let encoded = base64_encode(src);
        body["init_image_base64"] = json!(encoded);
        body["strength"] = json!(req.strength);
    }

    Ok(body)
}

fn base64_encode(bytes: &[u8]) -> String {
    // Use the same table from credentials.rs (we can't depend on it directly, so repeat here).
    const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64[((n >> 18) & 0x3f) as usize] as char);
        out.push(B64[((n >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 { out.push(B64[((n >> 6) & 0x3f) as usize] as char); } else { out.push('='); }
        if chunk.len() > 2 { out.push(B64[(n & 0x3f) as usize] as char); } else { out.push('='); }
    }
    out
}

fn parse_generate_response(raw: serde_json::Value) -> Result<AiGenerateResult, AiError> {
    // OpenArt returns: { data: [ { url: "...", seed: N }, ... ], ... }
    let data = raw.get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| AiError::Parse { service: "OpenArt.ai", message: "missing 'data' array".into() })?;

    let mut images = Vec::new();
    for item in data {
        let url = item.get("url").and_then(|u| u.as_str())
            .ok_or_else(|| AiError::Parse { service: "OpenArt.ai", message: "image item missing 'url'".into() })?;
        let seed = item.get("seed").and_then(|s| s.as_u64());

        let png_bytes = fetch_image_bytes(url)?;
        let (width, height) = image_dimensions(&png_bytes)?;

        images.push(AiGeneratedImage { png_bytes, width, height, seed });
    }

    if images.is_empty() {
        return Err(AiError::Parse { service: "OpenArt.ai", message: "no images in response".into() });
    }

    let effective_prompt = raw.get("prompt").and_then(|p| p.as_str()).unwrap_or("").to_string();

    Ok(AiGenerateResult { images, effective_prompt })
}

fn parse_models_response(raw: serde_json::Value) -> Result<Vec<(String, String)>, AiError> {
    // OpenArt returns: { data: [ { id: "...", name: "..." }, ... ] }
    let data = raw.get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| AiError::Parse { service: "OpenArt.ai", message: "missing 'data' array in models response".into() })?;

    let mut models = Vec::new();
    for item in data {
        let id = item.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let name = item.get("name").and_then(|v| v.as_str()).unwrap_or(&id).to_string();
        if !id.is_empty() {
            models.push((id, name));
        }
    }
    Ok(models)
}

// ── image fetching / decoding ─────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
fn fetch_image_bytes(url: &str) -> Result<Vec<u8>, AiError> {
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(60))
        .build();
    let resp = agent
        .get(url)
        .set("User-Agent", "PhotoCraft-dev/1.0")
        .call()
        .map_err(|e| network_err(e))?;
    let status = resp.status();
    if status != 200 {
        return Err(AiError::ApiError { service: "OpenArt.ai", status, message: "image download failed".into() });
    }
    let mut bytes = Vec::new();
    resp.into_reader().read_to_end(&mut bytes)
        .map_err(|e| AiError::Network(format!("image read failed: {e}")))?;
    Ok(bytes)
}

#[cfg(target_arch = "wasm32")]
fn fetch_image_bytes(_url: &str) -> Result<Vec<u8>, AiError> {
    Err(AiError::Other { service: "OpenArt.ai", message: "not available in wasm".into() })
}

fn image_dimensions(png_bytes: &[u8]) -> Result<(u32, u32), AiError> {
    // Read PNG IHDR: 8 byte signature + 4 len + 4 "IHDR" + 4 width + 4 height
    if png_bytes.len() < 24 {
        return Err(AiError::ImageDecode("image data too small".into()));
    }
    if !png_bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        // JPEG or other — estimate from JPEG SOF
        return jpeg_dimensions(png_bytes);
    }
    let w = u32::from_be_bytes(png_bytes[16..20].try_into().map_err(|_| AiError::ImageDecode("bad width bytes".into()))?);
    let h = u32::from_be_bytes(png_bytes[20..24].try_into().map_err(|_| AiError::ImageDecode("bad height bytes".into()))?);
    Ok((w, h))
}

fn jpeg_dimensions(bytes: &[u8]) -> Result<(u32, u32), AiError> {
    let mut i = 0;
    while i + 4 < bytes.len() {
        if bytes[i] != 0xFF { break; }
        let marker = bytes[i + 1];
        if matches!(marker, 0xC0 | 0xC1 | 0xC2) {
            // SOF: 2 len + 1 precision + 2 height + 2 width
            if i + 9 < bytes.len() {
                let h = u16::from_be_bytes([bytes[i + 5], bytes[i + 6]]) as u32;
                let w = u16::from_be_bytes([bytes[i + 7], bytes[i + 8]]) as u32;
                return Ok((w, h));
            }
        }
        if i + 4 > bytes.len() { break; }
        let len = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]) as usize;
        i += 2 + len;
    }
    // Fallback: unknown dimensions
    Ok((0, 0))
}

// ── error helpers ─────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
fn network_err(e: ureq::Error) -> AiError {
    match e {
        ureq::Error::Status(status, resp) => {
            let msg = resp.into_string().unwrap_or_else(|_| "unknown".into());
            AiError::ApiError { service: "OpenArt.ai", status, message: msg }
        }
        ureq::Error::Transport(t) => AiError::Network(t.to_string()),
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_dimensions_valid() {
        // Minimal valid 1×1 PNG header
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&[0, 0, 0, 13]); // IHDR length
        png.extend_from_slice(b"IHDR");
        png.extend_from_slice(&1u32.to_be_bytes()); // width
        png.extend_from_slice(&1u32.to_be_bytes()); // height
        let (w, h) = image_dimensions(&png).unwrap();
        assert_eq!((w, h), (1, 1));
    }

    #[test]
    fn png_dimensions_too_small_errors() {
        assert!(image_dimensions(b"\x89PNG\r\n\x1a").is_err());
    }

    #[test]
    fn parse_generate_response_missing_data() {
        let raw = serde_json::json!({ "status": "ok" });
        assert!(parse_generate_response(raw).is_err());
    }

    #[test]
    fn parse_models_response_empty_data() {
        let raw = serde_json::json!({ "data": [] });
        let models = parse_models_response(raw).unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn no_api_key_returns_error() {
        // use_env_var=false and empty stored key → NoApiKey without consulting env.
        let svc = OpenArtService::new(crate::credentials::ServiceCredPrefs {
            use_env_var: false,
            stored_key: String::new(),
        });
        let req = AiGenerateRequest::text_to_image("a cat", 512, 512);
        let err = svc.generate(&req);
        assert!(matches!(err, Err(AiError::NoApiKey { .. })));
    }
}

// Needed for std::io::Read in fetch_image_bytes
#[cfg(not(target_arch = "wasm32"))]
use std::io::Read;
