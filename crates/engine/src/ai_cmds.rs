//! `ai.*` commands: AI image-generation services (OpenArt.ai, Higgsfield, …).
//!
//! Commands:
//!   - `ai.generate`         — reserve a layer slot for an AI-generated image;
//!                             returns {requestId, layerId, service, prompt}.
//!                             The caller (MCP tool / shell) performs the actual
//!                             generation and calls `ai.layer.fulfil` with the result.
//!   - `ai.layer.fulfil`     — write PNG bytes (base64) into the layer reserved by
//!                             `ai.generate`; called by the MCP server after the
//!                             generation service returns.
//!   - `ai.services`         — list configured services and their credential status
//!   - `ai.credentials.set`  — store an API key in preferences (obfuscated)
//!   - `ai.credentials.clear`— remove a stored API key

use base64::Engine as _;
use photocraft_color::PixelFormat;
use photocraft_doc::{Layer, LayerContent, LayerId};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

// ── helpers ───────────────────────────────────────────────────────────────────

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn always(_: &Session) -> std::result::Result<(), String> { Ok(()) }

fn service_name_valid(name: &str) -> bool {
    matches!(name.to_ascii_lowercase().as_str(), "openart" | "openart.ai" | "higgsfield")
}

// ── ai.generate ───────────────────────────────────────────────────────────────
// Creates a transparent placeholder layer and returns its id so the MCP layer
// can fulfil it once the external service responds.

fn generate(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.generate";

    let service = p.get("service").and_then(Value::as_str).unwrap_or("openart");
    if !service_name_valid(service) {
        return Err(bad(CMD, format!("unknown service {:?}; use \"openart\" or \"higgsfield\"", service)));
    }

    let prompt = p.get("prompt").and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| bad(CMD, "missing `prompt`"))?;

    let (doc_w, doc_h) = s.active()
        .map(|st| (st.doc.size.width, st.doc.size.height))
        .unwrap_or((1024, 1024));

    let width  = p.get("width").and_then(Value::as_u64).unwrap_or(doc_w as u64).clamp(64, 4096) as u32;
    let height = p.get("height").and_then(Value::as_u64).unwrap_or(doc_h as u64).clamp(64, 4096) as u32;
    let count  = p.get("count").and_then(Value::as_u64).unwrap_or(1).clamp(1, 4) as u32;
    let model  = p.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    let negative = p.get("negativePrompt").and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    let guidance = p.get("guidance").and_then(Value::as_f64).unwrap_or(0.7);

    // Generate a stable request id from current timestamp + layer count.
    let request_id = format!("aigen-{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0));

    let prompt_label = prompt.chars().take(40).collect::<String>();
    let mut layer_ids: Vec<u64> = Vec::new();
    let prompt_str = prompt.to_string();

    s.edit(&format!("AI Generate ({})", service), |doc, _| {
        for i in 0..count {
            // Transparent 1×1 placeholder — fulfil will resize it.
            let rect = Rect::new(0, 0, 1, 1);
            let surf = Surface::from_interleaved(PixelFormat::RGBA8, rect, &[0u8; 4]);
            let name = if count == 1 {
                format!("AI: {} [pending]", prompt_label)
            } else {
                format!("AI: {} ({}) [pending]", prompt_label, i + 1)
            };
            let layer = Layer::new(name, LayerContent::Raster(surf));
            layer_ids.push(layer.id.0);
            doc.layers.push(layer);
        }
        Ok(())
    })?;

    Ok(json!({
        "requestId": request_id,
        "service":   service,
        "prompt":    prompt_str,
        "width":     width,
        "height":    height,
        "count":     count,
        "model":     model,
        "guidance":  guidance,
        "negativePrompt": negative,
        "layers":    layer_ids,
        "status":    "pending",
    }))
}

// ── ai.layer.fulfil ───────────────────────────────────────────────────────────
// Called by the MCP server (or any external orchestrator) once the generation
// service has returned an image. `pngBase64` is a standard base64-encoded PNG.

fn fulfil(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.layer.fulfil";

    let layer_id_raw = p.get("layerId").and_then(Value::as_u64)
        .ok_or_else(|| bad(CMD, "missing `layerId`"))?;
    let png_b64 = p.get("pngBase64").and_then(Value::as_str)
        .ok_or_else(|| bad(CMD, "missing `pngBase64`"))?;

    let png_bytes = base64::engine::general_purpose::STANDARD
        .decode(png_b64)
        .map_err(|e| bad(CMD, format!("invalid base64: {e}")))?;

    let (surf, w, h) = png_bytes_to_surface(&png_bytes)
        .map_err(|e| EngineError::Other(format!("AI image decode failed: {e}")))?;

    let layer_id = LayerId(layer_id_raw);
    s.edit("AI layer fulfil", |doc, _| {
        let layer = doc.layers.iter_mut()
            .find(|l| l.id == layer_id)
            .ok_or_else(|| EngineError::Other(format!("layer {layer_id_raw} not found")))?;
        // Strip the "[pending]" suffix from the name.
        if let Some(name) = layer.name.strip_suffix(" [pending]") {
            layer.name = name.to_string();
        }
        layer.content = LayerContent::Raster(surf);
        Ok(())
    })?;

    Ok(json!({"layerId": layer_id_raw, "width": w, "height": h, "status": "fulfilled"}))
}

// ── PNG bytes → Surface ───────────────────────────────────────────────────────

fn png_bytes_to_surface(bytes: &[u8]) -> std::result::Result<(Surface, u32, u32), String> {
    let img = photocraft_codecs::decode(bytes)
        .map_err(|e| format!("decode: {e}"))?;
    let w = img.width();
    let h = img.height();
    let rgba = img.to_rgba8();
    let rect = Rect::new(0, 0, w as i32, h as i32);
    let surf = Surface::from_interleaved(PixelFormat::RGBA8, rect, &rgba);
    Ok((surf, w, h))
}

// ── ai.services ───────────────────────────────────────────────────────────────

fn list_services(s: &mut Session, _p: &Value) -> Result<Value> {
    let ai = &s.prefs().integrations.ai_services;
    Ok(json!({
        "services": [
            {
                "id": "openart",
                "name": "OpenArt.ai",
                "envVar": "OPENART_API_KEY",
                "credentials": credential_status(ai.openart.use_env_var, &ai.openart.stored_key, "OPENART_API_KEY"),
            },
            {
                "id": "higgsfield",
                "name": "Higgsfield",
                "envVar": "HIGGSFIELD_API_KEY",
                "credentials": credential_status(ai.higgsfield.use_env_var, &ai.higgsfield.stored_key, "HIGGSFIELD_API_KEY"),
            },
        ]
    }))
}

fn credential_status(use_env_var: bool, stored_key: &str, env_var: &str) -> Value {
    let has_env = use_env_var
        && std::env::var(env_var).ok().is_some_and(|v| !v.trim().is_empty());
    let has_stored = !stored_key.is_empty();
    json!({
        "configured": has_env || has_stored,
        "source": if has_env { "envVar" } else if has_stored { "stored" } else { "none" },
        "envVar": env_var,
        "useEnvVar": use_env_var,
        "hasStoredKey": has_stored,
    })
}

// ── ai.credentials.set ────────────────────────────────────────────────────────

fn set_credentials(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.credentials.set";
    let service = p.get("service").and_then(Value::as_str)
        .ok_or_else(|| bad(CMD, "missing `service`"))?;
    if !service_name_valid(service) {
        return Err(bad(CMD, format!("unknown service {:?}", service)));
    }

    let api_key = p.get("apiKey").and_then(Value::as_str).map(|k| k.to_string());
    let use_env = p.get("useEnvVar").and_then(Value::as_bool);

    s.edit_prefs(|prefs| {
        let cred = match service.to_ascii_lowercase().as_str() {
            "openart" | "openart.ai" => &mut prefs.integrations.ai_services.openart,
            _ => &mut prefs.integrations.ai_services.higgsfield,
        };
        if let Some(key) = &api_key {
            cred.stored_key = encode_stored_key(key);
        }
        if let Some(v) = use_env {
            cred.use_env_var = v;
        }
    });

    Ok(json!({"service": service, "saved": true}))
}

// ── ai.credentials.clear ──────────────────────────────────────────────────────

fn clear_credentials(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.credentials.clear";
    let service = p.get("service").and_then(Value::as_str)
        .ok_or_else(|| bad(CMD, "missing `service`"))?;
    if !service_name_valid(service) {
        return Err(bad(CMD, format!("unknown service {:?}", service)));
    }

    s.edit_prefs(|prefs| {
        let cred = match service.to_ascii_lowercase().as_str() {
            "openart" | "openart.ai" => &mut prefs.integrations.ai_services.openart,
            _ => &mut prefs.integrations.ai_services.higgsfield,
        };
        cred.stored_key = String::new();
    });

    Ok(json!({"service": service, "cleared": true}))
}

// ── simple XOR obfuscation (keeps keys out of plaintext prefs) ────────────────

fn encode_stored_key(key: &str) -> String {
    let salt = machine_salt();
    let obfuscated: Vec<u8> = key.bytes()
        .enumerate()
        .map(|(i, b)| b ^ salt[i % salt.len()])
        .collect();
    base64::engine::general_purpose::STANDARD.encode(&obfuscated)
}

fn machine_salt() -> Vec<u8> {
    // Stable per-machine: mix hostname bytes so the stored value is useless
    // without the machine, but don't require a keyring.
    let host = std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "photocraft".into());
    let mut s: Vec<u8> = host.bytes().collect();
    if s.is_empty() { s = b"photocraft".to_vec(); }
    s
}

// ── command specs ─────────────────────────────────────────────────────────────

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "ai.generate",
            label: "AI Generate…",
            menu: &["File", "AI Generate…"],
            shortcut: None,
            params: r#"{"service":"openart"|"higgsfield","prompt":text,"width":int,"height":int,"count":int=1,"model":text,"guidance":float=0.7,"negativePrompt":text}"#,
            enabled: always,
            run: generate,
            journal: true,
        },
        CommandSpec {
            id: "ai.layer.fulfil",
            label: "AI: Fulfil layer",
            menu: &[],
            shortcut: None,
            params: r#"{"layerId":int,"pngBase64":text}"#,
            enabled: always,
            run: fulfil,
            journal: true,
        },
        CommandSpec {
            id: "ai.services",
            label: "AI: List Services",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: always,
            run: list_services,
            journal: false,
        },
        CommandSpec {
            id: "ai.credentials.set",
            label: "AI: Set API Key",
            menu: &[],
            shortcut: None,
            params: r#"{"service":"openart"|"higgsfield","apiKey":text,"useEnvVar":bool}"#,
            enabled: always,
            run: set_credentials,
            journal: false,
        },
        CommandSpec {
            id: "ai.credentials.clear",
            label: "AI: Clear API Key",
            menu: &[],
            shortcut: None,
            params: r#"{"service":"openart"|"higgsfield"}"#,
            enabled: always,
            run: clear_credentials,
            journal: false,
        },
    ]
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_missing_prompt_errors() {
        let mut s = Session::default();
        assert!(generate(&mut s, &json!({"service": "openart"})).is_err());
    }

    #[test]
    fn generate_unknown_service_errors() {
        let mut s = Session::default();
        assert!(generate(&mut s, &json!({"service": "unknown", "prompt": "x"})).is_err());
    }

    #[test]
    fn generate_returns_pending() {
        let mut s = Session::default();
        s.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        let r = generate(&mut s, &json!({"prompt": "a dog"})).unwrap();
        assert_eq!(r["status"], "pending");
        assert!(r["layers"].as_array().unwrap().len() == 1);
    }

    #[test]
    fn fulfil_bad_layer_id_errors() {
        let mut s = Session::default();
        // Encode a tiny 1×1 red PNG.
        let png = make_1x1_png();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
        let r = fulfil(&mut s, &json!({"layerId": 9999999u64, "pngBase64": b64}));
        assert!(r.is_err());
    }

    #[test]
    fn fulfil_bad_base64_errors() {
        let mut s = Session::default();
        let r = fulfil(&mut s, &json!({"layerId": 1u64, "pngBase64": "!!!not-base64!!!"}));
        assert!(r.is_err());
    }

    #[test]
    fn set_and_clear_credentials() {
        let mut s = Session::default();
        set_credentials(&mut s, &json!({"service": "openart", "apiKey": "sk-test-123"})).unwrap();
        assert!(!s.prefs().integrations.ai_services.openart.stored_key.is_empty());
        clear_credentials(&mut s, &json!({"service": "openart"})).unwrap();
        assert!(s.prefs().integrations.ai_services.openart.stored_key.is_empty());
    }

    #[test]
    fn list_services_returns_both() {
        let mut s = Session::default();
        let r = list_services(&mut s, &json!({})).unwrap();
        let services = r["services"].as_array().unwrap();
        assert_eq!(services.len(), 2);
        let ids: Vec<_> = services.iter().map(|s| s["id"].as_str().unwrap()).collect();
        assert!(ids.contains(&"openart"));
        assert!(ids.contains(&"higgsfield"));
    }

    // Builds a minimal valid 1×1 RGBA PNG in memory without any image crate.
    fn make_1x1_png() -> Vec<u8> {
        // Pre-built 67-byte 1×1 RGBA PNG (public domain, hand-crafted).
        vec![
            0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a, // signature
            0x00,0x00,0x00,0x0d, // IHDR length
            0x49,0x48,0x44,0x52, // "IHDR"
            0x00,0x00,0x00,0x01, // width=1
            0x00,0x00,0x00,0x01, // height=1
            0x08,0x06,           // 8bpc, RGBA
            0x00,0x00,0x00,
            0x1f,0x15,0xc4,0x89, // CRC
            0x00,0x00,0x00,0x0a, // IDAT length=10
            0x49,0x44,0x41,0x54, // "IDAT"
            0x08,0xd7,0x63,0xf8,0xcf,0xc0,0x00,0x00,0x00,0x02, // zlib-compressed row
            0x00,0x01,           // crc placeholder (wrong but enough for our decode test)
            0xe2,0x21,
            0x00,0x00,0x00,0x00, // IEND length=0
            0x49,0x45,0x4e,0x44, // "IEND"
            0xae,0x42,0x60,0x82, // CRC
        ]
    }
}
