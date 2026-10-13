//! `ai.*` commands: AI image-generation services (OpenArt.ai, Higgsfield, …).
//!
//! Commands:
//!   - `ai.generate`         — generate image(s) and place the result as new layer(s)
//!   - `ai.models`           — list available models for a service
//!   - `ai.services`         — list configured services and their credential status
//!   - `ai.credentials.set`  — store an API key in preferences (obfuscated)
//!   - `ai.credentials.clear`— remove a stored API key

use photocraft_ai_services::{
    AiGenerateRequest, AiServiceKind,
    credentials::{ServiceCredPrefs, encode_stored_key},
    openart::OpenArtService,
    types::{AiError, AiService},
};
use photocraft_color::PixelFormat;
use photocraft_doc::{Layer, LayerContent};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

// ── helpers ───────────────────────────────────────────────────────────────────

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn ai_err(e: AiError) -> EngineError {
    EngineError::Other(e.to_string())
}

fn always(_: &Session) -> std::result::Result<(), String> { Ok(()) }

fn service_kind(name: &str) -> std::result::Result<AiServiceKind, EngineError> {
    match name.to_ascii_lowercase().as_str() {
        "openart" | "openart.ai" => Ok(AiServiceKind::OpenArt),
        "higgsfield" => Ok(AiServiceKind::Higgsfield),
        other => Err(bad("ai", format!("unknown service {:?}; use \"openart\" or \"higgsfield\"", other))),
    }
}

fn service_prefs(s: &Session, kind: AiServiceKind) -> ServiceCredPrefs {
    let ai = &s.prefs().integrations.ai_services;
    match kind {
        AiServiceKind::OpenArt => ServiceCredPrefs {
            use_env_var: ai.openart.use_env_var,
            stored_key: ai.openart.stored_key.clone(),
        },
        AiServiceKind::Higgsfield => ServiceCredPrefs {
            use_env_var: ai.higgsfield.use_env_var,
            stored_key: ai.higgsfield.stored_key.clone(),
        },
    }
}

fn credential_status(prefs: &ServiceCredPrefs, kind: AiServiceKind) -> Value {
    let has_env = prefs.use_env_var
        && std::env::var(kind.env_var()).ok().is_some_and(|v| !v.trim().is_empty());
    let has_stored = !prefs.stored_key.is_empty();
    json!({
        "configured": has_env || has_stored,
        "source": if has_env { "envVar" } else if has_stored { "stored" } else { "none" },
        "envVar": kind.env_var(),
        "useEnvVar": prefs.use_env_var,
        "hasStoredKey": has_stored,
    })
}

// ── PNG bytes → Surface ───────────────────────────────────────────────────────

fn png_bytes_to_surface(bytes: &[u8]) -> Result<(Surface, u32, u32)> {
    let img = photocraft_codecs::decode(bytes)
        .map_err(|e| EngineError::Other(format!("AI image decode failed: {e}")))?;
    let w = img.width();
    let h = img.height();
    let rgba = img.to_rgba8();
    let rect = Rect::new(0, 0, w as i32, h as i32);
    let surf = Surface::from_interleaved(PixelFormat::RGBA8, rect, &rgba);
    Ok((surf, w, h))
}

// ── ai.generate ───────────────────────────────────────────────────────────────

fn generate(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.generate";

    let service_name = p.get("service").and_then(Value::as_str).unwrap_or("openart");
    let kind = service_kind(service_name)?;

    let prompt = p.get("prompt").and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| bad(CMD, "missing `prompt`"))?;

    let (doc_w, doc_h) = s.active()
        .map(|st| (st.doc.size.width, st.doc.size.height))
        .unwrap_or((1024, 1024));

    let width  = p.get("width").and_then(Value::as_u64).unwrap_or(doc_w as u64).min(4096).max(64) as u32;
    let height = p.get("height").and_then(Value::as_u64).unwrap_or(doc_h as u64).min(4096).max(64) as u32;
    let count  = p.get("count").and_then(Value::as_u64).unwrap_or(1).min(4).max(1) as u32;
    let model  = p.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    let guidance = p.get("guidance").and_then(Value::as_f64).unwrap_or(0.7) as f32;
    let negative = p.get("negativePrompt").and_then(Value::as_str).map(|s| s.to_string());

    let req = AiGenerateRequest {
        prompt: prompt.to_string(),
        negative_prompt: negative,
        width,
        height,
        count,
        model,
        guidance,
        source_image: None,
        strength: 0.75,
    };

    let prefs = service_prefs(s, kind);
    let result = run_generate(kind, prefs, &req)?;

    let mut layer_ids: Vec<u64> = Vec::new();
    let label = format!("AI Generate ({})", kind.name());
    let prompt_str = req.prompt.clone();
    s.edit(&label, |doc, _| {
        for (i, img) in result.images.iter().enumerate() {
            let (surf, _w, _h) = png_bytes_to_surface(&img.png_bytes)?;
            let name = if count == 1 {
                format!("AI: {}", prompt_str.chars().take(40).collect::<String>())
            } else {
                format!("AI: {} ({})", prompt_str.chars().take(40).collect::<String>(), i + 1)
            };
            let layer = Layer::new(name, LayerContent::Raster(surf));
            layer_ids.push(layer.id.0);
            doc.layers.push(layer);
        }
        Ok(())
    })?;

    Ok(json!({
        "service": service_name,
        "images": layer_ids.len(),
        "layers": layer_ids,
        "prompt": req.prompt,
    }))
}

// ── dispatch to service ───────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
fn run_generate(
    kind: AiServiceKind,
    prefs: ServiceCredPrefs,
    req: &AiGenerateRequest,
) -> Result<photocraft_ai_services::types::AiGenerateResult> {
    match kind {
        AiServiceKind::OpenArt => OpenArtService::new(prefs).generate(req).map_err(ai_err),
        AiServiceKind::Higgsfield => {
            Err(EngineError::Other("Higgsfield integration is not yet implemented".into()))
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn run_generate(
    _kind: AiServiceKind,
    _prefs: ServiceCredPrefs,
    _req: &AiGenerateRequest,
) -> Result<photocraft_ai_services::types::AiGenerateResult> {
    Err(EngineError::Other("AI generation is not available in the browser build".into()))
}

// ── ai.models ─────────────────────────────────────────────────────────────────

fn list_models(s: &mut Session, p: &Value) -> Result<Value> {
    let service_name = p.get("service").and_then(Value::as_str).unwrap_or("openart");
    let kind = service_kind(service_name)?;
    let prefs = service_prefs(s, kind);
    let models = run_list_models(kind, prefs)?;
    Ok(json!({
        "service": service_name,
        "models": models.iter().map(|(id, name)| json!({"id": id, "name": name})).collect::<Vec<_>>(),
    }))
}

#[cfg(not(target_arch = "wasm32"))]
fn run_list_models(kind: AiServiceKind, prefs: ServiceCredPrefs) -> Result<Vec<(String, String)>> {
    match kind {
        AiServiceKind::OpenArt => OpenArtService::new(prefs).list_models().map_err(ai_err),
        AiServiceKind::Higgsfield => Ok(Vec::new()),
    }
}

#[cfg(target_arch = "wasm32")]
fn run_list_models(_kind: AiServiceKind, _prefs: ServiceCredPrefs) -> Result<Vec<(String, String)>> {
    Ok(Vec::new())
}

// ── ai.services ───────────────────────────────────────────────────────────────

fn list_services(s: &mut Session, _p: &Value) -> Result<Value> {
    let ai = &s.prefs().integrations.ai_services;
    let openart_prefs = ServiceCredPrefs {
        use_env_var: ai.openart.use_env_var,
        stored_key: ai.openart.stored_key.clone(),
    };
    let higgsfield_prefs = ServiceCredPrefs {
        use_env_var: ai.higgsfield.use_env_var,
        stored_key: ai.higgsfield.stored_key.clone(),
    };
    Ok(json!({
        "services": [
            {
                "id": "openart",
                "name": AiServiceKind::OpenArt.name(),
                "envVar": AiServiceKind::OpenArt.env_var(),
                "credentials": credential_status(&openart_prefs, AiServiceKind::OpenArt),
            },
            {
                "id": "higgsfield",
                "name": AiServiceKind::Higgsfield.name(),
                "envVar": AiServiceKind::Higgsfield.env_var(),
                "credentials": credential_status(&higgsfield_prefs, AiServiceKind::Higgsfield),
            },
        ]
    }))
}

// ── ai.credentials.set ────────────────────────────────────────────────────────

fn set_credentials(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.credentials.set";
    let service_name = p.get("service").and_then(Value::as_str)
        .ok_or_else(|| bad(CMD, "missing `service`"))?;
    let kind = service_kind(service_name)?;

    let api_key = p.get("apiKey").and_then(Value::as_str).map(|k| k.to_string());
    let use_env = p.get("useEnvVar").and_then(Value::as_bool);

    s.edit_prefs(|prefs| {
        let cred = match kind {
            AiServiceKind::OpenArt => &mut prefs.integrations.ai_services.openart,
            AiServiceKind::Higgsfield => &mut prefs.integrations.ai_services.higgsfield,
        };
        if let Some(key) = &api_key {
            cred.stored_key = encode_stored_key(key);
        }
        if let Some(use_env_val) = use_env {
            cred.use_env_var = use_env_val;
        }
    });

    Ok(json!({"service": service_name, "saved": true}))
}

// ── ai.credentials.clear ──────────────────────────────────────────────────────

fn clear_credentials(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "ai.credentials.clear";
    let service_name = p.get("service").and_then(Value::as_str)
        .ok_or_else(|| bad(CMD, "missing `service`"))?;
    let kind = service_kind(service_name)?;

    s.edit_prefs(|prefs| {
        let cred = match kind {
            AiServiceKind::OpenArt => &mut prefs.integrations.ai_services.openart,
            AiServiceKind::Higgsfield => &mut prefs.integrations.ai_services.higgsfield,
        };
        cred.stored_key = String::new();
    });

    Ok(json!({"service": service_name, "cleared": true}))
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
            id: "ai.models",
            label: "AI: List Models",
            menu: &[],
            shortcut: None,
            params: r#"{"service":"openart"|"higgsfield"}"#,
            enabled: always,
            run: list_models,
            journal: false,
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
    use serde_json::json;

    #[test]
    fn service_kind_variants() {
        assert_eq!(service_kind("openart").unwrap(), AiServiceKind::OpenArt);
        assert_eq!(service_kind("openart.ai").unwrap(), AiServiceKind::OpenArt);
        assert_eq!(service_kind("higgsfield").unwrap(), AiServiceKind::Higgsfield);
        assert!(service_kind("unknown-service").is_err());
    }

    #[test]
    fn generate_missing_prompt_errors() {
        let mut session = Session::default();
        let p = json!({"service": "openart", "width": 512, "height": 512});
        let r = generate(&mut session, &p);
        assert!(r.is_err());
    }

    #[test]
    fn list_services_returns_both() {
        let mut session = Session::default();
        let r = list_services(&mut session, &json!({})).unwrap();
        let services = r["services"].as_array().unwrap();
        assert_eq!(services.len(), 2);
        let ids: Vec<_> = services.iter().map(|s| s["id"].as_str().unwrap()).collect();
        assert!(ids.contains(&"openart"));
        assert!(ids.contains(&"higgsfield"));
    }

    #[test]
    fn set_and_clear_credentials() {
        let mut session = Session::default();
        set_credentials(&mut session, &json!({"service": "openart", "apiKey": "sk-test-123"})).unwrap();
        assert!(!session.prefs().integrations.ai_services.openart.stored_key.is_empty());
        clear_credentials(&mut session, &json!({"service": "openart"})).unwrap();
        assert!(session.prefs().integrations.ai_services.openart.stored_key.is_empty());
    }
}
