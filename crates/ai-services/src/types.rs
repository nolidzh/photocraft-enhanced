//! Shared types for AI service adapters.

use serde::{Deserialize, Serialize};

/// Which AI service backend to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AiServiceKind {
    OpenArt,
    Higgsfield,
}

impl AiServiceKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::OpenArt => "OpenArt.ai",
            Self::Higgsfield => "Higgsfield",
        }
    }

    /// The environment variable name for this service's API key.
    pub fn env_var(self) -> &'static str {
        match self {
            Self::OpenArt => "OPENART_API_KEY",
            Self::Higgsfield => "HIGGSFIELD_API_KEY",
        }
    }

    /// The preferences key used to store the encrypted API key.
    pub fn prefs_key(self) -> &'static str {
        match self {
            Self::OpenArt => "openart",
            Self::Higgsfield => "higgsfield",
        }
    }
}

/// A request to generate one or more images.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiGenerateRequest {
    /// Text prompt describing the desired image.
    pub prompt: String,
    /// Optional negative prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    /// Target width in pixels (service may round to its supported sizes).
    pub width: u32,
    /// Target height in pixels.
    pub height: u32,
    /// Number of images to generate (1–4; service may cap lower).
    #[serde(default = "default_count")]
    pub count: u32,
    /// Model / style identifier (service-specific; empty = service default).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    /// Creativity / guidance scale (0.0–1.0; service normalises internally).
    #[serde(default = "default_guidance")]
    pub guidance: f32,
    /// Optional source image bytes (JPEG or PNG) for img2img / inpainting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_image: Option<Vec<u8>>,
    /// img2img strength (0.0 = keep source, 1.0 = ignore source).
    #[serde(default = "default_strength")]
    pub strength: f32,
}

fn default_count() -> u32 { 1 }
fn default_guidance() -> f32 { 0.7 }
fn default_strength() -> f32 { 0.75 }

impl AiGenerateRequest {
    /// Construct a minimal text-to-image request.
    pub fn text_to_image(prompt: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            prompt: prompt.into(),
            negative_prompt: None,
            width,
            height,
            count: 1,
            model: String::new(),
            guidance: 0.7,
            source_image: None,
            strength: 0.75,
        }
    }
}

/// A single generated image.
#[derive(Clone, Debug)]
pub struct AiGeneratedImage {
    /// Raw PNG bytes of the generated image.
    pub png_bytes: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Seed used for this image (if reported by the service).
    pub seed: Option<u64>,
}

/// The result of a generation request.
#[derive(Clone, Debug)]
pub struct AiGenerateResult {
    pub images: Vec<AiGeneratedImage>,
    /// The actual prompt used (may be modified by the service).
    pub effective_prompt: String,
}

/// Errors from AI service calls.
#[derive(Debug, thiserror::Error)]
pub enum AiError {
    #[error("no API key configured for {service}; set {env_var} or add a key in Preferences › Integrations")]
    NoApiKey { service: &'static str, env_var: &'static str },
    #[error("{service} API error {status}: {message}")]
    ApiError { service: &'static str, status: u16, message: String },
    #[error("network error: {0}")]
    Network(String),
    #[error("{service} returned an unrecognised response: {message}")]
    Parse { service: &'static str, message: String },
    #[error("image decode failed: {0}")]
    ImageDecode(String),
    #[error("{service}: {message}")]
    Other { service: &'static str, message: String },
}

/// Common interface for all AI image-generation services.
pub trait AiService: Send + Sync {
    fn kind(&self) -> AiServiceKind;

    /// Generate images from a text (or img2img) request.
    /// This is a blocking call; callers run it on a thread pool.
    fn generate(&self, req: &AiGenerateRequest) -> Result<AiGenerateResult, AiError>;

    /// List available models / styles. Returns `(id, display_name)` pairs.
    /// May return an empty list if the service does not expose a model list.
    fn list_models(&self) -> Result<Vec<(String, String)>, AiError>;
}
