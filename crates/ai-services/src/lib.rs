//! AI image-generation service adapters for PhotoCraft Enhanced.
//!
//! Each service (OpenArt.ai, Higgsfield, …) implements the [`AiService`] trait.
//! The engine's `ai_cmds` module registers commands that call these adapters.
//!
//! # Credentials
//!
//! API keys can come from two sources, tried in this order:
//! 1. Environment variable (e.g. `OPENART_API_KEY`).
//! 2. The encrypted value stored in PhotoCraft's preferences (`integrations.aiServices.<name>.apiKey`).
//!
//! The user chooses their preferred source in Edit › Preferences › Integrations.
//! If the environment variable is set it always wins (useful for server/CI use), unless the
//! user explicitly disables env-var lookup for a service in preferences.

#![forbid(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unimplemented,
    clippy::todo,
    clippy::unreachable
)]

pub mod credentials;
pub mod openart;
pub mod types;

pub use credentials::{AiCredentials, CredentialSource};
pub use types::{AiError, AiGenerateRequest, AiGenerateResult, AiService, AiServiceKind};
