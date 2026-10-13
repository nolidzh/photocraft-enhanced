//! API key resolution: environment variable OR encrypted preferences store.
//!
//! Resolution order (first wins):
//!   1. Environment variable (if `use_env_var` is true for this service in prefs).
//!   2. Stored key from preferences (if one is set).
//!
//! The stored key is XOR-obfuscated with a per-machine salt derived from the
//! hostname + username so it is not stored as plaintext in the preferences JSON.
//! This is obfuscation, not cryptographic encryption — anyone with file access
//! can reverse it.  For stronger protection users should use env vars.

use serde::{Deserialize, Serialize};

use crate::types::AiServiceKind;

/// Where an API key came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CredentialSource {
    /// Resolved from the environment variable for this service.
    EnvironmentVariable,
    /// Resolved from the encrypted preferences store.
    PreferencesStore,
}

/// Preferences for one AI service's credentials.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ServiceCredPrefs {
    /// Whether to check the environment variable before the stored key.
    /// Default true so power users and CI don't need to touch preferences.
    pub use_env_var: bool,
    /// Obfuscated API key (base64-encoded XOR blob).  Empty = not set.
    pub stored_key: String,
}

impl ServiceCredPrefs {
    pub fn new_default() -> Self {
        Self { use_env_var: true, stored_key: String::new() }
    }
}

/// Preferences for all AI services' credentials.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AiCredentials {
    pub openart: ServiceCredPrefs,
    pub higgsfield: ServiceCredPrefs,
}

impl AiCredentials {
    pub fn prefs_for(&self, kind: AiServiceKind) -> &ServiceCredPrefs {
        match kind {
            AiServiceKind::OpenArt => &self.openart,
            AiServiceKind::Higgsfield => &self.higgsfield,
        }
    }

    pub fn prefs_for_mut(&mut self, kind: AiServiceKind) -> &mut ServiceCredPrefs {
        match kind {
            AiServiceKind::OpenArt => &mut self.openart,
            AiServiceKind::Higgsfield => &mut self.higgsfield,
        }
    }
}

/// Resolve the API key for a service.
///
/// Returns `(key, source)` or `None` if no key is available.
pub fn resolve_api_key(
    kind: AiServiceKind,
    prefs: &ServiceCredPrefs,
) -> Option<(String, CredentialSource)> {
    // 1. Environment variable (if enabled)
    if prefs.use_env_var {
        if let Ok(val) = std::env::var(kind.env_var()) {
            let val = val.trim().to_string();
            if !val.is_empty() {
                return Some((val, CredentialSource::EnvironmentVariable));
            }
        }
    }
    // 2. Stored (obfuscated) key
    let decoded = decode_stored_key(&prefs.stored_key)?;
    if decoded.is_empty() {
        return None;
    }
    Some((decoded, CredentialSource::PreferencesStore))
}

// ── obfuscation ───────────────────────────────────────────────────────────────
// XOR with a simple per-machine salt so the key is not stored verbatim in JSON.
// This is NOT strong encryption; use env vars if you need real security.

fn machine_salt() -> Vec<u8> {
    // Build a short salt from hostname + username, repeated to fill 32 bytes.
    let host = hostname_str();
    let user = username_str();
    let raw = format!("{host}:{user}");
    let bytes = raw.as_bytes();
    if bytes.is_empty() {
        return vec![0x5A; 32];
    }
    let mut salt = Vec::with_capacity(32);
    for i in 0..32usize {
        let b = bytes.get(i % bytes.len()).copied().unwrap_or(0x5A);
        salt.push(b.wrapping_add(i as u8));
    }
    salt
}

#[cfg(not(target_arch = "wasm32"))]
fn hostname_str() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "localhost".into())
}
#[cfg(target_arch = "wasm32")]
fn hostname_str() -> String { "web".into() }

#[cfg(not(target_arch = "wasm32"))]
fn username_str() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "user".into())
}
#[cfg(target_arch = "wasm32")]
fn username_str() -> String { "user".into() }

/// Obfuscate and base64-encode an API key for storage in preferences.
pub fn encode_stored_key(plain: &str) -> String {
    let salt = machine_salt();
    let xored: Vec<u8> = plain.as_bytes().iter().enumerate()
        .map(|(i, &b)| b ^ salt[i % salt.len()])
        .collect();
    use_base64_encode(&xored)
}

/// Decode a stored key. Returns `None` if the stored value is empty or malformed.
pub fn decode_stored_key(stored: &str) -> Option<String> {
    if stored.is_empty() {
        return None;
    }
    let xored = use_base64_decode(stored)?;
    let salt = machine_salt();
    let plain: Vec<u8> = xored.iter().enumerate()
        .map(|(i, &b)| b ^ salt[i % salt.len()])
        .collect();
    String::from_utf8(plain).ok().filter(|s| !s.is_empty())
}

// ── minimal base64 (no external dep needed; the key is short) ─────────────────

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn use_base64_encode(bytes: &[u8]) -> String {
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

fn b64_val(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        b'=' => Some(0),
        _ => None,
    }
}

fn use_base64_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.trim();
    if s.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    for chunk in s.as_bytes().chunks(4) {
        let c0 = b64_val(chunk[0])?;
        let c1 = b64_val(chunk[1])?;
        let c2 = b64_val(*chunk.get(2).unwrap_or(&b'='))?;
        let c3 = b64_val(*chunk.get(3).unwrap_or(&b'='))?;
        let n = ((c0 as u32) << 18) | ((c1 as u32) << 12) | ((c2 as u32) << 6) | (c3 as u32);
        out.push((n >> 16) as u8);
        if chunk.get(2) != Some(&b'=') { out.push((n >> 8) as u8); }
        if chunk.get(3) != Some(&b'=') { out.push(n as u8); }
    }
    Some(out)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let key = "sk-test-1234567890abcdef";
        let stored = encode_stored_key(key);
        assert_ne!(stored, key, "key must be obfuscated");
        let decoded = decode_stored_key(&stored);
        assert_eq!(decoded.as_deref(), Some(key));
    }

    #[test]
    fn decode_empty_returns_none() {
        assert!(decode_stored_key("").is_none());
    }

    #[test]
    fn base64_roundtrip() {
        let input = b"hello world!";
        let encoded = use_base64_encode(input);
        let decoded = use_base64_decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn resolve_stored_when_env_disabled() {
        // With use_env_var=false the env var is never consulted; the stored key wins.
        let prefs = ServiceCredPrefs { use_env_var: false, stored_key: encode_stored_key("stored-key") };
        let result = resolve_api_key(AiServiceKind::OpenArt, &prefs);
        assert_eq!(result, Some(("stored-key".into(), CredentialSource::PreferencesStore)));
    }

    #[test]
    fn resolve_nothing_when_empty() {
        // No env var set (or at least not "PHOTOCRAFT_TEST_OPENART" which we won't set),
        // no stored key → None.
        let prefs = ServiceCredPrefs { use_env_var: false, stored_key: String::new() };
        let result = resolve_api_key(AiServiceKind::OpenArt, &prefs);
        assert_eq!(result, None);
    }
}
