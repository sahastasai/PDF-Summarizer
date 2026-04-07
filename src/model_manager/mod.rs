//! Model Manager Module
//!
//! Handles automatic downloading, caching, and validation of LLaMA 3 models.

pub mod downloader;
pub mod cache;
pub mod validator;

pub use downloader::ModelDownloader;
pub use cache::ModelCache;
pub use validator::ModelValidator;

use anyhow::{Context, Result};
use std::path::PathBuf;
use tracing::{info, warn};

/// Default model configurations
/// Gated model (requires HF token)
pub const GATED_MODEL_ID: &str = "meta-llama/Meta-Llama-3-8B-Instruct";
/// Open model (no token required) - TinyLlama is a smaller, open model for testing
/// For production, consider using a larger open model like "microsoft/phi-2" or "mistralai/Mistral-7B-v0.1"
pub const OPEN_MODEL_ID: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";

pub const DEFAULT_CACHE_DIR_NAME: &str = ".pdf_summarizer";
pub const MODEL_SUBDIR: &str = "models";

/// Get the default cache directory
pub fn get_default_cache_dir() -> Result<PathBuf> {
    let home = dirs::home_dir()
        .or_else(|| dirs::data_local_dir())
        .context("Could not determine home directory")?;
    
    Ok(home.join(DEFAULT_CACHE_DIR_NAME).join(MODEL_SUBDIR))
}

/// Get the default model path for a specific model
pub fn get_model_path(model_id: &str) -> Result<PathBuf> {
    let cache_dir = get_default_cache_dir()?;
    // Sanitize model name for filesystem
    let safe_name = model_id
        .replace('/', "-")
        .replace('\\', "-")
        .replace(':', "-");
    Ok(cache_dir.join(safe_name))
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model ID (HuggingFace format)
    pub model_id: String,
    /// Local path to the model
    pub local_path: PathBuf,
    /// Whether the model is fully downloaded
    pub is_complete: bool,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Required files present
    pub files_present: Vec<String>,
    /// Missing files
    pub files_missing: Vec<String>,
}

/// Ensure a model is available, downloading if necessary
pub async fn ensure_model_available(
    model_path: Option<PathBuf>,
    hf_token: Option<String>,
) -> Result<PathBuf> {
    // If a path is provided, validate it exists
    if let Some(path) = model_path {
        if path.exists() {
            info!("Using provided model path: {:?}", path);
            let validator = ModelValidator::new();
            validator.validate(&path)?;
            return Ok(path);
        } else {
            anyhow::bail!("Provided model path does not exist: {:?}", path);
        }
    }
    
    // Determine which model to use based on HF token availability
    let (model_id, model_name) = if hf_token.is_some() {
        (GATED_MODEL_ID, "LLaMA 3 8B Instruct")
    } else {
        info!("No HuggingFace token provided, using open model (TinyLlama-1.1B)");
        info!("For better quality, provide --hf-token to use LLaMA 3 8B Instruct");
        (OPEN_MODEL_ID, "TinyLlama 1.1B Chat")
    };
    
    // Check if model already exists in cache
    let default_path = get_model_path(model_id)?;
    
    if default_path.exists() {
        info!("Found existing model at: {:?}", default_path);
        let validator = ModelValidator::new();
        
        match validator.validate(&default_path) {
            Ok(_) => {
                info!("Model validation successful");
                return Ok(default_path);
            }
            Err(e) => {
                warn!("Existing model failed validation: {}", e);
                warn!("Will attempt to re-download...");
            }
        }
    }
    
    // Need to download the model
    info!("Model not found locally. Will download {}...", model_name);
    
    let cache = ModelCache::new(get_default_cache_dir()?)?;
    let downloader = ModelDownloader::new(hf_token);
    
    let model_path = downloader
        .download_model(model_id, &cache)
        .await?;
    
    // Validate the downloaded model
    let validator = ModelValidator::new();
    validator.validate(&model_path)?;
    
    info!("Model ready at: {:?}", model_path);
    Ok(model_path)
}

/// Synchronous version of ensure_model_available
pub fn ensure_model_available_sync(
    model_path: Option<PathBuf>,
    hf_token: Option<String>,
) -> Result<PathBuf> {
    tokio::runtime::Runtime::new()
        .context("Failed to create async runtime")?
        .block_on(ensure_model_available(model_path, hf_token))
}
