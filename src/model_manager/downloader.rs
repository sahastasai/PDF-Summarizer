//! Model Downloader
//!
//! Downloads LLaMA 3 models from HuggingFace Hub.

use anyhow::{Context, Result};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, USER_AGENT};
use serde::Deserialize;
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, warn};

use super::cache::ModelCache;

/// HuggingFace API base URL
const HF_API_URL: &str = "https://huggingface.co/api/models";
const HF_CDN_URL: &str = "https://huggingface.co";

/// Required files for a complete LLaMA model
const REQUIRED_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
];

/// SafeTensor file patterns (for reference)
#[allow(dead_code)]
const SAFETENSOR_PATTERNS: &[&str] = &[
    "model.safetensors",
    "model-00001-of-",
];

/// Model file info from HuggingFace API
#[derive(Debug, Deserialize)]
struct HfFileInfo {
    #[serde(rename = "rfilename")]
    filename: String,
    size: Option<u64>,
}

/// Model info from HuggingFace API
#[derive(Debug, Deserialize)]
struct HfModelInfo {
    siblings: Vec<HfFileInfo>,
}

/// Model downloader
pub struct ModelDownloader {
    /// HuggingFace API token (optional, but needed for gated models)
    hf_token: Option<String>,
    /// HTTP client
    client: reqwest::Client,
}

impl ModelDownloader {
    /// Create a new model downloader
    pub fn new(hf_token: Option<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3600)) // 1 hour timeout for large files
            .build()
            .expect("Failed to create HTTP client");
        
        Self { hf_token, client }
    }
    
    /// Create authorization headers
    fn create_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_static("pdf-summarizer/0.1.0"));
        
        if let Some(ref token) = self.hf_token {
            if let Ok(auth_value) = HeaderValue::from_str(&format!("Bearer {}", token)) {
                headers.insert(AUTHORIZATION, auth_value);
            }
        }
        
        headers
    }
    
    /// Get model info from HuggingFace
    async fn get_model_info(&self, model_id: &str) -> Result<HfModelInfo> {
        let url = format!("{}/{}", HF_API_URL, model_id);
        
        debug!("Fetching model info from: {}", url);
        
        let response = self.client
            .get(&url)
            .headers(self.create_headers())
            .send()
            .await
            .context("Failed to fetch model info")?;
        
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            
            if status.as_u16() == 401 || status.as_u16() == 403 {
                anyhow::bail!(
                    "Access denied to model '{}'. This is likely a gated model.\n\
                     Please:\n\
                     1. Visit https://huggingface.co/{} and accept the license\n\
                     2. Create an access token at https://huggingface.co/settings/tokens\n\
                     3. Set HF_TOKEN environment variable or use --hf-token flag",
                    model_id, model_id
                );
            }
            
            anyhow::bail!("Failed to fetch model info: {} - {}", status, text);
        }
        
        response.json::<HfModelInfo>().await.context("Failed to parse model info")
    }
    
    /// Get list of files to download
    fn get_files_to_download(&self, model_info: &HfModelInfo) -> Vec<(String, u64)> {
        let mut files = Vec::new();
        
        for file in &model_info.siblings {
            let filename = &file.filename;
            let size = file.size.unwrap_or(0);
            
            // Include required config files
            if REQUIRED_FILES.iter().any(|&req| filename == req) {
                files.push((filename.clone(), size));
                continue;
            }
            
            // Include safetensor files
            if filename.ends_with(".safetensors") {
                files.push((filename.clone(), size));
                continue;
            }
            
            // Include special tokens and generation config
            if filename == "special_tokens_map.json" 
                || filename == "generation_config.json"
                || filename == "tokenizer.model"
            {
                files.push((filename.clone(), size));
            }
        }
        
        files
    }
    
    /// Download a single file
    async fn download_file(
        &self,
        model_id: &str,
        filename: &str,
        dest_path: &PathBuf,
        progress: &ProgressBar,
    ) -> Result<()> {
        let url = format!("{}/{}/resolve/main/{}", HF_CDN_URL, model_id, filename);
        
        debug!("Downloading: {} -> {:?}", url, dest_path);
        
        let response = self.client
            .get(&url)
            .headers(self.create_headers())
            .send()
            .await
            .context("Failed to start download")?;
        
        if !response.status().is_success() {
            anyhow::bail!("Failed to download {}: {}", filename, response.status());
        }
        
        let total_size = response.content_length().unwrap_or(0);
        progress.set_length(total_size);
        progress.set_message(filename.to_string());
        
        // Create parent directory if needed
        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Download with progress
        let temp_path = dest_path.with_extension("download");
        let mut file = tokio::fs::File::create(&temp_path).await?;
        let mut stream = response.bytes_stream();
        
        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Failed to read chunk")?;
            file.write_all(&chunk).await?;
            progress.inc(chunk.len() as u64);
        }
        
        file.flush().await?;
        drop(file);
        
        // Rename to final path
        tokio::fs::rename(&temp_path, dest_path).await?;
        
        progress.finish_with_message(format!("{} - Complete", filename));
        
        Ok(())
    }
    
    /// Download a model to the cache
    pub async fn download_model(&self, model_id: &str, cache: &ModelCache) -> Result<PathBuf> {
        info!("Preparing to download model: {}", model_id);
        
        // Check for HF token for gated models
        if self.hf_token.is_none() {
            warn!(
                "No HuggingFace token provided. If '{}' is a gated model, download will fail.\n\
                 Set HF_TOKEN environment variable or use --hf-token flag.",
                model_id
            );
        }
        
        // Get model info
        println!("Fetching model information...");
        let model_info = self.get_model_info(model_id).await?;
        
        // Determine files to download
        let files = self.get_files_to_download(&model_info);
        
        if files.is_empty() {
            anyhow::bail!("No downloadable files found for model: {}", model_id);
        }
        
        let total_size: u64 = files.iter().map(|(_, s)| *s).sum();
        info!(
            "Found {} files to download (total: {:.2} GB)",
            files.len(),
            total_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        
        // Create model directory
        let model_dir = cache.get_or_create_model_dir(model_id)?;
        
        // Setup progress bars
        let multi_progress = MultiProgress::new();
        let overall_style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] {msg}")
            .unwrap();
        let file_style = ProgressStyle::default_bar()
            .template("  {spinner:.cyan} [{bar:40.cyan/blue}] {bytes}/{total_bytes} {msg}")
            .unwrap()
            .progress_chars("#>-");
        
        let overall_progress = multi_progress.add(ProgressBar::new(files.len() as u64));
        overall_progress.set_style(overall_style);
        overall_progress.set_message(format!("Downloading {} files...", files.len()));
        
        // Download each file
        for (i, (filename, _size)) in files.iter().enumerate() {
            let dest_path = model_dir.join(filename);
            
            // Skip if file already exists with correct size
            if dest_path.exists() {
                if let Ok(metadata) = std::fs::metadata(&dest_path) {
                    if metadata.len() > 0 {
                        debug!("Skipping existing file: {}", filename);
                        overall_progress.inc(1);
                        continue;
                    }
                }
            }
            
            let file_progress = multi_progress.add(ProgressBar::new(0));
            file_progress.set_style(file_style.clone());
            
            overall_progress.set_message(format!(
                "Downloading file {}/{}: {}",
                i + 1,
                files.len(),
                filename
            ));
            
            match self.download_file(model_id, filename, &dest_path, &file_progress).await {
                Ok(_) => {
                    overall_progress.inc(1);
                }
                Err(e) => {
                    file_progress.abandon_with_message(format!("Failed: {}", filename));
                    // Clean up partial download
                    let _ = std::fs::remove_file(&dest_path);
                    let _ = std::fs::remove_file(dest_path.with_extension("download"));
                    return Err(e);
                }
            }
        }
        
        overall_progress.finish_with_message("Download complete!");
        
        info!("Model downloaded successfully to: {:?}", model_dir);
        Ok(model_dir)
    }
}

/// Check if a HuggingFace token is available
pub fn get_hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
}
