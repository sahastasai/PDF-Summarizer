//! Model Cache Management
//!
//! Manages the local cache directory for downloaded models.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// Model cache manager
pub struct ModelCache {
    /// Root cache directory
    cache_dir: PathBuf,
}

impl ModelCache {
    /// Create a new model cache manager
    pub fn new(cache_dir: PathBuf) -> Result<Self> {
        // Ensure cache directory exists
        if !cache_dir.exists() {
            info!("Creating cache directory: {:?}", cache_dir);
            fs::create_dir_all(&cache_dir)
                .with_context(|| format!("Failed to create cache directory: {:?}", cache_dir))?;
        }

        Ok(Self { cache_dir })
    }

    /// Get the cache directory path
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Get the path for a specific model
    pub fn model_path(&self, model_name: &str) -> PathBuf {
        // Sanitize model name for filesystem
        let safe_name = model_name
            .replace('/', "-")
            .replace('\\', "-")
            .replace(':', "-");
        self.cache_dir.join(safe_name)
    }

    /// Check if a model exists in cache
    pub fn has_model(&self, model_name: &str) -> bool {
        let path = self.model_path(model_name);
        path.exists() && path.is_dir()
    }

    /// Get model directory, creating if necessary
    pub fn get_or_create_model_dir(&self, model_name: &str) -> Result<PathBuf> {
        let path = self.model_path(model_name);

        if !path.exists() {
            debug!("Creating model directory: {:?}", path);
            fs::create_dir_all(&path)
                .with_context(|| format!("Failed to create model directory: {:?}", path))?;
        }

        Ok(path)
    }

    /// List all cached models
    pub fn list_models(&self) -> Result<Vec<String>> {
        let mut models = Vec::new();

        if self.cache_dir.exists() {
            for entry in fs::read_dir(&self.cache_dir)? {
                let entry = entry?;
                if entry.path().is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        models.push(name.to_string());
                    }
                }
            }
        }

        Ok(models)
    }

    /// Get total cache size in bytes
    pub fn total_size(&self) -> Result<u64> {
        Self::dir_size(&self.cache_dir)
    }

    /// Calculate directory size recursively
    fn dir_size(path: &Path) -> Result<u64> {
        let mut size = 0;

        if path.is_dir() {
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_dir() {
                    size += Self::dir_size(&path)?;
                } else {
                    size += entry.metadata()?.len();
                }
            }
        }

        Ok(size)
    }

    /// Remove a model from cache
    pub fn remove_model(&self, model_name: &str) -> Result<()> {
        let path = self.model_path(model_name);

        if path.exists() {
            info!("Removing model from cache: {:?}", path);
            fs::remove_dir_all(&path)
                .with_context(|| format!("Failed to remove model: {:?}", path))?;
        }

        Ok(())
    }

    /// Clear entire cache
    pub fn clear(&self) -> Result<()> {
        if self.cache_dir.exists() {
            info!("Clearing cache directory: {:?}", self.cache_dir);
            fs::remove_dir_all(&self.cache_dir)?;
            fs::create_dir_all(&self.cache_dir)?;
        }

        Ok(())
    }

    /// Get a temporary download path for a file
    pub fn temp_download_path(&self, model_name: &str, filename: &str) -> PathBuf {
        self.model_path(model_name)
            .join(format!("{}.download", filename))
    }

    /// Finalize a download by renaming temp file
    pub fn finalize_download(&self, temp_path: &Path, final_path: &Path) -> Result<()> {
        fs::rename(temp_path, final_path).with_context(|| {
            format!(
                "Failed to finalize download: {:?} -> {:?}",
                temp_path, final_path
            )
        })
    }
}

/// Format bytes as human-readable string
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_model_path_sanitization() {
        let temp = tempdir().unwrap();
        let cache = ModelCache::new(temp.path().to_path_buf()).unwrap();

        let path = cache.model_path("meta-llama/Llama-3-8B");
        assert!(!path.to_string_lossy().contains('/'));
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(500), "500 bytes");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1536), "1.50 KB");
    }
}
