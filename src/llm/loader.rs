//! Model loading utilities
//!
//! Handles loading LLaMA weights from SafeTensors format into Burn tensors.

use anyhow::{Context, Result};
use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use tracing::{debug, info, warn};

use super::config::LlamaConfig;

/// Cached SafeTensor data for efficient weight loading
pub struct SafeTensorData {
    /// Raw bytes of all safetensor files
    buffers: Vec<Vec<u8>>,
    /// Mapping from weight name to buffer index
    name_to_buffer: HashMap<String, usize>,
}

impl SafeTensorData {
    /// Load all safetensor files from a directory
    pub fn load(model_path: &PathBuf) -> Result<Self> {
        let mut buffers = Vec::new();
        let mut name_to_buffer = HashMap::new();

        // Find all .safetensors files
        let mut files: Vec<PathBuf> = std::fs::read_dir(model_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
            .collect();
        files.sort();

        if files.is_empty() {
            anyhow::bail!("No .safetensors files found in {:?}", model_path);
        }

        info!("Loading {} safetensor files", files.len());

        for file_path in files {
            let mut file = File::open(&file_path)
                .with_context(|| format!("Failed to open {:?}", file_path))?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;

            let buffer_idx = buffers.len();

            // Parse to get tensor names
            let tensors = SafeTensors::deserialize(&buffer)
                .with_context(|| format!("Failed to parse {:?}", file_path))?;

            for name in tensors.names() {
                name_to_buffer.insert(name.to_string(), buffer_idx);
            }

            buffers.push(buffer);
            debug!("Loaded {:?}", file_path);
        }

        info!("Found {} weight tensors", name_to_buffer.len());
        Ok(Self {
            buffers,
            name_to_buffer,
        })
    }

    /// Get a 1D tensor by name
    pub fn get_tensor_1d<B: Backend>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, 1>> {
        let buffer_idx = self
            .name_to_buffer
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Weight not found: {}", name))?;

        let tensors = SafeTensors::deserialize(&self.buffers[*buffer_idx])?;
        let tensor_view = tensors.tensor(name)?;
        let shape = tensor_view.shape();
        let data = tensor_view.data();

        // Verify shape
        if shape.len() != 1 {
            anyhow::bail!("Expected 1D tensor for {}, got shape {:?}", name, shape);
        }

        let floats = self.convert_to_f32(data, tensor_view.dtype())?;
        let tensor_data = burn::tensor::TensorData::new(floats, vec![shape[0]]);
        Ok(Tensor::from_data(tensor_data, device))
    }

    /// Get a 2D tensor by name
    pub fn get_tensor_2d<B: Backend>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, 2>> {
        let buffer_idx = self
            .name_to_buffer
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Weight not found: {}", name))?;

        let tensors = SafeTensors::deserialize(&self.buffers[*buffer_idx])?;
        let tensor_view = tensors.tensor(name)?;
        let shape = tensor_view.shape();
        let data = tensor_view.data();

        // Verify shape
        if shape.len() != 2 {
            anyhow::bail!("Expected 2D tensor for {}, got shape {:?}", name, shape);
        }

        let floats = self.convert_to_f32(data, tensor_view.dtype())?;
        let tensor_data = burn::tensor::TensorData::new(floats, vec![shape[0], shape[1]]);
        Ok(Tensor::from_data(tensor_data, device))
    }

    /// Check if a weight exists
    pub fn has_weight(&self, name: &str) -> bool {
        self.name_to_buffer.contains_key(name)
    }

    /// Convert raw bytes to f32 based on dtype
    fn convert_to_f32(&self, data: &[u8], dtype: safetensors::Dtype) -> Result<Vec<f32>> {
        match dtype {
            safetensors::Dtype::F32 => Ok(data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            safetensors::Dtype::F16 => Ok(data
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect()),
            safetensors::Dtype::BF16 => Ok(data
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    half::bf16::from_bits(bits).to_f32()
                })
                .collect()),
            dtype => anyhow::bail!("Unsupported dtype: {:?}", dtype),
        }
    }
}

/// Load a Linear layer with weights from SafeTensors
pub fn load_linear<B: Backend>(
    data: &SafeTensorData,
    weight_name: &str,
    in_features: usize,
    out_features: usize,
    device: &B::Device,
) -> Result<Linear<B>> {
    debug!(
        "Loading linear: {} [{} x {}]",
        weight_name, out_features, in_features
    );

    // HuggingFace stores weights as [out_features, in_features]
    let weight = data.get_tensor_2d::<B>(weight_name, device)?;

    // Verify shape
    let [rows, cols] = weight.dims();
    if rows != out_features || cols != in_features {
        anyhow::bail!(
            "Shape mismatch for {}: expected [{}, {}], got [{}, {}]",
            weight_name,
            out_features,
            in_features,
            rows,
            cols
        );
    }

    // Burn Linear layers expect weights as [in_features, out_features]
    // but HuggingFace stores them as [out_features, in_features], so we need to transpose
    let weight_transposed = weight.swap_dims(0, 1);

    // Create Linear with no bias
    let mut linear = LinearConfig::new(in_features, out_features)
        .with_bias(false)
        .init(device);

    // Replace the weight
    linear.weight = Param::from_tensor(weight_transposed);

    Ok(linear)
}

/// Load an Embedding layer with weights from SafeTensors
pub fn load_embedding<B: Backend>(
    data: &SafeTensorData,
    weight_name: &str,
    vocab_size: usize,
    embedding_dim: usize,
    device: &B::Device,
) -> Result<Embedding<B>> {
    debug!(
        "Loading embedding: {} [{} x {}]",
        weight_name, vocab_size, embedding_dim
    );

    let weight = data.get_tensor_2d::<B>(weight_name, device)?;

    // Verify shape
    let [rows, cols] = weight.dims();
    if rows != vocab_size || cols != embedding_dim {
        anyhow::bail!(
            "Shape mismatch for {}: expected [{}, {}], got [{}, {}]",
            weight_name,
            vocab_size,
            embedding_dim,
            rows,
            cols
        );
    }

    let mut embedding = EmbeddingConfig::new(vocab_size, embedding_dim).init(device);
    embedding.weight = Param::from_tensor(weight);

    Ok(embedding)
}

/// Load RMSNorm weight from SafeTensors (returns just the weight tensor)
pub fn load_rms_norm_weight<B: Backend>(
    data: &SafeTensorData,
    weight_name: &str,
    hidden_size: usize,
    device: &B::Device,
) -> Result<Tensor<B, 1>> {
    debug!("Loading rms_norm: {} [{}]", weight_name, hidden_size);

    let weight = data.get_tensor_1d::<B>(weight_name, device)?;

    // Verify shape
    let [size] = weight.dims();
    if size != hidden_size {
        anyhow::bail!(
            "Shape mismatch for {}: expected [{}], got [{}]",
            weight_name,
            hidden_size,
            size
        );
    }

    Ok(weight)
}

/// Model loader that orchestrates loading all weights
pub struct ModelLoader {
    /// Path to model directory
    pub model_path: PathBuf,
    /// Model configuration
    pub config: LlamaConfig,
    /// Cached weight data
    pub weights: Option<SafeTensorData>,
}

impl ModelLoader {
    /// Create a new model loader
    pub fn new(model_path: PathBuf) -> Result<Self> {
        let config_path = model_path.join("config.json");
        let config = if config_path.exists() {
            LlamaConfig::from_file(&config_path).context("Failed to load model config")?
        } else {
            warn!("No config.json found, using default LLaMA 3 8B config");
            LlamaConfig::llama3_8b()
        };

        Ok(Self {
            model_path,
            config,
            weights: None,
        })
    }

    /// Get model configuration
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// Load weights into memory
    pub fn load_weights(&mut self) -> Result<()> {
        if self.weights.is_none() {
            info!("Loading model weights from {:?}", self.model_path);
            self.weights = Some(SafeTensorData::load(&self.model_path)?);
        }
        Ok(())
    }

    /// Get reference to loaded weights
    pub fn weights(&self) -> Result<&SafeTensorData> {
        self.weights
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Weights not loaded. Call load_weights() first."))
    }

    /// Check if model files exist
    pub fn validate(&self) -> Result<()> {
        if !self.model_path.exists() {
            anyhow::bail!("Model path does not exist: {:?}", self.model_path);
        }

        // Check for safetensor files
        let has_safetensors = std::fs::read_dir(&self.model_path)?
            .filter_map(|e| e.ok())
            .any(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            });

        if !has_safetensors {
            anyhow::bail!("No .safetensors files found in {:?}", self.model_path);
        }

        Ok(())
    }

    /// Get the tokenizer path
    pub fn tokenizer_path(&self) -> PathBuf {
        self.model_path.join("tokenizer.json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_f32() {
        let data = SafeTensorData {
            buffers: vec![],
            name_to_buffer: HashMap::new(),
        };

        // Test f32 conversion
        let bytes: Vec<u8> = vec![0, 0, 128, 63]; // 1.0f32 in little endian
        let result = data
            .convert_to_f32(&bytes, safetensors::Dtype::F32)
            .unwrap();
        assert_eq!(result, vec![1.0f32]);
    }
}
