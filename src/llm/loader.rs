//! Model loading utilities
//!
//! Handles loading LLaMA 3 weights from SafeTensors format.

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use tracing::{info, warn};

use super::config::LlamaConfig;

/// Model loader for LLaMA weights
pub struct ModelLoader {
    /// Path to model directory
    model_path: PathBuf,
    /// Model configuration
    config: LlamaConfig,
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

        Ok(Self { model_path, config })
    }

    /// Get model configuration
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// Find all safetensor files in the model directory
    pub fn find_weight_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        for entry in std::fs::read_dir(&self.model_path)? {
            let entry = entry?;
            let path = entry.path();
            if path
                .extension()
                .map(|e| e == "safetensors")
                .unwrap_or(false)
            {
                files.push(path);
            }
        }

        if files.is_empty() {
            anyhow::bail!("No .safetensors files found in {:?}", self.model_path);
        }

        files.sort();
        info!("Found {} weight files", files.len());
        Ok(files)
    }

    /// Load weight names from safetensor files
    pub fn load_weight_names(&self) -> Result<HashMap<String, PathBuf>> {
        let files = self.find_weight_files()?;
        let mut weight_map = HashMap::new();

        for file_path in files {
            let mut file = File::open(&file_path)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;

            let tensors =
                SafeTensors::deserialize(&buffer).context("Failed to deserialize safetensors")?;

            for name in tensors.names() {
                weight_map.insert(name.to_string(), file_path.clone());
            }
        }

        info!("Found {} weights", weight_map.len());
        Ok(weight_map)
    }

    /// Load a specific tensor by name
    pub fn load_tensor<B: burn::prelude::Backend>(
        &self,
        name: &str,
        weight_files: &HashMap<String, PathBuf>,
        device: &B::Device,
    ) -> Result<burn::tensor::Tensor<B, 2>> {
        let file_path = weight_files
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Weight not found: {}", name))?;

        let mut file = File::open(file_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let tensors = SafeTensors::deserialize(&buffer)?;
        let tensor_view = tensors.tensor(name)?;

        let shape = tensor_view.shape();
        let data = tensor_view.data();

        // Convert based on dtype
        let tensor = match tensor_view.dtype() {
            safetensors::Dtype::F32 => {
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                burn::tensor::Tensor::<B, 1>::from_floats(floats.as_slice(), device)
            }
            safetensors::Dtype::F16 => {
                let floats: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::f16::from_bits(bits).to_f32()
                    })
                    .collect();
                burn::tensor::Tensor::<B, 1>::from_floats(floats.as_slice(), device)
            }
            safetensors::Dtype::BF16 => {
                let floats: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        half::bf16::from_bits(bits).to_f32()
                    })
                    .collect();
                burn::tensor::Tensor::<B, 1>::from_floats(floats.as_slice(), device)
            }
            dtype => anyhow::bail!("Unsupported dtype: {:?}", dtype),
        };

        // Reshape to 2D
        let shape_2d = if shape.len() == 1 {
            [shape[0], 1]
        } else if shape.len() == 2 {
            [shape[0], shape[1]]
        } else {
            anyhow::bail!("Unsupported tensor rank: {}", shape.len());
        };

        Ok(tensor.reshape(shape_2d))
    }

    /// Check if model files exist
    pub fn validate(&self) -> Result<()> {
        if !self.model_path.exists() {
            anyhow::bail!("Model path does not exist: {:?}", self.model_path);
        }

        let weight_files = self.find_weight_files()?;
        if weight_files.is_empty() {
            anyhow::bail!("No weight files found in {:?}", self.model_path);
        }

        Ok(())
    }

    /// Get the tokenizer path
    pub fn tokenizer_path(&self) -> PathBuf {
        self.model_path.join("tokenizer.json")
    }
}

/// Weight mapping from HuggingFace naming to our model
pub fn get_weight_mapping(layer_idx: usize) -> HashMap<String, String> {
    let mut mapping = HashMap::new();

    let prefix = format!("model.layers.{}", layer_idx);

    // Attention weights
    mapping.insert(
        format!("{}.self_attn.q_proj.weight", prefix),
        format!("layers.{}.self_attn.q_proj.weight", layer_idx),
    );
    mapping.insert(
        format!("{}.self_attn.k_proj.weight", prefix),
        format!("layers.{}.self_attn.k_proj.weight", layer_idx),
    );
    mapping.insert(
        format!("{}.self_attn.v_proj.weight", prefix),
        format!("layers.{}.self_attn.v_proj.weight", layer_idx),
    );
    mapping.insert(
        format!("{}.self_attn.o_proj.weight", prefix),
        format!("layers.{}.self_attn.o_proj.weight", layer_idx),
    );

    // MLP weights
    mapping.insert(
        format!("{}.mlp.gate_proj.weight", prefix),
        format!("layers.{}.mlp.gate_proj.weight", layer_idx),
    );
    mapping.insert(
        format!("{}.mlp.up_proj.weight", prefix),
        format!("layers.{}.mlp.up_proj.weight", layer_idx),
    );
    mapping.insert(
        format!("{}.mlp.down_proj.weight", prefix),
        format!("layers.{}.mlp.down_proj.weight", layer_idx),
    );

    // Layer norms
    mapping.insert(
        format!("{}.input_layernorm.weight", prefix),
        format!("layers.{}.input_layernorm.weight", layer_idx),
    );
    mapping.insert(
        format!("{}.post_attention_layernorm.weight", prefix),
        format!("layers.{}.post_attention_layernorm.weight", layer_idx),
    );

    mapping
}
