//! LLaMA 3 Model Configuration
//!
//! Defines the hyperparameters for different LLaMA 3 model sizes.

use serde::{Deserialize, Serialize};

/// LLaMA 3 model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaConfig {
    /// Vocabulary size
    pub vocab_size: usize,

    /// Hidden dimension
    pub hidden_size: usize,

    /// Intermediate size (FFN)
    pub intermediate_size: usize,

    /// Number of attention heads
    pub num_attention_heads: usize,

    /// Number of key-value heads (for GQA)
    pub num_key_value_heads: usize,

    /// Number of transformer layers
    pub num_hidden_layers: usize,

    /// RMS normalization epsilon
    pub rms_norm_eps: f64,

    /// Maximum sequence length
    pub max_position_embeddings: usize,

    /// RoPE theta base
    pub rope_theta: f64,

    /// Beginning of sequence token ID
    pub bos_token_id: u32,

    /// End of sequence token ID
    pub eos_token_id: u32,

    /// Padding token ID
    pub pad_token_id: u32,

    /// Tie word embeddings
    pub tie_word_embeddings: bool,

    /// Head dimension (computed)
    pub head_dim: usize,
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self::llama3_8b()
    }
}

impl LlamaConfig {
    /// LLaMA 3 8B configuration
    pub fn llama3_8b() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            num_hidden_layers: 32,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 8192,
            rope_theta: 500000.0,
            bos_token_id: 128000,
            eos_token_id: 128001,
            pad_token_id: 128002,
            tie_word_embeddings: false,
            head_dim: 128,
        }
    }

    /// LLaMA 3 70B configuration  
    pub fn llama3_70b() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 8192,
            intermediate_size: 28672,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            num_hidden_layers: 80,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 8192,
            rope_theta: 500000.0,
            bos_token_id: 128000,
            eos_token_id: 128001,
            pad_token_id: 128002,
            tie_word_embeddings: false,
            head_dim: 128,
        }
    }

    /// Load config from a JSON file (HuggingFace format)
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: serde_json::Value = serde_json::from_str(&content)?;

        Ok(Self {
            vocab_size: config["vocab_size"].as_u64().unwrap_or(128256) as usize,
            hidden_size: config["hidden_size"].as_u64().unwrap_or(4096) as usize,
            intermediate_size: config["intermediate_size"].as_u64().unwrap_or(14336) as usize,
            num_attention_heads: config["num_attention_heads"].as_u64().unwrap_or(32) as usize,
            num_key_value_heads: config["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
            num_hidden_layers: config["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
            rms_norm_eps: config["rms_norm_eps"].as_f64().unwrap_or(1e-5),
            max_position_embeddings: config["max_position_embeddings"].as_u64().unwrap_or(8192)
                as usize,
            rope_theta: config["rope_theta"].as_f64().unwrap_or(500000.0),
            bos_token_id: config["bos_token_id"].as_u64().unwrap_or(128000) as u32,
            eos_token_id: config["eos_token_id"].as_u64().unwrap_or(128001) as u32,
            pad_token_id: config["pad_token_id"].as_u64().unwrap_or(128002) as u32,
            tie_word_embeddings: config["tie_word_embeddings"].as_bool().unwrap_or(false),
            head_dim: config["head_dim"].as_u64().unwrap_or(128) as usize,
        })
    }

    /// Compute the head dimension
    pub fn compute_head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the number of KV heads per attention head (for GQA)
    pub fn kv_heads_ratio(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }
}
