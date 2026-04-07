//! Core layer implementations for LLaMA 3
//!
//! Implements RMSNorm, SwiGLU, and other fundamental layers.

use anyhow::Result;
use burn::module::Param;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation;

use super::loader::{load_embedding, load_linear, load_rms_norm_weight, SafeTensorData};

/// RMS Normalization layer
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Weight parameter
    weight: Param<Tensor<B, 1>>,
    /// Epsilon for numerical stability
    #[module(skip)]
    eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    /// Create a new RMS normalization layer with random weights
    pub fn new(device: &B::Device, hidden_size: usize, eps: f64) -> Self {
        let weight = Tensor::ones([hidden_size], device);
        Self {
            weight: Param::from_tensor(weight),
            eps,
        }
    }

    /// Create RmsNorm from pre-loaded weight tensor
    pub fn from_tensor(weight: Tensor<B, 1>, eps: f64) -> Self {
        Self {
            weight: Param::from_tensor(weight),
            eps,
        }
    }

    /// Load from SafeTensor data
    pub fn from_safetensors(
        data: &SafeTensorData,
        weight_name: &str,
        hidden_size: usize,
        eps: f64,
        device: &B::Device,
    ) -> Result<Self> {
        let weight = load_rms_norm_weight::<B>(data, weight_name, hidden_size, device)?;
        Ok(Self::from_tensor(weight, eps))
    }

    /// Forward pass
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Compute RMS: sqrt(mean(x^2) + eps)
        // mean_dim keeps the dimension with size 1, so [batch, seq, hidden] -> [batch, seq, 1]
        let variance = x.clone().powf_scalar(2.0).mean_dim(2);
        let rms = (variance + self.eps).sqrt();

        // Normalize: x / rms (broadcasting happens automatically)
        let x_norm = x / rms;

        // Scale by weight: broadcast weight [hidden] to [1, 1, hidden]
        let weight = self
            .weight
            .val()
            .unsqueeze_dim::<2>(0)
            .unsqueeze_dim::<3>(0);
        x_norm * weight
    }
}

/// Configuration for RmsNorm
#[derive(Config, Debug)]
pub struct RmsNormConfig {
    pub hidden_size: usize,
    #[config(default = "1e-5")]
    pub eps: f64,
}

impl RmsNormConfig {
    /// Initialize the RMS normalization layer with random weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> RmsNorm<B> {
        RmsNorm::new(device, self.hidden_size, self.eps)
    }
}

/// SwiGLU activation (used in LLaMA FFN)
#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    /// Gate projection
    gate_proj: Linear<B>,
    /// Up projection
    up_proj: Linear<B>,
    /// Down projection
    down_proj: Linear<B>,
}

impl<B: Backend> MLP<B> {
    /// Create a new MLP layer with random weights
    pub fn new(device: &B::Device, hidden_size: usize, intermediate_size: usize) -> Self {
        let gate_proj = LinearConfig::new(hidden_size, intermediate_size)
            .with_bias(false)
            .init(device);
        let up_proj = LinearConfig::new(hidden_size, intermediate_size)
            .with_bias(false)
            .init(device);
        let down_proj = LinearConfig::new(intermediate_size, hidden_size)
            .with_bias(false)
            .init(device);

        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Create MLP from pre-loaded Linear layers
    pub fn from_linears(gate_proj: Linear<B>, up_proj: Linear<B>, down_proj: Linear<B>) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Load from SafeTensor data
    pub fn from_safetensors(
        data: &SafeTensorData,
        layer_prefix: &str,
        hidden_size: usize,
        intermediate_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let gate_proj = load_linear::<B>(
            data,
            &format!("{}.mlp.gate_proj.weight", layer_prefix),
            hidden_size,
            intermediate_size,
            device,
        )?;
        let up_proj = load_linear::<B>(
            data,
            &format!("{}.mlp.up_proj.weight", layer_prefix),
            hidden_size,
            intermediate_size,
            device,
        )?;
        let down_proj = load_linear::<B>(
            data,
            &format!("{}.mlp.down_proj.weight", layer_prefix),
            intermediate_size,
            hidden_size,
            device,
        )?;

        Ok(Self::from_linears(gate_proj, up_proj, down_proj))
    }

    /// Forward pass with SwiGLU activation
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = self.gate_proj.forward(x.clone());
        let gate = activation::silu(gate);
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

/// Configuration for MLP
#[derive(Config, Debug)]
pub struct MLPConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
}

impl MLPConfig {
    /// Initialize the MLP layer with random weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLP<B> {
        MLP::new(device, self.hidden_size, self.intermediate_size)
    }
}

/// Token embedding layer
#[derive(Module, Debug)]
pub struct TokenEmbedding<B: Backend> {
    embedding: Embedding<B>,
}

impl<B: Backend> TokenEmbedding<B> {
    /// Create a new token embedding layer with random weights
    pub fn new(device: &B::Device, vocab_size: usize, hidden_size: usize) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, hidden_size).init(device);
        Self { embedding }
    }

    /// Create from pre-loaded Embedding
    pub fn from_embedding(embedding: Embedding<B>) -> Self {
        Self { embedding }
    }

    /// Load from SafeTensor data
    pub fn from_safetensors(
        data: &SafeTensorData,
        weight_name: &str,
        vocab_size: usize,
        hidden_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let embedding = load_embedding::<B>(data, weight_name, vocab_size, hidden_size, device)?;
        Ok(Self::from_embedding(embedding))
    }

    /// Forward pass
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embedding.forward(input_ids)
    }
}

/// Configuration for TokenEmbedding
#[derive(Config, Debug)]
pub struct TokenEmbeddingConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
}

impl TokenEmbeddingConfig {
    /// Initialize the token embedding layer with random weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> TokenEmbedding<B> {
        TokenEmbedding::new(device, self.vocab_size, self.hidden_size)
    }
}

/// LM Head for output projection
#[derive(Module, Debug)]
pub struct LMHead<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> LMHead<B> {
    /// Create a new LM head with random weights
    pub fn new(device: &B::Device, hidden_size: usize, vocab_size: usize) -> Self {
        let linear = LinearConfig::new(hidden_size, vocab_size)
            .with_bias(false)
            .init(device);
        Self { linear }
    }

    /// Create from pre-loaded Linear layer
    pub fn from_linear(linear: Linear<B>) -> Self {
        Self { linear }
    }

    /// Load from SafeTensor data
    pub fn from_safetensors(
        data: &SafeTensorData,
        weight_name: &str,
        hidden_size: usize,
        vocab_size: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let linear = load_linear::<B>(data, weight_name, hidden_size, vocab_size, device)?;
        Ok(Self::from_linear(linear))
    }

    /// Forward pass
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.linear.forward(x)
    }
}

/// Configuration for LMHead
#[derive(Config, Debug)]
pub struct LMHeadConfig {
    pub hidden_size: usize,
    pub vocab_size: usize,
}

impl LMHeadConfig {
    /// Initialize the LM head with random weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> LMHead<B> {
        LMHead::new(device, self.hidden_size, self.vocab_size)
    }
}
