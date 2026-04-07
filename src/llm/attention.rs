//! Grouped Query Attention implementation for LLaMA 3
//!
//! Implements multi-head attention with Rotary Position Embeddings (RoPE).

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use super::config::LlamaConfig;
use super::layers::{RmsNorm, RmsNormConfig};

/// Rotary Position Embeddings
#[derive(Module, Debug)]
pub struct RotaryEmbedding<B: Backend> {
    /// Cosine cache
    cos_cache: Tensor<B, 2>,
    /// Sine cache
    sin_cache: Tensor<B, 2>,
    /// Head dimension
    #[module(skip)]
    head_dim: usize,
}

impl<B: Backend> RotaryEmbedding<B> {
    /// Create new rotary embeddings
    pub fn new(device: &B::Device, head_dim: usize, max_seq_len: usize, base: f64) -> Self {
        // Compute inverse frequencies
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (base.powf(i as f64 / head_dim as f64)) as f32)
            .collect();

        let inv_freq_tensor = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device);

        // Compute position indices
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions_tensor = Tensor::<B, 1>::from_floats(positions.as_slice(), device);

        // Compute freqs: [seq_len, head_dim/2]
        let freqs = positions_tensor.unsqueeze_dim::<2>(1) * inv_freq_tensor.unsqueeze_dim::<2>(0);

        // Concatenate to get [seq_len, head_dim]
        let freqs = Tensor::cat(vec![freqs.clone(), freqs], 1);

        let cos_cache = freqs.clone().cos();
        let sin_cache = freqs.sin();

        Self {
            cos_cache,
            sin_cache,
            head_dim,
        }
    }

    /// Apply rotary embeddings to query and key tensors
    pub fn forward(
        &self,
        q: Tensor<B, 4>, // [batch, heads, seq, head_dim]
        k: Tensor<B, 4>, // [batch, kv_heads, seq, head_dim]
        position_offset: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let seq_len = q.dims()[2];

        // Get the relevant portion of cos/sin cache
        let cos = self
            .cos_cache
            .clone()
            .slice([position_offset..position_offset + seq_len, 0..self.head_dim])
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0);

        let sin = self
            .sin_cache
            .clone()
            .slice([position_offset..position_offset + seq_len, 0..self.head_dim])
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0);

        let q_rot = self.apply_rotary(q, cos.clone(), sin.clone());
        let k_rot = self.apply_rotary(k, cos, sin);

        (q_rot, k_rot)
    }

    /// Apply rotary embedding to a single tensor
    fn apply_rotary(&self, x: Tensor<B, 4>, cos: Tensor<B, 4>, sin: Tensor<B, 4>) -> Tensor<B, 4> {
        let half = self.head_dim / 2;
        let dims = x.dims();

        // Split x into two halves
        let x1 = x
            .clone()
            .slice([0..dims[0], 0..dims[1], 0..dims[2], 0..half]);
        let x2 = x.slice([0..dims[0], 0..dims[1], 0..dims[2], half..self.head_dim]);

        // Rotate: [-x2, x1]
        let x_rotated = Tensor::cat(vec![x2.clone().neg(), x1.clone()], 3);

        // Get full x for multiplication
        let x_full = Tensor::cat(vec![x1, x2], 3);

        // Apply rotation: x * cos + rotate(x) * sin
        x_full * cos + x_rotated * sin
    }
}

/// Grouped Query Attention
#[derive(Module, Debug)]
pub struct GroupedQueryAttention<B: Backend> {
    /// Query projection
    q_proj: Linear<B>,
    /// Key projection
    k_proj: Linear<B>,
    /// Value projection
    v_proj: Linear<B>,
    /// Output projection
    o_proj: Linear<B>,
    /// Rotary embeddings
    rotary: RotaryEmbedding<B>,
    /// Number of attention heads
    #[module(skip)]
    num_heads: usize,
    /// Number of key-value heads
    #[module(skip)]
    num_kv_heads: usize,
    /// Head dimension
    #[module(skip)]
    head_dim: usize,
    /// Hidden size
    #[module(skip)]
    hidden_size: usize,
}

impl<B: Backend> GroupedQueryAttention<B> {
    /// Create new grouped query attention
    pub fn new(device: &B::Device, config: &LlamaConfig) -> Self {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        let q_proj = LinearConfig::new(hidden_size, num_heads * head_dim)
            .with_bias(false)
            .init(device);

        let k_proj = LinearConfig::new(hidden_size, num_kv_heads * head_dim)
            .with_bias(false)
            .init(device);

        let v_proj = LinearConfig::new(hidden_size, num_kv_heads * head_dim)
            .with_bias(false)
            .init(device);

        let o_proj = LinearConfig::new(num_heads * head_dim, hidden_size)
            .with_bias(false)
            .init(device);

        let rotary = RotaryEmbedding::new(
            device,
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        );

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_size,
        }
    }

    /// Forward pass
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 4>>,
        position_offset: usize,
        kv_cache: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
    ) -> (Tensor<B, 3>, Option<(Tensor<B, 4>, Tensor<B, 4>)>) {
        let [batch_size, seq_len, _] = hidden_states.dims();

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states.clone());
        let k = self.k_proj.forward(hidden_states.clone());
        let v = self.v_proj.forward(hidden_states);

        // Reshape to [batch, seq, heads, head_dim] then transpose to [batch, heads, seq, head_dim]
        let q = q
            .reshape([batch_size, seq_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Apply rotary embeddings
        let (q, k) = self.rotary.forward(q, k, position_offset);

        // Handle KV cache
        let (k, v, new_kv_cache) = if let Some((k_cache, v_cache)) = kv_cache {
            let k = Tensor::cat(vec![k_cache, k], 2);
            let v = Tensor::cat(vec![v_cache, v], 2);
            (k.clone(), v.clone(), Some((k, v)))
        } else {
            (k.clone(), v.clone(), Some((k, v)))
        };

        // Expand KV heads to match query heads (for GQA)
        let kv_ratio = self.num_heads / self.num_kv_heads;
        let k = self.repeat_kv(k, kv_ratio);
        let v = self.repeat_kv(v, kv_ratio);

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(k.swap_dims(2, 3)) / scale;

        // Apply attention mask (causal mask)
        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights + mask
        } else {
            attn_weights
        };

        // Softmax
        let attn_weights = burn::tensor::activation::softmax(attn_weights, 3);

        // Apply attention to values
        let attn_output = attn_weights.matmul(v);

        // Reshape back to [batch, seq, hidden]
        let attn_output = attn_output.swap_dims(1, 2).reshape([
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ]);

        // Output projection
        let output = self.o_proj.forward(attn_output);

        (output, new_kv_cache)
    }

    /// Repeat KV heads to match query heads
    fn repeat_kv(&self, x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
        if n_rep == 1 {
            return x;
        }

        let [batch, num_kv_heads, seq_len, head_dim] = x.dims();

        // Expand and reshape
        let x = x.unsqueeze_dim::<5>(2); // [batch, kv_heads, 1, seq, head_dim]
        let x = x.repeat_dim(2, n_rep); // [batch, kv_heads, n_rep, seq, head_dim]
        x.reshape([batch, num_kv_heads * n_rep, seq_len, head_dim])
    }
}

/// Transformer decoder layer
#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    /// Self attention
    self_attn: GroupedQueryAttention<B>,
    /// MLP
    mlp: super::layers::MLP<B>,
    /// Input layer norm
    input_layernorm: RmsNorm<B>,
    /// Post attention layer norm
    post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> DecoderLayer<B> {
    /// Create a new decoder layer
    pub fn new(device: &B::Device, config: &LlamaConfig) -> Self {
        let self_attn = GroupedQueryAttention::new(device, config);
        let mlp = super::layers::MLP::new(device, config.hidden_size, config.intermediate_size);
        let input_layernorm: RmsNorm<B> = RmsNormConfig::new(config.hidden_size)
            .with_eps(config.rms_norm_eps)
            .init(device);
        let post_attention_layernorm: RmsNorm<B> = RmsNormConfig::new(config.hidden_size)
            .with_eps(config.rms_norm_eps)
            .init(device);

        Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        }
    }

    /// Forward pass
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 4>>,
        position_offset: usize,
        kv_cache: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
    ) -> (Tensor<B, 3>, Option<(Tensor<B, 4>, Tensor<B, 4>)>) {
        // Self attention with residual
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let (hidden_states, new_kv_cache) =
            self.self_attn
                .forward(hidden_states, attention_mask, position_offset, kv_cache);
        let hidden_states = residual + hidden_states;

        // MLP with residual
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states);
        let hidden_states = residual + hidden_states;

        (hidden_states, new_kv_cache)
    }
}
