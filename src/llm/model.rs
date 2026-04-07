//! LLaMA 3 Model implementation
//!
//! Full model architecture combining all components.

use burn::prelude::*;
use tracing::{debug, info};

use super::attention::DecoderLayer;
use super::config::LlamaConfig;
use super::layers::{LMHead, RmsNorm, RmsNormConfig, TokenEmbedding};

/// KV Cache for efficient generation
pub type KVCache<B> = Vec<Option<(Tensor<B, 4>, Tensor<B, 4>)>>;

/// LLaMA 3 Model
#[derive(Module, Debug)]
pub struct Llama<B: Backend> {
    /// Token embeddings
    embed_tokens: TokenEmbedding<B>,
    /// Decoder layers
    layers: Vec<DecoderLayer<B>>,
    /// Final layer norm
    norm: RmsNorm<B>,
    /// LM head
    lm_head: LMHead<B>,
}

impl<B: Backend> Llama<B> {
    /// Create a new LLaMA model
    pub fn new(device: &B::Device, config: &LlamaConfig) -> Self {
        info!(
            "Initializing LLaMA model with {} layers",
            config.num_hidden_layers
        );

        let embed_tokens = TokenEmbedding::new(device, config.vocab_size, config.hidden_size);

        let layers: Vec<DecoderLayer<B>> = (0..config.num_hidden_layers)
            .map(|i| {
                debug!("Initializing layer {}", i);
                DecoderLayer::new(device, config)
            })
            .collect();

        let norm: RmsNorm<B> = RmsNormConfig::new(config.hidden_size)
            .with_eps(config.rms_norm_eps)
            .init(device);

        let lm_head = LMHead::new(device, config.hidden_size, config.vocab_size);

        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
        }
    }

    /// Forward pass
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 4>>,
        position_offset: usize,
        kv_cache: Option<KVCache<B>>,
        _config: &LlamaConfig,
    ) -> (Tensor<B, 3>, KVCache<B>) {
        let [batch_size, seq_len] = input_ids.dims();

        // Get embeddings
        let mut hidden_states = self.embed_tokens.forward(input_ids);

        // Create causal mask if not provided
        let attention_mask = attention_mask.unwrap_or_else(|| {
            self.create_causal_mask(
                &hidden_states.device(),
                batch_size,
                seq_len,
                position_offset,
            )
        });

        // Initialize KV cache
        let mut new_kv_cache: KVCache<B> = Vec::with_capacity(self.layers.len());
        let kv_cache = kv_cache.unwrap_or_else(|| vec![None; self.layers.len()]);

        // Forward through decoder layers
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_kv_cache = kv_cache.get(i).cloned().flatten();
            let (new_hidden_states, layer_new_kv_cache) = layer.forward(
                hidden_states,
                Some(attention_mask.clone()),
                position_offset,
                layer_kv_cache,
            );
            hidden_states = new_hidden_states;
            new_kv_cache.push(layer_new_kv_cache);
        }

        // Final layer norm
        let hidden_states = self.norm.forward(hidden_states);

        // LM head
        let logits = self.lm_head.forward(hidden_states);

        (logits, new_kv_cache)
    }

    /// Create a causal attention mask
    fn create_causal_mask(
        &self,
        device: &B::Device,
        batch_size: usize,
        seq_len: usize,
        position_offset: usize,
    ) -> Tensor<B, 4> {
        let total_len = position_offset + seq_len;

        // Create lower triangular mask
        let mask_data: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..total_len).map(move |j| {
                    if j <= i + position_offset {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();

        Tensor::<B, 2>::from_floats(mask_data.as_slice(), device)
            .reshape([seq_len, total_len])
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(0)
            .repeat_dim(0, batch_size)
    }

    /// Generate text using the model
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        config: &LlamaConfig,
    ) -> Tensor<B, 2, Int> {
        let device = input_ids.device();
        let [batch_size, initial_len] = input_ids.dims();

        let mut generated = input_ids.clone();
        let mut kv_cache: Option<KVCache<B>> = None;
        let mut position_offset = 0;

        for step in 0..max_new_tokens {
            // Get the input for this step
            let current_input = if step == 0 {
                generated.clone()
            } else {
                // Only use the last token for subsequent steps (with KV cache)
                let last_pos = generated.dims()[1] - 1;
                generated
                    .clone()
                    .slice([0..batch_size, last_pos..last_pos + 1])
            };

            // Forward pass
            let (logits, new_kv_cache) =
                self.forward(current_input, None, position_offset, kv_cache, config);
            kv_cache = Some(new_kv_cache);

            // Get logits for the last position
            let logits_dims = logits.dims();
            let last_logits = logits.slice([
                0..batch_size,
                logits_dims[1] - 1..logits_dims[1],
                0..config.vocab_size,
            ]);
            let last_logits = last_logits.squeeze::<2>(1);

            // Sample next token
            let next_token = self.sample_token(last_logits, temperature, top_p, top_k, &device);

            // Append to generated sequence
            let next_token_2d = next_token.clone().unsqueeze_dim::<2>(1);
            generated = Tensor::cat(vec![generated, next_token_2d], 1);

            // Update position offset
            if step == 0 {
                position_offset = initial_len;
            } else {
                position_offset += 1;
            }

            // Check for EOS token (simplified check for first batch item)
            let next_token_data = next_token.to_data();
            let next_token_val = next_token_data.iter::<i64>().next().unwrap_or(0);
            if next_token_val == config.eos_token_id as i64 {
                break;
            }
        }

        generated
    }

    /// Sample a token from logits
    fn sample_token(
        &self,
        logits: Tensor<B, 2>,
        temperature: f32,
        _top_p: f32,
        _top_k: usize,
        _device: &B::Device,
    ) -> Tensor<B, 1, Int> {
        // Apply temperature
        let logits = if temperature > 0.0 && temperature != 1.0 {
            logits / temperature
        } else {
            logits
        };

        // Apply softmax to get probabilities
        let probs = burn::tensor::activation::softmax(logits.clone(), 1);

        // Sample from the distribution (using argmax for deterministic sampling)
        // In production, you'd want to use proper multinomial sampling
        let argmax = probs.argmax(1);
        argmax.squeeze(1)
    }
}

/// Model output with additional information
#[derive(Debug)]
pub struct LlamaOutput<B: Backend> {
    /// Output logits
    pub logits: Tensor<B, 3>,
    /// Updated KV cache
    pub kv_cache: KVCache<B>,
}
