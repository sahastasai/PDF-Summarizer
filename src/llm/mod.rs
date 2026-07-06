//! LLaMA 3 Model implementation using Burn
//!
//! This module implements the LLaMA 3 architecture for text generation.

pub mod attention;
pub mod config;
pub mod layers;
pub mod loader;
pub mod model;

pub use config::LlamaConfig;
pub use loader::{ModelLoader, SafeTensorData};
pub use model::Llama;
