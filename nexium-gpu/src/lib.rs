#![allow(dead_code)]

pub mod adapter;
mod astc_async;
mod astc_gpu;
pub mod bcn_encode;
pub mod bundle_cache;
pub mod commands;
pub mod compute;
mod depth;
pub mod descriptor;
pub mod deswizzle;
pub mod draw;
pub mod gpu_health;
mod max_clocks;
pub mod pipeline;
pub mod presentation;
pub mod pitch_oracle;
pub mod renderer;
pub mod rt_cache;
mod rt_writeback;
pub mod shader;
pub mod tex_invalidate;
pub mod texture;
pub mod texture_manifest;
pub mod texture_mips;

pub use renderer::{
    PipelinedPresentCompletion, PipelinedPresentFrame, PipelinedPresentReadback,
    PipelinedPresentSubmission, PresentDepth, Renderer,
};
