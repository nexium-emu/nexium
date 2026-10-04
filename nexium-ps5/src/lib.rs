#![cfg_attr(not(all(target_os = "freebsd", target_vendor = "sony")), allow(unused))]

#[cfg(all(target_os = "freebsd", target_vendor = "sony", feature = "title"))]
pub mod affinity;
#[cfg(all(target_os = "freebsd", target_vendor = "sony", feature = "title"))]
mod audio;
#[cfg(all(target_os = "freebsd", target_vendor = "sony", feature = "title"))]
mod clock;
#[cfg(all(target_os = "freebsd", target_vendor = "sony"))]
mod console;
#[cfg(all(target_os = "freebsd", target_vendor = "sony", feature = "title"))]
mod display;
#[cfg(all(target_os = "freebsd", target_vendor = "sony", feature = "title"))]
pub mod filemap;
#[cfg(all(target_os = "freebsd", target_vendor = "sony", feature = "title"))]
mod frontend;
#[cfg(any(test, all(target_os = "freebsd", target_vendor = "sony", feature = "title")))]
#[allow(dead_code)]
#[path = "../../nexium-android/src/library.rs"]
mod library;
#[cfg(all(target_os = "freebsd", target_vendor = "sony", feature = "title"))]
mod menu;
#[cfg(any(test, all(target_os = "freebsd", target_vendor = "sony", feature = "title")))]
mod menu_view;
#[cfg(any(test, all(target_os = "freebsd", target_vendor = "sony", feature = "title")))]
#[allow(dead_code)]
#[path = "../../nexium-android/src/ui.rs"]
mod ui;
#[cfg(all(target_os = "freebsd", target_vendor = "sony"))]
pub mod klog;
#[cfg(all(target_os = "freebsd", target_vendor = "sony"))]
mod probe;
#[cfg(all(target_os = "freebsd", target_vendor = "sony", feature = "title"))]
mod probe_mem;
#[cfg(all(target_os = "freebsd", target_vendor = "sony", feature = "title"))]
mod probe_io;
#[cfg(all(target_os = "freebsd", target_vendor = "sony", feature = "title"))]
mod probe_filemap;
#[cfg(all(target_os = "freebsd", target_vendor = "sony", feature = "title"))]
mod probe_jit;
#[cfg(all(target_os = "freebsd", target_vendor = "sony", feature = "title"))]
mod probe_vk;
#[cfg(all(target_os = "freebsd", target_vendor = "sony", feature = "title"))]
mod pad;
#[cfg(all(target_os = "freebsd", target_vendor = "sony", feature = "title"))]
pub mod sampler;
#[cfg(all(target_os = "freebsd", target_vendor = "sony"))]
mod shim;
#[cfg(all(target_os = "freebsd", target_vendor = "sony"))]
pub mod sys;

#[cfg(all(target_os = "freebsd", target_vendor = "sony"))]
pub use console::*;
