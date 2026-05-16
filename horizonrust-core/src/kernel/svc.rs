use super::Kernel;
use crate::common::result::SUCCESS;

pub fn dispatch(kernel: &mut Kernel, imm: u16) -> u32 {
    match imm {
        0x01 => svc_set_heap_size(kernel),
        0x06 => svc_query_memory(kernel),
        0x07 => svc_exit_process(kernel),
        0x0d => svc_map_shared_memory(kernel),
        0x0e => svc_unmap_shared_memory(kernel),
        0x10 => svc_signal_event(kernel),
        0x12 => svc_wait_synchronization(kernel),
        0x13 => svc_cancel_synchronization(kernel),
        0x15 => svc_send_sync_request(kernel),
        0x16 => svc_send_sync_request_with_user_buffer(kernel),
        0x19 => svc_get_thread_id(kernel),
        0x1a => svc_break(kernel),
        0x1b => svc_output_debug_string(kernel),
        0x1f => svc_connect_to_named_port(kernel),
        0x21 => svc_get_info(kernel),
        0x25 => svc_map_physical_memory(kernel),
        0x26 => svc_unmap_physical_memory(kernel),
        0x29 => svc_create_session(kernel),
        0x2b => svc_reply_and_receive(kernel),
        0x2d => svc_create_event(kernel),
        0x31 => svc_create_shared_memory(kernel),
        0x41 => svc_map_transfer_memory(kernel),
        0x46 => svc_create_thread(kernel),
        0x47 => svc_start_thread(kernel),
        0x48 => svc_exit_thread(kernel),
        0x4a => svc_sleep_thread(kernel),
        0x50 => svc_flush_data_cache(kernel),
        _ => {
            log::warn!("unknown SVC: {:#04x}", imm);
            0x4201
        }
    }
}

fn svc_set_heap_size(kernel: &mut Kernel) -> u32 { log::trace!("svc_set_heap_size"); SUCCESS }
fn svc_query_memory(kernel: &mut Kernel) -> u32 { log::trace!("svc_query_memory"); SUCCESS }
fn svc_exit_process(kernel: &mut Kernel) -> u32 { log::trace!("svc_exit_process"); SUCCESS }
fn svc_map_shared_memory(kernel: &mut Kernel) -> u32 { log::trace!("svc_map_shared_memory"); SUCCESS }
fn svc_unmap_shared_memory(kernel: &mut Kernel) -> u32 { log::trace!("svc_unmap_shared_memory"); SUCCESS }
fn svc_signal_event(kernel: &mut Kernel) -> u32 { log::trace!("svc_signal_event"); SUCCESS }
fn svc_wait_synchronization(kernel: &mut Kernel) -> u32 { log::trace!("svc_wait_synchronization"); SUCCESS }
fn svc_cancel_synchronization(kernel: &mut Kernel) -> u32 { log::trace!("svc_cancel_synchronization"); SUCCESS }
fn svc_send_sync_request(kernel: &mut Kernel) -> u32 { log::trace!("svc_send_sync_request"); SUCCESS }
fn svc_send_sync_request_with_user_buffer(kernel: &mut Kernel) -> u32 { log::trace!("svc_send_sync_request_with_user_buffer"); SUCCESS }
fn svc_get_thread_id(kernel: &mut Kernel) -> u32 { log::trace!("svc_get_thread_id"); SUCCESS }
fn svc_break(kernel: &mut Kernel) -> u32 { log::trace!("svc_break"); SUCCESS }
fn svc_output_debug_string(kernel: &mut Kernel) -> u32 { log::trace!("svc_output_debug_string"); SUCCESS }
fn svc_connect_to_named_port(kernel: &mut Kernel) -> u32 { log::trace!("svc_connect_to_named_port"); SUCCESS }
fn svc_get_info(kernel: &mut Kernel) -> u32 { log::trace!("svc_get_info"); SUCCESS }
fn svc_map_physical_memory(kernel: &mut Kernel) -> u32 { log::trace!("svc_map_physical_memory"); SUCCESS }
fn svc_unmap_physical_memory(kernel: &mut Kernel) -> u32 { log::trace!("svc_unmap_physical_memory"); SUCCESS }
fn svc_create_session(kernel: &mut Kernel) -> u32 { log::trace!("svc_create_session"); SUCCESS }
fn svc_reply_and_receive(kernel: &mut Kernel) -> u32 { log::trace!("svc_reply_and_receive"); SUCCESS }
fn svc_create_event(kernel: &mut Kernel) -> u32 { log::trace!("svc_create_event"); SUCCESS }
fn svc_create_shared_memory(kernel: &mut Kernel) -> u32 { log::trace!("svc_create_shared_memory"); SUCCESS }
fn svc_map_transfer_memory(kernel: &mut Kernel) -> u32 { log::trace!("svc_map_transfer_memory"); SUCCESS }
fn svc_create_thread(kernel: &mut Kernel) -> u32 { log::trace!("svc_create_thread"); SUCCESS }
fn svc_start_thread(kernel: &mut Kernel) -> u32 { log::trace!("svc_start_thread"); SUCCESS }
fn svc_exit_thread(kernel: &mut Kernel) -> u32 { log::trace!("svc_exit_thread"); SUCCESS }
fn svc_sleep_thread(kernel: &mut Kernel) -> u32 { log::trace!("svc_sleep_thread"); SUCCESS }
fn svc_flush_data_cache(kernel: &mut Kernel) -> u32 { log::trace!("svc_flush_data_cache"); SUCCESS }
