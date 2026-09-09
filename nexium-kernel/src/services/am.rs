use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_common::result::SUCCESS;
use std::sync::atomic::{AtomicU32, Ordering};

pub mod msg {
    pub const EXIT_REQUESTED: u32 = 1;
    pub const EXIT: u32 = 4;
    pub const FOCUS_STATE_CHANGED: u32 = 15;
    pub const RESUME: u32 = 16;
    pub const OPERATION_MODE_CHANGED: u32 = 30;
    pub const PERFORMANCE_MODE_CHANGED: u32 = 31;
    pub const REQUEST_TO_DISPLAY: u32 = 51;
}

pub const FOCUS_STATE_IN_FOCUS: u8 = 1;

pub(crate) fn mode_trace_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NEXIUM_MODE_TRACE").is_some())
}

pub(crate) fn default_display_resolution() -> (u32, u32) {
    if crate::hid_state::is_docked() {
        (1920, 1080)
    } else {
        (1280, 720)
    }
}

pub const APPLET_MESSAGE_AVAILABLE_RC: u32 = 0;
pub const APPLET_NO_MESSAGES_RC: u32 = 0x680;

pub struct AppletService;

impl AppletService {
    pub fn new() -> Self {
        Self
    }

    pub fn dispatch(&self, cmd_id: u32) -> u32 {
        log::debug!(
            "am cmd: {} (legacy path - prefer AM helpers in svc.rs)",
            cmd_id
        );
        SUCCESS
    }
}

impl Default for AppletService {
    fn default() -> Self {
        Self::new()
    }
}

pub fn proxy_subsession(port_name: &str, cmd_id: u32) -> Option<&'static str> {
    match (port_name, cmd_id) {
        ("appletAE" | "appletOE", 0) => Some("IApplicationProxy"),
        ("appletAE" | "appletOE", 100) => Some("ISystemAppletProxy"),
        ("appletAE" | "appletOE", 200) => Some("ILibraryAppletProxy"),
        ("appletAE" | "appletOE", 201) => Some("ILibraryAppletProxy"),
        ("appletAE" | "appletOE", 300) => Some("IOverlayAppletProxy"),
        ("appletAE" | "appletOE", 350) => Some("IApplicationProxy"),
        ("IApplicationProxy", 0) => Some("ICommonStateGetter"),
        ("IApplicationProxy", 1) => Some("ISelfController"),
        ("IApplicationProxy", 2) => Some("IWindowController"),
        ("IApplicationProxy", 3) => Some("IAudioController"),
        ("IApplicationProxy", 4) => Some("IDisplayController"),
        ("IApplicationProxy", 10) => Some("IProcessWindingController"),
        ("IApplicationProxy", 11) => Some("ILibraryAppletCreator"),
        ("IApplicationProxy", 20) => Some("IApplicationFunctions"),
        ("IApplicationProxy", 1000) => Some("IDebugFunctions"),
        ("ISystemAppletProxy", 0) => Some("ICommonStateGetter"),
        ("ISystemAppletProxy", 1) => Some("ISelfController"),
        ("ISystemAppletProxy", 2) => Some("IWindowController"),
        ("ISystemAppletProxy", 3) => Some("IAudioController"),
        ("ISystemAppletProxy", 4) => Some("IDisplayController"),
        ("ISystemAppletProxy", 10) => Some("IProcessWindingController"),
        ("ISystemAppletProxy", 11) => Some("ILibraryAppletCreator"),
        ("ISystemAppletProxy", 20) => Some("IApplicationFunctions"),
        ("ISystemAppletProxy", 21) => Some("IHomeMenuFunctions"),
        ("ISystemAppletProxy", 22) => Some("IGlobalStateController"),
        ("ISystemAppletProxy", 23) => Some("IApplicationCreator"),
        ("ISystemAppletProxy", 1000) => Some("IDebugFunctions"),
        ("ILibraryAppletProxy", 0) => Some("ICommonStateGetter"),
        ("ILibraryAppletProxy", 1) => Some("ISelfController"),
        ("ILibraryAppletProxy", 2) => Some("IWindowController"),
        ("ILibraryAppletProxy", 3) => Some("IAudioController"),
        ("ILibraryAppletProxy", 4) => Some("IDisplayController"),
        ("ILibraryAppletProxy", 10) => Some("IProcessWindingController"),
        ("ILibraryAppletProxy", 11) => Some("ILibraryAppletCreator"),
        ("ILibraryAppletProxy", 20) => Some("ILibraryAppletSelfAccessor"),
        ("ILibraryAppletProxy", 21) => Some("IProcessWindingController"),
        ("ILibraryAppletProxy", 1000) => Some("IDebugFunctions"),
        ("IOverlayAppletProxy", 0) => Some("ICommonStateGetter"),
        ("IOverlayAppletProxy", 1) => Some("ISelfController"),
        ("IOverlayAppletProxy", 2) => Some("IWindowController"),
        ("IOverlayAppletProxy", 3) => Some("IAudioController"),
        ("IOverlayAppletProxy", 4) => Some("IDisplayController"),
        ("IOverlayAppletProxy", 10) => Some("IProcessWindingController"),
        ("IOverlayAppletProxy", 11) => Some("ILibraryAppletCreator"),
        ("IOverlayAppletProxy", 20) => Some("IOverlayFunctions"),
        ("IOverlayAppletProxy", 1000) => Some("IDebugFunctions"),
        ("ILibraryAppletCreator", 0) => Some("ILibraryAppletAccessor"),
        ("ILibraryAppletCreator", 10) => Some("IStorage"),
        ("ILibraryAppletCreator", 11) => Some("IStorage"),
        ("IApplicationCreator", 0) => Some("IApplicationAccessor"),
        ("ILibraryAppletAccessor", 101) => Some("IStorageOut"),
        ("IStorage", 0 | 1) => Some("IStorageAccessor"),
        ("IStorageOut", 0 | 1) => Some("IStorageAccessorOut"),
        ("IApplicationFunctions", 1) => Some("ILaunchParamStorage"),
        ("ILaunchParamStorage", 0) => Some("ILaunchParamStorageAccessor"),
        ("acc:u0" | "acc:u1" | "acc:aa", 5) => Some("IProfile"),
        ("acc:u0" | "acc:u1" | "acc:aa", 101) => Some("IManagerForApplication"),
        ("IManagerForApplication", 2) => Some("IAsyncContext"),
        ("nifm:u" | "nifm:a" | "nifm:s", 4) | ("nifm:u" | "nifm:a" | "nifm:s", 5) => {
            Some("IGeneralService")
        }
        ("IGeneralService", 2) => Some("IScanRequest"),
        ("IGeneralService", 4) => Some("IRequest"),
        ("ssl", 0) => Some("ISslContext"),
        ("ISslContext" | "ISslContextForSystem", 2 | 100) => Some("ISslConnection"),
        ("lm", 0) => Some("ILogService"),
        _ => None,
    }
}

pub const ACCOUNT_UID: [u8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

pub fn dispatch_command(
    kernel: &mut Kernel,
    port_name: &str,
    cmd_id: u32,
) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match port_name {
        "ICommonStateGetter" => common_state_getter(kernel, cmd_id),
        "ISelfController" => self_controller(kernel, cmd_id),
        "IWindowController" => window_controller(kernel, cmd_id),
        "IAudioController" => audio_controller(cmd_id),
        "IDisplayController" => display_controller(cmd_id),
        "IProcessWindingController" => process_winding_controller(cmd_id),
        "ILibraryAppletCreator" => library_applet_creator(cmd_id),
        "ILibraryAppletAccessor" => library_applet_accessor(kernel, cmd_id),
        "ILibraryAppletSelfAccessor" => library_applet_self_accessor(kernel, cmd_id),
        "IApplicationFunctions" => application_functions(kernel, cmd_id),
        "IApplicationCreator" => application_creator(cmd_id),
        "IApplicationAccessor" => application_accessor(kernel, cmd_id),
        "IHomeMenuFunctions" => home_menu_functions(kernel, cmd_id),
        "IGlobalStateController" => global_state_controller(cmd_id),
        "IDebugFunctions" => debug_functions(cmd_id),
        "IStorage" => storage(cmd_id),
        "IStorageOut" => storage(cmd_id),
        "IStorageAccessor" => storage_accessor(cmd_id),
        "IStorageAccessorOut" => storage_accessor_out(cmd_id),
        "ILaunchParamStorage" => storage(cmd_id),
        "ILaunchParamStorageAccessor" => launch_param_storage_accessor(cmd_id),
        "acc:u0" | "acc:u1" | "acc:aa" => account_service(cmd_id),
        "IProfile" => profile(cmd_id),
        "IManagerForApplication" => manager_for_application(cmd_id),
        "IApmManager" => apm_manager(cmd_id),
        "IApmSession" => apm_session(cmd_id),
        "IAsyncContext" => async_context(kernel, cmd_id),
        "IGeneralService" => general_service(cmd_id),
        "IRequest" => nifm_request(kernel, cmd_id),
        "IScanRequest" => ok_empty(),
        "ssl" => ssl_service(cmd_id),
        "ISslContext" | "ISslContextForSystem" => ssl_context(cmd_id),
        "ISslConnection" => ssl_connection(cmd_id),
        "ILogService" => ok_empty(),
        "IOverlayFunctions" => overlay_functions(cmd_id),
        "ILockAccessor" => lock_accessor(cmd_id),
        "IAppletCommonFunctions" => applet_common_functions(cmd_id),
        _ => None,
    }
}

fn alloc_event(kernel: &mut Kernel, slot: &mut Option<u32>, name: &str) -> u32 {
    if let Some(h) = *slot {
        h
    } else {
        let h = kernel.handles.create_handle(HandleType::Event);
        kernel.event_signals.insert(h, false);
        *slot = Some(h);
        log::debug!("am: allocated event {} = {:#x}", name, h);
        h
    }
}

fn ok(data: Vec<u8>) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    Some((0, data, Vec::new()))
}

fn ok_with_handle(data: Vec<u8>, handle: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    Some((0, data, vec![handle]))
}

fn ok_empty() -> Option<(u32, Vec<u8>, Vec<u32>)> {
    Some((0, Vec::new(), Vec::new()))
}

fn err(rc: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    Some((rc, Vec::new(), Vec::new()))
}

fn common_state_getter(kernel: &mut Kernel, cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 => {
            let first = kernel.applet_message_event.is_none();
            let mut slot = kernel.applet_message_event;
            let h = alloc_event(kernel, &mut slot, "AppletMessageEvent");
            kernel.applet_message_event = slot;
            if first {
                queue_message(kernel, msg::FOCUS_STATE_CHANGED);
            }
            log::debug!(
                "ICommonStateGetter.GetEventHandle → {:#x} (initial focus msg queued={})",
                h,
                first
            );
            ok_with_handle(Vec::new(), h)
        }
        1 => {
            if let Some(m) = kernel.applet_messages.pop_front() {
                log::debug!("ICommonStateGetter.ReceiveMessage → {}", m);
                if kernel.applet_messages.is_empty() {
                    if let Some(h) = kernel.applet_message_event {
                        kernel.event_signals.insert(h, false);
                    }
                }
                ok(m.to_le_bytes().to_vec())
            } else {
                log::debug!("ICommonStateGetter.ReceiveMessage (none) → 0x680");
                err(APPLET_NO_MESSAGES_RC)
            }
        }
        2 => {
            log::debug!("ICommonStateGetter.GetThisAppletKind → 0");
            ok(0u32.to_le_bytes().to_vec())
        }
        3 | 4 => ok_empty(),
        5 => {
            let mode: u8 = if crate::hid_state::is_docked() { 1 } else { 0 };
            if mode_trace_enabled() {
                log::warn!("[mode-trace] GetOperationMode -> {mode}");
            }
            ok(mode.to_le_bytes().to_vec())
        }
        6 => {
            let mode: u32 = if crate::hid_state::is_docked() { 1 } else { 0 };
            if mode_trace_enabled() {
                log::warn!("[mode-trace] GetPerformanceMode -> {mode}");
            }
            ok(mode.to_le_bytes().to_vec())
        }
        7 => ok(0u8.to_le_bytes().to_vec()),
        8 => ok(0u8.to_le_bytes().to_vec()),
        9 => ok(kernel.applet_focus_state.to_le_bytes().to_vec()),
        10 | 11 | 12 => ok_empty(),
        13 => {
            let mut slot = kernel.acquired_sleep_lock_event;
            let h = alloc_event(kernel, &mut slot, "AcquiredSleepLockEvent");
            kernel.acquired_sleep_lock_event = slot;
            ok_with_handle(Vec::new(), h)
        }
        14 => ok(0u64.to_le_bytes().to_vec()),
        50 => ok(0u8.to_le_bytes().to_vec()),
        51 | 52 | 53 | 54 => ok_empty(),
        55 => ok(0u8.to_le_bytes().to_vec()),
        60 => {
            let (width, height) = default_display_resolution();
            if mode_trace_enabled() {
                log::warn!("[mode-trace] GetDefaultDisplayResolution -> {width}x{height}");
            }
            let mut buf = [0u8; 8];
            buf[0..4].copy_from_slice(&width.to_le_bytes());
            buf[4..8].copy_from_slice(&height.to_le_bytes());
            ok(buf.to_vec())
        }
        61 => {
            let mut slot = kernel.display_resolution_change_event;
            let h = alloc_event(kernel, &mut slot, "DisplayResolutionChangeEvent");
            kernel.display_resolution_change_event = slot;
            ok_with_handle(Vec::new(), h)
        }
        66 | 67 => ok_empty(),
        68 => ok(0u32.to_le_bytes().to_vec()),
        80 | 90 => ok_empty(),
        91 => ok(0u32.to_le_bytes().to_vec()),
        200 => ok(0u32.to_le_bytes().to_vec()),
        300 => ok(0u8.to_le_bytes().to_vec()),
        400 | 401 | 500 | 900 => ok_empty(),
        _ => {
            log::warn!(
                "ICommonStateGetter.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
                cmd
            );
            ok_empty()
        }
    }
}

fn self_controller(kernel: &mut Kernel, cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 | 1 | 2 | 3 | 4 => ok_empty(),
        9 => {
            let mut slot = kernel.library_applet_launchable_event;
            let h = alloc_event(kernel, &mut slot, "LibraryAppletLaunchableEvent");
            kernel.library_applet_launchable_event = slot;
            kernel.event_signals.insert(h, true);
            ok_with_handle(Vec::new(), h)
        }
        10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 => ok_empty(),
        40 => ok(1u64.to_le_bytes().to_vec()),
        41 => ok_empty(),
        42 | 43 => {
            let mut buf = [0u8; 16];
            buf[0..8].copy_from_slice(&1u64.to_le_bytes());
            buf[8..16].copy_from_slice(&0u64.to_le_bytes());
            ok(buf.to_vec())
        }
        44 => {
            let mut buf = [0u8; 16];
            buf[0..8].copy_from_slice(&1u64.to_le_bytes());
            buf[8..16].copy_from_slice(&2u64.to_le_bytes());
            ok(buf.to_vec())
        }
        50 | 51 => ok_empty(),
        60 | 61 | 62 | 63 | 64 | 65 => ok_empty(),
        66 => ok(0u32.to_le_bytes().to_vec()),
        67 => ok(0u8.to_le_bytes().to_vec()),
        68 => ok_empty(),
        69 => ok(0u8.to_le_bytes().to_vec()),
        80 => ok_empty(),
        81 => ok(0u64.to_le_bytes().to_vec()),
        90 => ok(0u64.to_le_bytes().to_vec()),
        91 => {
            let mut slot = kernel.accumulated_suspended_tick_event;
            let h = alloc_event(kernel, &mut slot, "AccumulatedSuspendedTickChangedEvent");
            kernel.accumulated_suspended_tick_event = slot;
            ok_with_handle(Vec::new(), h)
        }
        100 | 110 | 120 | 130 => ok_empty(),
        _ => {
            log::warn!(
                "ISelfController.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
                cmd
            );
            ok_empty()
        }
    }
}

fn window_controller(_kernel: &mut Kernel, cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 | 10 | 11 | 12 | 13 => ok_empty(),
        1 => ok(1u64.to_le_bytes().to_vec()),
        _ => {
            log::warn!(
                "IWindowController.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
                cmd
            );
            ok_empty()
        }
    }
}

fn audio_controller(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 | 1 => ok_empty(),
        2 => ok(0u32.to_le_bytes().to_vec()),
        3 => ok_empty(),
        4 => ok(0u32.to_le_bytes().to_vec()),
        _ => {
            log::warn!(
                "IAudioController.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
                cmd
            );
            ok_empty()
        }
    }
}

fn display_controller(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19
        | 20 | 21 | 22 | 23 | 24 => ok_empty(),
        _ => {
            log::warn!(
                "IDisplayController.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
                cmd
            );
            ok_empty()
        }
    }
}

fn process_winding_controller(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 | 11 | 21 | 22 | 23 | 30 | 40 | 41 => ok_empty(),
        _ => {
            log::warn!("IProcessWindingController.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)", cmd);
            ok_empty()
        }
    }
}

fn library_applet_creator(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    log::warn!(
        "ILibraryAppletCreator.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
        cmd
    );
    ok_empty()
}

fn swkbd_pending() -> bool {
    pending_applet_id() == APPLET_ID_SWKBD
}

fn swkbd_read_initial_text(kernel: &mut Kernel) -> String {
    let Some((workbuf_addr, workbuf_size)) = kernel.swkbd_workbuf else {
        return String::new();
    };
    let Some((offset, units)) = crate::swkbd_state::config_initial_span() else {
        return String::new();
    };
    let units = units.min(4096) as usize;
    let byte_len = units * 2;
    if (offset as u64).saturating_add(byte_len as u64) > workbuf_size {
        log::warn!(
            "swkbd: initial string span +{:#x}x{} exceeds workbuf size {:#x}",
            offset,
            units,
            workbuf_size
        );
        return String::new();
    }
    let mut bytes = vec![0u8; byte_len];
    if kernel
        .address_space
        .read(workbuf_addr + offset as u64, &mut bytes)
        .is_err()
    {
        log::warn!(
            "swkbd: failed to read initial string at {:#x}+{:#x}",
            workbuf_addr,
            offset
        );
        return String::new();
    }
    let mut u16s = Vec::with_capacity(units);
    for pair in bytes.chunks_exact(2) {
        let u = u16::from_le_bytes([pair[0], pair[1]]);
        if u == 0 {
            break;
        }
        u16s.push(u);
    }
    String::from_utf16_lossy(&u16s)
}

fn library_applet_accessor(kernel: &mut Kernel, cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 => {
            let h = kernel.handles.create_handle(HandleType::Event);
            if swkbd_pending() && !crate::swkbd_state::is_completed() {
                kernel.event_signals.insert(h, false);
                kernel.swkbd_state_changed_events.push(h);
                log::debug!("swkbd: state-changed event {:#x} (deferred completion)", h);
            } else {
                kernel.event_signals.insert(h, true);
            }
            ok_with_handle(Vec::new(), h)
        }
        1 => {
            if swkbd_pending() {
                ok(vec![u8::from(crate::swkbd_state::is_completed())])
            } else {
                ok(vec![1u8])
            }
        }
        10 => {
            if swkbd_pending() {
                let initial = swkbd_read_initial_text(kernel);
                crate::swkbd_state::start(initial);
            } else if pending_applet_id() == APPLET_ID_OFFLINE_WEB {
                log::warn!(
                    "offline HTML viewer unavailable; returning a WindowClosed result to the game"
                );
            }
            ok_empty()
        }
        20 | 25 => {
            if swkbd_pending() && !crate::swkbd_state::is_completed() {
                crate::swkbd_state::complete(false, "");
                crate::swkbd_state::request_gui_cancel();
                kernel.signal_swkbd_state_changed();
                log::info!("swkbd: canceled by guest (accessor cmd {})", cmd);
            }
            ok_empty()
        }
        106 => {
            if swkbd_pending() {
                let h = kernel.handles.create_handle(HandleType::Event);
                kernel.event_signals.insert(h, false);
                kernel.swkbd_interactive_event = Some(h);
                log::debug!("swkbd: interactive-out event {:#x} (never signaled)", h);
                ok_with_handle(Vec::new(), h)
            } else {
                log::warn!(
                    "ILibraryAppletAccessor.cmd_106 UNHANDLED → returning empty SUCCESS (likely wrong)"
                );
                ok_empty()
            }
        }
        30 => {
            kernel.applet_focus_state = FOCUS_STATE_IN_FOCUS;
            let exit_on_result = std::env::var_os("NEXIUM_APPLET_EXIT_ON_RESULT").is_some();
            let msg = if exit_on_result {
                msg::EXIT
            } else {
                msg::RESUME
            };
            queue_message(kernel, msg);
            log::info!(
                "ILibraryAppletAccessor.GetResult → applet complete, queued {}, focus=InFocus",
                msg
            );
            ok_empty()
        }
        26 | 50 | 51 | 60 | 90 | 91 | 100 | 102 | 103 | 110 | 120 | 150 | 160 => ok_empty(),
        _ => {
            log::warn!(
                "ILibraryAppletAccessor.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
                cmd
            );
            ok_empty()
        }
    }
}

fn library_applet_self_accessor(kernel: &mut Kernel, cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 | 1 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 19 => ok_empty(),
        2 => {
            let h = kernel.handles.create_handle(HandleType::Event);
            kernel.event_signals.insert(h, false);
            ok_with_handle(Vec::new(), h)
        }
        20 | 25 | 30 | 40 | 50 | 60 | 100 | 110 | 120 => ok_empty(),
        _ => {
            log::warn!("ILibraryAppletSelfAccessor.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)", cmd);
            ok_empty()
        }
    }
}

fn application_functions(kernel: &mut Kernel, cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        1 => err(0x480),
        10 | 12 => ok_empty(),
        20 => ok(0u64.to_le_bytes().to_vec()),
        21 => {
            let lang_code = u64::from_le_bytes(*b"en-US\0\0\0");
            ok(lang_code.to_le_bytes().to_vec())
        }
        22 => ok_empty(),
        23 => ok(vec![0u8; 16]),
        24 => ok(vec![0u8; 16]),
        25 => ok(0u64.to_le_bytes().to_vec()),
        26 => ok(vec![0u8; 16]),
        27 => ok(vec![0u8; 16]),
        28 => ok(vec![0u8; 16]),
        30 | 31 | 32 | 33 => ok_empty(),
        40 => ok(1u8.to_le_bytes().to_vec()),
        50 => ok(vec![0u8; 16]),
        60 => ok_empty(),
        65 => ok(0u8.to_le_bytes().to_vec()),
        66 | 67 | 70 | 71 | 72 | 80 | 90 => ok_empty(),
        100 | 101 | 102 | 103 => ok_empty(),
        110 => ok(0u8.to_le_bytes().to_vec()),
        111 => ok(0u64.to_le_bytes().to_vec()),
        120 => ok(vec![0u8; 16]),
        121 => ok(0i32.to_le_bytes().to_vec()),
        123 => ok((-1i32).to_le_bytes().to_vec()),
        124 => ok_empty(),
        130 => {
            let mut slot = kernel.gpu_error_detected_event;
            let h = alloc_event(kernel, &mut slot, "GpuErrorDetectedSystemEvent");
            kernel.gpu_error_detected_event = slot;
            ok_with_handle(Vec::new(), h)
        }
        140 => {
            let mut slot = kernel.friend_invitation_event;
            let h = alloc_event(kernel, &mut slot, "FriendInvitationStorageChannelEvent");
            kernel.friend_invitation_event = slot;
            ok_with_handle(Vec::new(), h)
        }
        150 => {
            let mut slot = kernel.notification_event;
            let h = alloc_event(kernel, &mut slot, "NotificationStorageChannelEvent");
            kernel.notification_event = slot;
            ok_with_handle(Vec::new(), h)
        }
        131 | 141 | 160 | 170 | 1000 | 1001 => ok_empty(),
        _ => {
            log::warn!(
                "IApplicationFunctions.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
                cmd
            );
            ok_empty()
        }
    }
}

fn application_creator(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    log::warn!(
        "IApplicationCreator.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
        cmd
    );
    ok_empty()
}

fn application_accessor(_kernel: &mut Kernel, cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    log::warn!(
        "IApplicationAccessor.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
        cmd
    );
    ok_empty()
}

fn home_menu_functions(_kernel: &mut Kernel, cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        10 | 11 | 12 | 13 | 20 | 21 | 30 => ok_empty(),
        _ => {
            log::warn!(
                "IHomeMenuFunctions.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
                cmd
            );
            ok_empty()
        }
    }
}

fn global_state_controller(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    log::warn!(
        "IGlobalStateController.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
        cmd
    );
    ok_empty()
}

fn debug_functions(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    log::warn!(
        "IDebugFunctions.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
        cmd
    );
    ok_empty()
}

fn storage(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 => ok_empty(),
        _ => {
            log::warn!(
                "IStorage.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
                cmd
            );
            ok_empty()
        }
    }
}

fn storage_accessor(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 => ok(0u64.to_le_bytes().to_vec()),
        10 | 11 => ok_empty(),
        _ => {
            log::warn!(
                "IStorageAccessor.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
                cmd
            );
            ok_empty()
        }
    }
}

static PENDING_APPLET_ID: AtomicU32 = AtomicU32::new(0);
static CONTROLLER_SELECTED_ID: AtomicU32 = AtomicU32::new(0);

pub const APPLET_ID_CONTROLLER: u32 = 0x0c;
pub const APPLET_ID_SWKBD: u32 = 0x11;
pub const APPLET_ID_OFFLINE_WEB: u32 = 0x17;

pub fn set_pending_applet_id(id: u32) {
    PENDING_APPLET_ID.store(id, Ordering::Relaxed);
}

pub fn pending_applet_id() -> u32 {
    PENDING_APPLET_ID.load(Ordering::Relaxed)
}

pub fn set_controller_selected_id(id: u32) {
    CONTROLLER_SELECTED_ID.store(id, Ordering::Relaxed);
}

pub fn applet_out_data() -> Vec<u8> {
    match PENDING_APPLET_ID.load(Ordering::Relaxed) {
        APPLET_ID_CONTROLLER => {
            let mut v = vec![0u8; 0xc];
            v[0] = 1;
            let sel = CONTROLLER_SELECTED_ID.load(Ordering::Relaxed);
            v[4..8].copy_from_slice(&sel.to_le_bytes());
            v
        }
        APPLET_ID_SWKBD => crate::swkbd_state::out_data(),
        APPLET_ID_OFFLINE_WEB => {
            let mut result = vec![0u8; 0x1010];
            result[..4].copy_from_slice(&4u32.to_le_bytes());
            result
        }
        _ => Vec::new(),
    }
}

fn storage_accessor_out(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 => ok((applet_out_data().len() as u64).to_le_bytes().to_vec()),
        10 | 11 => ok_empty(),
        _ => {
            log::warn!(
                "IStorageAccessorOut.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
                cmd
            );
            ok_empty()
        }
    }
}

pub const LAUNCH_PARAMETER_SIZE: u64 = 0x88;

fn launch_param_storage_accessor(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 => ok(LAUNCH_PARAMETER_SIZE.to_le_bytes().to_vec()),
        10 | 11 => ok_empty(),
        _ => {
            log::warn!("ILaunchParamStorageAccessor.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)", cmd);
            ok_empty()
        }
    }
}

fn account_service(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 => ok(1u32.to_le_bytes().to_vec()),
        1 => ok(vec![1u8]),
        2 | 3 => ok_empty(),
        4 => ok(ACCOUNT_UID.to_vec()),
        100 | 102 | 103 | 110 | 140 | 141 | 160 => ok_empty(),
        150 => ok(vec![0u8]),
        _ => {
            log::warn!("acc.cmd_{} → returning empty SUCCESS (likely wrong)", cmd);
            ok_empty()
        }
    }
}

fn build_profile_base() -> Vec<u8> {
    let mut out = vec![0u8; 0x38];
    out[0..16].copy_from_slice(&ACCOUNT_UID);
    let name = b"nexium";
    out[0x18..0x18 + name.len()].copy_from_slice(name);
    out
}

fn profile(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 | 1 => ok(build_profile_base()),
        10 => ok(0u32.to_le_bytes().to_vec()),
        11 => ok_empty(),
        _ => {
            log::warn!(
                "IProfile.cmd_{} → returning empty SUCCESS (likely wrong)",
                cmd
            );
            ok_empty()
        }
    }
}

fn manager_for_application(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 => ok(vec![0u8]),
        1 => ok(0x0102_0304_0506_0708u64.to_le_bytes().to_vec()),
        3 => ok(0u64.to_le_bytes().to_vec()),
        160 => ok_empty(),
        _ => {
            log::warn!(
                "IManagerForApplication.cmd_{} → returning empty SUCCESS (likely wrong)",
                cmd
            );
            ok_empty()
        }
    }
}

fn apm_manager(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        1 => {
            let mode: i32 = if crate::hid_state::is_docked() { 1 } else { 0 };
            if mode_trace_enabled() {
                log::warn!("[mode-trace] IApmManager.GetPerformanceMode -> {mode}");
            }
            ok(mode.to_le_bytes().to_vec())
        }
        6 => ok(vec![0u8]),
        _ => {
            log::warn!(
                "IApmManager.cmd_{} -> returning empty SUCCESS (likely wrong)",
                cmd
            );
            ok_empty()
        }
    }
}

fn apm_session(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 | 2 => ok_empty(),
        1 => ok(0x0002_0003u32.to_le_bytes().to_vec()),
        _ => {
            log::warn!(
                "IApmSession.cmd_{} -> returning empty SUCCESS (likely wrong)",
                cmd
            );
            ok_empty()
        }
    }
}

fn async_context(kernel: &mut Kernel, cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 => {
            let h = kernel.handles.create_handle(HandleType::Event);
            kernel.event_signals.insert(h, true);
            ok_with_handle(Vec::new(), h)
        }
        1 => ok_empty(),
        2 => ok(vec![1u8]),
        3 => ok_empty(),
        _ => ok_empty(),
    }
}

fn nifm_online() -> bool {
    std::env::var("NEXIUM_NIFM_ONLINE").ok().as_deref() != Some("0")
}

fn host_ip_address() -> [u8; 4] {
    use std::sync::OnceLock;
    static ADDRESS: OnceLock<[u8; 4]> = OnceLock::new();
    *ADDRESS.get_or_init(|| {
        std::net::UdpSocket::bind("0.0.0.0:0")
            .and_then(|socket| {
                socket.connect("8.8.8.8:53")?;
                socket.local_addr()
            })
            .ok()
            .and_then(|address| match address {
                std::net::SocketAddr::V4(address) => Some(address.ip().octets()),
                std::net::SocketAddr::V6(_) => None,
            })
            .filter(|octets| octets != &[0, 0, 0, 0])
            .unwrap_or([192, 168, 0, 2])
    })
}

const NIFM_REQUEST_NOT_SUBMITTED: u32 = 1;
const NIFM_REQUEST_ACCEPTED: u32 = 3;
const RESULT_NIFM_NETWORK_COMMUNICATION_DISABLED: u32 = (110u32) | (1111u32 << 9);

static NIFM_REQUEST_STATE: AtomicU32 = AtomicU32::new(NIFM_REQUEST_NOT_SUBMITTED);

fn general_service(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        1 => ok(1u64.to_le_bytes().to_vec()),
        4 => {
            NIFM_REQUEST_STATE.store(NIFM_REQUEST_NOT_SUBMITTED, Ordering::Relaxed);
            ok_empty()
        }
        12 => {
            if nifm_online() {
                ok(host_ip_address().to_vec())
            } else {
                ok(vec![0u8; 4])
            }
        }
        15 => ok(vec![0u8; 0x16]),
        17 => ok(vec![1u8]),
        20 | 21 => ok(vec![u8::from(nifm_online())]),
        18 => {
            if nifm_online() {
                ok(vec![1u8, 3u8, 4u8])
            } else {
                ok(vec![0u8, 0u8, 0u8])
            }
        }
        22 => ok(vec![0u8]),
        _ => ok_empty(),
    }
}

fn nifm_request(kernel: &mut Kernel, cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 => {
            let state = NIFM_REQUEST_STATE.load(Ordering::Relaxed);
            ok(state.to_le_bytes().to_vec())
        }
        1 => {
            if nifm_online() {
                if NIFM_REQUEST_STATE.load(Ordering::Relaxed) != NIFM_REQUEST_NOT_SUBMITTED {
                    NIFM_REQUEST_STATE.store(NIFM_REQUEST_ACCEPTED, Ordering::Relaxed);
                }
                ok_empty()
            } else {
                err(RESULT_NIFM_NETWORK_COMMUNICATION_DISABLED)
            }
        }
        2 => {
            let h1 = kernel.handles.create_handle(HandleType::Event);
            let h2 = kernel.handles.create_handle(HandleType::Event);
            kernel.event_signals.insert(h1, true);
            kernel.event_signals.insert(h2, true);
            kernel.nvdrv_sync_events.insert(h1);
            kernel.nvdrv_sync_events.insert(h2);
            Some((0, Vec::new(), vec![h1, h2]))
        }
        3 => {
            NIFM_REQUEST_STATE.store(NIFM_REQUEST_NOT_SUBMITTED, Ordering::Relaxed);
            ok_empty()
        }
        4 => {
            let state = if nifm_online() {
                NIFM_REQUEST_ACCEPTED
            } else {
                NIFM_REQUEST_NOT_SUBMITTED
            };
            NIFM_REQUEST_STATE.store(state, Ordering::Relaxed);
            ok_empty()
        }
        5 | 6 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 23 | 24 | 25 => ok_empty(),
        _ => ok_empty(),
    }
}

fn ssl_service(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        1 => ok(0u32.to_le_bytes().to_vec()),
        2 | 3 => ok(0u32.to_le_bytes().to_vec()),
        5 => ok_empty(),
        6 | 7 | 8 | 9 => ok_empty(),
        101 | 102 => ok(0u64.to_le_bytes().to_vec()),
        _ => ok_empty(),
    }
}

fn ssl_context(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 | 9 | 10 | 11 => ok_empty(),
        1 => ok(0i32.to_le_bytes().to_vec()),
        3 => ok(0u32.to_le_bytes().to_vec()),
        4 | 5 | 8 | 12 | 13 | 14 => ok(1u64.to_le_bytes().to_vec()),
        _ => ok_empty(),
    }
}

fn ssl_connection(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    match cmd {
        0 => ok((-1i32).to_le_bytes().to_vec()),
        3 | 4 | 5 | 6 | 19 | 20 | 22 | 24 | 25 | 26 | 27 | 28 | 29 => ok_empty(),
        8 | 11 | 12 | 13 | 14 | 15 | 16 | 18 | 23 | 30 | 31 => ok(0u32.to_le_bytes().to_vec()),
        17 => ok((-1i32).to_le_bytes().to_vec()),
        _ => ok_empty(),
    }
}

fn overlay_functions(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    log::warn!(
        "IOverlayFunctions.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
        cmd
    );
    ok_empty()
}

fn lock_accessor(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    log::warn!(
        "ILockAccessor.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
        cmd
    );
    ok_empty()
}

fn applet_common_functions(cmd: u32) -> Option<(u32, Vec<u8>, Vec<u32>)> {
    log::warn!(
        "IAppletCommonFunctions.cmd_{} UNHANDLED → returning empty SUCCESS (likely wrong)",
        cmd
    );
    ok_empty()
}

pub fn queue_message(kernel: &mut Kernel, msg: u32) {
    kernel.applet_messages.push_back(msg);
    if let Some(h) = kernel.applet_message_event {
        kernel.event_signals.insert(h, true);
        kernel.threads.signal_handle(h);
    }
    log::debug!(
        "am: queued message {} (queue len {})",
        msg,
        kernel.applet_messages.len()
    );
}
