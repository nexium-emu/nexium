use crate::hid_vibration::{self, VibrationValue};
use crate::kernel::handles::HandleType;
use crate::kernel::Kernel;
use nexium_ipc::IpcCtx;

#[inline]
fn new_event(kernel: &mut Kernel) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, false);
    h
}

#[inline]
fn new_signaled_event(kernel: &mut Kernel) -> u32 {
    let h = kernel.handles.create_handle(HandleType::Event);
    kernel.event_signals.insert(h, true);
    h
}

pub fn activate_debug_pad(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {}
pub fn activate_touch_screen(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {}
pub fn activate_mouse(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {}
pub fn add_mouse_wheel_delta(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _delta: i32) {}
pub fn activate_debug_mouse(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {}
pub fn activate_keyboard(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {}
pub fn send_keyboard_lock_key_event(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _flags: u32,
    _aruid: u64,
) {
}

pub fn acquire_xpad_id_event_handle(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _basic_xpad_id: u64,
) -> u32 {
    new_event(kernel)
}
pub fn release_xpad_id_event_handle(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _basic_xpad_id: u64,
) {
}
pub fn activate_xpad(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _basic_xpad_id: u32, _aruid: u64) {}
pub fn get_xpad_ids(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}
pub fn activate_joy_xpad(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _joy_xpad_id: u32) {}
pub fn get_joy_xpad_lifo_handle(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _joy_xpad_id: u32,
) -> u32 {
    new_event(kernel)
}
pub fn get_joy_xpad_ids(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> u64 {
    0
}

pub fn activate_six_axis_sensor(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _joy_xpad_id: u32) {}
pub fn deactivate_six_axis_sensor(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _joy_xpad_id: u32) {}
pub fn get_six_axis_sensor_lifo_handle(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _joy_xpad_id: u32,
) -> u32 {
    new_event(kernel)
}
pub fn activate_joy_six_axis_sensor(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _joy_xpad_id: u32) {}
pub fn deactivate_joy_six_axis_sensor(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _joy_xpad_id: u32,
) {
}
pub fn get_joy_six_axis_sensor_lifo_handle(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _joy_xpad_id: u32,
) -> u32 {
    new_event(kernel)
}
pub fn start_six_axis_sensor(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _handle: u32, _aruid: u64) {
}
pub fn stop_six_axis_sensor(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _handle: u32, _aruid: u64) {}

const SIX_AXIS_CALIBRATION_SIZE: usize = 0x744;

fn six_axis_device_key(handle: u32) -> u32 {
    handle & 0x00FF_FF00
}

fn write_output(kernel: &mut Kernel, ctx: &IpcCtx, bytes: &[u8]) {
    let target = ctx
        .recv_buffers
        .iter()
        .chain(ctx.recv_statics.iter())
        .find(|buffer| buffer.size > 0 && buffer.addr != 0)
        .copied();
    if let Some(buffer) = target {
        let mut out = vec![0u8; (buffer.size as usize).min(0x1000)];
        let len = bytes.len().min(out.len());
        out[..len].copy_from_slice(&bytes[..len]);
        let _ = kernel.address_space.write(buffer.addr, &out);
    }
}

fn read_input(kernel: &mut Kernel, ctx: &IpcCtx, size: usize) -> Option<Vec<u8>> {
    let source = ctx
        .send_buffers
        .iter()
        .chain(ctx.send_statics.iter())
        .find(|buffer| buffer.size > 0 && buffer.addr != 0)
        .copied()?;
    let mut bytes = vec![0u8; (source.size as usize).min(size)];
    kernel.address_space.read(source.addr, &mut bytes).ok()?;
    Some(bytes)
}

fn six_axis_ic_information() -> Vec<u8> {
    let gyro_min = [0.95f32, -0.003, -0.003, -0.003, 0.95, -0.003, -0.003, -0.003, 0.95];
    let gyro_max = [1.05f32, 0.003, 0.003, 0.003, 1.05, 0.003, 0.003, 0.003, 1.05];
    let accel_min = [0.95f32, -0.016, -0.016, -0.016, 0.95, -0.016, -0.016, -0.016, 0.95];
    let accel_max = [1.05f32, 0.016, 0.016, 0.016, 1.05, 0.016, 0.016, 0.016, 1.05];
    let mut values = Vec::with_capacity(50);
    values.push(2000.0f32);
    values.extend_from_slice(&[-10.0, -10.0, -10.0, 10.0, 10.0, 10.0]);
    values.extend_from_slice(&gyro_min);
    values.extend_from_slice(&gyro_max);
    values.push(8.0);
    values.extend_from_slice(&[-0.0612, -0.0612, -0.0612, 0.0612, 0.0612, 0.0612]);
    values.extend_from_slice(&accel_min);
    values.extend_from_slice(&accel_max);
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

pub fn is_six_axis_sensor_fusion_enabled(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _handle: u32,
    _aruid: u64,
) -> bool {
    true
}
pub fn enable_six_axis_sensor_fusion(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _enabled: bool,
    _handle: u32,
    _aruid: u64,
) {
}
pub fn set_six_axis_sensor_fusion_parameters(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _handle: u32,
    _f0: u32,
    _f1: u32,
    _aruid: u64,
) {
}
pub fn get_six_axis_sensor_fusion_parameters(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _handle: u32,
    _aruid: u64,
) -> (u32, u32) {
    (0.03f32.to_bits(), 0.4f32.to_bits())
}
pub fn reset_six_axis_sensor_fusion_parameters(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _handle: u32,
    _aruid: u64,
) {
}
pub fn set_accelerometer_parameters(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _handle: u32,
    _f0: u32,
    _f1: u32,
    _aruid: u64,
) {
}
pub fn get_accelerometer_parameters(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _handle: u32,
    _aruid: u64,
) -> (u32, u32) {
    (0, 0)
}
pub fn reset_accelerometer_parameters(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _handle: u32,
    _aruid: u64,
) {
}
pub fn set_accelerometer_play_mode(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _handle: u32,
    _mode: u32,
    _aruid: u64,
) {
}
pub fn get_accelerometer_play_mode(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _handle: u32,
    _aruid: u64,
) -> u32 {
    0
}
pub fn reset_accelerometer_play_mode(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _handle: u32,
    _aruid: u64,
) {
}
pub fn set_gyroscope_zero_drift_mode(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    handle: u32,
    mode: u32,
    _aruid: u64,
) {
    kernel.services.hid.six_axis_zero_drift.insert(handle, mode);
}
pub fn get_gyroscope_zero_drift_mode(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    handle: u32,
    _aruid: u64,
) -> u32 {
    kernel.services.hid.six_axis_zero_drift.get(&handle).copied().unwrap_or(1)
}
pub fn reset_gyroscope_zero_drift_mode(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    handle: u32,
    _aruid: u64,
) {
    kernel.services.hid.six_axis_zero_drift.remove(&handle);
}
pub fn is_six_axis_sensor_at_rest(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    handle: u32,
    _aruid: u64,
) -> bool {
    crate::hid_state::sixaxis_handle_slot(handle)
        .and_then(|(_, index)| crate::hid_state::sixaxis_route(crate::hid_motion::connected_sources())[index])
        .map_or(true, crate::hid_motion::source_at_rest)
}
pub fn is_firmware_update_available_for_six_axis_sensor(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _handle: u32,
    _aruid: u64,
) -> bool {
    false
}
pub fn enable_six_axis_sensor_unaltered_passthrough(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    enabled: bool,
    handle: u32,
    _aruid: u64,
) {
    if enabled {
        kernel.services.hid.six_axis_passthrough.insert(handle);
    } else {
        kernel.services.hid.six_axis_passthrough.remove(&handle);
    }
    crate::hid_state::get_hid_state().lock().set_sixaxis_passthrough(handle, enabled);
    log::debug!("HID::EnableSixAxisSensorUnalteredPassthrough handle={:#x} enabled={}", handle, enabled);
}
pub fn is_six_axis_sensor_unaltered_passthrough_enabled(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    handle: u32,
    _aruid: u64,
) -> bool {
    kernel.services.hid.six_axis_passthrough.contains(&handle)
}
pub fn store_six_axis_sensor_calibration_parameter(
    kernel: &mut Kernel,
    ctx: &mut IpcCtx,
    _s: u32,
    handle: u32,
    _aruid: u64,
) {
    if let Some(bytes) = read_input(kernel, ctx, SIX_AXIS_CALIBRATION_SIZE) {
        kernel.services.hid.six_axis_calibration.insert(six_axis_device_key(handle), bytes);
    }
}
pub fn load_six_axis_sensor_calibration_parameter(
    kernel: &mut Kernel,
    ctx: &mut IpcCtx,
    _s: u32,
    handle: u32,
    _aruid: u64,
) {
    let bytes = kernel
        .services
        .hid
        .six_axis_calibration
        .get(&six_axis_device_key(handle))
        .cloned()
        .unwrap_or_else(|| vec![0u8; SIX_AXIS_CALIBRATION_SIZE]);
    write_output(kernel, ctx, &bytes);
}
pub fn get_six_axis_sensor_ic_information(
    kernel: &mut Kernel,
    ctx: &mut IpcCtx,
    _s: u32,
    _handle: u32,
    _aruid: u64,
) {
    write_output(kernel, ctx, &six_axis_ic_information());
}
pub fn reset_is_six_axis_sensor_device_newly_assigned(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    handle: u32,
    _aruid: u64,
) {
    if crate::hid_state::get_hid_state().lock().reset_sixaxis_newly_assigned(handle) {
        log::info!("HID::ResetIsSixAxisSensorDeviceNewlyAssigned handle={:#x}", handle);
    }
}

pub fn activate_gesture(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _unk: u32, _aruid: u64) {}
pub fn set_gesture_output_ranges(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _w: u32,
    _h: u32,
    _aruid: u64,
) {
}

pub fn set_supported_npad_style_set(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    style_set: u32,
    _aruid: u64,
) {
    kernel.services.hid.npad_style_set = style_set;
    kernel.services.hid.p1_assignment_joy_dual = None;
    crate::hid_state::apply_controller_applet_style(style_set);
    let events: Vec<u32> = kernel.services.hid.style_change_events.clone();
    for h in events {
        kernel.event_signals.insert(h, true);
        kernel.threads.signal_handle(h);
    }
}
pub fn get_supported_npad_style_set(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _aruid: u64,
) -> u32 {
    kernel.services.hid.npad_style_set
}
pub fn set_supported_npad_id_type(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _aruid: u64,
    ids: &[u8],
) {
    let count = ids.len() / 4;
    log::debug!("HID::SetSupportedNpadIdType npad_count={}", count);
}
pub fn activate_npad(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {
    let events: Vec<u32> = kernel.services.hid.style_change_events.clone();
    for h in events {
        kernel.event_signals.insert(h, true);
        kernel.threads.signal_handle(h);
    }
}
pub fn deactivate_npad(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {}
pub fn acquire_npad_style_set_update_event_handle(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _npad_id: u32,
    _aruid: u64,
    _unk: u64,
) -> u32 {
    let h = new_signaled_event(kernel);
    kernel.services.hid.style_change_events.push(h);
    h
}
pub fn disconnect_npad(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _npad_id: u32, _aruid: u64) {}
pub fn get_player_led_pattern(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, npad_id: u32) -> u64 {
    match npad_id {
        0 => 0b0001,
        1 => 0b0011,
        2 => 0b0111,
        3 => 0b1111,
        _ => 0,
    }
}
pub fn activate_npad_with_revision(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _revision: i32,
    _aruid: u64,
) {
}

pub fn set_npad_joy_hold_type(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _aruid: u64,
    ty: u64,
) {
    kernel.services.hid.set_npad_joy_hold_type(ty);
}
pub fn get_npad_joy_hold_type(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _aruid: u64,
) -> u64 {
    kernel.services.hid.get_npad_joy_hold_type()
}

fn signal_style_change_events(kernel: &mut Kernel) {
    {
        let state = crate::hid_state::get_hid_state();
        let mut hid = state.lock();
        if hid.shmem_va.is_some() {
            let cur = hid.input;
            hid.tick(cur);
        }
    }
    let events: Vec<u32> = kernel.services.hid.style_change_events.clone();
    for h in events {
        kernel.event_signals.insert(h, true);
        kernel.threads.signal_handle(h);
    }
}

pub fn set_npad_joy_assignment_mode_single_by_default(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    npad_id: u32,
    aruid: u64,
) {
    kernel
        .services
        .hid
        .set_npad_assignment_single_by_default(npad_id, aruid);
    signal_style_change_events(kernel);
}
pub fn set_npad_joy_assignment_mode_single(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    npad_id: u32,
    aruid: u64,
    device_type: i64,
) {
    kernel
        .services
        .hid
        .set_npad_assignment_single(npad_id, aruid, device_type);
    signal_style_change_events(kernel);
}
pub fn set_npad_joy_assignment_mode_dual(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    npad_id: u32,
    aruid: u64,
) {
    kernel.services.hid.set_npad_assignment_dual(npad_id, aruid);
    signal_style_change_events(kernel);
}
pub fn merge_single_joy_as_dual_joy(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    npad_id_l: u32,
    npad_id_r: u32,
    aruid: u64,
) {
    kernel
        .services
        .hid
        .merge_single_joy_as_dual_joy(npad_id_l, npad_id_r, aruid);
    signal_style_change_events(kernel);
}

pub fn start_lr_assignment_mode(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {}
pub fn stop_lr_assignment_mode(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {}

pub fn set_npad_handheld_activation_mode(
    kernel: &mut Kernel,
    _ctx: &mut IpcCtx,
    _session: u32,
    _aruid: u64,
    mode: u64,
) {
    kernel.services.hid.set_npad_handheld_activation_mode(mode);
}
pub fn get_npad_handheld_activation_mode(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _aruid: u64,
) -> i64 {
    kernel.services.hid.npad_handheld_activation_mode as i64
}
pub fn swap_npad_assignment(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _a: u32,
    _b: u32,
    _aruid: u64,
) {
}
pub fn is_unintended_home_button_input_protection_enabled(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _npad_id: u32,
    _aruid: u64,
) -> bool {
    false
}
pub fn enable_unintended_home_button_input_protection(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _enabled: bool,
    _npad_id: u32,
    _aruid: u64,
) {
}
pub fn set_npad_joy_assignment_mode_single_with_destination(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _npad_id: u32,
    _aruid: u64,
    _device_type: i64,
) -> (bool, u32) {
    (false, 0)
}
pub fn set_npad_analog_stick_use_center_clamp(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _clamp: bool,
    _aruid: u64,
) {
}
pub fn set_npad_capture_button_assignment(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _style: u32,
    _aruid: u64,
    _button_set: u64,
) {
}
pub fn clear_npad_capture_button_assignment(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _aruid: u64,
) {
}
pub fn set_npad_gc_analog_stick8bit_raw_value(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _a: u32,
    _b: u32,
    _aruid: u64,
) {
}

pub fn get_vibration_device_info(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    handle: u32,
) -> (u32, u32) {
    hid_vibration::device_info(handle)
}
pub fn send_vibration_value(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    handle: u32,
    v0: u32,
    v1: u32,
    v2: u32,
    v3: u32,
    _aruid: u64,
) {
    hid_vibration::submit(
        handle,
        VibrationValue::from_words([v0, v1, v2, v3]),
        kernel.services.hid.vibration_allowed(),
    );
}
pub fn get_actual_vibration_value(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    handle: u32,
    _aruid: u64,
) -> Vec<u8> {
    hid_vibration::actual(handle).to_le_bytes().to_vec()
}
pub fn create_active_vibration_device_list(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn permit_vibration(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32, permit: bool) {
    kernel.services.hid.vibration_permitted = permit;
    if !permit && !kernel.services.hid.vibration_session {
        hid_vibration::silence();
    }
}
pub fn is_vibration_permitted(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> bool {
    kernel.services.hid.vibration_permitted
}
pub fn send_vibration_values(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _aruid: u64,
    handles: &[u8],
    values: &[u8],
) {
    hid_vibration::submit_batch(handles, values, kernel.services.hid.vibration_allowed());
}
pub fn send_vibration_gc_erm_command(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    handle: u32,
    _aruid: u64,
    cmd: u64,
) {
    hid_vibration::submit_erm(handle, cmd, kernel.services.hid.vibration_allowed());
}
pub fn get_actual_vibration_gc_erm_command(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    handle: u32,
    _aruid: u64,
) -> u64 {
    hid_vibration::erm_command(handle)
}
pub fn begin_permit_vibration_session(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {
    kernel.services.hid.vibration_session = true;
}
pub fn end_permit_vibration_session(kernel: &mut Kernel, _c: &mut IpcCtx, _s: u32) {
    kernel.services.hid.vibration_session = false;
    if !kernel.services.hid.vibration_permitted {
        hid_vibration::silence();
    }
}
pub fn is_vibration_device_mounted(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _handle: u32,
    _aruid: u64,
) -> bool {
    true
}
pub fn send_vibration_value_in_bool(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    value: bool,
    handle: u32,
    _aruid: u64,
) {
    hid_vibration::submit_bool(handle, value, kernel.services.hid.vibration_allowed());
}
pub fn send_vibration_value_in_mode(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _handle: u32,
    _v0: u32,
    _v1: u32,
    _v2: u32,
    _v3: u32,
    _aruid: u64,
) {
}
pub fn send_vibration_values_in_mode(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _aruid: u64,
    _mode: u64,
) {
}

pub fn activate_console_six_axis_sensor(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {}
pub fn start_console_six_axis_sensor(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _handle: u64,
    _aruid: u64,
) {
}
pub fn stop_console_six_axis_sensor(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _handle: u64,
    _aruid: u64,
) {
}
pub fn activate_seven_six_axis_sensor(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {}
pub fn start_seven_six_axis_sensor(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {}
pub fn stop_seven_six_axis_sensor(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {}
pub fn initialize_seven_six_axis_sensor(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _aruid: u64,
    _t0: u64,
    _t1: u64,
) {
}
pub fn finalize_seven_six_axis_sensor(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {}
pub fn set_seven_six_axis_sensor_fusion_strength(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _f: u32,
    _aruid: u64,
) {
}
pub fn get_seven_six_axis_sensor_fusion_strength(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _aruid: u64,
) -> u32 {
    0
}
pub fn reset_seven_six_axis_sensor_timestamp(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _aruid: u64,
) {
}
pub fn force_activate_console_six_axis_sensor(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _aruid: u64,
) {
}
pub fn force_deactivate_console_six_axis_sensor(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _aruid: u64,
) {
}

pub fn enable_npad_imu(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _npad_id: u32, _aruid: u64) {}
pub fn disable_npad_imu(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {}

pub fn is_usb_full_key_controller_enabled(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> bool {
    false
}
pub fn enable_usb_full_key_controller(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _enabled: bool) {}
pub fn is_usb_full_key_controller_connected(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _npad_id: u32,
) -> bool {
    false
}
pub fn has_battery(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _npad_id: u32) -> bool {
    false
}
pub fn has_left_right_battery(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _npad_id: u32,
) -> (bool, bool) {
    (false, false)
}
pub fn get_npad_interface_type(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _npad_id: u32) -> u8 {
    1
}
pub fn get_npad_left_right_interface_type(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _npad_id: u32,
) -> (u8, u8) {
    (1, 1)
}
pub fn get_npad_of_highest_battery_level(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _aruid: u64,
) -> u32 {
    0
}

pub fn get_palma_connection_handle(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _npad_id: u32,
    _aruid: u64,
) -> u64 {
    0
}
pub fn initialize_palma(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _palma: u64) {}
pub fn acquire_palma_operation_complete_event(
    kernel: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _palma: u64,
) -> u32 {
    new_event(kernel)
}
pub fn get_palma_operation_info(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _palma: u64) -> u64 {
    0
}
pub fn play_palma_activity(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _palma: u64, _unk: u64) {}
pub fn set_palma_fr_mode_type(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _palma: u64, _ty: u64) {}
pub fn read_palma_step(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _palma: u64) {}
pub fn enable_palma_step(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _enabled: bool, _palma: u64) {}
pub fn reset_palma_step(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _palma: u64) {}
pub fn read_palma_application_section(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _palma: u64,
    _off: u64,
    _size: u64,
) {
}
pub fn write_palma_application_section(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _palma: u64,
    _off: u64,
    _size: u64,
) {
}
pub fn read_palma_unique_code(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _palma: u64) {}
pub fn set_palma_unique_code_invalid(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _palma: u64) {}
pub fn write_palma_activity_entry(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _palma: u64,
    _a: u64,
    _b: u64,
    _cc: u64,
    _d: u64,
) {
}
pub fn write_palma_rgb_led_pattern_entry(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _palma: u64,
    _unk: u64,
) {
}
pub fn write_palma_wave_entry(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _palma: u64,
    _wave_set: u64,
    _unk: u64,
    _t: u64,
    _size: u64,
) {
}
pub fn set_palma_data_base_identification_version(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _version: i32,
    _palma: u64,
) {
}
pub fn get_palma_data_base_identification_version(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _palma: u64,
) {
}
pub fn suspend_palma_feature(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _features: u32,
    _palma: u64,
) {
}
pub fn get_palma_operation_result(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _palma: u64) {}
pub fn read_palma_play_log(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _unk: u16, _palma: u64) {}
pub fn reset_palma_play_log(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _unk: u16, _palma: u64) {}
pub fn set_is_palma_all_connectable(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _connectable: bool,
    _aruid: u64,
) {
}
pub fn set_is_palma_paired_connectable(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _connectable: bool,
    _aruid: u64,
) {
}
pub fn pair_palma(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _palma: u64) {}
pub fn set_palma_boost_mode(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _enabled: bool) {}
pub fn cancel_write_palma_wave_entry(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _palma: u64) {}
pub fn enable_palma_boost_mode(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _enabled: bool,
    _aruid: u64,
) {
}
pub fn get_palma_bluetooth_address(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _palma: u64) {}
pub fn set_disallowed_palma_connection(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {}

pub fn set_npad_communication_mode(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _aruid: u64,
    _mode: i64,
) {
}
pub fn get_npad_communication_mode(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) -> i64 {
    3
}
pub fn set_touch_screen_configuration(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _c0: u64,
    _c1: u64,
    _aruid: u64,
) {
}
pub fn is_firmware_update_needed_for_notification(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _unk: i32,
    _aruid: u64,
) -> bool {
    false
}
pub fn set_touch_screen_output_ranges(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _w: i32,
    _h: i32,
    _aruid: u64,
) {
}
pub fn enable_nx_touch_screen_emulation_for_touch_enter(
    _k: &mut Kernel,
    _c: &mut IpcCtx,
    _s: u32,
    _flag: u32,
    _aruid: u64,
) {
}
pub fn activate_digitizer(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _aruid: u64) {}

pub fn get_debug_pad_generic_pad_map(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_debug_pad_generic_pad_map(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn reset_debug_pad_generic_pad_map(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_debug_pad_keyboard_map(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_debug_pad_keyboard_map(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn reset_debug_pad_keyboard_map(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_full_key_generic_pad_map(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _flag: u32) {}
pub fn set_full_key_generic_pad_map(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _flag: u32) {}
pub fn reset_full_key_generic_pad_map(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _flag: u32) {}
pub fn get_full_key_keyboard_map(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _flag: u32) {}
pub fn set_full_key_keyboard_map(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _flag: u32) {}
pub fn reset_full_key_keyboard_map(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _flag: u32) {}
pub fn get_debug_pad_generic_pad_map_2(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_debug_pad_generic_pad_map_2(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn get_debug_pad_keyboard_map_2(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_debug_pad_keyboard_map_2(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32) {}
pub fn set_mouse_library_version(_k: &mut Kernel, _c: &mut IpcCtx, _s: u32, _v: u64, _aruid: u64) {}
