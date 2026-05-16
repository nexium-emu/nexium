use gilrs::Gilrs;

#[derive(Clone, Copy, Debug)]
pub struct InputSnapshot {
    pub a_pressed: bool,
    pub b_pressed: bool,
    pub x_pressed: bool,
    pub y_pressed: bool,
    pub l_pressed: bool,
    pub r_pressed: bool,
    pub zl_pressed: bool,
    pub zr_pressed: bool,
    pub plus_pressed: bool,
    pub minus_pressed: bool,
    pub dpad_up: bool,
    pub dpad_down: bool,
    pub dpad_left: bool,
    pub dpad_right: bool,
    pub stick_left_x: f32,
    pub stick_left_y: f32,
    pub stick_right_x: f32,
    pub stick_right_y: f32,
}

impl InputSnapshot {
    pub fn new() -> Self {
        Self {
            a_pressed: false,
            b_pressed: false,
            x_pressed: false,
            y_pressed: false,
            l_pressed: false,
            r_pressed: false,
            zl_pressed: false,
            zr_pressed: false,
            plus_pressed: false,
            minus_pressed: false,
            dpad_up: false,
            dpad_down: false,
            dpad_left: false,
            dpad_right: false,
            stick_left_x: 0.0,
            stick_left_y: 0.0,
            stick_right_x: 0.0,
            stick_right_y: 0.0,
        }
    }

    pub fn update_from_gamepad(gilrs: &mut Gilrs) -> Self {
        let mut snapshot = InputSnapshot::new();

        for (_id, gamepad) in gilrs.gamepads() {
            use gilrs::Button::*;
            use gilrs::Axis::*;

            snapshot.a_pressed = gamepad.is_pressed(South);
            snapshot.b_pressed = gamepad.is_pressed(East);
            snapshot.x_pressed = gamepad.is_pressed(West);
            snapshot.y_pressed = gamepad.is_pressed(North);
            snapshot.l_pressed = gamepad.is_pressed(LeftTrigger);
            snapshot.r_pressed = gamepad.is_pressed(RightTrigger);
            snapshot.zl_pressed = gamepad.is_pressed(LeftTrigger2);
            snapshot.zr_pressed = gamepad.is_pressed(RightTrigger2);
            snapshot.plus_pressed = gamepad.is_pressed(Start);
            snapshot.minus_pressed = gamepad.is_pressed(Select);
            snapshot.dpad_up = gamepad.is_pressed(DPadUp);
            snapshot.dpad_down = gamepad.is_pressed(DPadDown);
            snapshot.dpad_left = gamepad.is_pressed(DPadLeft);
            snapshot.dpad_right = gamepad.is_pressed(DPadRight);

            snapshot.stick_left_x = gamepad.value(LeftStickX);
            snapshot.stick_left_y = gamepad.value(LeftStickY);
            snapshot.stick_right_x = gamepad.value(RightStickX);
            snapshot.stick_right_y = gamepad.value(RightStickY);

            break;
        }

        snapshot
    }
}

impl Default for InputSnapshot {
    fn default() -> Self {
        Self::new()
    }
}
