#[derive(Copy, Clone, Debug)]
pub struct HidState {
    pub buttons: u32,
    pub touch_x: u32,
    pub touch_y: u32,
    pub touch_pressed: bool,
}

pub struct HidShared {
    pub state: HidState,
}

impl HidShared {
    pub fn new() -> Self {
        Self {
            state: HidState {
                buttons: 0,
                touch_x: 0,
                touch_y: 0,
                touch_pressed: false,
            },
        }
    }

    pub fn update_state(&mut self, state: HidState) {
        self.state = state;
    }

    pub fn get_state(&self) -> HidState {
        self.state
    }
}

impl Default for HidShared {
    fn default() -> Self {
        Self::new()
    }
}
