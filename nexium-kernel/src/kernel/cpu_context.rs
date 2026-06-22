pub struct CpuContext {
    pub x_regs: [u64; 31],
    pub pc: u64,
    pub sp: u64,
}

impl CpuContext {
    pub fn new() -> Self {
        Self {
            x_regs: [0; 31],
            pc: 0,
            sp: 0,
        }
    }

    pub fn get_arg(&self, idx: usize) -> u64 {
        if idx < 8 {
            self.x_regs[idx]
        } else {
            0
        }
    }

    pub fn set_ret0(&mut self, val: u64) {
        self.x_regs[0] = val;
    }

    pub fn set_ret1(&mut self, val: u64) {
        self.x_regs[1] = val;
    }

    pub fn set_ret2(&mut self, val: u64) {
        self.x_regs[2] = val;
    }

    pub fn set_ret3(&mut self, val: u64) {
        self.x_regs[3] = val;
    }
}

impl Default for CpuContext {
    fn default() -> Self {
        Self::new()
    }
}
