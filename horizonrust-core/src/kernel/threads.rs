#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ThreadState {
    Created,
    Ready,
    Running,
    Sleeping,
    WaitingHandle,
    Exited,
}

pub struct ThreadCtx {
    pub id: u32,
    pub state: ThreadState,
    pub regs: [u64; 31],
    pub pc: u64,
    pub sp: u64,
    pub tpidrro_el0: u64,
}

pub struct Threads {
    threads: Vec<ThreadCtx>,
}

impl Threads {
    pub fn new() -> Self {
        Self { threads: Vec::new() }
    }

    pub fn create_thread(&mut self, id: u32) {
        let ctx = ThreadCtx {
            id,
            state: ThreadState::Created,
            regs: [0u64; 31],
            pc: 0,
            sp: 0,
            tpidrro_el0: 0,
        };
        self.threads.push(ctx);
    }

    pub fn get_thread(&self, id: u32) -> Option<&ThreadCtx> {
        self.threads.iter().find(|t| t.id == id)
    }

    pub fn get_thread_mut(&mut self, id: u32) -> Option<&mut ThreadCtx> {
        self.threads.iter_mut().find(|t| t.id == id)
    }
}

impl Default for Threads {
    fn default() -> Self {
        Self::new()
    }
}
