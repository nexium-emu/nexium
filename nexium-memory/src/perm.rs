use bitflags::bitflags;

bitflags! {
    #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
    pub struct Perm: u8 {
        const R = 0b001;
        const W = 0b010;
        const X = 0b100;
    }
}

impl Perm {
    pub const RO: Perm = Perm::R;
    pub const RW: Perm = Perm::R.union(Perm::W);
    pub const RX: Perm = Perm::R.union(Perm::X);
    pub const RWX: Perm = Perm::R.union(Perm::W).union(Perm::X);
}

impl core::fmt::Display for Perm {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let r = if self.contains(Perm::R) { 'r' } else { '-' };
        let w = if self.contains(Perm::W) { 'w' } else { '-' };
        let x = if self.contains(Perm::X) { 'x' } else { '-' };
        write!(f, "{r}{w}{x}")
    }
}
