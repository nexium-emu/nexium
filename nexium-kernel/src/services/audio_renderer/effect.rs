#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum UsageState {
    #[default]
    Invalid,
    New,
    Enabled,
    Disabled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EffectState {
    effect_type: u8,
    usage: UsageState,
    status: u8,
}

impl Default for EffectState {
    fn default() -> Self {
        Self {
            effect_type: 0,
            usage: UsageState::Invalid,
            status: 4,
        }
    }
}

impl EffectState {
    pub fn update(
        &mut self,
        effect_type: u8,
        is_new: bool,
        enabled: bool,
        renderer_active: bool,
    ) -> u8 {
        if effect_type != self.effect_type {
            *self = Self {
                effect_type,
                ..Self::default()
            };
        }
        if effect_type == 0 {
            *self = Self::default();
            return self.status;
        }
        if is_new {
            self.usage = UsageState::New;
        }
        self.status = if renderer_active {
            if self.usage == UsageState::Disabled {
                4
            } else {
                3
            }
        } else if self.usage == UsageState::New {
            3
        } else {
            4
        };
        if renderer_active {
            self.usage = if enabled {
                UsageState::Enabled
            } else {
                UsageState::Disabled
            };
        }
        self.status
    }
}

#[cfg(test)]
mod tests {
    use super::EffectState;

    #[test]
    fn enabled_effect_becomes_removable_after_disabled_update() {
        let mut effect = EffectState::default();
        assert_eq!(effect.update(2, true, true, true), 3);
        assert_eq!(effect.update(2, false, true, true), 3);
        assert_eq!(effect.update(2, false, false, true), 3);
        assert_eq!(effect.update(2, false, false, true), 4);
        assert_eq!(effect.update(2, false, false, true), 4);
    }

    #[test]
    fn new_disabled_effect_waits_for_active_processing() {
        let mut effect = EffectState::default();
        assert_eq!(effect.update(3, true, false, false), 3);
        assert_eq!(effect.update(3, false, false, false), 3);
        assert_eq!(effect.update(3, false, false, true), 3);
        assert_eq!(effect.update(3, false, false, true), 4);
    }

    #[test]
    fn stopped_renderer_releases_processed_effects() {
        let mut effect = EffectState::default();
        assert_eq!(effect.update(5, true, true, true), 3);
        assert_eq!(effect.update(5, false, true, false), 4);
        assert_eq!(effect.update(5, false, true, true), 3);
    }

    #[test]
    fn reused_effect_slot_clears_disabled_usage() {
        let mut effect = EffectState::default();
        assert_eq!(effect.update(2, true, false, true), 3);
        assert_eq!(effect.update(2, false, false, true), 4);
        assert_eq!(effect.update(2, true, true, true), 3);
        assert_eq!(effect.update(2, false, false, true), 3);
        assert_eq!(effect.update(2, false, false, true), 4);
        assert_eq!(effect.update(3, false, true, true), 3);
        assert_eq!(effect.update(0, false, false, true), 4);
        assert_eq!(effect.update(3, false, true, true), 3);
    }

    #[test]
    fn disabling_one_effect_preserves_other_active_slots() {
        let mut effects = [EffectState::default(); 2];
        for effect in &mut effects {
            assert_eq!(effect.update(2, true, true, true), 3);
        }
        for expected in [3, 4, 4] {
            assert_eq!(effects[0].update(2, false, false, true), expected);
            assert_eq!(effects[1].update(2, false, true, true), 3);
        }
    }
}
