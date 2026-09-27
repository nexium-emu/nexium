use std::sync::Arc;

use nexium_memory::{AddressSpace, HostRegionChange, HostRegionLease};

use crate::{Cpu, CpuBackendKind};

#[derive(Clone, Debug, Default)]
pub struct CpuSystemConfig {
    #[cfg(feature = "backend-rustarmic")]
    pub rustarmic: ::rustarmic::EngineConfig,
}

#[cfg(feature = "backend-rustarmic")]
impl From<::rustarmic::EngineConfig> for CpuSystemConfig {
    fn from(rustarmic: ::rustarmic::EngineConfig) -> Self {
        Self { rustarmic }
    }
}

pub struct CpuSystem {
    backend: CpuBackendKind,
    address_space: Arc<AddressSpace>,
    config: CpuSystemConfig,
}

impl CpuSystem {
    pub fn new<C: Into<CpuSystemConfig>>(
        backend: CpuBackendKind,
        address_space: Arc<AddressSpace>,
        config: C,
    ) -> Result<Self, String> {
        if !backend.is_compiled_in() {
            return Err(format!("{} backend is not compiled in", backend.label()));
        }
        Ok(Self {
            backend,
            address_space,
            config: config.into(),
        })
    }

    pub fn backend(&self) -> CpuBackendKind {
        self.backend
    }

    pub fn address_space(&self) -> Arc<AddressSpace> {
        Arc::clone(&self.address_space)
    }

    pub fn create_core(&self, core_id: u64) -> Result<CpuCore, String> {
        let mut cpu = match self.backend {
            CpuBackendKind::Rustarmic => {
                #[cfg(feature = "backend-rustarmic")]
                {
                    Cpu::new_rustarmic_with_config(&self.config.rustarmic)?
                }
                #[cfg(not(feature = "backend-rustarmic"))]
                {
                    return Err("Rustarmic backend not compiled in".to_string());
                }
            }
            CpuBackendKind::Dynarmic | CpuBackendKind::Nce => Cpu::new(self.backend)?,
        };
        cpu.set_core_id(core_id);
        let leases = self.address_space.host_region_leases();
        for region in self.address_space.host_regions() {
            unsafe {
                cpu.map_host(region.base, region.size, region.perm, region.host_ptr)?;
            }
        }
        let generation = self.address_space.generation();
        let _ = &self.config;
        Ok(CpuCore {
            cpu,
            address_space: Arc::clone(&self.address_space),
            generation,
            core_id,
            leases,
        })
    }
}

pub struct CpuCore {
    cpu: Cpu,
    address_space: Arc<AddressSpace>,
    generation: u64,
    core_id: u64,
    leases: Vec<HostRegionLease>,
}

impl CpuCore {
    pub fn core_id(&self) -> u64 {
        self.core_id
    }

    pub fn cpu(&self) -> &Cpu {
        &self.cpu
    }

    pub fn cpu_mut(&mut self) -> &mut Cpu {
        &mut self.cpu
    }

    pub fn sync_mappings(&mut self) -> Result<(), String> {
        let changes = self
            .address_space
            .host_region_changes_since(self.generation);
        for change in changes.changes {
            match change {
                HostRegionChange::Upsert(region) => {
                    unsafe {
                        let _ = self.cpu.unmap_host(region.base, region.size);
                        self.cpu.map_host(
                            region.base,
                            region.size,
                            region.perm,
                            region.host_ptr,
                        )?;
                    }
                    self.leases.retain(|lease| lease.base() != region.base);
                    if let Some(lease) = self.address_space.host_region_lease_at(region.base) {
                        self.leases.push(lease);
                    }
                }
                HostRegionChange::Invalidate { base, size } => {
                    self.cpu.invalidate_range(base, size);
                }
                HostRegionChange::Remove { base, size } => unsafe {
                    let _ = self.cpu.unmap_host(base, size);
                    self.leases.retain(|lease| lease.base() != base);
                },
            }
        }
        self.generation = changes.generation;
        Ok(())
    }
}
