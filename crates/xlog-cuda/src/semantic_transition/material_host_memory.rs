//! Cold host allocation preflight against the current inherited Linux envelope.

use super::*;

/// Workspace promised by the original pool but not yet entered by its codec.
/// This is not another capacity pool or a measurement of physical backing.
pub(super) struct MaterialHostCommitment {
    budget: SemanticReplayMaterialBudget,
    bytes: u64,
    entered: std::sync::atomic::AtomicBool,
}

impl MaterialHostCommitment {
    pub(super) fn is_entered(&self) -> bool {
        self.entered.load(std::sync::atomic::Ordering::Acquire)
    }

    pub(super) fn new(
        budget: &SemanticReplayMaterialBudget,
        bytes: u64,
    ) -> Result<Self, SemanticTransitionError> {
        let mut pending = budget
            .owner
            .host_commitments
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        *pending = pending
            .checked_add(bytes)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        Ok(Self {
            budget: budget.clone(),
            bytes,
            entered: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub(super) fn enter(&self) -> Result<(), SemanticTransitionError> {
        let mut pending = self
            .budget
            .owner
            .host_commitments
            .lock()
            .map_err(|_| SemanticTransitionError::Poisoned)?;
        if !self.entered.load(std::sync::atomic::Ordering::Acquire) {
            *pending = pending
                .checked_sub(self.bytes)
                .ok_or(SemanticTransitionError::ObservationMismatch)?;
            self.entered
                .store(true, std::sync::atomic::Ordering::Release);
        }
        Ok(())
    }
}

impl Drop for MaterialHostCommitment {
    fn drop(&mut self) {
        if !self.entered.load(std::sync::atomic::Ordering::Acquire) {
            let mut pending = self
                .budget
                .owner
                .host_commitments
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *pending = pending
                .checked_sub(self.bytes)
                .expect("original unentered host commitment");
        }
    }
}

impl SemanticReplayMaterialBudget {
    pub(super) fn require_host_allocation(
        &self,
        additional: u64,
    ) -> Result<(), SemanticTransitionError> {
        #[cfg(target_os = "linux")]
        {
            // Covers the bounded proc buffers and simultaneously live path
            // decoding/growth, including the diagnostic before any new backing.
            let _metadata = self.reserve_auxiliary(128 * 1024)?;
            let pending = *self
                .owner
                .host_commitments
                .lock()
                .map_err(|_| SemanticTransitionError::Poisoned)?;
            let requested = additional
                .checked_add(pending)
                .ok_or(SemanticTransitionError::GenerationExhausted)?;
            let available = linux::headroom()?;
            if requested > available {
                return Err(publication_input_error(&format!(
                    "original host allocation requires {requested} additional/unbacked bytes but the current inherited envelope has {available} bytes"
                )));
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = additional;
        Ok(())
    }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::io::Read;
    use std::path::{Component, Path, PathBuf};

    fn text<'a>(path: &Path, buffer: &'a mut [u8]) -> Result<&'a str, SemanticTransitionError> {
        let mut file = std::fs::File::open(path)
            .map_err(|error| runtime_error("host envelope read", error))?;
        let mut length = 0;
        while length < buffer.len() {
            let count = file
                .read(&mut buffer[length..])
                .map_err(|error| runtime_error("host envelope read", error))?;
            if count == 0 {
                return std::str::from_utf8(&buffer[..length])
                    .map_err(|_| publication_input_error("host envelope is not UTF-8"));
            }
            length += count;
        }
        let mut extra = [0];
        if file
            .read(&mut extra)
            .map_err(|error| runtime_error("host envelope read", error))?
            != 0
        {
            return Err(publication_input_error(
                "host envelope exceeds its original bounded reader",
            ));
        }
        std::str::from_utf8(&buffer[..length])
            .map_err(|_| publication_input_error("host envelope is not UTF-8"))
    }

    fn number(value: &str) -> Result<u64, SemanticTransitionError> {
        value
            .parse()
            .map_err(|_| publication_input_error("host envelope has an invalid byte count"))
    }

    fn limit(path: &Path) -> Result<Option<u64>, SemanticTransitionError> {
        let mut buffer = [0; 64];
        let value = text(path, &mut buffer)?.trim();
        if value == "max" {
            Ok(None)
        } else {
            number(value).map(Some)
        }
    }

    fn current(path: &Path) -> Result<u64, SemanticTransitionError> {
        let mut buffer = [0; 64];
        number(text(path, &mut buffer)?.trim())
    }

    fn mount_path(value: &str) -> Result<PathBuf, SemanticTransitionError> {
        if value.len() > 4096 {
            return Err(publication_input_error(
                "host envelope mount path exceeds its bound",
            ));
        }
        let mut decoded = Vec::with_capacity(value.len());
        let bytes = value.as_bytes();
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index] == b'\\' {
                let escape = bytes.get(index + 1..index + 4).ok_or_else(|| {
                    publication_input_error("host envelope mount escape is incomplete")
                })?;
                let byte = match escape {
                    b"040" => b' ',
                    b"011" => b'\t',
                    b"012" => b'\n',
                    b"134" => b'\\',
                    _ => {
                        return Err(publication_input_error(
                            "host envelope mount escape is unsupported",
                        ));
                    }
                };
                decoded.push(byte);
                index += 4;
            } else {
                decoded.push(bytes[index]);
                index += 1;
            }
        }
        let decoded = String::from_utf8(decoded)
            .map_err(|_| publication_input_error("host envelope mount path is not UTF-8"))?;
        Ok(PathBuf::from(decoded))
    }

    pub(super) fn headroom() -> Result<u64, SemanticTransitionError> {
        let mut memory = [0; 8192];
        let memory = text(Path::new("/proc/meminfo"), &mut memory)?;
        let available = memory
            .lines()
            .find_map(|line| line.strip_prefix("MemAvailable:"))
            .ok_or_else(|| {
                publication_input_error("host envelope has no available-memory observation")
            })?;
        let mut fields = available.split_whitespace();
        let mut available =
            number(fields.next().ok_or_else(|| {
                publication_input_error("available-memory observation is empty")
            })?)?
            .checked_mul(1024)
            .ok_or(SemanticTransitionError::GenerationExhausted)?;
        if fields.next() != Some("kB") || fields.next().is_some() {
            return Err(publication_input_error(
                "available-memory observation has another unit",
            ));
        }
        let mut group = [0; 8192];
        let group = text(Path::new("/proc/self/cgroup"), &mut group)?;
        let group = group
            .lines()
            .find_map(|line| line.strip_prefix("0::"))
            .ok_or_else(|| {
                publication_input_error(
                    "host allocation requires the original unified cgroup hierarchy",
                )
            })?;
        let group = Path::new(group);
        if !group.is_absolute()
            || group
                .components()
                .any(|part| matches!(part, Component::ParentDir | Component::CurDir))
        {
            return Err(publication_input_error(
                "host envelope has an invalid cgroup path",
            ));
        }
        let mut mounts = [0; 65536];
        let mounts = text(Path::new("/proc/self/mountinfo"), &mut mounts)?;
        let mut matched = false;
        for line in mounts.lines() {
            let Some((prefix, suffix)) = line.split_once(" - ") else {
                continue;
            };
            if suffix.split_whitespace().next() != Some("cgroup2") {
                continue;
            }
            let mut fields = prefix.split_whitespace().skip(3);
            let root = mount_path(
                fields
                    .next()
                    .ok_or_else(|| publication_input_error("cgroup mount has no root"))?,
            )?;
            let mount = mount_path(
                fields
                    .next()
                    .ok_or_else(|| publication_input_error("cgroup mount has no target"))?,
            )?;
            let Ok(relative) = group.strip_prefix(&root) else {
                continue;
            };
            let mut scope = mount.join(relative);
            if !scope.starts_with(&mount) {
                return Err(publication_input_error(
                    "cgroup membership escaped its original mount",
                ));
            }
            matched = true;
            loop {
                // memory.current already includes the live source/output and
                // descendants. Never subtract their pool receipts again.
                if scope
                    .join("memory.max")
                    .try_exists()
                    .map_err(|error| runtime_error("host envelope stat", error))?
                {
                    let resident_limit = limit(&scope.join("memory.max"))?;
                    let resident = current(&scope.join("memory.current"))?;
                    let swap_limit = limit(&scope.join("memory.swap.max"))?;
                    let swap = current(&scope.join("memory.swap.current"))?;
                    if let Some(maximum) = resident_limit {
                        available = available.min(maximum.saturating_sub(resident));
                    }
                    if let (Some(resident_limit), Some(swap_limit)) = (resident_limit, swap_limit) {
                        let total = resident_limit
                            .checked_add(swap_limit)
                            .ok_or(SemanticTransitionError::GenerationExhausted)?;
                        let used = resident
                            .checked_add(swap)
                            .ok_or(SemanticTransitionError::GenerationExhausted)?;
                        available = available.min(total.saturating_sub(used));
                    }
                } else if scope != mount {
                    return Err(publication_input_error(
                        "original cgroup has no memory-controller envelope",
                    ));
                }
                if scope == mount {
                    break;
                }
                if !scope.pop() || !scope.starts_with(&mount) {
                    return Err(publication_input_error(
                        "cgroup ancestor left its original mount",
                    ));
                }
            }
        }
        if !matched {
            return Err(publication_input_error(
                "original cgroup membership has no matching unified mount",
            ));
        }
        Ok(available)
    }
}
