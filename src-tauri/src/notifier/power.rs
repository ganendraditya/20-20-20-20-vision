/// OS-Level Display Sleep Prevention (Presence-Aware Smart Keep-Awake)
/// Holds display awake only when the user is actively present and looking at screen.
/// Gracefully releases the sleep assertion when the user looks away, leaves, or turns off monitoring.

#[derive(Default)]
pub struct SleepBlocker {
    #[cfg(target_os = "macos")]
    assertion_id: Option<u32>,
    #[cfg(target_os = "windows")]
    is_active: bool,
}

impl SleepBlocker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Acquire native OS display sleep inhibition
    pub fn acquire(&mut self) {
        #[cfg(target_os = "macos")]
        {
            if self.assertion_id.is_none() {
                if let Some(id) = macos_impl::create_display_assertion() {
                    self.assertion_id = Some(id);
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            if !self.is_active {
                windows_impl::set_display_required(true);
                self.is_active = true;
            }
        }
    }

    /// Release native OS display sleep inhibition, allowing normal OS power management
    pub fn release(&mut self) {
        #[cfg(target_os = "macos")]
        {
            if let Some(id) = self.assertion_id.take() {
                macos_impl::release_display_assertion(id);
            }
        }

        #[cfg(target_os = "windows")]
        {
            if self.is_active {
                windows_impl::set_display_required(false);
                self.is_active = false;
            }
        }
    }

    /// Returns true if sleep inhibitor is currently holding display awake
    pub fn is_active(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            self.assertion_id.is_some()
        }
        #[cfg(target_os = "windows")]
        {
            self.is_active
        }
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            false
        }
    }
}

impl Drop for SleepBlocker {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(target_os = "macos")]
mod macos_impl {
    use std::ffi::c_void;

    // Opaque Core Foundation type — never constructed or dereferenced directly.
    enum CFString {}
    type CFStringRef = *const CFString;
    type IOPMAssertionID = u32;
    type IOReturn = i32;

    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        fn IOPMAssertionCreateWithName(
            assertion_type: CFStringRef,
            assertion_level: u32,
            assertion_name: CFStringRef,
            assertion_id: *mut IOPMAssertionID,
        ) -> IOReturn;

        fn IOPMAssertionRelease(assertion_id: IOPMAssertionID) -> IOReturn;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFStringCreateWithCString(
            alloc: *const c_void,
            c_str: *const i8,
            encoding: u32,
        ) -> CFStringRef;
        fn CFRelease(cf: *const c_void);
    }

    const K_CF_STRING_ENCODING_UTF8: u32 = 0x08000100;
    const K_IOPM_ASSERTION_LEVEL_ON: u32 = 255;

    pub fn create_display_assertion() -> Option<u32> {
        let type_str = std::ffi::CString::new("PreventUserIdleDisplaySleep").ok()?;
        let name_str = std::ffi::CString::new("420vision: Attention-Aware Display Sleep Prevention").ok()?;

        unsafe {
            let cf_type = CFStringCreateWithCString(std::ptr::null(), type_str.as_ptr(), K_CF_STRING_ENCODING_UTF8);
            let cf_name = CFStringCreateWithCString(std::ptr::null(), name_str.as_ptr(), K_CF_STRING_ENCODING_UTF8);

            if cf_type.is_null() || cf_name.is_null() {
                if !cf_type.is_null() { CFRelease(cf_type as *const c_void); }
                if !cf_name.is_null() { CFRelease(cf_name as *const c_void); }
                return None;
            }

            let mut assertion_id: IOPMAssertionID = 0;
            let ret = IOPMAssertionCreateWithName(cf_type, K_IOPM_ASSERTION_LEVEL_ON, cf_name, &mut assertion_id);

            CFRelease(cf_type as *const c_void);
            CFRelease(cf_name as *const c_void);

            if ret == 0 {
                Some(assertion_id)
            } else {
                eprintln!("[420vision::power] Failed to create IOPM assertion: {}", ret);
                None
            }
        }
    }

    pub fn release_display_assertion(id: u32) {
        unsafe {
            let ret = IOPMAssertionRelease(id);
            if ret != 0 {
                eprintln!("[420vision::power] Failed to release IOPM assertion: {}", ret);
            }
        }
    }
}

#[cfg(target_os = "windows")]
mod windows_impl {
    const ES_CONTINUOUS: u32 = 0x80000000;
    const ES_DISPLAY_REQUIRED: u32 = 0x00000002;

    extern "system" {
        fn SetThreadExecutionState(es_flags: u32) -> u32;
    }

    /// Set thread display execution state on Windows.
    /// Note: SetThreadExecutionState is thread-affine; acquire() and release()
    /// must be called from the same thread context.
    pub fn set_display_required(enable: bool) {
        unsafe {
            let flags = if enable {
                ES_CONTINUOUS | ES_DISPLAY_REQUIRED
            } else {
                ES_CONTINUOUS
            };
            let prev = SetThreadExecutionState(flags);
            if prev == 0 {
                eprintln!("[420vision::power] SetThreadExecutionState failed (returned 0)");
            }
        }
    }
}
