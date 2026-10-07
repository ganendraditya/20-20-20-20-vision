use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

static SILENT_MODE: AtomicBool = AtomicBool::new(false);

pub struct Notifier;

impl Notifier {
    /// Toggle silent mode (used during testing to prevent intrusive OS notifications)
    pub fn set_silent_mode(silent: bool) {
        SILENT_MODE.store(silent, Ordering::Relaxed);
    }

    /// Check if silent mode is enabled
    pub fn is_silent() -> bool {
        SILENT_MODE.load(Ordering::Relaxed)
    }

    /// Send non-blocking native desktop notification
    pub fn send(title: &str, message: &str) {
        if Self::is_silent() {
            return;
        }

        let title_owned = title.to_string();
        let message_owned = message.to_string();

        thread::spawn(move || {
            #[cfg(target_os = "macos")]
            {
                // Native macOS notification banner via osascript using positional argv
                // (Prevents AppleScript injection: strings are passed as raw argv, never interpolated)
                let script = r#"
                    on run argv
                        display notification (item 2 of argv) with title (item 1 of argv)
                    end run
                "#;
                let _ = Command::new("osascript")
                    .arg("-e")
                    .arg(script)
                    .arg(&title_owned)
                    .arg(&message_owned)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status();
            }

            #[cfg(target_os = "windows")]
            {
                // Native Windows Action Center toast via PowerShell using environment variables & CREATE_NO_WINDOW
                let script = r#"
                    [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
                    $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
                    $textNodes = $template.GetElementsByTagName("text")
                    $textNodes.Item(0).AppendChild($template.CreateTextNode($env:TOAST_TITLE)) > $null
                    $textNodes.Item(1).AppendChild($template.CreateTextNode($env:TOAST_MSG)) > $null
                    $toast = [Windows.UI.Notifications.ToastNotification]::new($template)
                    $appId = "{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
                    try {
                        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("com.ganendraditya.vision420").Show($toast)
                    } catch {
                        [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($appId).Show($toast)
                    }
                "#;
                let mut cmd = Command::new("powershell");
                cmd.args(["-WindowStyle", "Hidden", "-Command", script])
                    .env("TOAST_TITLE", &title_owned)
                    .env("TOAST_MSG", &message_owned)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null());

                #[cfg(target_os = "windows")]
                {
                    use std::os::windows::process::CommandExt;
                    cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW to eliminate black console flash
                }

                let _ = cmd.status();
            }
        });
    }

    pub fn notify_stare_warning() {
        Self::send(
            "Blink Prompt (Dry Eye Alert)",
            "You have stared for >8 seconds without blinking. Rest your eyelids briefly.",
        );
    }

    pub fn notify_break_time() {
        Self::send(
            "20-20-20 Break Time",
            "20 minutes of screen presence reached. Look at an object at least 20 feet (6 meters) away for 20 seconds.",
        );
    }
}
