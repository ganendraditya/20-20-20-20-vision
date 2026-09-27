use std::process::Command;
use std::thread;

pub struct Notifier;

impl Notifier {
    /// Send non-blocking native desktop notification
    pub fn send(title: &str, message: &str) {
        let title_owned = title.to_string();
        let message_owned = message.to_string();

        thread::spawn(move || {
            #[cfg(target_os = "macos")]
            {
                // Native macOS notification banner via osascript
                let clean_title = title_owned.replace('"', "\\\"");
                let clean_msg = message_owned.replace('"', "\\\"");
                let script = format!(
                    r#"display notification "{}" with title "{}""#,
                    clean_msg, clean_title
                );
                let _ = Command::new("osascript")
                    .arg("-e")
                    .arg(script)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status();
            }

            #[cfg(target_os = "windows")]
            {
                // Native Windows Action Center toast via PowerShell BurntToast or Windows.UI.Notifications
                let clean_title = title_owned.replace('"', "`\"");
                let clean_msg = message_owned.replace('"', "`\"");
                let script = format!(
                    r#"[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null; $template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02); $textNodes = $template.GetElementsByTagName("text"); $textNodes.Item(0).AppendChild($template.CreateTextNode("{}")) > $null; $textNodes.Item(1).AppendChild($template.CreateTextNode("{}")) > $null; $toast = [Windows.UI.Notifications.ToastNotification]::new($template); [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("420vision").Show($toast);"#,
                    clean_title, clean_msg
                );
                let _ = Command::new("powershell")
                    .args(["-WindowStyle", "Hidden", "-Command", &script])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status();
            }
        });
    }

    pub fn notify_stare_warning() {
        Self::send(
            "👁️ Kedip Yuk! (Dry Eye Alert)",
            "Kamu sudah >8 detik belum kedip. Istirahatkan kelopak matamu sejenak.",
        );
    }

    pub fn notify_break_time() {
        Self::send(
            "✨ 20-20-20-20 Break Time! ✨",
            "20 Menit layar tercapai! Tatap objek sejauh 20 kaki (6m) selama 20 detik & kedip 20x.",
        );
    }
}
