fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        && std::env::var_os("CARGO_CFG_DEBUG_ASSERTIONS").is_some()
    {
        cc::Build::new()
            .file("src/capture/reference_macos.m")
            .flag("-fobjc-arc")
            .compile("vision420_reference");
        for framework in ["AVFoundation", "CoreMedia", "CoreVideo", "Foundation"] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
        println!("cargo:rerun-if-changed=src/capture/reference_macos.m");
    }
    tauri_build::build()
}
