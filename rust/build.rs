use std::env;
use std::path::PathBuf;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();
    let is_ios = target.contains("apple-ios");
    let enable = is_ios || env::var("KOKORO_TVM_LINK_TVM").ok().as_deref() == Some("1");
    if !enable {
        return;
    }

    let ios_dynamic = env::var("KOKORO_TVM_IOS_DYNAMIC").ok().as_deref() == Some("1");
    let use_dynamic = is_ios && ios_dynamic;

    if is_ios {
        let mode = if use_dynamic { "dynamic" } else { "static" };
        println!(
            "cargo:warning=kokoro-tvm: iOS build detected ({} linking). If you see \
             `ffi.ModuleLoadFromFile not found`, ensure tvm-ffi/runtime are linked \
             and not dead-stripped (static) or bundled with rpath (dynamic).",
            mode
        );
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let suffix = if is_ios { "build-ios" } else { "build" };

    let tvm_dir = PathBuf::from(env::var("TVM_BUILD_DIR").unwrap_or_else(|_| {
        manifest_dir
            .join("..")
            .join("reference")
            .join("tvm")
            .join(suffix)
            .to_string_lossy()
            .to_string()
    }));
    let tvm_ffi_dir = PathBuf::from(env::var("TVM_FFI_BUILD_DIR").unwrap_or_else(|_| {
        manifest_dir
            .join("..")
            .join("reference")
            .join("tvm")
            .join("3rdparty")
            .join("tvm-ffi")
            .join(suffix)
            .to_string_lossy()
            .to_string()
    }));

    println!("cargo:rerun-if-env-changed=KOKORO_TVM_LINK_TVM");
    println!("cargo:rerun-if-env-changed=KOKORO_TVM_IOS_DYNAMIC");
    println!("cargo:rerun-if-env-changed=TVM_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=TVM_FFI_BUILD_DIR");

    println!("cargo:rustc-link-search=native={}", tvm_dir.display());
    println!("cargo:rustc-link-search=native={}", tvm_dir.join("lib").display());
    println!("cargo:rustc-link-search=native={}", tvm_ffi_dir.display());
    println!("cargo:rustc-link-search=native={}", tvm_ffi_dir.join("lib").display());

    if use_dynamic {
        println!("cargo:rustc-link-lib=dylib=tvm_runtime");
        println!("cargo:rustc-link-lib=dylib=tvm_ffi");
        if target.contains("apple") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/Frameworks");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/Frameworks");
        }
    } else {
        println!("cargo:rustc-link-lib=static=tvm_runtime");
        let ffi_lib = if is_ios { "tvm_ffi_static" } else { "tvm_ffi" };
        println!("cargo:rustc-link-lib=static={}", ffi_lib);
        if is_ios {
            force_load_ios_static(&tvm_dir, "libtvm_runtime.a");
            force_load_ios_static(&tvm_ffi_dir, "libtvm_ffi_static.a");
        }
    }

    if target.contains("apple") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=c++");
    }
}

fn force_load_ios_static(base_dir: &PathBuf, lib_name: &str) {
    let candidates = [
        base_dir.join(lib_name),
        base_dir.join("lib").join(lib_name),
    ];
    if let Some(path) = candidates.iter().find(|path| path.exists()) {
        println!("cargo:rustc-link-arg=-Wl,-force_load,{}", path.display());
    } else {
        println!(
            "cargo:warning=Missing {} for force-load in {}",
            lib_name,
            base_dir.display()
        );
    }
}
