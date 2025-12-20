use kokoro_tvm::{shutdown, status, synthesize, warmup};

#[test]
fn runtime_uninitialized_flow() {
    let current = status().expect("status call failed");
    assert!(!current.initialized);

    let warmup_err = warmup().expect_err("warmup should fail before init");
    assert!(
        warmup_err.to_string().contains("not initialized"),
        "unexpected warmup error: {warmup_err}"
    );

    let synth_err = synthesize("a", 1.0).expect_err("synthesize should fail before init");
    assert!(
        synth_err.to_string().contains("not initialized"),
        "unexpected synth error: {synth_err}"
    );

    shutdown().expect("shutdown should succeed even if not initialized");
    let after = status().expect("status call failed");
    assert!(!after.initialized);
}
