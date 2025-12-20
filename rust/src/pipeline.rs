//! TVM inference pipeline for Kokoro TTS.
//!
//! NOTE: This module is a work-in-progress. The tvm-ffi 0.1.0-alpha.0 crate
//! has API limitations that need to be addressed before full functionality.

use crate::constants::{SAMPLES_PER_FRAME, STATIC_AUDIO_LEN, STATIC_TEXT_LEN};
use crate::preprocessing::{build_alignment_with_pred, create_masks, pad_input_ids};
use anyhow::Result;
use ndarray::{s, Array1, Array2, Array3, ArrayD, ArrayView3};
use std::collections::HashMap;
use std::ffi::c_void;
use std::mem;
use std::path::{Path, PathBuf};
use tvm_ffi::dtype::AsDLDataType;

/// Helper macro to convert tvm_ffi::Error to anyhow::Error
macro_rules! tvm_err {
    ($expr:expr, $msg:expr) => {
        $expr.map_err(|e| anyhow::anyhow!("{}: {:?}", $msg, e))
    };
}

/// Extract a tensor from a function output.
/// Handles both direct Tensor returns and Array (tuple) returns.
fn extract_tensor_from_output(output: &tvm_ffi::Any, index: usize) -> Result<tvm_ffi::Tensor> {
    // First, try to convert directly to Tensor
    let type_index = output.type_index();
    let output_view = tvm_ffi::AnyView::from(output);

    // Check if it's a Tensor (type index for Tensor)
    if type_index == tvm_ffi::TypeIndex::kTVMFFITensor as i32 {
        return tvm_err!(
            output_view.try_into(),
            "Failed to convert output to Tensor"
        );
    }

    // Otherwise, assume it's an Array and use ffi.ArrayGetItem
    let array_get_item = tvm_err!(
        tvm_ffi::Function::get_global("ffi.ArrayGetItem"),
        "Failed to get ffi.ArrayGetItem function"
    )?;

    // Use call_packed with AnyView since &Any doesn't implement ArgIntoRef
    let index_val = index as i64;
    let args = [output_view, tvm_ffi::AnyView::from(&index_val)];

    let element = tvm_err!(
        array_get_item.call_packed(&args),
        format!("Failed to get element {} from array", index)
    )?;

    tvm_err!(element.try_into(), format!("Failed to convert array element {} to Tensor", index))
}

/// Create a tensor with bool dtype from a slice of u8 values.
/// TVM's bool dtype is kDLBool (code=6) with 8 bits.
fn create_bool_tensor(
    data: &[u8],
    shape: &[i64],
    device: tvm_ffi::DLDevice,
) -> Result<tvm_ffi::Tensor> {
    use tvm_ffi::{DLDataType, DLDataTypeCode};

    // Create bool dtype: kDLBool = 6, 8 bits, 1 lane
    let bool_dtype = DLDataType::new(DLDataTypeCode::kDLBool, 8, 1);
    tensor_from_bytes(data, shape, bool_dtype, device)
}

fn tensor_alloc(
    shape: &[i64],
    dtype: tvm_ffi::DLDataType,
    device: tvm_ffi::DLDevice,
) -> Result<tvm_ffi::Tensor> {
    use tvm_ffi::{CPUNDAlloc, DLDeviceType, Tensor};

    if device.device_type == DLDeviceType::kDLCPU {
        return Ok(Tensor::from_nd_alloc(CPUNDAlloc {}, shape, dtype, device));
    }

    let alloc = tvm_err!(
        tvm_ffi::Function::get_global("runtime.TVMTensorAllocWithScope"),
        "Failed to get runtime.TVMTensorAllocWithScope"
    )?;
    let shape_obj = tvm_ffi::Shape::from(shape);
    let mem_scope: Option<tvm_ffi::String> = None;
    let alloc_args = [
        tvm_ffi::AnyView::from(&shape_obj),
        tvm_ffi::AnyView::from(&dtype),
        tvm_ffi::AnyView::from(&device),
        tvm_ffi::AnyView::from(&mem_scope),
    ];
    let tensor_any = tvm_err!(
        alloc.call_packed(&alloc_args),
        "Failed to allocate device tensor"
    )?;
    tvm_err!(tensor_any.try_into(), "Failed to convert allocated tensor")
}

fn tensor_copy_from_bytes(
    tensor: &tvm_ffi::Tensor,
    data_ptr: *mut c_void,
    nbytes: usize,
) -> Result<()> {
    let copy_from = tvm_err!(
        tvm_ffi::Function::get_global("runtime.TVMTensorCopyFromBytes"),
        "Failed to get runtime.TVMTensorCopyFromBytes"
    )?;
    let copy_args = [
        tvm_ffi::AnyView::from(tensor),
        tvm_ffi::AnyView::from(&data_ptr),
        tvm_ffi::AnyView::from(&nbytes),
    ];
    tvm_err!(
        copy_from.call_packed(&copy_args),
        "Failed to copy data into tensor"
    )?;
    Ok(())
}

fn tensor_copy_to_bytes(
    tensor: &tvm_ffi::Tensor,
    data_ptr: *mut c_void,
    nbytes: usize,
) -> Result<()> {
    let copy_to = tvm_err!(
        tvm_ffi::Function::get_global("runtime.TVMTensorCopyToBytes"),
        "Failed to get runtime.TVMTensorCopyToBytes"
    )?;
    let copy_args = [
        tvm_ffi::AnyView::from(tensor),
        tvm_ffi::AnyView::from(&data_ptr),
        tvm_ffi::AnyView::from(&nbytes),
    ];
    tvm_err!(
        copy_to.call_packed(&copy_args),
        "Failed to copy data from tensor"
    )?;
    Ok(())
}

fn expected_nbytes(shape: &[i64], dtype: tvm_ffi::DLDataType) -> Result<usize> {
    let numel: usize = shape
        .iter()
        .map(|&x| usize::try_from(x).map_err(|_| anyhow::anyhow!("Invalid shape dimension")))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .product();
    let item_size = (dtype.bits as usize * dtype.lanes as usize + 7) / 8;
    Ok(numel * item_size)
}

fn tensor_from_bytes(
    data: &[u8],
    shape: &[i64],
    dtype: tvm_ffi::DLDataType,
    device: tvm_ffi::DLDevice,
) -> Result<tvm_ffi::Tensor> {
    use tvm_ffi::DLDeviceType;

    let expected_size = expected_nbytes(shape, dtype)?;
    if data.len() != expected_size {
        return Err(anyhow::anyhow!(
            "Tensor size mismatch: expected {}, got {}",
            expected_size,
            data.len()
        ));
    }

    let mut tensor = tensor_alloc(shape, dtype, device)?;
    if device.device_type == DLDeviceType::kDLCPU {
        unsafe {
            let dst = tensor.data_ptr_mut() as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
        return Ok(tensor);
    }

    tensor_copy_from_bytes(&tensor, data.as_ptr() as *mut c_void, data.len())?;
    Ok(tensor)
}

fn tensor_from_slice_device<T: AsDLDataType>(
    slice: &[T],
    shape: &[i64],
    device: tvm_ffi::DLDevice,
) -> Result<tvm_ffi::Tensor> {
    use tvm_ffi::{DLDeviceType, Tensor};

    let expected_len: usize = shape
        .iter()
        .map(|&x| usize::try_from(x).map_err(|_| anyhow::anyhow!("Invalid shape dimension")))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .product();
    if slice.len() != expected_len {
        return Err(anyhow::anyhow!(
            "Tensor length mismatch: expected {}, got {}",
            expected_len,
            slice.len()
        ));
    }

    if device.device_type == DLDeviceType::kDLCPU {
        return tvm_err!(
            Tensor::from_slice(slice, shape),
            "Failed to create CPU tensor"
        );
    }

    let dtype = T::DL_DATA_TYPE;
    let tensor = tensor_alloc(shape, dtype, device)?;
    let nbytes = slice.len() * mem::size_of::<T>();
    tensor_copy_from_bytes(&tensor, slice.as_ptr() as *mut c_void, nbytes)?;
    Ok(tensor)
}

fn tensor_to_vec_f32(tensor: &tvm_ffi::Tensor) -> Result<Vec<f32>> {
    use tvm_ffi::{DLDataType, DLDataTypeCode, DLDeviceType};

    let f32_dtype = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);
    if tensor.dtype() != f32_dtype {
        return Err(anyhow::anyhow!(
            "Tensor dtype mismatch: expected f32, got {:?}",
            tensor.dtype()
        ));
    }

    if tensor.device().device_type == DLDeviceType::kDLCPU {
        return Ok(
            tvm_err!(tensor.data_as_slice::<f32>(), "Failed to read tensor data")?.to_vec(),
        );
    }

    let numel = tensor.numel();
    let mut data = vec![0f32; numel];
    let nbytes = numel * mem::size_of::<f32>();
    tensor_copy_to_bytes(tensor, data.as_mut_ptr() as *mut c_void, nbytes)?;
    Ok(data)
}

/// Convert a tensor to an owned ndarray (f32).
fn tensor_to_array_d(tensor: &tvm_ffi::Tensor) -> Result<ArrayD<f32>> {
    let shape: Vec<usize> = tensor.shape().iter().map(|&dim| dim as usize).collect();
    let data = tensor_to_vec_f32(tensor)?;
    ArrayD::from_shape_vec(shape, data)
        .map_err(|e| anyhow::anyhow!("Failed to reshape tensor data: {}", e))
}

/// TVM-based inference pipeline for Kokoro TTS.
///
/// This pipeline orchestrates the following TVM-compiled modules:
/// - BERT encoder: input_ids, attention_mask -> d_en
/// - Duration predictor: d_en, style, lengths, mask -> (duration_logits, d)
/// - F0N predictor: en, style, frame_lengths -> (F0, N)
/// - Text encoder: input_ids, lengths, mask -> t_en
/// - Decoder: asr, F0, N, style -> audio
pub struct KokoroPipeline {
    // Module functions - these are the callable entry points from Relax VMs
    f_bert: tvm_ffi::Function,
    f_duration: tvm_ffi::Function,
    f_f0n: tvm_ffi::Function,
    f_text_enc: tvm_ffi::Function,
    decoder_fns: HashMap<usize, tvm_ffi::Function>,
    decoder_bucket_lens: Vec<usize>,

    // Keep VM modules alive - they own the functions
    #[allow(dead_code)]
    vms: Vec<tvm_ffi::Module>,

    // Device info
    device: tvm_ffi::DLDevice,
}

/// Intermediate tensors captured from a forward pass.
pub struct PipelineTrace {
    pub audio: Vec<f32>,
    pub cur_len: usize,
    pub frames: usize,
    pub d_en: Array3<f32>,
    pub duration_logits: ArrayD<f32>,
    pub d: Array3<f32>,
    pub pred_dur: Array1<i64>,
    pub f0: Array2<f32>,
    pub n: Array2<f32>,
    pub t_en: Array3<f32>,
    pub asr: Array3<f32>,
    pub decoder_bucket_len: usize,
}

// Allocator type constants (matching Python VirtualMachine class)
#[allow(dead_code)]
const NAIVE_ALLOCATOR: i32 = 1;
const POOLED_ALLOCATOR: i32 = 2;


impl KokoroPipeline {
    fn artifact_extension(device: &str) -> &'static str {
        if cfg!(target_os = "macos") {
            if device == "metal" {
                "dylib"
            } else {
                "so"
            }
        } else {
            "so"
        }
    }

    fn required_artifact_paths(base_dir: &Path, ext: &str) -> Vec<PathBuf> {
        vec![
            base_dir.join(format!("bert_compiled.{}", ext)),
            base_dir.join(format!("duration_compiled.{}", ext)),
            base_dir.join(format!("f0n_compiled.{}", ext)),
            base_dir.join(format!("text_encoder_compiled.{}", ext)),
        ]
    }

    fn missing_required_artifacts(base_dir: &Path, ext: &str) -> Vec<PathBuf> {
        Self::required_artifact_paths(base_dir, ext)
            .into_iter()
            .filter(|path| !path.exists())
            .collect()
    }

    fn candidate_artifact_dirs(base_dir: &Path) -> Vec<PathBuf> {
        let base_dir = if base_dir.is_file() {
            base_dir.parent().unwrap_or(base_dir).to_path_buf()
        } else {
            base_dir.to_path_buf()
        };
        let mut candidates = vec![
            base_dir.clone(),
            base_dir.join("tvm_output"),
            base_dir.join("Resources"),
            base_dir.join("Resources").join("tvm_output"),
            base_dir.join("Contents").join("Resources"),
            base_dir.join("Contents").join("Resources").join("tvm_output"),
        ];
        if let Some(parent) = base_dir.parent() {
            candidates.push(parent.to_path_buf());
            candidates.push(parent.join("tvm_output"));
            candidates.push(parent.join("Resources"));
            candidates.push(parent.join("Resources").join("tvm_output"));
        }
        candidates
    }

    /// Resolve a directory containing compiled artifacts, useful for app bundle paths.
    pub fn load_from_artifacts_dir(base_dir: impl AsRef<Path>, device: &str) -> Result<Self> {
        let base_dir = base_dir.as_ref();
        let ext = Self::artifact_extension(device);

        let mut details = Vec::new();
        for candidate in Self::candidate_artifact_dirs(base_dir) {
            let missing = Self::missing_required_artifacts(&candidate, ext);
            if missing.is_empty() {
                return Self::load(&candidate, device);
            }

            let status = if candidate.exists() {
                format!("missing {:?}", missing)
            } else {
                "directory missing".to_string()
            };
            details.push(format!("{:?}: {}", candidate, status));
        }

        Err(anyhow::anyhow!(
            "No compiled artifacts found for device '{}'. Tried:\n{}",
            device,
            details.join("\n")
        ))
    }

    /// Load TVM modules from a directory.
    ///
    /// Args:
    ///     lib_dir: Directory containing compiled .so/.dylib files
    ///     device: Target device ("llvm", "metal", "cuda")
    pub fn load(lib_dir: &Path, device: &str) -> Result<Self> {
        // Try to load libtvm_runtime to register Relax VM loader
        Self::init_relax_runtime()?;

        let ext = Self::artifact_extension(device);

        let device_type = Self::device_type_code(device);
        let device_id = 0i32;
        let device = Self::device_from_str(device)?;

        // Load and wrap encoder modules
        let bert_path = lib_dir.join(format!("bert_compiled.{}", ext));
        let duration_path = lib_dir.join(format!("duration_compiled.{}", ext));
        let f0n_path = lib_dir.join(format!("f0n_compiled.{}", ext));
        let text_enc_path = lib_dir.join(format!("text_encoder_compiled.{}", ext));

        println!("  Loading BERT from {:?}...", bert_path);
        let (bert_vm, f_bert) =
            Self::load_relax_module(device_type, device_id, &bert_path, "bert_forward")?;

        println!("  Loading Duration from {:?}...", duration_path);
        let (duration_vm, f_duration) =
            Self::load_relax_module(device_type, device_id, &duration_path, "duration_forward")?;

        println!("  Loading F0N from {:?}...", f0n_path);
        let (f0n_vm, f_f0n) =
            Self::load_relax_module(device_type, device_id, &f0n_path, "f0n_forward")?;

        println!("  Loading Text Encoder from {:?}...", text_enc_path);
        let (text_enc_vm, f_text_enc) =
            Self::load_relax_module(device_type, device_id, &text_enc_path, "text_encoder_forward")?;


        let mut vms = vec![bert_vm, duration_vm, f0n_vm, text_enc_vm];

        // Load decoder(s) - check for bucketed decoders first
        let mut decoder_fns = HashMap::new();
        let mut decoder_bucket_lens = Vec::new();

        // Try to find bucketed decoders (decoder_compiled_seq256.so, etc.)
        for entry in std::fs::read_dir(lib_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            if name_str.starts_with("decoder_compiled_seq") && name_str.ends_with(ext) {
                let prefix = "decoder_compiled_seq";
                let suffix = format!(".{}", ext);
                if let Some(num_str) = name_str
                    .strip_prefix(prefix)
                    .and_then(|s| s.strip_suffix(&suffix))
                {
                    if let Ok(bucket_len) = num_str.parse::<usize>() {
                        println!(
                            "  Loading Decoder bucket {} from {:?}...",
                            bucket_len,
                            entry.path()
                        );
                        let (decoder_vm, f_decoder) = Self::load_relax_module(
                            device_type,
                            device_id,
                            &entry.path(),
                            "decoder_forward",
                        )?;
                        vms.push(decoder_vm);
                        decoder_fns.insert(bucket_len, f_decoder);
                        decoder_bucket_lens.push(bucket_len);
                    }
                }
            }

        }

        // Fall back to default decoder if no buckets found
        if decoder_fns.is_empty() {
            // Try decoder_llvm_{STATIC_AUDIO_LEN}.so first (explicit bucket naming)
            let llvm_decoder_path = lib_dir.join(format!("decoder_llvm_{}.{}", STATIC_AUDIO_LEN, ext));
            let decoder_path = if llvm_decoder_path.exists() {
                println!("  Loading Decoder from {:?} (LLVM bucket)...", llvm_decoder_path);
                llvm_decoder_path
            } else {
                // Fall back to decoder_compiled.so
                let path = lib_dir.join(format!("decoder_compiled.{}", ext));
                println!("  Loading Decoder from {:?}...", path);
                path
            };

            let (decoder_vm, f_decoder) = Self::load_relax_module(
                device_type,
                device_id,
                &decoder_path,
                "decoder_forward",
            )?;
            vms.push(decoder_vm);
            decoder_fns.insert(STATIC_AUDIO_LEN, f_decoder);
            decoder_bucket_lens.push(STATIC_AUDIO_LEN);
        }


        decoder_bucket_lens.sort();
        println!(
            "Loaded {} decoder bucket(s): {:?}",
            decoder_bucket_lens.len(),
            decoder_bucket_lens
        );

        Ok(Self {
            f_bert,
            f_duration,
            f_f0n,
            f_text_enc,
            decoder_fns,
            decoder_bucket_lens,
            vms,
            device,
        })
    }

    /// Get device type code from string
    fn device_type_code(device: &str) -> i32 {
        match device {
            "llvm" | "cpu" => 1,  // kDLCPU = 1
            "cuda" | "gpu" => 2,  // kDLCUDA = 2
            "metal" => 8,         // kDLMetal = 8
            _ => 1,               // Default to CPU
        }
    }

    fn device_from_str(device: &str) -> Result<tvm_ffi::DLDevice> {
        use tvm_ffi::DLDeviceType;

        let device_type = match device {
            "llvm" | "cpu" => DLDeviceType::kDLCPU,
            "cuda" | "gpu" => DLDeviceType::kDLCUDA,
            "metal" => DLDeviceType::kDLMetal,
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported device '{}'",
                    device
                ))
            }
        };
        Ok(tvm_ffi::DLDevice::new(device_type, 0))
    }

    /// Load a Relax module and extract a function via VirtualMachine.
    ///
    /// This follows the same initialization sequence as Python's VirtualMachine class:
    /// 1. Load the compiled library
    /// 2. Call vm_load_executable() to get the Executable module
    /// 3. Call vm_initialization(device_type, device_id, allocator_type) to set up devices
    /// 4. Get the entry point function from the initialized VM
    fn load_relax_module(
        device_type: i32,
        device_id: i32,
        path: &Path,
        func_name: &str,
    ) -> Result<(tvm_ffi::Module, tvm_ffi::Function)> {
        // Step 1: Load the compiled library
        let lib = tvm_err!(
            tvm_ffi::Module::load_from_file(path.to_str().unwrap()),
            format!("Failed to load module from {:?}", path)
        )?;

        // Step 2: Get vm_load_executable from the loaded library and call it
        let vm_load_exec = tvm_err!(
            lib.get_function("vm_load_executable"),
            "Could not find 'vm_load_executable'. Is this a Relax model?"
        )?;

        let exec_result = tvm_err!(
            vm_load_exec.call_tuple(()),
            "Failed to execute vm_load_executable"
        )?;

        let exec_module: tvm_ffi::Module = tvm_err!(
            exec_result.try_into(),
            "Failed to convert vm_load_executable result to Module"
        )?;

        // Step 3: Get vm_initialization and call it to set up devices
        let vm_init = tvm_err!(
            exec_module.get_function("vm_initialization"),
            "Could not find 'vm_initialization' in executable"
        )?;

        // Initialize with: (device_type, device_id, allocator_type)
        // We use POOLED_ALLOCATOR for better performance
        // Also add CPU device for shape functions (required by Relax VM)
        let cpu_device_type = 1i32;  // kDLCPU
        let cpu_device_id = 0i32;

        if device_type != cpu_device_type {
            // Initialize with both target device and CPU
            tvm_err!(
                vm_init.call_tuple((
                    device_type, device_id, POOLED_ALLOCATOR,
                    cpu_device_type, cpu_device_id, POOLED_ALLOCATOR
                )),
                "Failed to initialize the Virtual Machine"
            )?;
        } else {
            // CPU only
            tvm_err!(
                vm_init.call_tuple((device_type, device_id, POOLED_ALLOCATOR)),
                "Failed to initialize the Virtual Machine"
            )?;
        }

        // Step 4: Get the entry point function from the VM
        let func = tvm_err!(
            exec_module.get_function(func_name),
            format!("Failed to get function '{}'", func_name)
        )?;

        Ok((exec_module, func))
    }


    /// Select the smallest decoder bucket that fits the given frame count.
    fn select_decoder_bucket(&self, frames: usize) -> usize {
        for &bucket in &self.decoder_bucket_lens {
            if bucket >= frames {
                return bucket;
            }
        }
        *self.decoder_bucket_lens.last().unwrap_or(&STATIC_AUDIO_LEN)
    }

    /// Run full inference.
    ///
    /// Args:
    ///     input_ids: Token IDs (including start/end tokens)
    ///     ref_s: Style embedding [256] (flat slice from the [1, 256] array)
    ///     speed: Speech speed multiplier (1.0 = normal)
    ///
    /// Returns:
    ///     Intermediate tensors including the final audio samples
    pub fn forward_trace(
        &self,
        input_ids: &[i64],
        ref_s: &[f32],
        speed: f32,
    ) -> Result<PipelineTrace> {
        if ref_s.len() < 256 {
            return Err(anyhow::anyhow!(
                "ref_s must have at least 256 elements, got {}",
                ref_s.len()
            ));
        }
        // --- Preprocessing ---
        let (input_ids_arr, cur_len) = pad_input_ids(input_ids, STATIC_TEXT_LEN);
        let (_text_mask, attention_mask) = create_masks(cur_len, STATIC_TEXT_LEN);
        let device = self.device;

        // Style embeddings: s = ref_s[:, 128:], style_128 = ref_s[:, :128]
        let style_128: Vec<f32> = ref_s[..128].to_vec();
        let s: Vec<f32> = ref_s[128..].to_vec();

        // Convert to TVM tensors
        let input_ids_tvm = tvm_err!(
            tensor_from_slice_device(
                input_ids_arr.as_slice().unwrap(),
                &[1, STATIC_TEXT_LEN as i64],
                device,
            ),
            "Failed to create input_ids tensor"
        )?;
        let attention_mask_tvm = tvm_err!(
            tensor_from_slice_device(
                attention_mask.as_slice().unwrap(),
                &[1, STATIC_TEXT_LEN as i64],
                device,
            ),
            "Failed to create attention_mask tensor"
        )?;

        // --- BERT: input_ids, attention_mask -> d_en [1, 512, seq_len] ---
        let bert_out = tvm_err!(
            self.f_bert.call_tuple((&input_ids_tvm, &attention_mask_tvm)),
            "BERT forward failed"
        )?;
        // BERT returns a tuple, extract the first element (d_en)
        let d_en_tvm: tvm_ffi::Tensor = extract_tensor_from_output(&bert_out, 0)?;
        let d_en: Array3<f32> = tensor_to_array_d(&d_en_tvm)?
            .into_dimensionality()
            .map_err(|e| anyhow::anyhow!("Expected BERT output to be 3D: {}", e))?;

        // --- Duration: d_en, s, lengths, mask -> (duration_logits, d) ---
        let s_tvm = tvm_err!(
            tensor_from_slice_device(&s, &[1, 128], device),
            "Failed to create style tensor"
        )?;
        let lengths_tvm = tvm_err!(
            tensor_from_slice_device(&[cur_len as i64], &[1], device),
            "Failed to create lengths tensor"
        )?;

        // Create text mask tensor with bool dtype (kDLBool)
        // Mask is true (1) where padding exists (i >= cur_len)
        let text_mask_u8: Vec<u8> = (0..STATIC_TEXT_LEN)
            .map(|i| if i >= cur_len { 1u8 } else { 0u8 })
            .collect();
        let text_mask_tvm =
            create_bool_tensor(&text_mask_u8, &[1, STATIC_TEXT_LEN as i64], device)?;

        let duration_out = tvm_err!(
            self.f_duration
                .call_tuple((&d_en_tvm, &s_tvm, &lengths_tvm, &text_mask_tvm)),
            "Duration forward failed"
        )?;

        let duration_logits_tvm: tvm_ffi::Tensor = extract_tensor_from_output(&duration_out, 0)?;
        let d_tvm: tvm_ffi::Tensor = extract_tensor_from_output(&duration_out, 1)?;

        let duration_logits = tensor_to_array_d(&duration_logits_tvm)?;
        let d_np: Array3<f32> = tensor_to_array_d(&d_tvm)?
            .into_dimensionality()
            .map_err(|e| anyhow::anyhow!("Expected duration output to be 3D: {}", e))?;

        // --- Alignment computation ---
        let (full_aln, actual_audio_len, pred_dur) =
            build_alignment_with_pred(&duration_logits, cur_len, speed);

        // Compute en = d.T @ alignment
        let d_transposed = d_np.view().permuted_axes([0, 2, 1]);
        let en = Self::matmul_3d(&d_transposed, &full_aln.view());

        // --- F0N: en, s, frame_lengths -> (F0, N) ---
        let en_tvm = tvm_err!(
            tensor_from_slice_device(
                en.as_slice().unwrap(),
                &[1, 640, STATIC_AUDIO_LEN as i64],
                device,
            ),
            "Failed to create en tensor"
        )?;
        let frame_lengths_tvm = tvm_err!(
            tensor_from_slice_device(&[actual_audio_len as i64], &[1], device),
            "Failed to create frame_lengths tensor"
        )?;

        let f0n_out = tvm_err!(
            self.f_f0n.call_tuple((&en_tvm, &s_tvm, &frame_lengths_tvm)),
            "F0N forward failed"
        )?;
        let f0_tvm: tvm_ffi::Tensor = extract_tensor_from_output(&f0n_out, 0)?;
        let n_tvm: tvm_ffi::Tensor = extract_tensor_from_output(&f0n_out, 1)?;
        let f0_np: Array2<f32> = tensor_to_array_d(&f0_tvm)?
            .into_dimensionality()
            .map_err(|e| anyhow::anyhow!("Expected f0 output to be 2D: {}", e))?;
        let n_np: Array2<f32> = tensor_to_array_d(&n_tvm)?
            .into_dimensionality()
            .map_err(|e| anyhow::anyhow!("Expected n output to be 2D: {}", e))?;

        // --- Text Encoder: input_ids, lengths, mask -> t_en ---
        let t_en_out = tvm_err!(
            self.f_text_enc
                .call_tuple((&input_ids_tvm, &lengths_tvm, &text_mask_tvm)),
            "Text encoder forward failed"
        )?;
        let t_en_tvm: tvm_ffi::Tensor = extract_tensor_from_output(&t_en_out, 0)?;
        let t_en: Array3<f32> = tensor_to_array_d(&t_en_tvm)?
            .into_dimensionality()
            .map_err(|e| anyhow::anyhow!("Expected text encoder output to be 3D: {}", e))?;

        let asr = Self::matmul_3d(&t_en.view(), &full_aln.view());

        // --- Decoder: asr, F0, N, style[:128] -> audio ---
        let bucket_len = self.select_decoder_bucket(actual_audio_len);
        let expected_f0_len = bucket_len * 2;
        let f0_len = f0_np.shape().get(1).copied().unwrap_or(0);
        let n_len = n_np.shape().get(1).copied().unwrap_or(0);
        if f0_len < expected_f0_len || n_len < expected_f0_len {
            return Err(anyhow::anyhow!(
                "F0/N length mismatch: expected {}, got f0={} n={}",
                expected_f0_len,
                f0_len,
                n_len
            ));
        }

        let asr_b = asr.slice(s![.., .., ..bucket_len]).to_owned();
        let f0_b = f0_np.slice(s![.., ..expected_f0_len]).to_owned();
        let n_b = n_np.slice(s![.., ..expected_f0_len]).to_owned();

        let asr_tvm = tvm_err!(
            tensor_from_slice_device(
                asr_b.as_slice().unwrap(),
                &[1, 512, bucket_len as i64],
                device,
            ),
            "Failed to create asr tensor"
        )?;
        let f0_b_tvm = tvm_err!(
            tensor_from_slice_device(
                f0_b.as_slice().unwrap(),
                &[1, (bucket_len * 2) as i64],
                device,
            ),
            "Failed to create f0 tensor"
        )?;
        let n_b_tvm = tvm_err!(
            tensor_from_slice_device(
                n_b.as_slice().unwrap(),
                &[1, (bucket_len * 2) as i64],
                device,
            ),
            "Failed to create n tensor"
        )?;
        let style_128_tvm = tvm_err!(
            tensor_from_slice_device(&style_128, &[1, 128], device),
            "Failed to create style_128 tensor"
        )?;

        let f_decoder = self
            .decoder_fns
            .get(&bucket_len)
            .ok_or_else(|| anyhow::anyhow!("No decoder found for bucket size {}", bucket_len))?;

        let audio_out = tvm_err!(
            f_decoder.call_tuple((&asr_tvm, &f0_b_tvm, &n_b_tvm, &style_128_tvm)),
            "Decoder forward failed"
        )?;
        let audio_tvm: tvm_ffi::Tensor = extract_tensor_from_output(&audio_out, 0)?;
        let audio_data = tensor_to_vec_f32(&audio_tvm)?;

        let target_samples = actual_audio_len * SAMPLES_PER_FRAME;
        let trim_len = target_samples.min(audio_data.len());
        let audio = audio_data[..trim_len].to_vec();

        Ok(PipelineTrace {
            audio,
            cur_len,
            frames: actual_audio_len,
            d_en,
            duration_logits,
            d: d_np,
            pred_dur,
            f0: f0_np,
            n: n_np,
            t_en,
            asr,
            decoder_bucket_len: bucket_len,
        })
    }

    /// Run full inference.
    ///
    /// Args:
    ///     input_ids: Token IDs (including start/end tokens)
    ///     ref_s: Style embedding [256] (flat slice from the [1, 256] array)
    ///     speed: Speech speed multiplier (1.0 = normal)
    ///
    /// Returns:
    ///     Audio samples as Vec<f32>
    pub fn forward(&self, input_ids: &[i64], ref_s: &[f32], speed: f32) -> Result<Vec<f32>> {
        Ok(self.forward_trace(input_ids, ref_s, speed)?.audio)
    }

    /// Batch matrix multiplication for 3D arrays.
    /// [1, M, K] @ [1, K, N] -> [1, M, N]
    fn matmul_3d(a: &ArrayView3<f32>, b: &ArrayView3<f32>) -> Array3<f32> {
        let m = a.shape()[1];
        let n = b.shape()[2];
        let k = a.shape()[2];

        let mut result = Array3::<f32>::zeros((1, m, n));

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[[0, i, l]] * b[[0, l, j]];
                }
                result[[0, i, j]] = sum;
            }
        }

        result
    }

    /// Initialize the Relax VM runtime by loading libtvm_runtime.
    #[cfg(not(target_os = "ios"))]
    fn init_relax_runtime() -> Result<()> {
        use std::sync::Once;
        static INIT: Once = Once::new();
        static mut INIT_RESULT: Option<String> = None;

        INIT.call_once(|| {
            let lib_names = if cfg!(target_os = "macos") {
                vec!["libtvm.dylib", "libtvm_runtime.dylib"]
            } else {
                vec!["libtvm.so", "libtvm_runtime.so"]
            };

            for lib_name in &lib_names {
                match unsafe { libloading::Library::new(lib_name) } {
                    Ok(lib) => {
                        std::mem::forget(lib);
                        println!("  Loaded {} for Relax VM support", lib_name);
                        return;
                    }
                    Err(e) => {
                        eprintln!("  Could not load {}: {}", lib_name, e);
                    }
                }
            }

            unsafe {
                INIT_RESULT = Some("Failed to load TVM runtime library".to_string());
            }
        });

        unsafe {
            if let Some(ref err) = INIT_RESULT {
                Err(anyhow::anyhow!("{}", err))
            } else {
                Ok(())
            }
        }
    }

    #[cfg(target_os = "ios")]
    fn init_relax_runtime() -> Result<()> {
        // iOS disallows runtime dynamic loading; assume static registration.
        Ok(())
    }
}
