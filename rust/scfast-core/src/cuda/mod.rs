//! CUDA backend plumbing (feature `cuda`, docs/scfast-integration.md Phase 6).
//!
//! Built on `cudarc` with `dynamic-loading`: nothing links against libcuda, libnvrtc or
//! libcublas at build time. They are `dlopen`ed on first use, so this module compiles and
//! its CPU-side tests run on a machine without a GPU; [`probe`] reports whether a device
//! can actually be used.
//!
//! What lives on the device (per kernel; nothing persists between calls):
//!
//! | Kernel          | Device memory                                        | Products |
//! |-----------------|------------------------------------------------------|----------|
//! | PCA (`pca/cuda.rs`) | CSR of the HVG matrix (f32 values, u32 columns, i64 row offsets), gene mean / 1/sd, one cells x l and a few genes x l f32 buffers | NVRTC CSR SpMM kernels `zv` (Z V) and `ztu` (Z^T U, scatter with atomics), mean / 1/sd correction fused in |
//! | kNN (`knn/cuda.rs`) | centred cells x dims (f32), squared norms, one cells x block score buffer, block x cap candidate buffer | cuBLAS SGEMM for the cross terms, NVRTC `topk` (one block per query) |
//!
//! Dense steps of the randomised SVD (TSQR, the small SVD) and the exact re-ranking of kNN
//! candidates stay on the CPU (faer / f64), exactly as in the CPU paths.
//!
//! Errors: every driver / NVRTC / cuBLAS failure becomes [`Error::Gpu`] with the failing
//! step named; `CUDA_ERROR_OUT_OF_MEMORY` is reported as "out of device memory (...)" so
//! callers that resolved `device = "auto"` can fall back to the CPU and say why.
//!
//! The kernel sources are CUDA C strings compiled with NVRTC on first use per process and
//! device (the compiled module is cached in [`Gpu`]).

use std::collections::HashMap;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::device::DeviceInfo;
use crate::error::{Error, Result};

/// Threads per block for every kernel here. `topk` relies on it (shared-memory arrays).
pub(crate) const BLOCK: u32 = 256;
/// Largest candidate list `topk` keeps per query (local-memory arrays in the kernel).
pub(crate) const TOPK_CAP_MAX: usize = 128;

/// CUDA C source of the kernels, compiled with NVRTC. Kept in one translation unit so a
/// single module holds every function.
pub(crate) const KERNELS: &str = r#"
// Self-contained: no headers, so NVRTC needs no include path.
#define BLOCK 256
#define POS_INF __int_as_float(0x7f800000)
#define CAP_MAX 128

// Z V for the implicitly scaled operator Z = (X - 1 mu^T) diag(inv_sd):
//   out[i, c] = sum_j X[i, j] * vs[j, c] - shift[c]
// with vs = diag(inv_sd) V (genes x l, row-major) and shift = mu^T vs, both prepared on
// the host. X is CSR (cells x genes); out is column-major cells x l (faer's layout).
// One thread per (row, component); consecutive threads share a row and read consecutive
// components of vs, so the CSR row is read from cache and the vs reads coalesce.
extern "C" __global__ void zv(const long long* __restrict__ indptr,
                              const unsigned int* __restrict__ indices,
                              const float* __restrict__ vals,
                              const float* __restrict__ vs,
                              const float* __restrict__ shift,
                              float* __restrict__ out,
                              int n, int l) {
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (long long)n * l) return;
  int i = (int)(tid / l);
  int c = (int)(tid - (long long)i * l);
  long long s = indptr[i], e = indptr[i + 1];
  float acc = 0.0f;
  for (long long t = s; t < e; ++t) {
    acc += vals[t] * vs[(size_t)indices[t] * l + c];
  }
  out[(size_t)c * n + i] = acc - shift[c];
}

// X^T U scatter: acc[j, c] += X[i, j] * U[i, c] for every stored X[i, j].
// U is column-major cells x l, acc is row-major genes x l and must be zeroed first.
// Float atomics: the summation order is not fixed, so results are reproducible only to
// f32 rounding (the parity thresholds are far looser than that).
extern "C" __global__ void ztu(const long long* __restrict__ indptr,
                               const unsigned int* __restrict__ indices,
                               const float* __restrict__ vals,
                               const float* __restrict__ u,
                               float* __restrict__ acc,
                               int n, int l) {
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (long long)n * l) return;
  int i = (int)(tid / l);
  int c = (int)(tid - (long long)i * l);
  float ui = u[(size_t)c * n + i];
  if (ui == 0.0f) return;
  long long s = indptr[i], e = indptr[i + 1];
  for (long long t = s; t < e; ++t) {
    atomicAdd(&acc[(size_t)indices[t] * l + c], vals[t] * ui);
  }
}

// Z^T U = diag(inv_sd) (X^T U - mu colsum^T): the mean / 1/sd correction, on device.
// colsum[c] = sum_i U[i, c] comes from the host (it is exact there in f64).
extern "C" __global__ void ztu_finish(const float* __restrict__ acc,
                                      const float* __restrict__ mean,
                                      const float* __restrict__ inv_sd,
                                      const float* __restrict__ colsum,
                                      float* __restrict__ out,
                                      int p, int l) {
  long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (long long)p * l) return;
  int j = (int)(tid / l);
  int c = (int)(tid - (long long)j * l);
  out[tid] = (acc[tid] - mean[j] * colsum[c]) * inv_sd[j];
}

// Top-k candidate selection for kNN. scores is column-major n x nq: column q holds
// q_(q0+q) . x_r for every reference r (from cuBLAS). The squared-distance estimate up to
// the per-query constant |q|^2 is est = norms[r] - 2 scores[r]. One block per query: each
// thread scans a strided slice keeping its `cap` best (ascending by est, then index) in a
// sorted local array, then the block merges the per-thread lists by repeatedly taking the
// minimum of the list heads. Writes cap (index, est) pairs per query in ascending order;
// slots past n - 1 candidates hold index 0xffffffff.
extern "C" __global__ void topk(const float* __restrict__ scores,
                                const float* __restrict__ norms,
                                int n, int q0, int cap,
                                unsigned int* __restrict__ out_idx,
                                float* __restrict__ out_est) {
  __shared__ float sv[BLOCK];
  __shared__ unsigned int si[BLOCK];
  const int q = blockIdx.x;
  const int self_idx = q0 + q;
  const float* col = scores + (size_t)q * n;
  float best_v[CAP_MAX];
  unsigned int best_i[CAP_MAX];
  int cnt = 0;
  for (int r = threadIdx.x; r < n; r += BLOCK) {
    if (r == self_idx) continue;
    float v = norms[r] - 2.0f * col[r];
    unsigned int ri = (unsigned int)r;
    if (cnt == cap) {
      float wv = best_v[cap - 1];
      if (!(v < wv || (v == wv && ri < best_i[cap - 1]))) continue;
    }
    int pos = (cnt < cap) ? cnt : cap - 1;
    while (pos > 0 && (v < best_v[pos - 1] || (v == best_v[pos - 1] && ri < best_i[pos - 1]))) {
      best_v[pos] = best_v[pos - 1];
      best_i[pos] = best_i[pos - 1];
      --pos;
    }
    best_v[pos] = v;
    best_i[pos] = ri;
    if (cnt < cap) ++cnt;
  }
  int head = 0;
  for (int t = 0; t < cap; ++t) {
    sv[threadIdx.x] = (head < cnt) ? best_v[head] : POS_INF;
    si[threadIdx.x] = (head < cnt) ? best_i[head] : 0xffffffffu;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        int a = threadIdx.x, b = a + stride;
        if (sv[b] < sv[a] || (sv[b] == sv[a] && si[b] < si[a])) {
          sv[a] = sv[b];
          si[a] = si[b];
        }
      }
      __syncthreads();
    }
    unsigned int win = si[0];
    float winv = sv[0];
    if (head < cnt && best_i[head] == win) ++head;
    if (threadIdx.x == 0) {
      out_idx[(size_t)q * cap + t] = win;
      out_est[(size_t)q * cap + t] = winv;
    }
    __syncthreads();
  }
}
"#;

// ------------------------------------------------------------------ error mapping

pub(crate) fn drv_err(step: &str, e: DriverError) -> Error {
    if e.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
        return Error::gpu(format!("out of device memory ({step})"));
    }
    let what = e.error_string().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|_| format!("{:?}", e.0));
    Error::gpu(format!("{step}: {what}"))
}

pub(crate) fn blas_err(step: &str, e: cudarc::cublas::result::CublasError) -> Error {
    Error::gpu(format!("{step}: cuBLAS {:?}", e.0))
}

/// Runs `f`, turning a panic inside cudarc (a missing symbol in an old driver, a library
/// that fails to load) into `Error::Gpu` instead of unwinding through the caller.
pub(crate) fn guard<T>(step: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            Err(Error::gpu(format!("{step}: {msg}")))
        }
    }
}

// ------------------------------------------------------------------ probe

static PROBE: OnceLock<DeviceInfo> = OnceLock::new();

/// Probe once per process: are the three shared libraries loadable, does `cuInit` succeed,
/// how many devices, and what is device 0. Never panics.
pub(crate) fn probe() -> &'static DeviceInfo {
    PROBE.get_or_init(|| {
        let mut info = DeviceInfo { cuda_built: true, ..Default::default() };
        let r = panic::catch_unwind(|| -> std::result::Result<DeviceInfo, String> {
            // SAFETY: these only try to dlopen the libraries.
            if !unsafe { cudarc::driver::sys::is_culib_present() } {
                return Err("CUDA driver library (libcuda) not found".into());
            }
            if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
                return Err("NVRTC library (libnvrtc) not found".into());
            }
            if !unsafe { cudarc::cublas::sys::is_culib_present() } {
                return Err("cuBLAS library (libcublas) not found".into());
            }
            let count = CudaContext::device_count().map_err(|e| format!("cuInit / cuDeviceGetCount: {e:?}"))?;
            if count <= 0 {
                return Err("no CUDA device visible".into());
            }
            let ctx = CudaContext::new(0).map_err(|e| format!("CUDA device 0: {e:?}"))?;
            let name = ctx.name().ok();
            let cc = ctx.compute_capability().ok();
            let (free, total) = ctx.mem_get_info().map(|(f, t)| (Some(f), Some(t))).unwrap_or((None, None));
            Ok(DeviceInfo {
                cuda_built: true,
                cuda_available: true,
                device_count: count as usize,
                device_name: name,
                free_bytes: free,
                total_bytes: total,
                compute_capability: cc,
                cuda_error: None,
            })
        });
        match r {
            Ok(Ok(i)) => info = i,
            Ok(Err(msg)) => info.cuda_error = Some(msg),
            Err(_) => info.cuda_error = Some("the CUDA driver library could not be loaded (panic inside cudarc)".into()),
        }
        info
    })
}

// ------------------------------------------------------------------ device handle

/// One CUDA device: primary context, a stream, a cuBLAS handle and the compiled kernels.
/// Shared per ordinal for the life of the process.
pub(crate) struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    pub zv: CudaFunction,
    pub ztu: CudaFunction,
    pub ztu_finish: CudaFunction,
    pub topk: CudaFunction,
    pub ordinal: usize,
}

static GPUS: OnceLock<Mutex<HashMap<usize, Arc<Gpu>>>> = OnceLock::new();

impl Gpu {
    /// The handle for `ordinal`, created (context, NVRTC compile, cuBLAS handle) on first use.
    pub fn get(ordinal: usize) -> Result<Arc<Gpu>> {
        let map = GPUS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut guard_map = map.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(g) = guard_map.get(&ordinal) {
            return Ok(g.clone());
        }
        let g = guard("CUDA initialisation", || Gpu::create(ordinal))?;
        let g = Arc::new(g);
        guard_map.insert(ordinal, g.clone());
        Ok(g)
    }

    fn create(ordinal: usize) -> Result<Gpu> {
        let info = probe();
        if !info.cuda_available {
            return Err(Error::gpu(format!("no usable CUDA device: {}", info.cuda_error.as_deref().unwrap_or("unknown"))));
        }
        if ordinal >= info.device_count {
            return Err(Error::gpu(format!("CUDA device {ordinal} requested but only {} visible", info.device_count)));
        }
        let ctx = CudaContext::new(ordinal).map_err(|e| drv_err("cuDevicePrimaryCtxRetain", e))?;
        let stream = ctx.default_stream();
        let (major, minor) = ctx.compute_capability().map_err(|e| drv_err("compute capability", e))?;
        // Compile for the device's own architecture; if this NVRTC does not know it (NVRTC
        // older than the GPU), fall back to NVRTC's default target, which the driver JITs.
        let with_arch = CompileOptions {
            options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
            ..Default::default()
        };
        let ptx = match compile_ptx_with_opts(KERNELS, with_arch) {
            Ok(p) => p,
            Err(_) => compile_ptx_with_opts(KERNELS, CompileOptions::default())
                .map_err(|e| Error::gpu(format!("NVRTC compilation of the scfast kernels failed: {e:?}")))?,
        };
        let module = ctx.load_module(ptx).map_err(|e| drv_err("cuModuleLoadData", e))?;
        let f = |name: &str| module.load_function(name).map_err(|e| drv_err(&format!("cuModuleGetFunction({name})"), e));
        let zv = f("zv")?;
        let ztu = f("ztu")?;
        let ztu_finish = f("ztu_finish")?;
        let topk = f("topk")?;
        let blas = CudaBlas::new(stream.clone()).map_err(|e| blas_err("cublasCreate", e))?;
        Ok(Gpu { ctx, stream, blas, module, zv, ztu, ztu_finish, topk, ordinal })
    }

    /// Free device memory in bytes right now.
    pub fn free_bytes(&self) -> Result<usize> {
        self.ctx.mem_get_info().map(|(f, _)| f).map_err(|e| drv_err("cuMemGetInfo", e))
    }

    /// Refuse up front when a kernel's working set clearly does not fit, with a message
    /// that names the sizes (an allocation failure half-way would say the same, later).
    pub fn ensure_fits(&self, needed: usize, what: &str) -> Result<()> {
        let free = self.free_bytes()?;
        if needed > free {
            return Err(Error::gpu(format!(
                "out of device memory: {what} needs {:.2} GiB, {:.2} GiB free on {}",
                needed as f64 / (1u64 << 30) as f64,
                free as f64 / (1u64 << 30) as f64,
                self.name()
            )));
        }
        Ok(())
    }

    pub fn name(&self) -> String {
        self.ctx.name().unwrap_or_else(|_| format!("cuda:{}", self.ordinal))
    }

    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(&self, len: usize, step: &str) -> Result<CudaSlice<T>> {
        self.stream.alloc_zeros(len).map_err(|e| drv_err(step, e))
    }

    pub fn upload<T: cudarc::driver::DeviceRepr>(&self, src: &[T], step: &str) -> Result<CudaSlice<T>> {
        self.stream.clone_htod(src).map_err(|e| drv_err(step, e))
    }

    pub fn upload_into<T: cudarc::driver::DeviceRepr>(&self, src: &[T], dst: &mut CudaSlice<T>, step: &str) -> Result<()> {
        self.stream.memcpy_htod(src, dst).map_err(|e| drv_err(step, e))
    }

    pub fn download_into<T: cudarc::driver::DeviceRepr>(&self, src: &CudaSlice<T>, dst: &mut [T], step: &str) -> Result<()> {
        self.stream.memcpy_dtoh(src, dst).map_err(|e| drv_err(step, e))?;
        self.sync(step)
    }

    pub fn sync(&self, step: &str) -> Result<()> {
        self.stream.synchronize().map_err(|e| drv_err(step, e))
    }

    /// 1-D launch configuration covering `elems` threads with [`BLOCK`] threads per block.
    pub fn cfg_1d(elems: usize) -> LaunchConfig {
        let blocks = (elems as u64).div_ceil(BLOCK as u64) as u32;
        LaunchConfig { grid_dim: (blocks.max(1), 1, 1), block_dim: (BLOCK, 1, 1), shared_mem_bytes: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel source is a plain string; NVRTC only runs on a machine with the
    /// library. This checks the things that can be checked here: every entry point the
    /// Rust side loads is defined, and the compile-time constants agree.
    #[test]
    fn kernel_source_names_every_entry_point() {
        for name in ["zv", "ztu", "ztu_finish", "topk"] {
            assert!(KERNELS.contains(&format!("__global__ void {name}(")), "kernel {name} missing");
        }
        assert!(KERNELS.contains(&format!("#define BLOCK {BLOCK}")));
        assert!(KERNELS.contains(&format!("#define CAP_MAX {TOPK_CAP_MAX}")));
    }

    #[test]
    fn probe_without_hardware_explains_itself() {
        let info = probe();
        assert!(info.cuda_built);
        if !info.cuda_available {
            assert!(info.cuda_error.is_some());
            assert!(matches!(Gpu::get(0), Err(Error::Gpu(_))));
        }
    }

    #[test]
    fn cfg_1d_covers_all_elements() {
        for n in [0usize, 1, 255, 256, 257, 100_000] {
            let c = Gpu::cfg_1d(n);
            assert!(c.grid_dim.0 as usize * BLOCK as usize >= n);
            assert!(c.grid_dim.0 >= 1);
        }
    }
}
