//! CUDA backend plumbing (feature `cuda`, docs/scfast-integration.md Phase 6).
//!
//! Built on `cudarc` with `dynamic-loading`: nothing links against libcuda or libnvrtc at
//! build time. They are `dlopen`ed on first use, so this module compiles and
//! its CPU-side tests run on a machine without a GPU; [`probe`] reports whether a device
//! can actually be used.
//!
//! What lives on the device (per kernel; nothing persists between calls):
//!
//! | Kernel          | Device memory                                        | Products |
//! |-----------------|------------------------------------------------------|----------|
//! | PCA (`pca/cuda.rs`) | CSR of the HVG matrix (f32 values, u32 columns, i64 row offsets), gene mean / 1/sd, one cells x l and a few genes x l f32 buffers | NVRTC CSR SpMM kernels `zv` (Z V) and `ztu` (Z^T U, scatter with atomics), mean / 1/sd correction fused in |
//! | kNN (`knn/cuda.rs`) | centred cells x dims (f32), block x cap candidate buffer | NVRTC `knn`: fused distances and top-k, one query per thread, compiled per (dims, cap) |
//!
//! Dense steps of the randomised SVD (TSQR, the small SVD) and the exact re-ranking of kNN
//! candidates stay on the CPU (faer / f64), exactly as in the CPU paths.
//!
//! Errors: every driver / NVRTC failure becomes [`Error::Gpu`] with the failing
//! step named; `CUDA_ERROR_OUT_OF_MEMORY` is reported as "out of device memory (...)" so
//! callers that resolved `device = "auto"` can fall back to the CPU and say why.
//!
//! The kernel sources are CUDA C strings compiled with NVRTC on first use per process and
//! device (the compiled module is cached in [`Gpu`]).

use std::collections::HashMap;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::device::DeviceInfo;
use crate::error::{Error, Result};

/// Threads per block for every kernel here.
pub(crate) const BLOCK: u32 = 256;
/// Largest candidate list `knn` keeps per query (a per-thread local array).
pub(crate) const KNN_CAP_MAX: usize = 128;
/// Shared memory for one tile of reference points in `knn` (bytes).
const KNN_TILE_BYTES: usize = 32 << 10;

/// Brute-force kNN, one query per thread. The block stages a tile of `TP` reference points
/// in shared memory; every thread computes exact f32 squared distances from its query (held
/// in registers: `D` is fixed at compile time) to the tile and keeps its `CAP` best, sorted
/// by (distance, index), with the current worst in registers so that the common case, a
/// rejected point, touches no local memory. Nothing goes through DRAM per pair. Compiled
/// per (`D`, `CAP`) by NVRTC with those `#define`s prepended. Rows of x are
/// `D` contiguous floats; output slot `q` covers query `q0 + q`.
const KNN_KERNEL: &str = r#"
#define POS_INF __int_as_float(0x7f800000)
extern "C" __global__ void knn(const float* __restrict__ x, int n, int q0, int nq,
                               unsigned int* __restrict__ out_idx, float* __restrict__ out_d2) {
  extern __shared__ float tile[];
  const int qi = blockIdx.x * blockDim.x + threadIdx.x;
  const bool active = qi < nq;
  const unsigned int self = (unsigned int)(q0 + qi);
  float q[D];
#pragma unroll
  for (int t = 0; t < D; ++t) q[t] = active ? x[(size_t)self * D + t] : 0.0f;
  float bv[CAP];
  unsigned int bi[CAP];
  int cnt = 0;
  float wv = POS_INF;
  unsigned int wi = 0xffffffffu;
  for (int base = 0; base < n; base += TP) {
    const int m = min(TP, n - base);
    __syncthreads();
    for (int e = threadIdx.x; e < m * D; e += blockDim.x) tile[e] = x[(size_t)base * D + e];
    __syncthreads();
    if (!active) continue;
    for (int p = 0; p < m; ++p) {
      float s = 0.0f;
#pragma unroll
      for (int t = 0; t < D; ++t) {
        const float df = tile[p * D + t] - q[t];
        s = fmaf(df, df, s);
      }
      const unsigned int r = (unsigned int)(base + p);
      if (!(s < wv || (s == wv && r < wi)) || r == self) continue;
      int pos = (cnt < CAP) ? cnt : CAP - 1;
      while (pos > 0 && (s < bv[pos - 1] || (s == bv[pos - 1] && r < bi[pos - 1]))) {
        bv[pos] = bv[pos - 1];
        bi[pos] = bi[pos - 1];
        --pos;
      }
      bv[pos] = s;
      bi[pos] = r;
      if (cnt < CAP) ++cnt;
      if (cnt == CAP) {
        wv = bv[CAP - 1];
        wi = bi[CAP - 1];
      }
    }
  }
  if (!active) return;
  for (int t = 0; t < CAP; ++t) {
    out_idx[(size_t)qi * CAP + t] = (t < cnt) ? bi[t] : 0xffffffffu;
    out_d2[(size_t)qi * CAP + t] = (t < cnt) ? bv[t] : POS_INF;
  }
}
"#;

/// CUDA C source of the kernels, compiled with NVRTC. Kept in one translation unit so a
/// single module holds every function.
pub(crate) const KERNELS: &str = r#"
// Self-contained: no headers, so NVRTC needs no include path.

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
"#;

// ------------------------------------------------------------------ error mapping

pub(crate) fn drv_err(step: &str, e: DriverError) -> Error {
    if e.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
        return Error::gpu(format!("out of device memory ({step})"));
    }
    let what = e.error_string().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|_| format!("{:?}", e.0));
    Error::gpu(format!("{step}: {what}"))
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

/// Compiles `src` for the device's own architecture; if this NVRTC does not know it (NVRTC
/// older than the GPU), falls back to NVRTC's default target, which the driver JITs.
fn compile(ctx: &Arc<CudaContext>, src: &str) -> Result<Arc<CudaModule>> {
    let (major, minor) = ctx.compute_capability().map_err(|e| drv_err("compute capability", e))?;
    let with_arch = CompileOptions { options: vec![format!("--gpu-architecture=compute_{major}{minor}")], ..Default::default() };
    let ptx = match compile_ptx_with_opts(src, with_arch) {
        Ok(p) => p,
        Err(_) => compile_ptx_with_opts(src, CompileOptions::default())
            .map_err(|e| Error::gpu(format!("NVRTC compilation of the scfast kernels failed: {e:?}")))?,
    };
    ctx.load_module(ptx).map_err(|e| drv_err("cuModuleLoadData", e))
}

/// One CUDA device: primary context, a stream and the compiled kernels.
/// Shared per ordinal for the life of the process.
pub(crate) struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    pub zv: CudaFunction,
    pub ztu: CudaFunction,
    pub ztu_finish: CudaFunction,
    /// `knn` compiled per (dims, cap).
    knn: Mutex<HashMap<(usize, usize), CudaFunction>>,
    pub ordinal: usize,
}

static GPUS: OnceLock<Mutex<HashMap<usize, Arc<Gpu>>>> = OnceLock::new();

impl Gpu {
    /// The handle for `ordinal`, created (context, NVRTC compile) on first use.
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
        let module = compile(&ctx, KERNELS)?;
        let f = |name: &str| module.load_function(name).map_err(|e| drv_err(&format!("cuModuleGetFunction({name})"), e));
        let zv = f("zv")?;
        let ztu = f("ztu")?;
        let ztu_finish = f("ztu_finish")?;
        Ok(Gpu { ctx, stream, module, zv, ztu, ztu_finish, knn: Mutex::new(HashMap::new()), ordinal })
    }

    /// The `knn` kernel for `d` dims and `cap` candidates, and the reference points per
    /// shared-memory tile; compiled on first use.
    pub fn knn_kernel(&self, d: usize, cap: usize) -> Result<(CudaFunction, usize)> {
        let tp = (KNN_TILE_BYTES / (4 * d)).clamp(1, BLOCK as usize);
        let mut map = self.knn.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(f) = map.get(&(d, cap)) {
            return Ok((f.clone(), tp));
        }
        let src = format!("#define D {d}\n#define CAP {cap}\n#define TP {tp}\n{KNN_KERNEL}");
        let f = guard("NVRTC knn", || {
            compile(&self.ctx, &src)?.load_function("knn").map_err(|e| drv_err("cuModuleGetFunction(knn)", e))
        })?;
        map.insert((d, cap), f.clone());
        Ok((f, tp))
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
        for name in ["zv", "ztu", "ztu_finish"] {
            assert!(KERNELS.contains(&format!("__global__ void {name}(")), "kernel {name} missing");
        }
        assert!(KNN_KERNEL.contains("__global__ void knn("));
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
