// ctypes bridge for the T1--T3 experiment; no timing or synchronization here.
// Build: nvcc -std=c++17 -O3 -shared -Xcompiler -fPIC native.cu \
//             -lcublasLt -lcublas -o libgemm_bench.so
// Precision semantics: CUDA 13.1 cuBLAS documentation, sections 2.2.9--11,
// 2.4.7--8, 3.3.6--7, 3.3.13, 3.3.17, and 3.3.26.
// https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if CUDART_VERSION < 13010
#error "This experiment requires CUDA 13.1 or newer, including CUBLAS_GEMM_AUTOTUNE."
#endif

namespace {
constexpr int kRequestedCandidates = 32;
thread_local std::string creation_error;

std::string quoted(const std::string& s) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : s) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4)
                         << std::setfill('0') << static_cast<unsigned>(c) << std::dec;
    else out << c;
  }
  out << '"';
  return out.str();
}

std::string status_text(cublasStatus_t status) {
  const char* name = cublasGetStatusName(status);
  const char* detail = cublasGetStatusString(status);
  return std::string(name ? name : "unknown cuBLAS status") + " (" +
         std::to_string(static_cast<int>(status)) + "): " +
         (detail ? detail : "no description");
}

void check_blas(cublasStatus_t status, const char* api) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(api) + ": " + status_text(status));
}

void check_cuda(cudaError_t status, const char* api) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(api) + ": " + cudaGetErrorName(status) +
                             ": " + cudaGetErrorString(status));
}

#define BLAS(call) check_blas((call), #call)
#define CUDA(call) check_cuda((call), #call)

template <typename T>
struct Attribute {
  T value{};
  cublasStatus_t status = CUBLAS_STATUS_INTERNAL_ERROR;
  size_t bytes = 0;
  bool valid() const { return status == CUBLAS_STATUS_SUCCESS && bytes == sizeof(T); }
  std::string json() const {
    std::ostringstream out;
    out << "{\"value\":";
    if (valid()) out << +value;
    else out << "null";
    out << ",\"status\":" << static_cast<int>(status)
        << ",\"bytes_written\":" << bytes << '}';
    return out.str();
  }
};

template <typename T>
Attribute<T> config(const cublasLtMatmulAlgo_t& algo,
                    cublasLtMatmulAlgoConfigAttributes_t key) {
  Attribute<T> result;
  result.status = cublasLtMatmulAlgoConfigGetAttribute(
      &algo, key, &result.value, sizeof(T), &result.bytes);
  return result;
}

template <typename T>
Attribute<T> capability(const cublasLtMatmulAlgo_t& algo,
                        cublasLtMatmulAlgoCapAttributes_t key) {
  Attribute<T> result;
  result.status = cublasLtMatmulAlgoCapGetAttribute(
      &algo, key, &result.value, sizeof(T), &result.bytes);
  return result;
}

std::string algo_hex(const cublasLtMatmulAlgo_t& algo) {
  std::ostringstream out;
  const auto* bytes = reinterpret_cast<const unsigned char*>(&algo);
  for (size_t i = 0; i < sizeof(algo); ++i)
    out << std::hex << std::setw(2) << std::setfill('0') << unsigned(bytes[i]);
  return out.str();
}

struct Candidate {
  cublasLtMatmulHeuristicResult_t result{};
  std::string json;
};

struct Context {
  int64_t m = 0, n = 0, k = 0;
  size_t workspace_bytes = 0;
  cudaStream_t stream = nullptr;
  const void* a = nullptr;
  const void* b = nullptr;
  void* d = nullptr;
  void* workspace = nullptr;
  cublasLtHandle_t lt = nullptr;
  cublasHandle_t blas = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr, d_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  std::vector<Candidate> candidates;
  std::vector<std::string> rejected;
  int returned_candidates = 0;
  int cublas_version = 0, runtime_version = 0, driver_version = 0;
  int cublas_major = 0, cublas_minor = 0, cublas_patch = 0;
  size_t lt_version = 0;
  cublasMath_t actual_math = CUBLAS_DEFAULT_MATH;
  std::string error, info;

  ~Context() {
    // Destroying the cuBLAS handle synchronizes outstanding work before freeing
    // our workspace. Python should synchronize explicitly before teardown.
    if (blas) cublasDestroy(blas);
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (operation) cublasLtMatmulDescDestroy(operation);
    if (lt) cublasLtDestroy(lt);
    if (workspace) cudaFree(workspace);
  }

  void make_row_layout(cublasLtMatrixLayout_t* layout, uint64_t rows,
                       uint64_t cols, int64_t leading_dimension) {
    BLAS(cublasLtMatrixLayoutCreate(layout, CUDA_R_16F, rows, cols, leading_dimension));
    const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    BLAS(cublasLtMatrixLayoutSetAttribute(*layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &order, sizeof(order)));
  }

  void initialize() {
    BLAS(cublasCreate(&blas));
    BLAS(cublasGetVersion(blas, &cublas_version));
    BLAS(cublasGetProperty(MAJOR_VERSION, &cublas_major));
    BLAS(cublasGetProperty(MINOR_VERSION, &cublas_minor));
    BLAS(cublasGetProperty(PATCH_LEVEL, &cublas_patch));
    CUDA(cudaRuntimeGetVersion(&runtime_version));
    CUDA(cudaDriverGetVersion(&driver_version));
    BLAS(cublasSetStream(blas, stream));
    if (workspace_bytes) CUDA(cudaMalloc(&workspace, workspace_bytes));
    // cublasSetStream resets the workspace; always set workspace afterwards.
    BLAS(cublasSetWorkspace(blas, workspace, workspace_bytes));
    const cublasMath_t math_mode = static_cast<cublasMath_t>(
        CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    BLAS(cublasSetMathMode(blas, math_mode));
    BLAS(cublasGetMathMode(blas, &actual_math));
    if (actual_math != math_mode)
      throw std::runtime_error("cuBLAS math mode did not retain strict reduction setting");
    BLAS(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST));

    BLAS(cublasLtCreate(&lt));
    lt_version = cublasLtGetVersion();
    BLAS(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t transpose_a = CUBLAS_OP_N;
    const cublasOperation_t transpose_b = CUBLAS_OP_T;
    BLAS(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &transpose_a, sizeof(transpose_a)));
    BLAS(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &transpose_b, sizeof(transpose_b)));
    make_row_layout(&a_layout, m, k, k);
    make_row_layout(&b_layout, n, k, k);
    make_row_layout(&d_layout, m, n, n);
    BLAS(cublasLtMatmulPreferenceCreate(&preference));
    const uint64_t max_workspace = workspace_bytes;
    BLAS(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, sizeof(max_workspace)));
    // NONE is zero and remains permitted. INPLACE and OUTPUT_TYPE store/reduce
    // FP16 partial results, so they must not be enabled in this FP32 comparison.
    const uint32_t reduction_mask = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    BLAS(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction_mask, sizeof(reduction_mask)));
    const uint64_t implementation_mask = ~static_cast<uint64_t>(
        CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_16F);
    BLAS(cublasLtMatmulPreferenceSetAttribute(preference,
        CUBLASLT_MATMUL_PREF_IMPL_MASK, &implementation_mask, sizeof(implementation_mask)));

    std::array<cublasLtMatmulHeuristicResult_t, kRequestedCandidates> results{};
    BLAS(cublasLtMatmulAlgoGetHeuristic(lt, operation, a_layout, b_layout,
        d_layout, d_layout, preference, kRequestedCandidates, results.data(),
        &returned_candidates));
    for (int rank = 0; rank < returned_candidates; ++rank)
      inspect(results[rank], rank);
    make_info();
  }

  void inspect(const cublasLtMatmulHeuristicResult_t& result, int rank) {
    std::ostringstream out;
    out << "{\"heuristic_rank\":" << rank
        << ",\"heuristic_status\":" << static_cast<int>(result.state);
    std::string rejection;
    if (result.state != CUBLAS_STATUS_SUCCESS) {
      rejection = "heuristic returned " + status_text(result.state);
    } else {
      const auto& algo = result.algo;
      const auto reduction = config<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
      const auto flags = capability<uint64_t>(algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS);
      out << ",\"workspace_bytes\":" << result.workspaceSize
          << ",\"waves_count\":";
      if (std::isfinite(result.wavesCount)) out << result.wavesCount;
      else out << "null";
      out << ",\"algorithm_id\":" << config<int32_t>(algo, CUBLASLT_ALGO_CONFIG_ID).json()
          << ",\"tile_id\":" << config<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_TILE_ID).json()
          << ",\"stages_id\":" << config<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_STAGES_ID).json()
          << ",\"split_k\":" << config<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM).json()
          << ",\"reduction_scheme\":" << reduction.json()
          << ",\"cta_swizzling\":" << config<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING).json()
          << ",\"custom_option\":" << config<uint32_t>(algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION).json()
          << ",\"inner_shape_id\":" << config<uint16_t>(algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID).json()
          << ",\"cluster_shape_id\":" << config<uint16_t>(algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID).json()
          << ",\"numerical_impl_flags\":" << flags.json()
          << ",\"min_alignment_a\":" << capability<uint32_t>(algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES).json()
          << ",\"min_alignment_b\":" << capability<uint32_t>(algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES).json()
          << ",\"min_alignment_c\":" << capability<uint32_t>(algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES).json()
          << ",\"min_alignment_d\":" << capability<uint32_t>(algo, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES).json()
          << ",\"algo_blob_hex\":" << quoted(algo_hex(algo));
      if (!reduction.valid()) rejection = "could not audit reduction scheme";
      else if (reduction.value != CUBLASLT_REDUCTION_SCHEME_NONE &&
               reduction.value != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)
        rejection = "reduction scheme does not preserve FP32 intermediate results";
      else if (!flags.valid()) rejection = "could not audit numerical implementation flags";
      else if (flags.value & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_16F)
        rejection = "algorithm explicitly uses an FP16 accumulator";
      else if (result.workspaceSize > workspace_bytes)
        rejection = "heuristic workspace requirement exceeds allocation";

      cublasLtMatmulHeuristicResult_t checked{};
      const cublasStatus_t check_status = cublasLtMatmulAlgoCheck(
          lt, operation, a_layout, b_layout, d_layout, d_layout, &algo, &checked);
      out << ",\"algo_check_status\":" << static_cast<int>(check_status);
      if (check_status == CUBLAS_STATUS_SUCCESS) {
        out << ",\"algo_check_result_status\":" << static_cast<int>(checked.state)
            << ",\"algo_check_workspace_bytes\":" << checked.workspaceSize;
      }
      if (rejection.empty() && check_status != CUBLAS_STATUS_SUCCESS)
        rejection = "cublasLtMatmulAlgoCheck: " + status_text(check_status);
      else if (rejection.empty() && checked.state != CUBLAS_STATUS_SUCCESS)
        rejection = "cublasLtMatmulAlgoCheck result: " + status_text(checked.state);
      else if (rejection.empty() && checked.workspaceSize > workspace_bytes)
        rejection = "checked workspace requirement exceeds allocation";
    }
    if (rejection.empty()) {
      out << ",\"candidate_index\":" << candidates.size()
          << ",\"precision_approved\":true}";
      candidates.push_back(Candidate{result, out.str()});
    } else {
      out << ",\"precision_approved\":false,\"rejection\":" << quoted(rejection) << '}';
      rejected.push_back(out.str());
    }
  }

  void make_info() {
    std::ostringstream out;
    out << "{\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k
        << ",\"operation\":\"D=A@B.T\",\"storage\":\"row_major\""
        << ",\"input_type\":\"FP16\",\"output_type\":\"FP16\""
        << ",\"compute_type\":\"CUBLAS_COMPUTE_32F\",\"scale_type\":\"FP32\""
        << ",\"alpha\":1,\"beta\":0,\"workspace_bytes\":" << workspace_bytes
        << ",\"stream\":" << reinterpret_cast<uintptr_t>(stream)
        << ",\"cudart_header_version\":" << CUDART_VERSION
        << ",\"cuda_runtime_version\":" << runtime_version
        << ",\"cuda_driver_version\":" << driver_version
        << ",\"cublas_version\":" << cublas_version
        << ",\"cublas_major\":" << cublas_major
        << ",\"cublas_minor\":" << cublas_minor
        << ",\"cublas_patch\":" << cublas_patch
        << ",\"cublaslt_version\":" << lt_version
        << ",\"cublas_math_mode\":" << static_cast<int>(actual_math)
        << ",\"cublas_disallow_reduced_precision_reduction\":true"
        << ",\"lt_reduction_scheme_mask\":"
        << static_cast<uint32_t>(CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)
        << ",\"lt_reject_fp16_accumulator\":true"
        << ",\"autotune_algorithm_enum\":" << static_cast<int>(CUBLAS_GEMM_AUTOTUNE)
        << ",\"requested_candidates\":" << kRequestedCandidates
        << ",\"returned_candidates\":" << returned_candidates
        << ",\"usable_candidates\":" << candidates.size()
        << ",\"rejected_candidates\":[";
    for (size_t i = 0; i < rejected.size(); ++i) {
      if (i) out << ',';
      out << rejected[i];
    }
    out << "]}";
    info = out.str();
  }
};

Context* require_context(void* opaque) {
  if (!opaque) {
    creation_error = "null native GEMM context";
    return nullptr;
  }
  return static_cast<Context*>(opaque);
}

int record_status(Context* context, cublasStatus_t status, const char* api) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    context->error.clear();
    return 0;
  }
  context->error = std::string(api) + ": " + status_text(status);
  return static_cast<int>(status);
}
}  // namespace

extern "C" {

void* gemm_create(int64_t m, int64_t n, int64_t k, uint64_t workspace_bytes,
                  uint64_t stream, uint64_t a, uint64_t b, uint64_t d) {
  try {
    creation_error.clear();
    if (m <= 0 || n <= 0 || k <= 0 || m > INT_MAX || n > INT_MAX || k > INT_MAX)
      throw std::runtime_error("matrix dimensions must be positive and fit the cuBLAS int32 API");
    if (!a || !b || !d || a % 256 || b % 256 || d % 256)
      throw std::runtime_error("A, B and D must be non-null, 256-byte-aligned CUDA device pointers");
    if (workspace_bytes > std::numeric_limits<size_t>::max())
      throw std::runtime_error("workspace_bytes exceeds host size_t range");
    auto context = std::make_unique<Context>();
    context->m = m; context->n = n; context->k = k;
    context->workspace_bytes = static_cast<size_t>(workspace_bytes);
    context->stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream));
    context->a = reinterpret_cast<const void*>(static_cast<uintptr_t>(a));
    context->b = reinterpret_cast<const void*>(static_cast<uintptr_t>(b));
    context->d = reinterpret_cast<void*>(static_cast<uintptr_t>(d));
    int device = -1;
    CUDA(cudaGetDevice(&device));
    for (const void* pointer : {context->a, context->b, static_cast<const void*>(context->d)}) {
      cudaPointerAttributes attributes{};
      CUDA(cudaPointerGetAttributes(&attributes, pointer));
      if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
        throw std::runtime_error("A, B and D must reside on the current CUDA device");
    }
    context->initialize();
    return context.release();
  } catch (const std::exception& error) {
    creation_error = error.what();
    return nullptr;
  } catch (...) {
    creation_error = "unknown exception constructing native GEMM context";
    return nullptr;
  }
}

int gemm_candidate_count(void* opaque) {
  auto* context = require_context(opaque);
  return context ? static_cast<int>(context->candidates.size()) : -1;
}

const char* gemm_candidate_json(void* opaque, int index) {
  auto* context = require_context(opaque);
  if (!context) return nullptr;
  if (index < 0 || static_cast<size_t>(index) >= context->candidates.size()) {
    context->error = "candidate index is out of range";
    return nullptr;
  }
  return context->candidates[index].json.c_str();
}

int gemm_run_lt(void* opaque, int index) {
  auto* context = require_context(opaque);
  if (!context) return -1;
  if (index < 0 || static_cast<size_t>(index) >= context->candidates.size()) {
    context->error = "candidate index is out of range";
    return -1;
  }
  const float alpha = 1.0f, beta = 0.0f;
  const auto& candidate = context->candidates[index];
  return record_status(context, cublasLtMatmul(context->lt, context->operation,
      &alpha, context->a, context->a_layout, context->b, context->b_layout,
      &beta, context->d, context->d_layout, context->d, context->d_layout,
      &candidate.result.algo, context->workspace, context->workspace_bytes,
      context->stream), "cublasLtMatmul");
}

int gemm_run_autotune(void* opaque) {
  auto* context = require_context(opaque);
  if (!context) return -1;
  const float alpha = 1.0f, beta = 0.0f;
  // Column-major views avoid copies: D_col[N,M] = B_col[K,N]^T A_col[K,M].
  // This equals the row-major D = A @ B.T used by the Lt descriptors.
  // The first call tunes; subsequent calls reuse this handle's cached result.
  // Python must warm up outside capture and timed regions.
  return record_status(context, cublasGemmEx(context->blas, CUBLAS_OP_T, CUBLAS_OP_N,
      static_cast<int>(context->n), static_cast<int>(context->m), static_cast<int>(context->k),
      &alpha, context->b, CUDA_R_16F, static_cast<int>(context->k),
      context->a, CUDA_R_16F, static_cast<int>(context->k), &beta,
      context->d, CUDA_R_16F, static_cast<int>(context->n),
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_AUTOTUNE), "cublasGemmEx(CUBLAS_GEMM_AUTOTUNE)");
}

const char* gemm_error(void* opaque) {
  return opaque ? static_cast<Context*>(opaque)->error.c_str() : creation_error.c_str();
}

const char* gemm_info_json(void* opaque) {
  auto* context = require_context(opaque);
  return context ? context->info.c_str() : nullptr;
}

void gemm_destroy(void* opaque) { delete static_cast<Context*>(opaque); }

}  // extern "C"
