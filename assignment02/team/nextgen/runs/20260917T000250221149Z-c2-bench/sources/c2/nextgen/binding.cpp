#include <torch/extension.h>

void partial_cuda(at::Tensor q, at::Tensor kv, at::Tensor topk,
                  at::Tensor table, at::Tensor lengths, at::Tensor ks,
                  at::Tensor vs, at::Tensor partial, at::Tensor lse,
                  int64_t dql, int64_t splits, double sm_scale);
int64_t variant_id_cuda();
int64_t layout_map_check_cuda();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("partial", &partial_cuda, "Paged tcgen05 BF16 partial attention (no PDL)");
  m.def("variant_id", &variant_id_cuda, "Compiled dataflow variant, no runtime fallback");
  m.def("layout_map_check", &layout_map_check_cuda, "CPU exhaustive logical-to-partition mapping check");
}
