#include <torch/extension.h>

void partial_cuda(at::Tensor q, at::Tensor kv, at::Tensor topk,
                  at::Tensor table, at::Tensor lengths, at::Tensor ks,
                  at::Tensor vs, at::Tensor partial, at::Tensor lse,
                  int64_t dql, int64_t splits, double sm_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("partial", &partial_cuda, "Paged tcgen05 BF16 partial attention (no PDL)");
}
