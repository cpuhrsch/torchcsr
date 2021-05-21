#include <torch/extension.h>
#include <torch/library.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor torchcsr_softmax(Tensor tensor, int64_t dim, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(!dtype, "dtype not supported for CSR softmax.");
  return tensor;
}

TORCH_LIBRARY_IMPL(aten, SparseCsrCPU, m) {
  m.impl("softmax.int", torchcsr_softmax)
}
} // namespace at
