#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <sputnik/softmax/sparse_softmax.h>

using namespace torch::nn;
namespace F = torch::nn::functional;

namespace at {

Tensor torchcsr_softmax(const Tensor& tensor, int64_t dim, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(!dtype, "dtype not supported for CSR softmax.");
  SparseCsrTensorImpl* simpl = static_cast<SparseCsrTensorImpl*>(tensor.unsafeGetTensorImpl());
  Tensor output_values = simpl->values().clone().cuda().zero_();
  Tensor values = simpl->values().cuda();
  Tensor crow_indices = simpl->crow_indices().cuda();
  Tensor col_indices = simpl->col_indices().cuda();
  Tensor offsets = torch::cumsum(crow_indices, 0).to(torch::kInt32);
  std::cout << "offsets: " << offsets << std::endl;
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  at::cuda::setCurrentCUDAStream(defaultStream);
  sputnik::SparseSoftmax((int)(tensor.size(0)),
                         (int)(tensor.size(1)), 
                         (int)(simpl->nnz()), /* ignored by kernel? */
                         values.data_ptr<float>(),
                         crow_indices.data_ptr<int>(),
                         offsets.data_ptr<int>(),
                         col_indices.data_ptr<int>(),
                         output_values.data_ptr<float>(),
                         defaultStream);
  return sparse_csr_tensor(simpl->crow_indices().clone(),
                           simpl->col_indices().clone(),
                           output_values.cpu(),
                           tensor.sizes(),
                           tensor.options());
}

TORCH_LIBRARY_IMPL(aten, SparseCsrCPU, m) {
  m.impl("softmax.int", torchcsr_softmax);
}

} // namespace at
