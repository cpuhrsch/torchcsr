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
  std::cout << "Calling softmax" << std::endl;
  SparseCsrTensorImpl* simpl = static_cast<SparseCsrTensorImpl*>(tensor.unsafeGetTensorImpl());
  std::cout << "simpl->crow_indices(): " << simpl->crow_indices() << std::endl;
  std::cout << "simpl->col_indices(): " << simpl->col_indices() << std::endl;
  std::cout << "simpl->values(): " << simpl->values() << std::endl;
  std::cout << "simpl->nnz(): " << simpl->nnz() << std::endl;
  Tensor offsets = torch::tensor({0, 2}, torch::kInt32);
  Tensor output_values = simpl->values().clone();
  at::cuda::CUDAStream defaultStream = at::cuda::getDefaultCUDAStream();
  at::cuda::setCurrentCUDAStream(defaultStream);
  sputnik::SparseSoftmax((int)(tensor.size(0)),
                       (int)(tensor.size(1)), 
                        (int)(3),
                        simpl->values().data_ptr<float>(),
                        simpl->crow_indices().data_ptr<int>(),
                        offsets.data_ptr<int>(),
                        simpl->col_indices().data_ptr<int>(),
                        output_values.data_ptr<float>(),
             defaultStream);
  return output_values;
}

TORCH_LIBRARY_IMPL(aten, SparseCsrCPU, m) {
  m.impl("softmax.int", torchcsr_softmax);
}

} // namespace at
