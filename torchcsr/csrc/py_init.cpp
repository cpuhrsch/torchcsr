#include <torch/extension.h>


namespace py = pybind11;

using namespace at;

static c10::InferenceMode guard;


PYBIND11_MODULE(_torchcsr, m) {

}
