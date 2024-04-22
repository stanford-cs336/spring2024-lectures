#include <torch/extension.h>
torch::Tensor gelu(torch::Tensor x);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("gelu", torch::wrap_pybind_function(gelu), "gelu");
}