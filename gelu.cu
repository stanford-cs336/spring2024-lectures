#include <math.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

__global__ void gelu_kernel(float* in, float* out, int num_elements) {
    // Get the index into the tensor
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_elements) {  // To handle the case when n < numBlocks * blockDim
        // Do the actual computation
        out[i] = 0.5 * in[i] * (1.0 + tanh(0.79788456 * (in[i] + 0.044715 * in[i] * in[i] * in[i])));
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    // Compute ceil(a / b)
    return (a + b - 1) / b;
}

torch::Tensor gelu(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cuda());
    TORCH_CHECK(x.is_contiguous());

    // Allocate empty tensor
    torch::Tensor y = torch::empty_like(x);

    // Determine grid (elements divided into blocks)
    int num_elements = x.numel();
    int block_size = 1024;  // Number of threads
    int num_blocks = cdiv(num_elements, block_size);

    // Launch the kernel
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();  // Catch errors immediately

    return y;
}