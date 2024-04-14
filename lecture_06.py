from util import *
import torch
import triton
import triton.language as tl

def lecture():
    note("Let's first review the basics about GPUs.")
    introduction_to_gpus()

    note("Benchmarking and profiling are critical for understanding in general (not just seeing what's taking memory/time).")
    benchmarking()
    profiling()

    mixed_precision()

    simple_cuda_example()

    triton_vector_add()
    triton_fused_softmax()
    pytorch_compilation()

    matrix_multiplication()
    recomputation()


def recomputation():
    # Backwards
    pass


def simple_cuda_example():
    pass


def pytorch_compilation():
    see("https://towardsdatascience.com/how-pytorch-2-0-accelerates-deep-learning-with-operator-fusion-and-cpu-gpu-code-generation-35132a85bd26")

    
def print_gpu_specs():
    num_devices = torch.cuda.device_count()
    note(f"{num_devices} devices")
    for i in range(num_devices):
        note(torch.cuda.get_device_properties(i))


def introduction_to_gpus():
    note("Hardware")
    image("https://miro.medium.com/v2/resize:fit:2000/format:webp/1*6xoBKi5kL2dZpivFe1-zgw.jpeg")
    note("- Compute: streaming multiprocessor (SM) [A100: 108]")
    note("- Memory: DRAM [A100: 80GB], L2 cache [40MB], L1 cache (SRAM, for each SM)")
    note("- Bandwidth: [A100: 2039 GB/s]")

    note("Idealized example: compute f(i) for all i = 0, ..., N-1")
    
    note("Execution")
    note("- *Thread*: process individual index (i.e., f(i))")
    note("- *Thread block*: placed on an SM (should be >= 4x # SMs)")
    note("- *Grid*: consists of a collection of thread blocks")

    note("Notes about execution")
    note("- Each thread block is divided into warps, each containing 32 threads")
    note("- Can synchronize threads within a block, but not across blocks")
    note("- Threads within a block have shared memory")
    
    note("Recall that you can look at the specs on your actual GPU")
    print_gpu_specs()

    note("General NVIDIA reference")
    see("https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html")

    note("Arithmetic intensity: # ops / # bytes")
    note("- High: compute-bound")
    note("- Low: memory-bound")
    note("General rule: "
         "matrix multiplication is compute-bound, "
         "everything else is memory-bound")

    # Horace He's Making Deep Learning Go Brrrr from First Principles:
    see("https://horace.io/brrr_intro.html")

    note("Analogy: warehouse : DRAM :: factory : SRAM")
    image("https://horace.io/img/perf_intro/factory_bandwidth.png")
    image("https://horace.io/img/perf_intro/multi_operators.png")
    image("https://horace.io/img/perf_intro/operator_fusion.png")


def references():
    note("Some other helpful references")

    note("HetSys Course: Lecture 1: Programming heterogenous computing systems with GPUs")
    see("https://www.youtube.com/watch?v=8JGo2zylE80")
    note("HetSys Course: Lecture 2: SIMD processing and GPUs")
    see("https://www.youtube.com/watch?v=x1MA4MtO4Tc")
    note("HetSys Course: Lecture 3: GPU Software Hierarchy")
    see("https://www.youtube.com/watch?v=KGZ00J5MJz0")
    note("HetSys Course: Lecture 4: GPU Memory Hierarchy")
    see("https://www.youtube.com/watch?v=ZQKMZIP3Fzg")
    note("HetSys Course: Lecture 5: GPU performance considerations")
    see("https://www.youtube.com/watch?v=ODeprwr3Jho")
    note("HetSys Course: Lecture 6: GPU performance considerations")
    see("https://www.youtube.com/watch?v=Xp0HHpcDwUc")

    note("Overview of A100")
    see("https://jonathan-hui.medium.com/ai-chips-a100-gpu-with-nvidia-ampere-architecture-3034ed685e6e")


def mixed_precision():
    # TODO
    pass


def benchmarking():
    see("https://pytorch.org/tutorials/recipes/recipes/benchmark.html")


def profiling():
    see("https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html")
    see("https://www.youtube.com/watch?v=LuhJEEJQgUM")
    see("https://github.com/cuda-mode/lectures/blob/main/lecture1/pytorch_square.py")
    with torch.autograd.profiler.profile() as prof:
        # Code to be profiled
        torch.square()
        pass

    
def arithmetic_intensity():
    note("Rematerialization")
    note("Activation checkpointing")
    

def triton_vector_add():
    see("https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html")


def triton_fused_softmax():
    see("https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html")


def matrix_multiplication():
    pass 

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.device_print("foo", x)

    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    assert x.size() == y.size()
    output = torch.empty_like(x)

    assert x.is_cuda and y.is_cuda and output.is_cuda
    assert x.is_contiguous() and y.is_contiguous() and output.is_contiguous()

    num_elements = output.numel()
    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

    add_kernel[grid](x, y, output, num_elements, BLOCK_SIZE=1024)
    return output

n = 1024 * 1024
x = torch.randn(n, device='cuda')
y = torch.randn(n, device='cuda')
out = add(x, y)
out_ref = x + y
assert torch.equal(out, out_ref)
