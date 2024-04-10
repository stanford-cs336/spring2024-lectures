from util import *
import torch
import triton
import triton.language as tl

def lecture():
    introduction_to_gpus()

    benchmarking()
    profiling()

    triton_vector_add()
    triton_fused_softmax()


def introduction_to_gpus():
    see("https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html")

    # Horace He's Making Deep Learning Go Brrrr from First Principles:
    see("https://horace.io/brrr_intro.html")
    
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

def benchmarking():
    see("https://pytorch.org/tutorials/recipes/recipes/benchmark.html")


def profiling():
    see("https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html")


def triton_vector_add():
    see("https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html")


def triton_fused_softmax():
    see("https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html")


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
