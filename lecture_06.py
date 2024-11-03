import os
from util import *
from non_executable import *
import math
import time
from typing import List, Callable
import torch
import torch.nn as nn
import torch.functional as F
from torch.profiler import ProfilerActivity
from torch.utils.cpp_extension import load_inline
import triton
import triton.language as tl

# Set to 1 to run the kernel in interpreted mode (slow, for debugging).
# Set to 0 to run the kernel in compiled mode (fast, no debugging).
os.environ["TRITON_INTERPRET"] = "0"

def lecture_06():
    announcements()

    note("Last lecture: high-level overview of GPUs and performance"), see(lecture_05)
    note("This lecture: benchmarking/profiling + write kernels")

    if not torch.cuda.is_available():
        note("You should run this lecture on a GPU to get the full experience.")

    review_of_gpus()
    benchmarking_and_profiling()  # Important for understanding!

    kernel_fusion_motivation()
    cuda_kernels()  # Write kernels in CUDA/C++
    triton_kernels()  # Write kernels in Python
    pytorch_compilation()  # Don't write kernels at all?

    # More advanced computations
    triton_softmax_main()
    triton_matmul_main()

    note("## Summary")

    note("Gap between the programming model (PyTorch, Triton, PTX) and hardware => performance mysteries")

    note("Benchmarking for understanding scaling")
    note("Profiling for understanding internals of PyTorch functions (bottoms out with kernels)")
    note("Looking at PTX assembly to understand internals of CUDA kernels")

    note("5 ways to write a function: manual, PyTorch, compiled, CUDA, Triton")
    note("GeLU (element-wise), softmax (row-wise), matmul (complex aggregation)")

    note("Key principle: organize computation to minimize reads/writes")
    note("Key ideas: kernel fusion (warehouse/factory analogy), tiling (shared memory)")
    note("Automatic compilers (Triton, torch.compile) will get better over time")

    further_reading()


def announcements():
    note("Assignment 1 is due on [Monday April 16] + 3 late days.")
    note("Assignment 1 leaderboard"), see("https://github.com/stanford-cs336/spring2024-assignment1-basics-leaderboard")
    note("Assignment 2 is out"), see("https://github.com/stanford-cs336/spring2024-assignment2-systems")


def review_of_gpus():
    note("## Hardware")
    image("https://miro.medium.com/v2/resize:fit:2000/format:webp/1*6xoBKi5kL2dZpivFe1-zgw.jpeg", width=1)
    note("Compute: streaming multiprocessors (SMs) [A100: 108]")
    note("Memory:")
    note("- DRAM [A100: 80GB] - big, slow")
    note("- L2 cache [A100: 40MB]")
    note("- L1 cache [A100: 192KB per SM] - small, fast")

    note("You can look at the specs on your actual GPU.")
    print_gpu_specs()

    note("Basic structure: run f(i) for all i = 0, ..., N-1")

    note("## Execution model")
    image("https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png", width=0.5)
    note("- *Thread*: process individual index (i.e., f(i))")
    note("- *Thread block* (a.k.a. concurrent thread arrays): scheduled on a single SM")
    note("- *Grid*: collection of thread blocks")

    note("Why thread blocks? Shared memory.")
    note("- Intuition: group f(i)'s that read similar data together")
    note("- Threads within a thread block have shared memory (as fast as L1 cache) [A100: 164KB]")
    note("- Can synchronize threads (for reading/writing) within a block (but not across blocks)")

    note("### Hardware and execution interact.")
    image("https://developer-blogs.nvidia.com/wp-content/uploads/2019/06/pasted-image-0.png", width=0.25)
    note("Thread blocks scheduled onto SMs in waves.")
    note("Problem: last wave has fewer thread blocks, leaving some SMs idle (low occupancy).")
    note("Wave quantization: make number of thread blocks divide # SMs.")
    note("Rule of thumb: number of thread blocks should be >= 4x # SMs")
    note("Challenge: some aspects of hardware are hidden from the execution model (e.g., scheduling, # SMs).")

    note("### Arithmetic intensity: # FLOPs / # bytes")
    note("- If high, operation is compute-bound (good)")
    note("- If low, operation is memory-bound (bad)")
    note("General rule: "
         "matrix multiplication is compute-bound, "
         "everything else is memory-bound")


def benchmarking_and_profiling():
    note("IMPORTANT: benchmark/profile your code!")

    note("You can read spec sheets (marketing material) and papers, "
         "but performance depends on your library version, your hardware, your workload, "
         "so there is no substitute for benchmarking/profiling your code.")

    note("Example computation: running forward/backward passes on an MLP.")
    run_mlp(dim=128, num_layers=16, batch_size=128, num_steps=5)

    benchmarking()       # How long does it take?
    profiling()          # Where time is being spent?

    note("Every time you make a change, benchmark/profile!")


class MLP(nn.Module):
    """Simple MLP: linear -> GeLU -> linear -> GeLU -> ... -> linear -> GeLU"""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.gelu(x)
        return x


def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> Callable:
    # Define a model (with random weights)
    model = MLP(dim, num_layers).to(get_device())

    # Define an input (random)
    x = torch.randn(batch_size, dim, device=get_device())

    def run():
        # Run the model `num_steps` times (note: no optimizer updates)
        for step in range(num_steps):
            # Forward
            y = model(x).mean()

            # Backward
            y.backward()

    return run


def run_operation1(dim: int, operation: Callable) -> Callable:
    # Setup: create one random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda : operation(x)


def run_operation2(dim: int, operation: Callable) -> Callable:
    # Setup: create two random dim x dim matrices
    x = torch.randn(dim, dim, device=get_device())
    y = torch.randn(dim, dim, device=get_device())
    # Return a function to perform the operation
    return lambda : operation(x, y)


def benchmarking():
    note("Benchmarking measures the wall-clock time of performing some operation.")

    note("It only gives you end-to-end time, not where time is spent (profiling).")

    note("It is still useful for:")
    note("- comparing different implementations (which is faster?), and")
    note("- understanding how performance scales (e.g., with dimension).")

    note("Let's define a convenient function for benchmarking an arbitrary function.")
    benchmark("sleep", lambda : time.sleep(50 / 1000))

    note("### Benchmarking matrix multiplication")
    note("First, let us benchmark matrix multiplication of square matrices.")
    if torch.cuda.is_available():
        dims = (1024, 2048, 4096, 8192, 16384)
    else:
        dims = (1024, 2048)
    for dim in dims:
        benchmark(f"matmul(dim={dim})", run_operation2(dim=dim, operation=lambda a, b: a @ b))
    note("Times scale cubicly with dimension.")

    note("Let us benchmark our MLP!")
    dim = 256
    num_layers = 4
    batch_size = 256
    num_steps = 2

    benchmark("run_mlp", run_mlp(dim=dim, num_layers=num_layers, batch_size=batch_size, num_steps=num_steps))

    note("Scale the number of steps.")
    for scale in (2, 3, 4, 5):
        benchmark(f"run_mlp({scale}x num_steps)", run_mlp(dim=dim, num_layers=scale * num_layers, batch_size=batch_size, num_steps=scale * num_steps))

    note("Scale the number of layers.")
    for scale in (2, 3, 4, 5):
        benchmark(f"run_mlp({scale}x num_layers)", run_mlp(dim=dim, num_layers=scale * num_layers, batch_size=batch_size, num_steps=num_steps))

    note("Scale the batch size.")
    for scale in (2, 3, 4, 5):
        benchmark(f"run_mlp({scale}x batch_size)", run_mlp(dim=dim, num_layers=num_layers, batch_size=scale * batch_size, num_steps=num_steps))

    note("Scale the dimension.")
    for scale in (2, 3, 4, 5):
        benchmark(f"run_mlp({scale}x dim)", run_mlp(dim=scale * dim, num_layers=num_layers, batch_size=batch_size, num_steps=num_steps))

    note("The timings are not always predictable due to the non-homogenous nature of CUDA kernels, hardware, etc.")

    note("You can also use `torch.utils.benchmark`, which provides more amenities."), see("https://pytorch.org/tutorials/recipes/recipes/benchmark.html")
    note("We did not use this to make benchmarking more transparent.")


def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Time it for real now!
    times: List[float] = []
    for trial in range(num_trials):  # Do it multiple times to capture variance
        start_time = time.time()

        run()  # Actually perform computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

        end_time = time.time()
        times.append((end_time - start_time) * 1000)

    mean_time = mean(times)
    note(f"{description}: {list(map(round1, sorted(times)))} (mean {round1(mean_time)} ms)", pop_stack=True)


def profiling():
    note("While benchmarking looks at end-to-end time, profiling looks at where time is spent.")
    note("Obvious: profiling helps you understand where time is being spent.")
    note("Deeper: profiling helps you understand (what is being called).")

    note("PyTorch has a nice built-in profiler"), see("https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html")

    note("Let's profile some code to see what is going on under the hood.")
    profile("sleep", lambda : time.sleep(50 / 1000))  # Dummy function

    note("Let's start with some basic operations.")
    profile("add", run_operation2(dim=2048, operation=lambda a, b: a + b))
    profile("matmul", run_operation2(dim=2048, operation=lambda a, b: a @ b))
    profile("matmul(dim=128)", run_operation2(dim=128, operation=lambda a, b: a @ b))

    note("Observations")
    note("- You can see what CUDA kernels are actually being called.")
    note("- Different CUDA kernels are invoked depending on the tensor dimensions.")

    note("Name of CUDA kernel tells us something about the implementation.")
    note("Example: cutlass_80_simt_sgemm_256x128_8x4_nn_align1")
    note("- cutlass: NVIDIA's CUDA library for linear algebra")
    note("- 256x128: tile size")

    note("Let's now look at some composite operations.")
    profile("cdist", run_operation2(dim=2048, operation=lambda a, b: torch.cdist(a, b)))
    profile("gelu", run_operation2(dim=2048, operation=lambda a, b: torch.nn.functional.gelu(a + b)))
    profile("softmax", run_operation2(dim=2048, operation=lambda a, b: torch.nn.functional.softmax(a + b)))

    note("Now let's profile our MLP.")
    note("We will also visualize our stack trace using a flame graph, which reveals where time is being spent.")
    if torch.cuda.is_available():
        profile("mlp", run_mlp(dim=2048, num_layers=64, batch_size=1024, num_steps=2), with_stack=True)
    else:
        profile("mlp", run_mlp(dim=128, num_layers=16, batch_size=128, num_steps=2), with_stack=True)


def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
    # Warmup
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Run the code with the profiler
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # Output stack trace for visualization
            with_stack=with_stack,
            # Needed to export stack trace for visualization
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Print out table
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=80,
                                      row_limit=10)
    note(f"## {description}", pop_stack=True)
    note(table, verbatim=True, pop_stack=True)

    # Write stack trace visualization
    if with_stack:
        text_path = f"var/stacks_{description}.txt"
        svg_path = f"var/stacks_{description}.svg"
        prof.export_stacks(text_path, "self_cuda_time_total")
        create_flame_graph(text_path, svg_path)
        image(svg_path, width=1, pop_stack=True)


def create_flame_graph(in_path: str, out_path: str):
    """Create a flame graph from the profiler output in `in_path` and output a SVG file to `out_path`."""
    # https://www.brendangregg.com/flamegraphs.html
    if not os.path.exists("FlameGraph"):
        os.system("git clone https://github.com/brendangregg/FlameGraph")
    os.system(f"FlameGraph/flamegraph.pl --title \"CUDA time\" --countname \"us\" {in_path} > {out_path}")


def kernel_fusion_motivation():
    note("Horace He's blog post"), see("https://horace.io/brrr_intro.html")

    note("Analogy: warehouse : DRAM :: factory : SRAM")
    image("https://horace.io/img/perf_intro/factory_bandwidth.png", width=0.3)

    note("Each operation needs to read/compute/write:")
    image("https://horace.io/img/perf_intro/multi_operators.png", width=0.3)

    note("If we *fuse* the operations, only need to read/write once:")
    image("https://horace.io/img/perf_intro/operator_fusion.png", width=0.3)

    note("To see the effect of fusion, let's consider the GeLU activation function.")
    see("https://pytorch.org/docs/stable/generated/torch.nn.GELU.html")

    note("Let's consider two ways to compute GeLU:")
    x = torch.tensor([1.])  # Input

    note("1. The default PyTorch implementation (fused):")
    y1 = pytorch_gelu(x)  # Fused

    note("2. We can also write our own by hand (not fused):")
    y2 = manual_gelu(x)  # Not fused

    # Check that the implementations match
    assert torch.allclose(y1, y2)

    # Check more systematically
    check_equal(pytorch_gelu, manual_gelu)

    note("Let's benchmark.")
    benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu))
    benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu))
    note("The fused version is significantly faster.")

    note("Let's look under the hood.")
    profile("manual_gelu", run_operation1(dim=16384, operation=manual_gelu))
    profile("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu))
    note("The PyTorch just calls one kernel whereas the others are atomic (remember the warehouse/factory) ")


def cuda_kernels():
    note("Now let's open the box to understand what's going on inside a CUDA kernel by writing our own.")

    note("Let's write the GeLU function in CUDA.")
    cuda_gelu = create_cuda_gelu()

    note("Check correctness of our implementation.")
    if cuda_gelu is not None:
        check_equal(cuda_gelu, manual_gelu)

    note("Benchmark our CUDA version.")
    benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu))
    benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu))
    if cuda_gelu is not None:
        benchmark("cuda_gelu", run_operation1(dim=16384, operation=cuda_gelu))
        profile("cuda_gelu", run_operation1(dim=16384, operation=cuda_gelu))
    note("Our CUDA implementation is faster than manual, but not as good as PyTorch.")

    note("Elementwise operations are easy in CUDA (though you can still be smarter).")
    note("But most interesting operations (e.g., matmul, softmax, RMSNorm) require reading multiple values.")
    note("For that, you have to think about managing shared memory, etc.")


def create_cuda_gelu():
    note("CUDA is an extension of C/C++ with APIs for managing GPUs.")

    note("Simplified picture: write f(i), CUDA kernel computes f(i) for all i.")

    image("https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png", width=0.5)
    note("Grid: collection of thread blocks: numBlocks = (2, 4), blockDim = (1, 8)")
    note("Thread block: collection of threads: blockIdx = (0, 1)")
    note("Thread: single unit of operation: threadIdx = (0, 3).")

    note("You write code that a thread execute, using (blockIdx, blockDim, threadIdx) to determine what to do.")

    note("Set CUDA_LAUNCH_BLOCKING so that if there are errors, CUDA will tell you what went wrong.")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    note("The `load_inline` function makes it convenient to write CUDA code and "
         "bind it to a Python module for immediate use.")

    # CUDA code: has the full logic
    cuda_gelu_src = open("gelu.cu").read()

    # C++ code: defines the gelu function
    cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"

    note("Compile the CUDA code and bind it to a Python module.")
    ensure_directory_exists("var/cuda_gelu")
    if not torch.cuda.is_available():
        return None
    module = load_inline(
        cuda_sources=[cuda_gelu_src],
        cpp_sources=[cpp_gelu_src],
        functions=["gelu"],
        extra_cflags=["-O2"],
        verbose=True,
        name="inline_gelu",
        build_directory="var/cuda_gelu",
    )

    cuda_gelu = getattr(module, "gelu")
    return cuda_gelu


def triton_kernels():
    triton_introduction()
    triton_gelu_main()


def triton_introduction():
    note("Developed by OpenAI in 2021")
    see("https://openai.com/research/triton")

    note("Make GPU programming more accessible")
    note("- Write in Python")
    note("- Think about thread blocks rather than threads")

    note("What does Triton offer?")
    note("                                             CUDA      Triton")
    note("- Memory coalescing (transfer from DRAM)     manual    automatic")
    note("- Shared memory management                   manual    automatic")
    note("- Scheduling within SMs                      manual    automatic")
    note("- Scheduling across SMs                      manual    manual")

    note("Compiler does more work, can actually outperform PyTorch implementations!")


def triton_gelu_main():
    if not torch.cuda.is_available():
        return

    note("One big advantage of Triton is that you can step through the Python code.")

    note("Let's step through a Triton kernel.")
    x = torch.randn(8192, device=get_device())
    y1 = triton_gelu(x)

    print_ptx_main()  # Look at the generated instructions

    note("Check that it's correct.")
    check_equal(triton_gelu, manual_gelu)

    note("Let's now benchmark it compared to the PyTorch and CUDA implementations.")
    note("Remember to set TRITON_INTERPRET=0 for good performance.")
    benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu))
    benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu))
    benchmark("cuda_gelu", run_operation1(dim=16384, operation=create_cuda_gelu()))
    benchmark("triton_gelu", run_operation1(dim=16384, operation=triton_gelu))

    profile("triton_gelu", run_operation1(dim=16384, operation=triton_gelu))

    note("Our Triton implementation (triton_gelu):")
    note("- is almost as good as the PyTorch implementation (pytorch_gelu).")
    note("- is actually slower than our naive CUDA implementation (cuda_gelu).")

    note("Triton operates on blocks, CUDA operates on threads.")
    note("Blocks allows Triton compiler to do other optimizations (e.g., thread coarsening).")

    note("Everything is way faster than the manual implementation (manual_gelu).")


def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()

    # Allocate output tensor
    y = torch.empty_like(x)

    # Determine grid (elements divided into blocks)
    num_elements = x.numel()
    block_size = 1024  # Number of threads
    num_blocks = triton.cdiv(num_elements, block_size)

    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=block_size)

    return y


@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Input is at `x_ptr` and output is at `y_ptr`
    #     |        Block 0            |          Block 1          |      ...      |
    #                            BLOCK_SIZE                                 num_elements

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Indices where this thread block should operate
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Handle boundary
    mask = offsets < num_elements

    # Read
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute (tl.tanh doesn't exist, use tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    # Store
    tl.store(y_ptr + offsets, y, mask=mask)


def print_ptx_main():
    note("PTX (parallel thread execution) is like an assembly language for GPUs.")

    note("We can see the PTX code generated by Triton.")
    see("https://docs.nvidia.com/cuda/parallel-thread-execution/index.html")

    print_ptx("triton_gelu", triton_gelu_kernel)

    note("Observations:")
    note("- ld.global.* and st.global.* reads and writes from global memory")
    note("- %ctaid.x is block index, %tid.x is thread index")
    note("- %f* are floating point registers, %r* are integer registers")
    note("- One thread processes 8 elements at the same time (thread coarsening)")

    note("We can compare this to the CUDA code we wrote earlier:")
    see(get_local_url("var/cuda_gelu-ptx.txt"))
    note("To get this, you have to look at the nvcc command that's printed out, "
         "add `-ptx -o var/cuda_gelu-ptx.txt` and rerun it.")


def print_ptx(name: str, kernel):
    if os.environ.get("TRITON_INTERPRET") == "1":
        note("PTX is not generated when in interpret mode.")
        return

    """Print out the PTX code generated by Triton for the given `kernel`."""
    ptx_path = f"var/{name}-ptx.txt"
    with open(ptx_path, "w") as f:
        print(list(kernel.cache[0].values())[0].asm["ptx"], file=f)

    note("Let's go poke around at the PTX code.")
    see(get_local_url(ptx_path))


def pytorch_compilation():
    note("So far, we have seen three ways to write GeLU:")
    note("- Use the default PyTorch function")
    note("- Write it in Python"), see(manual_gelu)
    note("- Write it in CUDA"), see(create_cuda_gelu)
    note("- Write it in Triton"), see(triton_gelu)

    note("- Write it in Python and compile it into Triton")
    compiled_gelu = torch.compile(manual_gelu)

    note("Check correctness of our implementation.")
    check_equal(compiled_gelu, manual_gelu)

    if not torch.cuda.is_available():
        return

    note("Let's benchmark and profile it!")
    benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu))
    benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu))
    benchmark("cuda_gelu", run_operation1(dim=16384, operation=create_cuda_gelu()))
    benchmark("triton_gelu", run_operation1(dim=16384, operation=triton_gelu))
    benchmark("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu))

    note("Let's look under the hood")
    profile("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu))


def triton_softmax_main():
    note("So far, we've looked at elementwise operations in Triton (e.g., GeLU).")
    note("Now let us look at operations that aggregate over multiple values.")

    note("We will roughly follow the Triton fused softmax tutorial:"), see("https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html")

    note("Recall the softmax operation is used in attention and generating probabilities.")
    note("Normalize each row of a matrix:")
    note("[A1 A2 A3]   =>   [A1/A A2/A A3/A]", verbatim=True)
    note("[B1 B2 B3]   =>   [B1/B B2/B B3/B]", verbatim=True)

    note("Let's first start with the naive implementation and keep track of reads/writes.")
    x = torch.tensor([
        [5., 5, 5],
        [0, 0, 100],
    ], device=get_device())
    y1 = manual_softmax(x)

    if not torch.cuda.is_available():
        return

    note("Now let us write the Triton kernel.")
    y2 = triton_softmax(x)
    assert torch.allclose(y1, y2)

    note("Check our implementations are correct.")
    check_equal2(pytorch_softmax, manual_softmax)
    check_equal2(pytorch_softmax, triton_softmax)

    compiled_softmax = torch.compile(manual_softmax)

    note("Now let's benchmark everything.")
    benchmark("manual_softmax", run_operation1(dim=16384, operation=manual_softmax))
    benchmark("compiled_softmax", run_operation1(dim=16384, operation=compiled_softmax))
    benchmark("pytorch_softmax", run_operation1(dim=16384, operation=pytorch_softmax))
    benchmark("triton_softmax", run_operation1(dim=16384, operation=triton_softmax))

    note("Look under the hood using the profiler.")
    profile("manual_softmax", run_operation1(dim=16384, operation=manual_softmax))
    profile("compiled_softmax", run_operation1(dim=16384, operation=compiled_softmax))
    profile("pytorch_softmax", run_operation1(dim=16384, operation=pytorch_softmax))
    profile("triton_softmax", run_operation1(dim=16384, operation=triton_softmax))

    print_ptx("triton_softmax", triton_softmax_kernel)

    note("Observations:")
    note("- Triton outperforms everything!")


def manual_softmax(x: torch.Tensor):
    # M: number of rows, N: number of columns
    M, N = x.shape

    # Compute the max of each row (MN reads, M writes)
    x_max = x.max(dim=1)[0]

    # Subtract off the max (MN + M reads, MN writes)
    x = x - x_max[:, None]

    # Exponentiate (MN reads, MN writes)
    numerator = torch.exp(x)

    # Compute normalization constant (MN reads, M writes)
    denominator = numerator.sum(dim=1)

    # Normalize (MN reads, MN writes)
    y = numerator / denominator[:, None]

    # Total: 5MN + M reads, 3MN + 2M writes
    # In principle, should have MN reads, MN writes (speedup of 4x!)
    return y


def triton_softmax(x: torch.Tensor):
    # Allocate output tensor
    y = torch.empty_like(x)

    # Determine grid
    M, N = x.shape                          # Number of rows x number of columns
    block_size = triton.next_power_of_2(N)  # Each block contains all the columns
    num_blocks = M                          # Each block is a row

    # Launch kernel
    triton_softmax_kernel[(M,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size
    )

    return y


@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    assert num_cols <= BLOCK_SIZE

    # Process each row independently
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Read from global memory
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # Compute
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # Write back to global memory
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)


def triton_matmul_main():
    note("Matrix multipliction is perhaps the most optimized algorithm ever.")

    note("If you write matrix multiplication in CUDA, there's all sorts of crazy things you have to do.")
    see("https://github.com/openai/blocksparse/blob/master/src/matmul_op_gpu.cu")

    note("It's much easier in Triton.")
    see("https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html")

    note("       k                  j                     ", verbatim=True)
    note("  [ A1 A2 A3 ]       [ B1 B2 B3 ]   [ C1 C2 C3 ]", verbatim=True)
    note("i [ A4 A5 A6 ]  *  k [ B4 B5 B6 ] = [ C4 C5 C6 ]", verbatim=True)
    note("  [ A7 A8 A9 ]       [ B7 B8 B9 ]   [ C7 C8 C9 ]", verbatim=True)

    note("Naively: need MKN reads, MN writes")

    note("Computing C4 and C5 both need A4, A5, A6.")
    note("Can we read A4, A5, A6 from DRAM once to compute both?")
    note("Answer: yes, using shared memory!")

    note("## Tiling (leveraging shared memory)")

    note("Recall that shared memory is:")
    note("- fast (10x faster) and small(~100KB)")
    note("- shared between all the threads in a block.")
    image("https://miro.medium.com/v2/resize:fit:2000/format:webp/1*6xoBKi5kL2dZpivFe1-zgw.jpeg")

    note("Trivial: for small matrices, load all of A and B into shared memory, then could compute C.")
    note("Now we get MK + KN reads, MN writes")

    note("But what if we have big matrices...")

    image("https://www.researchgate.net/profile/Axel-Huebl/publication/320499173/figure/fig1/AS:614298980196359@1523471698396/Performance-critical-A-B-part-of-the-GEMM-using-a-tiling-strategy-A-thread-iterates.png", width=0.5)
    note("Key idea: divide the matrix into blocks.")
    note("For each block of A and block of B:")
    note("- load into shared memory,")
    note("- do mini-matrix multiplication,")
    note("- write the partial sum.")

    note("Animation of tiled matrix multiplication"), see("https://youtu.be/aMvCEEBIBto")

    note("## Leveraging L2 cache")

    note("Two ways of computing 9 elements of a matrix:")
    image("https://triton-lang.org/main/_images/grouped_vs_row_major_ordering.png", width=0.5)
    note("1. Loads 9 + 81 = 90 blocks")
    note("1. Loads 27 + 27 = 54 blocks")

    note("Process the blocks in an order that minimizes the reads.")

    note("Why write your own kernel for matrix multiplication (e.g., A @ B)?")
    note("Answer: fusion with another operation (e.g., gelu(A @ B))")

    if not torch.cuda.is_available():
        return
    note("Let's try it!")
    benchmark("pytorch_matmul", run_operation2(dim=16384, operation=torch.matmul))
    benchmark("triton_matmul", run_operation2(dim=16384, operation=triton_matmul))

    # Not working for some reason
    #print_ptx("triton_matmul", triton_matmul_kernel)


def further_reading():
    see("https://horace.io/brrr_intro.html")

    note("CUDA MODE Lecture 1: how to profile CUDA kernels in PyTorch")
    see("https://www.youtube.com/watch?v=LuhJEEJQgUM")
    note("CUDA MODE Lecture 2: Chapters 1-3 of PPMP book")
    see("https://www.youtube.com/watch?v=NQ-0D5Ti2dc")
    note("CUDA MODE Lecture 3: Getting started with CUDA for Python Programmers")
    see("https://www.youtube.com/watch?v=4sgKnKbR-WE")
    note("CUDA MODE Lecture 4: Compute and memory basics")
    see("https://www.youtube.com/watch?v=lTmYrKwjSOU")
    note("CUDA MODE Lecture 8: CUDA performance checklist")
    see("https://www.youtube.com/watch?v=SGhfUhlowB4")

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

    see("https://jonathan-hui.medium.com/ai-chips-a100-gpu-with-nvidia-ampere-architecture-3034ed685e6e")
    see("https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html")
    see("https://github.com/srush/gpu-puzzles")
    see("https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf")
    see("https://towardsdatascience.com/how-pytorch-2-0-accelerates-deep-learning-with-operator-fusion-and-cpu-gpu-code-generation-35132a85bd26")

############################################################

def print_gpu_specs():
    num_devices = torch.cuda.device_count()
    note(f"{num_devices} devices")
    for i in range(num_devices):
        properties = torch.cuda.get_device_properties(i)
        note(f"{i}: {properties}")


def pytorch_softmax(x: torch.Tensor):
    return torch.nn.functional.softmax(x, dim=-1)


def pytorch_gelu(x: torch.Tensor):
    # Use the tanh approximation to match our implementation
    return torch.nn.functional.gelu(x, approximate="tanh")


def manual_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))


def check_equal(f1, f2):
    x = torch.randn(2048, device=get_device())
    y1 = f1(x)
    y2 = f2(x)
    assert torch.allclose(y1, y2, atol=1e-6)


def check_equal2(f1, f2):
    x = torch.randn(2048, 2048, device=get_device())
    y1 = f1(x)
    y2 = f2(x)
    assert torch.allclose(y1, y2, atol=1e-6)


def get_local_url(path: str) -> str:
    #return f"http://localhost:8000/{path}"
    return "https://github.com/stanford-cs336/spring2024-lectures/blob/main/" + path;


# Copied from https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def triton_matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)

def triton_matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    triton_matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c


if __name__ == "__main__":
    init_content("lecture_06-content.js")
    lecture_06()
