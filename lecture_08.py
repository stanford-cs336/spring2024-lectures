import subprocess
import torch
import util
import time
import re
import math
from typing import List, Callable
from util import *
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.fsdp
import torch.multiprocessing as mp

def lecture_08():
    note("This lecture: distributed training across multiple GPUs")

    note("Situation: compute (arithmetic logic units) is spread out, communication is slow")

    if not torch.cuda.is_available():
        note("Please use a GPU to get the full experience.")

    note("Hierarchy")
    note("- L1 cache / shared memory (small, fast)")
    note("- L2 cache")
    note("- DRAM")
    note("- Single node, multi-GPU")
    note("- Multi-node, multi-GPU (big, slow)")

    note("Last week: reduce DRAM accesses via fusion/tiling")
    note("This week: reduce communication across GPUs/nodes via replication/sharding")

    note("Game: organize computation so that communication is minimized")

    note("## Part 1: building blocks of distributed communication/computation")
    hardware()                 # How nodes actually communicate
    collective_operations()    # Conceptual programming interface
    torch_distributed()        # How this is implemented in NCCL/PyTorch

    benchmarking()             # Estimate bandwidth
    cuda_streams()             # Used to overlap communication and computation

    note("## Part 2: distributed training")
    image("https://uvadlc-notebooks.readthedocs.io/en/latest/_images/parallelism_strategies_overview.svg")
    ddp()                      # Distributed data parallel (DDP)
    ddp_zero3()                # Distributed data parallel (DDP) + ZeRO stage 3
    tensor_parallelism()       # Cut up along the width dimension
    pipeline_parallelism()     # Cut up along the depth dimension

    note("## Summary")

    note("The game: trading off")
    note("- memory usage (store locally) and ")
    note("- communication (send across GPUs)")

    note("- Hardware is getting faster, will always have this hierarchical structure")
    note("- Many ways to parallelize: data, tensor, pipeline")

    further_reading()


def hardware():
    note("## Single GPU")
    image("https://miro.medium.com/v2/resize:fit:2000/format:webp/1*6xoBKi5kL2dZpivFe1-zgw.jpeg", width=1)
    note("Memory bandwidth for DRAM for H100 NVL is 7.8 TB/sec"), see("https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet")

    note("## Multi-node, multi-GPU")

    note("Traditionally:")
    image("https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2Fs42774-021-00098-3/MediaObjects/42774_2021_98_Fig1_HTML.png?as=webp", width=0.4)
    note("- GPUs on same node communicate via a PCI(e) bus (v7.0, 16 lanes => 242 GB/sec)"), see("https://en.wikipedia.org/wiki/PCI_Express")
    note("- GPUs on different nodes communicate via Ethernet (~200 MB/sec)")

    note("Both are too slow...")
    note("Key hardware advance: have GPUs connect *directly*, bypassing CPU")

    note("## InfiniBand")

    note("Standard developed in 1999; Mellanox created InfiniBand hardware, acquired by NVIDIA in 2019")
    note("Idea: Remote Direct Memory Access (RDMA) to connect nodes directly")
    image("https://lambdalabs.com/hubfs/Imported_Blog_Media/nvlink-diagram-update.png", width=0.6)

    note("## NVLink/NVSwitch")

    note("NVIDIA developed proprietary protocol since 2014")
    note("4.5x more bandwidth than InfiniBand")
    see("https://developer.nvidia.com/blog/upgrading-multi-gpu-interconnectivity-with-the-third-generation-nvidia-nvswitch/")

    note("Within a node: NVLink connects GPUs directly, bypass CPU")
    image("https://developer-blogs.nvidia.com/wp-content/uploads/2022/08/NVLink-generations-1.png", width=0.6)

    note("Across nodes: NVSwitch connects GPUs directly, bypass Ethernet")
    image("https://developer-blogs.nvidia.com/wp-content/uploads/2022/08/NVLink-all-to-all-connectivity-1.png", width=0.6)

    note("H100 (Hopper): 18 NVLink 4.0 links => 900GB/sec")

    note("Bonus: NVSwitch has SHARP acceleration, which halves the communication for all-reduce")
    image("https://developer-blogs.nvidia.com/wp-content/uploads/2022/08/NVLink-SHARP-acceleration-1.png", width=0.6)
    note("Will likely be many other hardware innovations in future")

    note("Let's check what our hardware setup is."), see("https://guide.ncloud-docs.com/docs/en/server-baremetal-a100-check-vpc")

    if torch.cuda.is_available():
        note_system(["nvidia-smi", "topo", "-m"])
        note("Note GPUs are connected via NV18, also connected to NICs (for PCIe)")


def collective_operations():
    note("Collective operations are the conceptual primitives used for distributed programming"), see("https://en.wikipedia.org/wiki/Collective_operation")

    note("- Collective means that specify communication pattern across many (e.g., 256) nodes")
    note("- These are classic in the parallel programming literature from the 1980s")
    note("- For SIMD (Single Instruction, Multiple Data) parallelism")
    note("- Better/faster abstraction than managing point-to-point communication yourself")

    note("Terminology:")
    note("- Rank: a device (e.g., GPU)")
    note("- World size: number of devices")

    note("## Broadcast"), image("https://pytorch.org/tutorials/_images/broadcast.png", width=0.3)

    note("## Scatter"), image("https://pytorch.org/tutorials/_images/scatter.png", width=0.3)

    note("## Gather"), image("https://pytorch.org/tutorials/_images/gather.png", width=0.3)

    note("## Reduce"), image("https://pytorch.org/tutorials/_images/reduce.png", width=0.3)

    note("## All-gather"), image("https://pytorch.org/tutorials/_images/all_gather.png", width=0.3)

    note("## Reduce-scatter"), image("https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/reducescatter.png", width=0.3)

    note("## All-reduce = reduce-scatter + all-gather"), image("https://pytorch.org/tutorials/_images/all_reduce.png", width=0.3)

    note("Framework to think about it:")
    note("- Reduce: performs some associative/commutative operation (sum, min, max)")
    note("- Broadcast/scatter is inverse of gather")
    note("- All: destination is all")


def torch_distributed():
    note("## PyTorch distributed library (`torch.distributed`)")

    note("Reference"), see("https://pytorch.org/docs/stable/distributed.html")

    note("- Provides clean interface for collective operations"), see(dist.all_reduce)
    note("- Backends: gloo (CPU), nccl (GPU)")
    note("- Also supports higher-level abstractions"), see(torch.distributed.fsdp.FullyShardedDataParallel)

    note("## NVIDIA Collective Communication Library (NCCL)")

    note("NCCL translates collective operations into low-level packets.")
    see("https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31880/")

    note("1. Detect toplogy of hardware (e.g., number of nodes, switches, NVLink/PCIe)")
    note("2. Optimize the path between ranks; ring (good bandwidth), tree (good latency)")
    image("https://developer-blogs.nvidia.com/wp-content/uploads/2019/02/DBtree.png", width=0.4)
    note("3. Launches CUDA kernels to send/receive data")

    note("## Examples")
    spawn(collective_operations_main, world_size=4)




def collective_operations_main(rank: int, world_size: int, content_path: str):
    """Try out some collective operations."""
    # Note: this function is running asynchronously for each process (world_size)

    setup(rank, world_size, content_path)

    # All-reduce
    if rank == 0:
        note("### All-reduce")
    dist.barrier()  # Waits for all processes to get to this point

    tensor = torch.tensor([0., 1, 2, 3], device=get_device(rank)) + rank  # Both input and output

    note(f"Rank {rank} [before all-reduce]: {tensor}", verbatim=True)
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)  # Modifies tensor in place
    note(f"Rank {rank} [after all-reduce]: {tensor}", verbatim=True)

    # Reduce-scatter
    if rank == 0:
        note("### Reduce-scatter")
    dist.barrier()

    input = torch.arange(world_size, dtype=torch.float32, device=get_device(rank)) + rank  # Input
    output = torch.empty(1, device=get_device(rank))  # Allocate output

    note(f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}", verbatim=True)
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    note(f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}", verbatim=True)

    # All-gather
    if rank == 0:
        note("### All-gather")
    dist.barrier()

    input = output  # Input is the output of reduce-scatter
    output = torch.empty(world_size, device=get_device(rank))  # Allocate output

    note(f"Rank {rank} [before all-gather]: input = {input}, output = {output}", verbatim=True)
    dist.all_gather_into_tensor(output_tensor=output, input_tensor=input, async_op=False)
    note(f"Rank {rank} [after all-gather]: input = {input}, output = {output}", verbatim=True)

    if rank == 0:
        note("Recall that all-reduce = reduce-scatter + all-gather!")

    cleanup()


def benchmarking():
    if not torch.cuda.is_available():
        return

    note("## Benchmarking"), see("https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py")

    note("Let's see how fast commmunication happens (will restrict to just one node).")

    note("### All-reduce")

    spawn(all_reduce, world_size=2, num_elements=1024**2)
    spawn(all_reduce, world_size=4, num_elements=1024**2)

    note("### Reduce-scatter")

    spawn(reduce_scatter, world_size=2, num_elements=1024**2)
    spawn(reduce_scatter, world_size=4, num_elements=1024**2)

    note("Reference on reasoning about operations:"), see("https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce")


def all_reduce(rank: int, world_size: int, content_path: str, num_elements: int):
    setup(rank, world_size, content_path)

    # Create tensor
    tensor = torch.randn(num_elements, device=get_device(rank))

    # Warmup
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here

    # All reduce
    start_time = time.time()
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    note(f"Rank {rank}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {round(duration * 1000)} ms")

    # Estimate the bandwidth
    size_bytes = tensor.element_size() * tensor.numel()
    sent_bytes = size_bytes * 2 * (world_size - 1)  # 2x because of send and receive
    total_duration = world_size * duration
    bandwidth = sent_bytes / total_duration
    note(f"Rank {rank}: all_reduce estimated bandwidth = {round(bandwidth / 1024**3)} GB/sec")

    cleanup()


def reduce_scatter(rank: int, world_size: int, content_path: str, num_elements: int):
    setup(rank, world_size, content_path)

    # Create tensor
    input = torch.randn(world_size, num_elements, device=get_device(rank))
    output = torch.empty(num_elements, device=get_device(rank))

    # Warmup
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here

    # All reduce
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    note(f"Rank {rank}: reduce_scatter(world_size={world_size}, num_elements={num_elements}) took {round(duration * 1000)} ms")

    # Estimate the bandwidth
    data_bytes = output.element_size() * output.numel()  # How much data in the output
    sent_bytes = data_bytes * (world_size - 1)  # How much needs to be sent
    total_duration = world_size * duration  # Total time for transmission
    bandwidth = sent_bytes / total_duration
    note(f"Rank {rank}: reduce_scatter estimated bandwidth = {round(bandwidth / 1024**3)} GB/sec")

    cleanup()


def cuda_streams():
    note("Execution model: when launch a kernel (e.g., matmul), grid -> blocks -> threads")
    note("Multiple kernels execute asynchronously (A || B)")
    note("Synchronization points: `cuda.torch.synchornize` or copy to CPU")
    note("Need a way to do more fine-grained synchronization and stay on GPU...")

    note("CUDA stream: a sequence of operations that execute in order")
    note("Different streams can execute concurrently")

    if not torch.cuda.is_available():
        return

    # Create streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    device = get_device()

    note("Simple example: need to do two operations")
    note("1. Matrix multiplication")
    note("2. Copy matrix from CPU to GPU")

    N = 16384
    X1 = torch.randn(N, N, device=device)
    X2 = torch.randn(N, N, device=device)
    Y = torch.randn(N, N)  # CPU

    def run1():
        # Do two operations sequentially
        Z1 = X1 @ X2
        Z2 = Y.to(device)
        return Z1 + Z2

    def run2():
        # Do two operations in parallel
        with torch.cuda.stream(stream1):
            Z1 = X1 @ X2
        with torch.cuda.stream(stream2):
            Z2 = Y.to(device)

        # Default stream needs to wait for streams to finish
        torch.cuda.current_stream().wait_stream(stream1)
        torch.cuda.current_stream().wait_stream(stream2)
        return Z1 + Z2

    benchmark("run1", run1)
    benchmark("run2", run2)


def ddp():
    note("## Distributed data parallel")

    # Generate data
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)

    spawn(ddp_main, world_size=1, data=data, num_layers=4, num_steps=5)
    spawn(ddp_main, world_size=2, data=data, num_layers=4, num_steps=5)
    spawn(ddp_main, world_size=4, data=data, num_layers=4, num_steps=5)

    note("Notes:")
    note("- Losses are different across nodes (computed on local data)")
    note("- Gradients are same, and therefore parameters are the same")


def ddp_main(rank: int, world_size: int, content_path: str, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size, content_path)

    # Get the slice of data for this rank
    batch_size = data.size(0) // world_size
    num_dim = data.size(1)
    start_index = rank * batch_size
    end_index = start_index + batch_size
    data = data[start_index:end_index].to(get_device(rank))

    # Create MLP: # gelu(gelu(x @ params[0]) @ params[1]) ...
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    for step in range(num_steps):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude

        # Backward pass
        loss.backward()

        # Sync gradients across workers (NEW!)
        if torch.cuda.is_available():
            for param in params:
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # Update parameters
        optimizer.step()

        note(f"Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[i]) for i in range(num_layers)]}", verbatim=True)


    cleanup()


def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> nn.Parameter:
    torch.random.manual_seed(0)  # For reproducibility
    return nn.Parameter(torch.randn(num_inputs, num_outputs, device=get_device(rank)) / math.sqrt(num_outputs))


def ddp_zero3():
    note("## Distributed data parallel (DDP) with ZeRO stage 3")

    image("https://production-media.paperswithcode.com/methods/Screen_Shot_2021-07-26_at_3.17.43_PM_3oyU7Qb.png", width=0.5)
    note("- ZeRO stage 1: shard optimizer state")
    note("- ZeRO stage 2: shard optimizer state + gradients")
    note("- ZeRO stage 3: shard optimizer state + gradients + parameters")

    # Generate data
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)

    spawn(ddp_zero3_main, world_size=4, data=data, num_layers=4, num_steps=5)


def ddp_zero3_main(rank: int, world_size: int, content_path: str, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size, content_path)

    # Get the slice of data for this rank
    batch_size = int_divide(data.size(0), world_size)
    num_dim = data.size(1)
    start_index = rank * batch_size
    end_index = start_index + batch_size
    data = data[start_index:end_index].to(get_device(rank))

    # Each rank handles a subset of layers:
    #       rank 0      |       rank 1      | ...
    # layer 0 | layer 1 | layer 2 | layer 3 | ...
    num_layers_per_rank = int_divide(num_layers, world_size)

    # Which layers this rank is responsible for
    rank_start_layer = rank * num_layers_per_rank
    rank_end_layer = rank_start_layer + num_layers_per_rank
    rank_layers = list(range(rank_start_layer, rank_end_layer))

    # Create MLP: # gelu(gelu(x @ params[0]) @ params[1]) ...

    # Each rank stores only parameters and optimizer state for layers `rank`
    params = []       # i -> parameters for layer i (or None)
    optimizers = []   # i -> optimizer state for layer i (or None)
    activations = []  # i -> activations after layer i (or None)
    for i in range(num_layers):
        i_rank = i // num_layers_per_rank  # Which rank is responsible for layer i

        if rank == i_rank:  # Layer i belongs to `rank`
            param = get_init_params(num_dim, num_dim, rank)
            params.append(param)
            optimizers.append(torch.optim.AdamW([param], lr=1e-3))
        else:
            params.append(None)
            optimizers.append(None)
        activations.append(None)

    # Note: since each layer belongs to exactly one rank, we can
    # replace (all-reduce, reduce-scatter) with (broadcast, reduce).

    for step in range(num_steps):
        # Forward pass
        x = data  # Start with data
        for i in range(num_layers):
            i_rank = i // num_layers_per_rank  # Which rank is responsible for layer i

            # Broadcast (in general, all-reduce) layer i params to all processes
            if rank != i_rank:  # Layer i does not belong to `rank`, allocate
                params[i] = nn.Parameter(torch.empty(num_dim, num_dim, device=get_device(rank)))
            dist.broadcast(tensor=params[i], src=i_rank)

            # Compute activations[i]
            x = x @ params[i]       # Linear layer
            x = F.gelu(x)           # Activation function
            activations[i] = x

            # Free memory if `rank` is not storing parameters of layer i
            if rank != i_rank:
                params[i] = None

        loss = x.square().mean()  # Loss function is average squared magnitude

        # Backward pass
        for i in range(num_layers - 1, 0, -1):
            i_rank = i // num_layers_per_rank  # Which rank is responsible for layer i

            # Broadcast (in general, all-reduce) layer i params to all processes
            if rank != i_rank:  # Layer i belongs to `rank`
                params[i] = nn.Parameter(torch.empty(num_dim, num_dim, device=get_device(rank)))
            dist.broadcast(tensor=params[i], src=i_rank)

            # This is janky...
            # Set up a local computation graph from activations[i - 1] to activations[i],
            # so we can just backprop on it to get params[i].grad and activations[i - 1].grad
            if i > 0:
                # 1. Don't propagate gradients back to previous layer
                activations[i - 1] = activations[i - 1].detach()
                # 2. Do compute gradient with respect to activations
                activations[i - 1].requires_grad_(True).retain_grad()

            # Reconstruct the local computation graph from layer i -> i+1
            x = activations[i - 1] if i > 0 else data
            x = x @ params[i]                              # Linear layer
            x = F.gelu(x)                                  # Activation function
            if i == num_layers - 1:
                subloss = x.square().mean()                # Compute actual loss
            else:
                subloss = (x * activations[i].grad).sum()  # Linear approximation so we get desired gradients

            # Compute gradients just for this layer (see detach, retain_grad above)
            subloss.backward()

            # Free memory (activations and gradients)
            activations[i] = None

            # Average gradients, send to the `i_rank` responsible for it
            # Broadcast layer to all processes
            if torch.cuda.is_available():
                dist.reduce(tensor=params[i].grad, dst=i_rank, op=dist.ReduceOp.AVG)

            if rank == i_rank:  # Layer i belongs to `rank`
                optimizers[i].step()

            # Free memory if `rank` is not storing it
            if rank != i_rank:
                params[i] = None

        note(f"Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[i]) for i in rank_layers]}", verbatim=True)

    cleanup()


def tensor_parallelism():
    note("## Tensor parallelism")

    note("Key idea: split the big matmul across ranks")
    note("Each rank will have the same data")

    # Create data
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)

    spawn(tensor_parallelism_main, world_size=4, data=data, num_layers=4)


def tensor_parallelism_main(rank: int, world_size: int, content_path: str, data: torch.Tensor, num_layers: int):
    setup(rank, world_size, content_path)

    # Note: no sharding of the data
    data = data.to(get_device(rank))
    batch_size = data.size(0)
    num_dim = data.size(1)
    sharded_num_dim = num_dim // world_size  # Shard `num_dim`

    # Create model (each rank gets 1/world_size of the parameters)
    params = [get_init_params(num_dim, sharded_num_dim, rank) for i in range(num_layers)]

    # Forward pass
    x = data
    for i in range(num_layers):
        # Compute activations (batch_size x sharded_num_dim)
        x = x @ params[i]  # Note: this is only on a slice of the parameters
        x = F.gelu(x)

        # Allocate memory for activations (world_size x batch_size x sharded_num_dim)
        activations = [torch.empty(batch_size, sharded_num_dim, device=get_device(rank)) for _ in range(world_size)]

        # Send via all gather
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        # Just concatenate them to get (batch_size x num_dim)
        x = torch.cat(activations, dim=1)

    note(f"Rank {rank}: forward pass produced activations {summarize_tensor(x)}", verbatim=True)

    # Backward pass: left as a homework exercise

    cleanup()


def pipeline_parallelism():
    note("## Pipeline parallelism")

    note("Key idea: each rank gets a subset of layers and all the data")
    image("https://pytorch.org/docs/stable/_images/pipe.png", width=0.5)

    # Create data
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)

    spawn(pipeline_parallelism_main, world_size=2, data=data, num_layers=4, num_micro_batches=4)
    spawn(pipeline_parallelism_main, world_size=4, data=data, num_layers=4, num_micro_batches=4)


def pipeline_parallelism_main(rank: int, world_size: int, content_path: str, data: torch.Tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size, content_path)

    # All the data
    data = data.to(get_device(rank))
    batch_size = data.size(0)
    num_dim = data.size(1)

    # Note: split up layers by rank
    num_layers_per_rank = int_divide(num_layers, world_size)

    micro_batch_size = int_divide(batch_size, num_micro_batches)

    # Each rank gets a subset of layers
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers_per_rank)]

    # Forward pass

    # Break up into micro batches to minimize the bubble
    if rank == 0:
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank)) for _ in range(num_micro_batches)]

    for x in micro_batches:
        # Get from previous rank
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # Do the compute
        for param in params:
            x = x @ param
            x = F.gelu(x)

        note(f"Rank {rank}: forward pass produced {summarize_tensor(x)}", verbatim=True)

        # Send to the next rank
        if rank + 1 < world_size:
            dist.send(tensor=x, dst=rank + 1)

    # Backward pass: left as a homework exercise

    # Note: we haven't done careful overlapping of the communication/computation yet

    cleanup()


def further_reading():
    note("Overview of different parallelism strategies"), see("https://github.com/stas00/ml-engineering/tree/master/training/model-parallelism")

    # Libraries
    note("PyTorch FSDP"), see("https://arxiv.org/pdf/2304.11277.pdf")
    note("DeepSpeed"), see("https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/")
    note("MegatronLM"), see("https://github.com/NVIDIA/Megatron-LM")
    note("GPT-NeoX"), see("https://github.com/EleutherAI/gpt-neox")
    note("Levanter"), see("https://github.com/stanford-crfm/levanter")

    # Profiling
    note("https://developer.nvidia.com/blog/nvidia-tools-extension-api-nvtx-annotation-tool-for-profiling-code-in-python-and-c-c/")

    # Collective operations
    note("https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html")
    note("https://pytorch.org/tutorials/intermediate/dist_tuto.html")

    # InfiniBand
    note("Infiniband (Wikipedia)"), see("https://en.wikipedia.org/wiki/InfiniBand")
    note("Introduction to InfiniBand (whitepaper)"), see("https://network.nvidia.com/pdf/whitepapers/IB_Intro_WP_190.pdf")
    note("Introduction to InfiniBand Networks"), see("https://www.youtube.com/watch?v=2gidd6lLiH8")
    note("InfiniBand"), see("https://www.youtube.com/watch?v=wecZb5lHkXk")

############################################################

def setup(rank: int, world_size: int, content_path: str):
    # This is where master lives (rank 0)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # For executable lecture, so we can write to the content file using `note`
    util.content_path = content_path


def cleanup():
    torch.distributed.destroy_process_group()


def int_divide(a: int, b: int):
    """Return a / b and throw an error if there's a remainder."""
    assert a % b == 0
    return a // b

def summarize_tensor(tensor: torch.Tensor) -> str:
    return "x".join(map(str, tensor.shape)) + "[" + str(round(tensor.view(-1)[0].item(), 4)) + "...]"


def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Time it for real now!
    times: List[float] = []
    for trial in range(num_trials):  # Do it multiple times to capture variance
        start_time = time.time()

        run()  # Actually perform computation
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

        end_time = time.time()
        times.append((end_time - start_time) * 1000)

    mean_time = mean(times)
    note(f"{description}: {list(map(round1, sorted(times)))} (mean {round1(mean_time)} ms)", pop_stack=True)


def spawn(main: Callable, world_size: int, *args, **kwargs):
    note(f"spawn(world_size = {world_size})")
    # Note: assume kwargs are in the same order as what main needs
    args = args + tuple(kwargs.values())
    mp.spawn(main, args=(world_size, util.content_path, *args), nprocs=world_size, join=True)


def note_system(command: List[str]):
    output = subprocess.check_output(command).decode('utf-8')
    output = remove_ansi_escape_sequences(output)
    note(output, verbatim=True)


def remove_ansi_escape_sequences(text):
    ansi_escape_pattern = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape_pattern.sub('', text)


if __name__ == "__main__":
    init_content("lecture_08-content.js")
    lecture_08()
