from util import *
from facts import *
from references import *
import torch.nn.functional as F
import timeit
import torch
import wandb
from typing import Iterable, Tuple
from torch import nn
import numpy as np

def lecture_02():
    note("## Pytorch primitives")

    note("We will build up all the primitives we need gradually.")

    note("Remember, this class is all about efficiency, so we need to think carefully about:")
    note("- Memory")
    note("- Compute")
    note("- Communication")

    note("Let's start with the nuts and bolts of Pytorch.")
    randomness()
    tensor_basics()
    floating_point()
    gpus()
    tensor_operations()
    flops()
    gradients()

    note("Pytorch modules (`nn.Module`) are classes that help construct the computation graph.")
    note("Pytorch has some pre-defined ones (nn.Embedding, nn.Linear, nn.LayerNorm, etc.).")
    note("But it is instructive to build them ourselves.")
    parameters()
    embeddings()
    custom_model()

    note("Now let's train some models.")
    data_loading()
    optimizer()
    train_loop()
    checkpointing()
    profiling()


def get_memory_usage(x: torch.Tensor):
    return x.numel() * x.element_size()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def get_promised_flop_per_sec(device: str) -> float:
    properties = torch.cuda.get_device_properties(device)
    if "A100" in properties.name:
        return a100_flop_per_sec
    if "H100" in properties.name:
        return h100_flop_per_sec
    raise ValueError(f"Unknown device: {device}")


def same_storage(x: torch.Tensor, y: torch.Tensor):
    return x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()




def randomness():
    note("Randomness shows up in many places: parameter initialization, dropout, data ordering, etc.")
    note("For reproducibility, we recommend you always fix the random seed for each use of randomness (or pass it in as an argument).")
    note("Determinism is particularly useful when debugging, so you can hunt down the bug.")
    note("There are three places to set the random seed, which you should do all at once just to be safe.")
    seed = 0
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def tensor_basics():
    note("## Tensors")
    see("https://pytorch.org/docs/stable/tensors.html")

    note("Tensors are the basic building block for storing parameters, activations, gradients, etc.")
    note("You can create tensors in multiple ways:")
    note("- Specify the entries (useful for toy examples); here's a 2x3 matrix (order 2 tensor)")
    x = torch.tensor([
        [1., 2, 3],
        [4, 5, 6],
    ])
    note("- All zeros (a 16x32 matrix)")
    x = torch.zeros(16, 32)
    note("- All ones (a 16x32 matrix)")
    x = torch.ones(16, 32)
    note("- Sample each entry of the 16x32 matrix from a univariate Gaussian N(0, 1)")
    x = torch.randn(16, 32)
    note("- Don't initialize the values (to save compute), "
         "because you want to use some custom logic to set the values later")
    x = torch.empty(16, 32)

    note("Let's think carefully about memory use, since we're often memory-bound.")
    note("Memory is determined by the (i) number of values and (ii) data type of each value.")
    x = torch.zeros(16, 32)
    assert x.dtype == torch.float32  # Default type
    assert x.size() == torch.Size([16, 32])
    assert x.numel() == 16 * 32
    assert x.element_size() == 4  # Float is 4 bytes
    assert get_memory_usage(x) == 16 * 32 * 4  # 2048 bytes

    note("One matrix in the feedforward layer of GPT-3:")
    assert get_memory_usage(torch.empty(12288 * 4, 12288)) == 2304 * 1024 * 1024  # 2.3 GB
    note("...which is a lot!")

    note("Tensors can also store integers (for representing tokenized text):")
    x = torch.tensor([1, 2, 3])
    assert x.dtype == torch.int64  # Default int type
    assert x.element_size() == 8  # int64 (long) is 8 bytes


def floating_point():
    note("## float32"), see("https://en.wikipedia.org/wiki/Single-precision_floating-point_format")
    note("The float32 data type (also known as fp32 or single precision) is the default.")
    note("In many scientific computing applications, float32 is viewed as a baseline; you could use double precision (float64) in some cases.")
    note("In deep learning, the needs are different.")

    note("## float16"), see("https://en.wikipedia.org/wiki/Half-precision_floating-point_format")
    note("The float16 data type (also known as fp16 or half precision) allows us to cut down the memory.")
    x = torch.zeros(16, 32, dtype=torch.float16)
    assert x.element_size() == 2
    note("However, the problem is that the dynamic range isn't great.")
    x = torch.tensor([1e-8], dtype=torch.float16)
    assert x == 0  # Underflow!
    note("If this happens when you train, you can get instability.")

    note("## bfloat16"), see("https://en.wikipedia.org/wiki/Bfloat16_floating-point_format")
    note("Google Brian developed the bfloat data type (brain floating point) in 2018 to address this issue.")
    note("bfloat16 uses the same memory as float16 but has the same dynamic range as float32!")
    note("The only catch is that the resolution is worse, but this matters less for deep learning.")
    x = torch.tensor([1e-8], dtype=torch.bfloat16)
    assert x != 0  # No underflow!

    note("Let's compare the dynamic ranges and memory usage of the different data types:")
    note(f"- float32: {torch.finfo(torch.float32)}")
    note(f"- float16: {torch.finfo(torch.float16)}")
    note(f"- bfloat16: {torch.finfo(torch.bfloat16)}")

    note("General rule of thumb:")
    note("- Training with float32 works, but requires lots of memory.")
    note("- Training with float16 is risky and you can get instability.")
    note("- Training with bfloat16 is safer, but you might want to upcast to float32 for certain things (more on this later).")


def gpus():
    note("## GPUs")

    note("By default, tensors are stored in CPU memory.")
    x = torch.zeros(32, 32)
    assert x.device == torch.device("cpu")

    note("However, in order to take advantage of the massive parallelism of GPUs, "
         "we need to move them to GPU memory.")
    image("https://www.researchgate.net/publication/338984158/figure/fig2/AS:854027243900928@1580627370716/Communication-between-host-CPU-and-GPU.png")

    note("Let's first see if we have any GPUs.")
    if not torch.cuda.is_available():
        return

    num_gpus = torch.cuda.device_count()
    note(f"We have {num_gpus} GPUs.")
    for i in range(num_gpus):
        properties = torch.cuda.get_device_properties(i)
        note(f"Device {i}: {properties}")

    memory_allocated = torch.cuda.memory_allocated()
    note(f"GPU memory used: {memory_allocated}")

    note("Move the tensor to GPU memory (device 0).")
    y = x.to("cuda:0")
    assert y.device == torch.device("cuda", 0)

    note("Create a tensor directly on the GPU:")
    z = torch.zeros(32, 32, device="cuda:0")

    new_memory_allocated = torch.cuda.memory_allocated()
    note(f"GPU memory used: {new_memory_allocated}")
    assert new_memory_allocated - memory_allocated == 2 * (32 * 32 * 4)


def tensor_operations():
    note("Most tensors are created from performing operations on other tensors.")
    note("Each operation has some memory and compute consequence.")

    note("## Storage")
    note("Pytorch tensors are really pointers into contiguous blocks of allocated memory "
         "with metadata describing how to get to any element of the tensor.")
    image("https://martinlwx.github.io/img/2D_tensor_strides.png")
    see("https://pytorch.org/docs/stable/generated/torch.Tensor.stride.html")
    x = torch.tensor([
        [1., 2, 3],
        [4, 5, 6],
    ])
    note("To go to the next row (dim 0), skip 3 elements.")
    assert x.stride(0) == 3
    note("To go to the next column (dim 1), skip 1 element.")
    assert x.stride(1) == 1

    note("## Slicing and dicing")
    note("Many operations simply provide a different *view* of the tensor.")
    note("This does not make a copy, and therefore mutations in one tensor affects the other.")
    y = x[0]
    assert torch.equal(y, torch.tensor([1., 2, 3]))
    assert same_storage(x, y)

    y = x[:, 1]
    assert torch.equal(y, torch.tensor([2, 5]))
    assert same_storage(x, y)

    y = x.view(3, 2)
    assert torch.equal(y, torch.tensor([[1, 2], [3, 4], [5, 6]]))
    assert same_storage(x, y)

    y = x.transpose(1, 0)
    assert torch.equal(y, torch.tensor([[1, 4], [2, 5], [3, 6]]))
    assert same_storage(x, y)

    note("Check that mutating x also mutates y.")
    x[0][0] = 100
    assert y[0][0] == 100

    note("Note that some views are non-contiguous, which means that further views aren't possible.")
    try:
        x.transpose(1, 0).view(2, 3)
        assert False
    except RuntimeError as e:
        assert "view size is not compatible with input tensor's size and stride" in str(e)
    note("One can use reshape, which makes a copy if needed.")
    y = x.reshape(3, 2)
    assert same_storage(x, y)
    y = x.transpose(1, 0).reshape(6)
    assert not same_storage(x, y)

    note("Now for the operations that make a copy...")

    note("# Elementwise operations")
    note("These operations apply some operation to each element of the tensor and "
         "return a (new) tensor of the same shape.")
    x = torch.tensor([1, 4, 9])
    assert torch.equal(x.pow(2), torch.tensor([1, 16, 81]))
    assert torch.equal(x.sqrt(), torch.tensor([1, 2, 3]))
    assert torch.equal(x.rsqrt(), torch.tensor([1, 1 / 2, 1 / 3]))  # i -> 1/sqrt(x_i)

    assert torch.equal(x + x, torch.tensor([2, 8, 18]))
    assert torch.equal(x * 2, torch.tensor([2, 8, 18]))
    assert torch.equal(x / 0.5, torch.tensor([2, 8, 18]))

    note("Dropout: zero out each element of the tensor with probability p.")
    x = torch.ones(16384)
    x = F.dropout(x, p=0.3)  # Dropout 0.3 fraction of elements
    # Note dropout rescales so the mean is unchanged
    assert torch.allclose(x.mean(), torch.tensor(1.), atol=0.02)
    # 0.3 fraction should be now zero
    assert torch.allclose(torch.sum(x == 0) / x.numel(), torch.tensor(0.3), atol=0.01)

    note("Take the upper triangular part of a matrix.")
    note("This is useful for computing an causal attention mask, "
         "where M[i, j] is the contribution of i to j.")
    assert torch.equal(torch.ones(3, 3).triu(), torch.tensor([
        [1, 1, 1],
        [0, 1, 1],
        [0, 0, 1]],
    ))

    note("## Aggregate operations")
    x = torch.tensor([
        [1., 2, 3],
        [4, 5, 6],
    ])
    note("By default, mean aggregates over the entire matrix.")
    assert torch.equal(torch.mean(x), torch.tensor(3.5))
    note("We can aggregate only over a subset of the dimensions by specifying `dim`.")
    assert torch.equal(torch.mean(x, dim=1), torch.tensor([2, 5]))
    note("Variance has the same form factor as mean.")
    assert torch.equal(torch.var(torch.tensor([-10., 10])), torch.tensor(200))  # Note: Bessel corrected

    note("## Batching")
    note("As a general rule, matrix multiplications are very optimized, "
         "so the more we can build up things into a single matrix operation, the better.")
    note("The `stack` operation adds a new dimension indexing the tensor we're stacking.")
    note("You can use `stack` given a set of data points, create a batch dimension.")
    assert torch.equal(torch.stack([x, x], dim=0), torch.tensor([
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
    ]))
    note("The `cat` operation concatenates two tensors along some dimension and does not add another dimension")
    note("This is useful for combining batching multiple matrix operations (e.g., Q, K, V in attention).")
    x = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
    ])
    assert torch.equal(torch.cat([x, x], dim=1), torch.tensor([
        [1, 2, 3, 1, 2, 3],
        [4, 5, 6, 4, 5, 6],
    ]))

    # TODO: chunk
    
    note("Squeezing and unsqueezing are trivial operations that simply add or a remove a dimension.")
    x = torch.tensor([1, 2, 3])
    note("Unsqueeze adds a dimension.")
    assert torch.equal(torch.unsqueeze(x, dim=0), torch.tensor([[1, 2, 3]]))
    note("Squeeze removes a dimension.")
    assert torch.equal(torch.squeeze(torch.unsqueeze(x, dim=0)), x)

    note("## Matrix multiplication")
    note("Finally, the bread and butter of deep learning: matrix multiplication.")
    note("Note that the first matrix could have an dimensions (batch, sequence length).")
    x = torch.ones(4, 4, 16, 32)
    y = torch.ones(32, 16)
    z = x @ y
    assert z.size() == torch.Size([4, 4, 16, 16])


def flops():
    note("## Compute")

    note("Having gone through all the operations, let us examine their computational cost.")

    note("A floating-point operation (FLOP) is a basic operation like addition (x + y) or multiplication (x y).")
    note("Two terribly confusing acronyms:")
    note("- FLOPs: floating-point operations (measure of compute)")
    note("- FLOP/s: floating-point operations per second (also written as FLOPS), "
         "which is used to measure the speed of a piece of hardware (more on this later).")

    note("## Linear model")
    note("As motivation, suppose you have a linear model.")
    note("- We have n points")
    note("- Each point is d-dimsional")
    note("- The linear model maps each d-dimensional vector to a k outputs")
    n = 16384
    d = 32768
    k = 8192

    device = get_device()
    x = torch.ones(n, d, device=device)
    w = torch.randn(d, k, device=device)
    y = x @ w
    note("We have one multiplication (x[i][j] * w[j][k]) and one addition per (i, j, k) triple.")
    num_flops = 2 * n * d * k
    note("Therefore the FLOPs is: {num_flops}")

    note("For a Transformer, this nice blog post explains:")
    see("https://www.adamcasson.com/posts/transformer-flops")

    note("## FLOPs of other operations")
    note("- Elementwise operation on a m x n matrix requires O(m n) FLOPs.")
    note("- Addition of two m x n matrices requires m n FLOPs.")
    note("In general, no other operation that you'd encounter in deep learning "
         "is as expensive as matrix multiplication for large enough matrices.")
    note("Counting FLOPs in training a complex Transformer might seem daunting at first, "
         "but all you have to do is to count the FLOPs of the matrix multiplications!")

    note("## Model FLOPs utilization (MFU)")

    note("How do our FLOPs calculations translate to wall-clock time (seconds)?")
    note("Let us time it!")
    num_trials = 3
    def run():
        x @ w
        # Wait until CUDA threads are done, or else timing won't work!
        torch.cuda.synchronize()
    total_time = timeit.timeit(run, number=num_trials) / num_trials
    note(f"Matrix multiplication takes {total_time} seconds.")
    model_flop_per_sec = num_flops / total_time

    note("Usually, a GPU is advertised as having a certain number of FLOP/s")
    note("- An A100 GPU has a peak performance of 312 teraFLOP/s"), see(a100_flop_per_sec)
    note("- An H100 GPU has a peak performance of 989 teraFLOP/s)"), see(h100_flop_per_sec)
    promised_flop_per_sec = get_promised_flop_per_sec(device)

    note("Model FLOPs Utilization (MFU) is actual FLOP/s over promised FLOP/s")
    mfu = model_flop_per_sec / promised_flop_per_sec
    note(f"MFU: {mfu:.2f}")


def gradients():
    note("## Gradients")

    note("So far, we've constructed tensors (which correspond to either parameters or data) "
         "and passed them through operations (forward).")
    note("Now, we're going to compute the gradient (backward).")

    note("As a simple example, let's consider the simple linear model: "
         "y = 0.5 (x * w - 5)^2")

    # Forward: compute values
    x = torch.tensor([1., 2, 3])
    assert x.requires_grad == False  # By default, no gradient
    w = torch.tensor([1., 1, 1], requires_grad=True)  # Want gradient
    pred_y = x @ w
    loss = 0.5 * (pred_y - 5).pow(2)
    assert w.grad is None

    # Backward: compute gradients
    loss.backward()
    assert loss.grad is None
    assert pred_y.grad is None
    assert x.grad is None
    assert torch.equal(w.grad, torch.tensor([1, 2, 3]))
    w.grad = None  # Free up the memory


def parameters():
    input_dim = 16384
    hidden_dim = 32

    W = nn.Parameter(torch.randn(input_dim, hidden_dim))
    note("Each model parameter behaves like a tensor.")
    assert isinstance(W, torch.Tensor)
    note("Can also access the underlying tensor:")
    assert type(W.data) == torch.Tensor

    note("Parameter initialization")
    x = nn.Parameter(torch.randn(input_dim))
    output = x @ W
    assert output.size() == torch.Size([hidden_dim])
    note("Note that each element of `output` scales as sqrt(num_inputs): {output[0]}.")
    note("Large values can cause gradients to blow up and cause training to be unstable.")
    note("We want an initialization that is invariant to `hidden_dim`.")
    note("To do that, we do something like Xavier initialization, which simply rescales")
    see("https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf")
    see("https://ai.stackexchange.com/questions/30491/is-there-a-proper-initialization-technique-for-the-weight-matrices-in-multi-head")
    W = nn.Parameter(torch.randn(input_dim, hidden_dim) / np.sqrt(input_dim))
    output = x @ W
    note("Now each element of `output` is constant: {output[0]}.")


def embeddings():
    note("Embeddings map token sequences (integer indices) to vectors.")
    V = 128 # Vocabulary size
    H = 64  # Hidden dimension
    embedding = nn.Embedding(V, H)

    note("Usually, each batch of data has B sequences of length L, each element is 0, ..., V-1.")
    B = 16  # Batch size
    L = 32  # Length of sequence
    x = torch.randint(V, (B, L))
    note("We can map each token to an embedding vector.")
    x = embedding(x)
    assert x.size() == torch.Size([B, L, H])


class Linear(nn.Module):
    """Simple linear layer."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


class Cruncher(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.hidden_layers = nn.ModuleList([
            Linear(dim, dim)
            for i in range(num_layers)
        ])
        self.final = Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply linear layers
        B, D = x.size()
        for layer in self.hidden_layers:
            x = layer(x)

        # Apply final head
        x = self.final(x)
        assert x.size() == torch.Size([B, 1])
        x = x.squeeze(-1)  # Remove the last dimension
        assert x.size() == torch.Size([B])

        return x

def custom_model():
    note("Let's define a simple model.")
    D = 64  # Dimension
    num_layers = 2
    model = Cruncher(dim=D, num_layers=num_layers)

    param_sizes = [
        (name, param.numel())
        for name, param in model.state_dict().items()
    ]
    assert param_sizes == [
        ("hidden_layers.0.weight", D * D),
        ("hidden_layers.1.weight", D * D),
        ("final.weight", D),
    ]
    num_parameters = sum(param.numel() for param in model.parameters())
    assert num_parameters == (D * D) + (D * D) + D

    note("Remember to move the model to the GPU.")
    device = get_device()
    model = model.to(device)

    note("Run the model on some data.")
    B = 8  # Batch size
    x = torch.randn(B, D, device=device)
    y = model(x)
    assert y.size() == torch.Size([B])


def get_batch(data: torch.Tensor, batch_size: int, sequence_length: int, device: str) -> torch.Tensor:
    start_indices = torch.randint(len(data) - sequence_length, (batch_size,))
    assert start_indices.size() == torch.Size([batch_size])
    x = torch.tensor([data[start:start + sequence_length] for start in start_indices])
    assert x.size() == torch.Size([batch_size, sequence_length])

    note("It can be useful to pin the memory of x and copy GPU to CPU asynchronously.")
    x = x.pin_memory().to(device, non_blocking=True)
    
    note("There are fancier things you can do to simultaneously the next batch of data from CPU to GPU while we return the current batch for processing.")
    # https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39?permalink_comment_id=3417135

    return x


def data_loading():
    note("In language modeling, we assume your data is one (long) sequence of integers (output by the tokenizer).")

    note("It is convenient to serialize them as numpy arrays (done by the tokenizer).")
    orig_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int16)
    orig_data.tofile("data.npy")

    note("You can load them back as numpy arrays.")
    data = np.memmap("data.npy", dtype=np.int16)
    assert np.array_equal(data, orig_data)

    note("A data loader yields a batch of sequences.")
    B = 2  # Batch size
    L = 4  # Length of sequence
    x = get_batch(data, batch_size=B, sequence_length=L, device=get_device())
    assert x.size() == torch.Size([B, L])


class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params: Iterable[nn.Parameter], lr: float = 0.01):
        super(AdaGrad, self).__init__(params, dict(lr=lr))

    def step(self):
        # TODO
        for group in self.param_groups:
            for p in group["params"]:
                p.data -= group["lr"] * p.grad.data


def optimizer():
    D = 4
    model = Cruncher(dim=D, num_layers=0).to(get_device())
    optimizer = AdaGrad(model.parameters(), lr=0.01)
    see(model.state_dict())

    note("Compute gradients")
    x = torch.randn(2, D, device=get_device())
    y = torch.tensor([4., 5.], device=get_device())
    pred_y = model(x)
    loss = F.mse_loss(input=pred_y, target=y)
    loss.backward()
    
    note("Take a step")
    optimizer.step()
    see(model.state_dict())

    note("Free up the memory (optional)")
    optimizer.zero_grad(set_to_none=True)

    # TODO: count flops, memory
    note("Compute the memory")
    note("Compute the FLOPs")


def train_loop():
    note("Let's do regression")

    note("Create a true function, which is linear with weights (0, 1, 2, ..., D-1).")
    D = 16
    true_w = torch.arange(D, dtype=torch.float32, device=get_device())
    def get_batch(B: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(B, D).to(get_device())
        true_y = x @ true_w
        return (x, true_y)

    note("Let's do a basic run")
    train("simple", get_batch, D=D, num_layers=0, B=4, num_train_steps=100, lr=0.01)

    # TODO
    note("Let's do some hyperparameter tuning")
    # Learning rate
    # Batch size
    # Number of layers, hidden size

    # Gradient clipping
    # Gradient accumulation
    # fp16


def train(name: str, get_batch,
          D: int, num_layers: int,
          B: int, num_train_steps: int, lr: float):
    wandb.init(project=f"lecture2-optimizer-{name}")

    model = Cruncher(dim=D, num_layers=0).to(get_device())
    optimizer = AdaGrad(model.parameters(), lr=0.01)

    for t in range(num_train_steps):
        # Get data
        x, y = get_batch(B=B)

        # Forward (inference)
        pred_y = model(x)

        # Compute loss
        loss = F.mse_loss(pred_y, y)
        wandb.log({"loss": loss.item()})

        # Backward (gradients)
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()


def checkpointing():
    model = Cruncher(dim=64, num_layers=3).to(get_device())
    optimizer = AdaGrad(model.parameters(), lr=0.01)

    note("During training, it is useful to periodically save your model (and optimizer state) to disk.")
    note("Long running jobs will certainly crash, and you don't want to lose all your progress.")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, "model_checkpoint.pt")

    note("Loading the model and optimizer state is quite simple.")
    loaded_checkpoint = torch.load("model_checkpoint.pt")
    #assert checkpoint == loaded_checkpoint


def profiling():
    # TODO: profiling
    pass
