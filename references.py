from util import *
from facts import *

shannon1950 = MethodSpec(
    name="Prediction and Entropy of Printed English", date="1950-09-15",
    url="https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf",
)

bengio2003 = MethodSpec(
    name="A Neural Probabilistic Language Model", date="2003-02-01",
    url="https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf",
)

susketver2014 = MethodSpec(
    name="Sequence to Sequence Learning with Neural Networks", date="2014-09-10",
    url="https://arxiv.org/pdf/1409.3215.pdf",
)

adam = MethodSpec(
    name="Adaptive Moment Estimation (Adam)", date="2014-12-22",
    url="https://arxiv.org/pdf/1412.6980.pdf",
    description=[
        "Introduces optimizer that combines momentum and RMSprop",
    ],
)

bahdanau2015_attention = MethodSpec(
    name="Neural Machine Translation by Jointly Learning to Align and Translate", date="2014-09-01",
    url="https://arxiv.org/pdf/1409.0473.pdf",
)


transformer = MethodSpec(
    name="Transformer architecture", date="2017-06-12",
    url="https://arxiv.org/pdf/1706.03762.pdf",
    description=[
        "Introduces the encoder-decoder Transformer architecture",
        "Scaled dot-product attention, multi-headed attention, sinusoidal positional embeddings",
        "Training with warmup learning rate",
        "Applied to machine translation",
    ],
)

adamw = MethodSpec(
    name="AdamW", date="2017-11-14",
    url="https://arxiv.org/pdf/1711.05101.pdf",
    description=[
        "Improves Adam by decoupling weight decay",
    ],
)

gpt_2 = ModelSpec(
    name="GPT-2", organization="OpenAI", date="2019-02-14",
    url="https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
    architecture=ArchitectureSpec(
        num_parameters=1.5e9,
    ),
)

t5 = ModelSpec(
    name="T5", organization="Google", date="2019-10-23",
    url="https://arxiv.org/pdf/1910.10683.pdf",
    description="Encoder-decoder, frames tasks as text-to-text",
    data=DataSpec(
        name="Colossal Cleaned Common Crawl (C4)",  # Introduced in this paper
        description="Filtering (Section 2.2)",
    ),
    architecture=ArchitectureSpec(
        num_parameters=11e9,
        description="Remove bias from feedforward layers",
    ),
)

############################################################
# 2020

kaplan_scaling_laws = MethodSpec(
    name="Scaling Laws for Neural Language Models (Kaplan+ 2020)", organization="OpenAI", date="2020-01-23",
    url="https://arxiv.org/pdf/2001.08361.pdf",
    description=[
        "Vary model size, dataset size, compute; get power laws",
        "Larger models require fewer tokens",
    ],
)

swiglu = MethodSpec(
    name="GLU Variants Improve Transformer", organization="Google", date="2020-02-14",
    url="https://arxiv.org/pdf/2002.05202.pdf",
    description=[
        "Experiments with different activation functions",
        "Activation functions: ReLU, GeLU, Swish",
        "Apply idea of gated units (GLU): ReGLU, GeGLU, SwiGLU",
        "FFN-SwiGLU = Swish(xW1) * xV) W2",
        "Have 3 matrices now, so make hidden dimension 2/3 of the 2 matrix version",
    ],
)

longformer = MethodSpec(
    name="Longformer", organization="AllenAI", date="2020-04-10",
    url="https://arxiv.org/pdf/2004.05150.pdf",
    description=[
        "Sliding window (local) attention",
        "Global attention to capture task-specific information",
    ],
)

megatron_lm = ModelSpec(
    name="MegatronLM", organization="NVIDIA", date="2020-04-09",
    url="https://arxiv.org/pdf/2104.04473.pdf",
    description=[
        "Compose tensor, pipeline, data parallelism",
        "Achieve 52% MFU on 1T parameter model on 3072 GPUs",
    ]
),

gpt_3 = ModelSpec(
    name="GPT-3", organization="OpenAI", date="2020-06-11",
    url="https://arxiv.org/pdf/2005.14165.pdf",
    data=DataSpec(
        num_tokens=300e9,  # 570GB (500B tokens)
    ),
    architecture=ArchitectureSpec(
        description="Same as GPT-2, but alternating sparse and dense attention layers",
        num_parameters=175e9, num_layers=96, dim_model=12288, num_heads=96, dim_head=128,
    ),
    training=TrainingSpec(
        batch_size_tokens=3.2e6, learning_rate=6e-5,
        hardware="V100s",
    )
)

mmlu = DataSpec(
    name="Massively Multilingual Language Understanding (MMLU)", organization="Berkeley", date="2020-09-07",
    url="https://arxiv.org/pdf/2009.03300.pdf",
    description=[
        "57 subjects, multiple-choice",
    ]
)

the_pile = DataSpec(
    name="The Pile", organization="EleutherAI", date="2020-12-31",
    url="https://arxiv.org/pdf/2101.00027.pdf",
    description="825GB text, 22 diverse subsets (CommonCrawl, PubMed, ArXiv, GitHub, StackExchange, USPTO, OpenWebText2, Books3, etc.)",
)

############################################################
# 2021

rope = MethodSpec(
    name="Rotary Positional Embedding (RoPE)", date="2021-04-20",
    url="https://arxiv.org/pdf/2104.09864.pdf",
    description=[
        "Encodes absolute position with rotation matrix, incorporate relative position dependency in self-attention",
        "Key: R W x, where R is a block-diagonal sequence of d/2 rotation matrices (equation 13)",
        "Extrapolates to longer sequences",
    ],
)

gpt_j = ModelSpec(
    name="GPT-J", organization="EleutherAI", date="2021-06-04",
    url="https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/",
    architecture=ArchitectureSpec(
        num_parameters=6.7e9,
        description="Attention and feedforward layers put in parallel",
    ),
    training=TrainingSpec(
        hardware="v3 256 TPUs (5.4 PFLOPs) for 5 weeks",
    ),
)

gopher = ModelSpec(
    name="Gopher", organization="DeepMind", date="2021-12-08",
    url="https://arxiv.org/pdf/2112.11446.pdf",
    data=DataSpec(
        name="MassiveText",
        num_tokens=300e9, # Total: 10.5 TB text
    ),
    architecture=ArchitectureSpec(
        num_parameters=380e9,
        num_layers=80, num_heads=128, dim_model=16384,
    ),
    training=TrainingSpec(
    ),
)

############################################################
# 2022

instruct_gpt = MethodSpec(
    name="Training language models to follow instructions with human feedback",
    url="https://arxiv.org/pdf/2203.02155.pdf",
)

chinchilla = ModelSpec(
    name="Chincilla", organization="DeepMind", date="2022-03-29",
    url="https://arxiv.org/pdf/2203.15556.pdf",
    description=[
        "Introduced the rigorous analysis scaling laws for language models",
        "Key improvement over Kaplan: tune learning rate for the compute budget",
        "Approach 1: for each model size, train with 4 learning rates, vary number of training tokens, fit lower envelope",
        "Approach 2 (IsoFLOP): for each model size, train with 9 training budgets, take last point",
        "Approach 3: fit parametric function L(N, D) = E + A/N^alpha + B/D^beta to data collected from approaches 1 and 2",
        "Conclusion: model and data should scale up at same rate",
        "Table 3: extrapolate to 10 trillion parameters",
    ],
    data=DataSpec(
        name="MassiveText, different data distribution",
        num_tokens=1.5e12,
    ),
    architecture=ArchitectureSpec(
        num_parameters=70e9,
    ),
    training=TrainingSpec(
    ),
)

palm = ModelSpec(
    name="PaLM", organization="Google", date="2022-04-05",
    url="https://arxiv.org/pdf/2204.02311.pdf",
    data=DataSpec(
        description="Social media conversations, webpages, books, GitHub, Wikipedia, news",
    ),
    architecture=ArchitectureSpec(
        num_parameters=540.35e9,
        description="SwiGLU, parallelize attention and feedforward layers, multi-query attention, RoPE, remove biases",
        num_layers=118, num_heads=48, dim_model=18432,
    ),
    training=TrainingSpec(
        hardware="6144 TPUv4, 46.2% MFU",
        optimizer="Adafactor without factorization",
        description="Introduced the term model FLOPs utilization (MFU) metric (observed tokens/sec / theoretical max tokens/sec)",
    ),
)

gpt_neox = ModelSpec(
    name="GPT-NeoX", organization="EleutherAI", date="2022-04-14",
    url="https://arxiv.org/pdf/2204.06745.pdf",
    description="",
    data=DataSpec(
        name="The Pile",
        references=[the_pile],
    ),
    architecture=ArchitectureSpec(
        num_parameters=20e9,
        description="Use RoPE, parallel attention and feedforward layers (15% throughput increase)",
    ),
    training=TrainingSpec(
        hardware="12x8 A100s",
    ),
)

opt_175b = ModelSpec(
    name="OPT", organization="Meta", date="2022-05-03",
    url="https://arxiv.org/pdf/2205.01068.pdf",
    data=DataSpec(
        description="The Pile, PushShift.io Reddit, deduplication",
    ),
    architecture=ArchitectureSpec(
        num_parameters=175e9,
    ),
    training=TrainingSpec(
        hardware="992 A100 80GB for 2 months, lots of hardware failures",
        description="FSDP with Megatron-LM, fp16 with loss scaling",
    ),
)

bloom = ModelSpec(
    name="BLOOM", organization="BigScience", date="2022-11-09",
    data=DataSpec(
        name="ROOTS",
    ),
    architecture=ArchitectureSpec(
        num_parameters=176e9,
        description="AliBi positional embeddings, embedding LayerNorm",
    ),
    training=TrainingSpec(
        hardware="48x8 A100s on Jean Zay supercomputer for 3.5 months",
        description="ZeRO stage 1"
    ),
)

############################################################
# 2023

llama = ModelSpec(
    name="LLaMA", organization="Meta", date="2023-02-27",
    url="https://arxiv.org/pdf/2302.13971.pdf",
    description=[
        "Train only on open data (detailed recipe that is replicated by RedPajama)",
        "Optimize for fast inference at 7B",
    ],
    data=DataSpec(
        description="CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, StackExchange",
        num_tokens=1.4e12,
    ),
    architecture=ArchitectureSpec(
        description="Pre-norm, SwiGLU, RoPE",
        num_parameters=65e9,
    ),
    training=TrainingSpec(
        hardware="2048 A100 80GB for 21 days",
        # https://proceedings.mlsys.org/paper_files/paper/2023/file/e851ca7b43815718fbbac8afb2246bf8-Paper-mlsys2023.pdf
    ),
)

gpt_4 = ModelSpec(
    name="GPT-4", organization="OpenAI", date="2023-03-15",
    url="https://arxiv.org/pdf/2303.08774.pdf",
    description=[
        "No details on the data or model architecture.",
    ]
)

gqa = MethodSpec(
    name="Grouped-Query Attention", organization="Google", date="2023-05-22",
    url="https://arxiv.org/pdf/2305.13245.pdf",
    description=[
        "Multi-query attention (MQA) speeds up, but less expressive",
        "GQA: use an intermediate (more than one, less than number of heads) number of key-value heads",
        "Experiments on T5",
    ]
)

lima = MethodSpec(
    name="LIMA: Less Is More for Alignment", date="2023-05-18",
    url="https://arxiv.org/pdf/2305.11206.pdf",
)
dpo = MethodSpec(
    name="Direct Preference Optimization: Your Language Model is Secretly a Reward Model", date="2023-05-29",
    url="https://arxiv.org/pdf/2305.18290.pdf",
)

llama2 = ModelSpec(
    name="Llama 2", organization="Meta", date="2023-07-18",
    url="https://arxiv.org/pdf/2307.09288.pdf",
    data=DataSpec(
        num_tokens=2e12,
    ),
    architecture=ArchitectureSpec(
    # TODO
    ),
),

mistral_7b = ModelSpec(
    name="Mistral-7B", organization="Mistral", date="2023-10-10",
    url="https://arxiv.org/pdf/2310.06825.pdf",
    data=DataSpec(
    ),
    architecture=ArchitectureSpec(
        num_parameters=7e9,
        description="GQA, sliding window attention",
    ),
)

amber = ModelSpec(
    name="Amber", organization="LLM360",
    url="https://arxiv.org/pdf/2309.16609.pdf",
    # TODO
)

############################################################
# 2024

deepseek_67b = ModelSpec(
    name="DeepSeek (67B)", organization="DeepSeek", date="2024-01-05",
    url="https://arxiv.org/pdf/2401.02954.pdf",
    data=DataSpec(
        name="DeepSeek",
        description="Common Crawl, The Stack, Reddit, etc.",
        num_tokens=2e12,
    ),
    architecture=ArchitectureSpec(
        description="LLaMA, but: for GQA increased depth",
        num_parameters=67e9,
        num_layers=95, dim_model=8192, num_heads=64,
    ),
    training=TrainingSpec(
        learning_rate=3.2e-4,
        description="Scaling laws: used non-embedding FLOPs with IsoFLOP",
    ),
)

mixtral = ModelSpec(
    name="Mixtral", organization="Mistral", date="2024-01-08",
    url="https://arxiv.org/pdf/2401.04088.pdf",
)

stable_lm_2 = ModelSpec(
    name="Stable LM 2 1.6B", organization="Stability AI", date="2024-01-19",
    url="https://drive.google.com/file/d/1JYJHszhS8EFChTbNAf8xmqhKjogWRrQF/view",
    data=DataSpec(
        num_tokens=2e12,
        description="Arxiv, PubMed, S2ORC, C4, OpenWebText2, RefinedWeb, StackExchange, EuroParl, RedPajama Wikipedia, The Stack, etc.",
        vocabulary_size=100_289,
    ),
    architecture=ArchitectureSpec(
        num_parameters=1.6e9,
        description="LLaMA, but: RoPE applied to first 25%, LayerNorm with learned biases, removed all biases except KQV projections",
    ),
    training=TrainingSpec(
        context_length=4096,
        num_epochs=2,
        num_flops=92_000 * a100_flop_per_sec,  # 2.87e19
        optimizer="AdamW",
        learning_rate="linear warmup to 1e-3, consine, inv-sqrt",
        weight_decay=0.1,
        hardware="512 A100s on Amazon, ZeRO stage 1, 54.5% MFU",
    ),
)
        
olmo_7b = ModelSpec(
    name="OLMo-7B", organization="AI2", date="2024-01-31",
    url="https://arxiv.org/pdf/2402.00838.pdf",
    data=DataSpec(
        name="subset of Dolma",
        description="Common Crawl, The Stack, Reddit, etc.",
        num_tokens=2.46e12,
        vocabulary_size=50_304,
    ),
    architecture=ArchitectureSpec(
        num_parameters=7e9,
        num_layers=32, dim_model=4096, num_heads=32,
        description="no biases, non-parametric layer norm, SwiGLU (8/3 d increased to closest multiple of 128)",
    ),
    training=TrainingSpec(
        context_length=2048,
        batch_size_tokens=4e6,
        description="mixed-precision training (FSDP, amp)",
        hardware="256x4 AMD MI250X on LUMI supercomputer, 27x8 A100s, 800Gbps interconnect",
    ),
),

megascale = MethodSpec(
    name="MegaScale", organization="Bytedance", date="2024-02-23",
    url="https://arxiv.org/pdf/2402.15627.pdf",
    description=[
        "55.2% MFU for 175B parameter model over 12,288 GPUs",
        "Combine data, tensor, pipeline, sequence parallelism",
        "Parallelize attention and feedforward layers, sliding window attention, LAMB optimizer",
    ],
)

nemotron_15b = ModelSpec(
    name="Nemotron-15B", organization="NVIDIA", date="2024-02-26",
    url="https://arxiv.org/pdf/2402.16819.pdf",
    data=DataSpec(
        num_tokens=8e12,
        description="70% English, 15% multilingual, 15% code",
    ),
    architecture=ArchitectureSpec(
        num_parameters=15e9,
        description="RoPE, squared ReLU activations, no bias, no dropout, GQA",
    ),
    training=TrainingSpec(
        hardware="384x8 H100s",
        description="After 8T tokens, train on higher quality sources + benchmark tasks",
    ),
)

griffin = ModelSpec(
    name="Griffin", organization="Google DeepMind", date="2024-03-01",
    url="https://arxiv.org/pdf/2402.19427.pdf",
    architecture=ArchitectureSpec(
        num_parameters=14e9,
        description="Gated linear recurrences + local attention",
    ),
)

yi_34b = ModelSpec(
    name="Yi-34B", organization="01.AI", date="2024-03-07",
    url="https://arxiv.org/pdf/2403.04652.pdf",
    # TODO
)

gemma = ModelSpec(
    name="Gemma", organization="Google DeepMind", date="2024-03-13",
    url="https://arxiv.org/pdf/2403.08295.pdf",
    data=DataSpec(
        num_tokens=6e12,
    ),
    architecture=ArchitectureSpec(
        num_parameters=7e9,
        description="MQA, RoPE, GeGLU, RMSNorm"
    ),
    training=TrainingSpec(
        hardware="4096 v4 TPUs",
        description="Use ZeRO-3 like techniques",
    ),
)

overtrained_scaling_laws = MethodSpec(
    name="Language models scale reliably with over-training and on downstream tasks", date="2024-03-13",
    url="https://arxiv.org/pdf/2403.08540.pdf",
    description=[
        "Chinchilla scaling laws focus on loss of the trained model, ignoring inference costs.",
        "Constant ratio of training tokens to parameters",
        "Extrapolate over 300x training compute to 1.4B model on 900B tokens",
        "Look at task performance rather than validation loss",
    ]
)

# https://github.com/lucidrains/x-transformers?tab=readme-ov-file#sandwich-norm

############################################################
# References

transformer_math = MethodSpec(
    name="Transformer Math 101", organization="EleutherAI", date="2023-04-03",
    url="https://blog.eleuther.ai/transformer-math/",
)

bahdanau_training_costs = MethodSpec(
    name="The FLOPs Calculus of Language Model Training", author="Dzmitry Bahdanau", date="2022-01-09",
    url="https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4",
)
