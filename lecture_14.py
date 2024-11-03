from util import *

def lecture_14():
    note("Previous lectures: data for pretraining => general capabilities")

    note("What if you want to add new capabilities to your language model?")

    # Types of capabilities
    note("- Solving tasks (e.g., information extraction)")
    note("- Instruction following and chat")

    note("- Long contexts (e.g., 4096 -> 100,000)")
    note("- Infilling (e.g., the cat __ the hat)")

    note("- Domain-specific capabilities (e.g., coding, math, medicine)")

    note("- Safety (e.g., refusal)")
    note("- Reasoning (e.g., chain of thought)")

    # Focus
    note("This lecture: fine-tune on *data* that exhibits the desired capabilities")

    note("Sources of data")
    note("- Annotators (e.g., Llama 2 instruction data)")
    note("- Real users (e.g., ShareGPT)")
    note("- Curated (e.g., from Common Crawl)")
    note("- Distilled from stronger model (e.g., synthetic data from GPT-4)")
    note("- Self-distillation (synthetic data from model you're training)")

    training_stages()

    tasks()                # Tasks, standard datasets
    instruction_chat()     # Instruction following and chat, various data

    long_context()         # Long context
    infilling()            # Infilling

    domains()              # Domain-specific knowledge and skills, various data

    reasoning()           # Reasoning, distillation
    self_distillation()   # Self-distillation

    note("Discussion on types of data")
    note("- Extract useful signals from the web")
    note("- Distillation from stronger model (GPT-4): cheap, scientifically interesting (oracle); "
         "be careful of licenses, playing catch up, not pushing things forward")
    note("- Self-distillation (constitutional AI, STaR): synthetic data, promising way to squeeze more out")

    note("## Summary")
    note("Add general capabilities just by adding data - very flexible!")
    note("Not a substitute for a strong base model (for generalization)")
    note("But for specific tasks, can get much smaller models to perform well")

    note("Data is the key ingredient that differentiates language models")
    note("Live service => raw data => processed data (conversion, filtering, deduplication)")
    note("Legal and ethical issues (e.g., copyright and privacy)")
    note("Much of this pipeline is heuristic, many opportunities to improve")

    note("Next time: alignment")


def training_stages():
    note("The textbook version:")
    note("1. Pre-training: train on raw text (e.g., documents from the web)")
    note("2. Mid-training (continued pre-training): enhance capabilities")
    note("3. Post-training: fine-tune on a particular task/dataset")

    note("Reality: lines are blurry")
    note("- Often there are multiple stages of training")
    note("- Train on general data, then train on clean data")
    note("- Mix in instruction data towards the end of training")

    note("Example (Dolma): (1) 2T tokens of Dolma 1.7, (2) 50B tokens on {Wikipedia, OpenWebMath, Flan}")
    see("https://blog.allenai.org/olmo-1-7-7b-a-24-point-improvement-on-mmlu-92b43f7d269d")
    image("https://miro.medium.com/v2/resize:fit:828/format:webp/1*QFZ9R3xZUH8stKchJz9G7w.png")  # Stage 1
    image("https://miro.medium.com/v2/resize:fit:828/format:webp/1*B_GIIKvnDKPFXEVb8Qd7Sw.png")  # Stage 2

    note("Note: base model doesn't mean just trained on web documents")


def tasks():
    note("TL;DR: convert lots of existing NLP datasets into prompts")

    note("Super-Natural Instructions [Wang+ 2022]"), see("https://arxiv.org/pdf/2204.07705")
    see("https://huggingface.co/datasets/Muennighoff/natural-instructions")
    note("Dataset: 1.6K+ tasks (Figure 2)")
    note("Fine-tune T5 on k-shot learning (Tk-instruct)")
    note("Tasks contributed by community (via GitHub)")
    note("Examples for each task are derived from existing datasets and converted into templatized prompts")
    note("Outperforms InstructGPT despite being much smaller(?)")

    note("Flan 2022 [Longpre+ 2023]"), see("https://arxiv.org/pdf/2301.13688")
    note("Dataset: 1.8K+ tasks")
    see("https://huggingface.co/datasets/Muennighoff/flan")
    note("Fine-tune T5 on zero-shot, few-shot, chain-of-thought versions of the dataset (Figure 7)")


def instruction_chat():
    note("TL;DR: more open-ended instructions, heavy use of synthetic data")

    note("Alpaca [Taori+ 2023]"), see("https://crfm.stanford.edu/2023/03/13/alpaca.html")
    note("Dataset of 52K examples from text-davinci-003 using self-instruct [Wang+ 2022]"), see("https://arxiv.org/pdf/2212.10560")
    note("Fine-tune LLaMA 7B on this dataset")

    note("Vicuna [LMSYS 2023]"), see("https://lmsys.org/blog/2023-03-30-vicuna/")
    note("Fine-tuned LLaMA on 70K conversations from ShareGPT (user-shared ChatGPT)"), see("https://sharegpt.com/")

    note("Baize [Xu+ 2023]"), see("https://arxiv.org/pdf/2304.01196")
    note("Generate dataset (111.5K examples) from GPT-3.5 using self-chat (seeded with Quora and StackOverflow questions)")
    note("Fine-tune LLaMA on this dataset")

    note("WizardLM [Xu+ 2023]"), see("https://arxiv.org/pdf/2304.12244")
    note("Evol-Instruct dataset ('evolve' questions to increase breadth/difficulty) (Figure 1)")
    note("Fine-tune LLaMA on this dataset")

    note("UltraLLaMA [Ding+ 2023]"), see("https://arxiv.org/pdf/2305.14233")
    note("UltraChat: 1.5M multi-turn dialogues across wide range of topics (Tables 2+3)"), see("https://huggingface.co/datasets/stingning/ultrachat")
    note("Fine-tune LLaMA on UltraChat")

    note("MAmmoTH2 [Yue 2024]"), see("https://arxiv.org/pdf/2405.03548")
    note("Curated WebInstruct, 10M instructions from Common Crawl")
    note("Filter: train fastText classifier on quiz sites")
    note("Extract: use GPT-4 and Mixtral to extract QA pairs")
    note("Fine-tune Mistral 7B on this data")
    note("Boosts math performance")

    note("OpenHermes 2.5")
    note("Agglomeration of many datasets"), see("https://huggingface.co/datasets/teknium/openhermes")
    note("Fine-tune Mistral 7B on 1M examples from GPT-4"), see("https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B")

    note("Llama 2 chat [Touvron+ 2023]"), see("https://arxiv.org/pdf/2307.09288")
    note("27,540 examples of high-quality instruction data from vendor-based annotations")
    note("Said was better than using the millions of examples from open datasets")
    note("Could have labeled less data and saved more effort for getting RLHF data")


def long_context():
    note("LLama 2 is trained on sequences of 4096 tokens")

    note("Demand for long contexts (want to do QA on books)")
    note("- Claude 2.1 has 100K tokens")
    note("- Gemini 1.5 has 1M tokens")

    note("Transformers scales quadratically with sequence length")
    note("Not efficient to pre-train on long contexts, want to add long context later")

    note("LongLoRA [2024]"), see("https://arxiv.org/pdf/2309.12307")
    note("Extends context length of Llama2 7B from 4K to 100K tokens")
    note("Use shifted sparse attention (Figure 2), positional interpolation [Chen+ 2023]")
    note("Trained on long documents - PG-19 (books) and Proof-Pile (math)")

    note("How to train on long contexts"), see("https://huggingface.co/blog/wenbopan/long-context-fine-tuning#long-text-data")
    note("anti-haystack"), see("https://huggingface.co/datasets/wenbopan/anti-haystack")


def infilling():
    note("Language models predict the next token given previous tokens")

    note("Infilling applications: writing assistant, code autocomplete")

    note("Goal: model that can only predict the next token => model that can infill")

    note("Enabling Language Models to Fill in the Blanks [Donahue+ 2020]"), see("https://arxiv.org/pdf/2005.05339")
    note("Data: take any text (stories, abstracts) and randomly mask out words")
    note("She ate [blank] for [blank] => leftover pasta [answer] lunch [answer]")
    note("Fine-tuned GPT-2 on this data")

    note("Efficient training of language models to fill in the middle [Bavarian+ 2022]"), see("https://arxiv.org/pdf/2207.14255")


def domains():
    # Code
    note("Code Llama [Rozière+ 2023 (Meta)]"), see("https://arxiv.org/pdf/2308.12950")
    image("https://pbs.twimg.com/media/F4TkZeFXoAEUzoM.png")
    note("Added many capabilities (Figure 2)")
    note("- Code and infilling (500B tokens): split into prefix-suffix-middle and suffix-prefix-middle")
    note("- Long context (20B tokens): 16384 tokens, changed period of RoPE embeddings")
    note("- Instruction following (5B tokens): proprietary dataset + self-instruct with execution feedback")
    note("- Rehearsal (to avoid forgetting): mix in 6% code and 2% text")

    # Math
    note("Llemma [Azerbayev+ 2024]"), see("https://arxiv.org/pdf/2310.10631")
    note("Proof-Pile-2: AlgebraicStack, OpenWebMath, arXiv"), see("https://huggingface.co/datasets/EleutherAI/proof-pile-2")
    note("Fine-tune Code Llama on Proof-Pile-2")
    note("Competes with Minerva 62B (Figure 1)")

    # Medicine
    note("PMC-LLaMA [Wu+ 2023]"), see("https://arxiv.org/pdf/2304.14454")
    note("Dataset: 4.8M biomedical academic papers, 30K medical textbooks, instructions (Figure 2)")
    note("Fine-tune LLaMA on this data")
    note("Outperforms ChatGPT on PubMedQA, MedMCQA, USMLE")


def reasoning():
    note("Distilling step by step [Hsieh+ 2023]"), see("https://arxiv.org/pdf/2305.02301")
    note("Prompt strong model (PaLM 540B) to get (input, rationale, output) examples (Figure 2)")
    note("Fine-tune weak model (T5) on [input => rationale] and [input => output]")

    note("Orca 2 [Mitra+ 2023]"), see("https://arxiv.org/pdf/2311.11045")
    note("Prompt strong model (GPT-4) to generate reasoning patterns: "
         "step-by-step processing, recall-then-generate, recall-reason-generate, extract-generate, direct-answer (Section 3)")
    note("Fine-tune weak model (LLama) on this data with prompt erasure (of the type of reasoning)") 

    note("Limitation: these works require a stronger model as an oracle...")


def self_distillation():
    # Constitutional AI
    note("Constitutional AI [Bai+ 2022]"), see("https://arxiv.org/pdf/2212.08073")

    note("Goal: improve safety of model without human feedback for harms")
    note("Motivation: getting humans labels for harmful content could be extra costly (emotional toll)")

    note("Constitution: set of principles with small number of examples")
    note("Example: Section 3.1")

    note("prompt => initial response => critique => revision => final response")

    note("Use constitution to ask model to critique and revise")
    note("Fine-tune on (prompt, final response) examples from this chain")

    # Consistency tuning
    note("Consistency tuning [Li+ 2023]"), see("https://arxiv.org/pdf/2310.01846")

    note("Motivation:")
    note("  User: what is 7+8? | ChatGPT: 15 (generator)")
    note("  User: 7+8 = 15? True or false | ChatGPT: False (validator)")

    note("Generate model responses from generator and validator")
    note("Fine-tune on the data where the two are consistent")
    note("Improves consistency (Table 3) and accuracy (Table 5)")

    # STaR
    note("STaR: self-taught reasoner [Zelikman+ 2022]"), see("https://arxiv.org/pdf/2203.14465")

    note("Chain-of-thought improves performance (input => rationale => output)")
    note("Getting (input, rationale, output) examples is expensive")

    note("Can we use (input, output) pairs?")

    note("Method:")
    note("- Generate rationale given input")
    note("- If lead to output, then keep (input, rationale, output)")
    note("- Otherwise, generate rationale given (input, output) [rationalization] (helps - Figure 4)")

    note("Fine-tune on this data")
    note("Works as well as 30x larger model on CommonsenseQA")

    note("Rumors of OpenAI's Q* algorithm..."), see("https://www.theatlantic.com/technology/archive/2023/11/openai-sam-altman-q-algorithm-breakthrough-project/676163/")


if __name__ == "__main__":
    init_content("lecture_14-content.js")
    lecture_14()
