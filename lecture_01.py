import os
import requests
from itertools import islice
from typing import List, Iterable
from util import *
from references import *
from facts import *
from data import preprocess, Document
from tokenization import Tokenizer, train_bpe, tokenization_unit
from sqlitedict import SqliteDict


def lecture_01():
    prelude()
    course_logistics()
    why_this_course()
    examples()
    brief_history()
    course_components()
    tokenization_unit()
    assignment1_overview()


def prelude():
    note("## CS336: Language Models From Scratch (Spring 2024)")

    note("What on earth is this program doing?")

    note("This is an *executable lecture*, "
         "a program whose execution delivers the content of a lecture.")

    note("Executable lectures make it possible to:")
    note("- view and run code (since everything is code!),")
    note("- see the hierarchical structure of the lecture (e.g., we're in the prelude), and")
    note("- jump to definitions and concepts."), see(gpt_3)

    note("It is an experiment.  Let's see how it goes!")


def course_logistics():
    see("https://stanford-cs336.github.io/spring2024/")

    note("This is a 5-unit class. "
         "You will write a lot of code (an order magnitude more than your average AI course).")
    note("This is the first time we're teaching this class. "
         "Please be patient with us and give us feedback!")

    note("## Cluster")
    note("Thanks to Together AI for providing compute.")
    note("Here's the guide on how to use the cluster: "
         "https://docs.google.com/document/d/1yLhnbclhOOL5_OBI_jBlhNh9xr3xRNTCgL5B-g-qQF4/edit")
    note("Start your assignments early, since the cluster will fill up close to the deadline!")

    note("There was a lot of interest in the class, "
         "so unfortunately we couldn't enroll everyone.")
    note("We will make all the assignments and lecture materials available online, "
         "so feel free to follow on your own.")
    note("We plan to offer this class again next year.")


def why_this_course():
    note("Philosophy: understanding via building")

    note("## Why you should take this course")
    note("You have an obsessive need to understand how things work.")
    note("You want to build up your research engineering muscles.")

    note("## Why you should not take this course")
    note("You actually want to get research done this quarter. "
         "(Talk to your advisor.)")
    note("You are interested in learning about the hottest new techniques in AI, "
         "e.g., diffusion, multimodality, long context, etc. "
         "(You could take a seminar class for that.)")
    note("You want to get good results on your own application domain. "
         "(You could just prompt GPT-4/Claude/Gemini.)")
    note("You need to build a language model for your own application. "
         "(You could fine-tune an existing model using standard packages.)")

    note("## Why this class exists")

    note("Problem: researchers are becoming disconnected.")
    note("10 years ago, researchers would implement and train their own models.")
    note("5 years ago, researchers would download a model (e.g., BERT) and fine-tune it.")
    note("Today, researchers just prompt GPT-4.")

    note("Moving up levels of abstractions boosts productivity, but")
    note("- These abstractions are leaky (contrast with operating systems).")
    note("- There is still fundamental research to be done at the lower levels.")

    note("## The landscape")

    note("There are language models...and then are (large) language models.")

    note("GPT-4 supposedly has 1.8T parameters."), see("https://www.hpcwire.com/2024/03/19/the-generative-ai-future-is-now-nvidias-huang-says")
    note("GPT-4 supposedly cost $100M to train."), see("https://www.wired.com/story/openai-ceo-sam-altman-the-age-of-giant-ai-models-is-already-over/")
    note("The GPT-4 technical report discloses no details.")
    image("images/gpt_4_section_2.png"), see("https://arxiv.org/pdf/2303.08774.pdf")

    note("## So what are we doing in this class?")

    note("We are obviously not building GPT-4 (or anything close).")
    note("But we hope to impart some of the skills and mindset, "
         "so that if you had the resources, at least you'd know where to start.")

    note("Key question: what can you learn at small scale that generalizes to large scale?")

    note("There are three types of knowledge:")
    note("- Mechanics: how things work (what a Transformer is, how FSDP works)")
    note("- Mindset: squeezing performance, thinking about scale (scaling laws)")
    note("- Intuitions: how to set hyperparameters, process data, to get good performance")

    note("We can teach mechanics and mindset (do transfer).")
    note("We cannot teach intuitions (do not necessarily transfer across scales).")

    note("You can tell a lot of stories about why something will work.")
    note("Reality: Some design decisions are simply not justifiable.")
    note("Example: Noam Shazeer paper that introduced SwiGLU (see last sentence of conclusion):")
    see("https://arxiv.org/pdf/2002.05202.pdf")

    note("## How to learn the material")
    note("- Implementing everything yourself (without looking!).")
    note("- Read the major language modeling papers to glean insights.")
    note("- Understand internals deeply by reading through code and thinking about what's happening.")
    note("- Run experiments to get intuition for what settings lead to what behavior.")


def examples():
    note("Here are some examples of language models in action.")

    response = query_model(
        model="gpt-4",
        prompt="Explain how you build a language model from scratch to a five-year old.",
    )
    see(gpt_4)
    note("### GPT-4 response")
    note(response, verbatim=True)

    response = query_model(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        prompt="Explain how you build a language model from scratch to a five-year old.",
    )
    see(mixtral)
    see("### Mixtral response")
    note(response, verbatim=True)

    note("They can mostly follow instructions, generate fluent and semantically relevant text.")
    note("How do they work?  How can we build one ourselves?")


def brief_history():
    note("Language model to measure the entropy of English"), see(shannon1950)
    note("Lots of work on n-gram language models (for machine translation, speech recognition)"),
    note("First neural language modeling"), see(bengio2003)
    note("Sequence-to-sequence modeling (for machine translation)"), see(susketver2014)
    note("Introduced attention mechanism (for machine translation)"), see(bahdanau2015_attention)
    note("Introduced the Transformer architecture (for machine translation)"), see(transformer)

    note("OpenAI's GPT-2 (1.5B): zero-shot learning, staged release"), see(gpt_2)
    note("Google's T5 (11B): cast everything as text-to-text"), see(t5)
    note("Kaplan's scaling laws"), see(kaplan_scaling_laws)
    note("OpenAI's GPT-3 (175B): in-context learning, closed"), see(gpt_3)

    note("EleutherAI's open datasets (The Pile) and models (GPT-J)"), see(the_pile), see(gpt_j)

    note("Meta's OPT (175B): GPT-3 replication, lots of hardware issues"), see(opt_175b)
    note("Hugging Face / BigScience's BLOOM: focused on data sourcing"), see(bloom)
    note("Google's PaLM (540B): massive scale, undertrained"), see(palm)
    note("DeepMind's Chinchilla (70B): compute-optimal scaling laws"), see(chinchilla)

    note("Meta's LLaMA (7B, .., 65B): overtrained, optimized the 7B"), see(llama)
    note("Mistral (7B): overtrained, very good 7B model"), see(mistral_7b)
    note("Many other open models: Yi, DeepSeek, Qwen, StableLM, OLMo, Gemma, etc.")
    note("Mixture of experts: Mistral's Mixtral, xAI's Grok, Databricks's DBRX")

    note("Frontier models:")
    note("- OpenAI's GPT-4"), see(gpt_4)
    note("- Anthropic's Claude 3")
    note("- Google's Gemini")

    note("Ecosystem graphs tracks latest models")
    see("https://crfm.stanford.edu/ecosystem-graphs/index.html?mode=table")

    note("Summary")
    note("- Interplay between open and closed models")
    note("- Emphasis on number of parameters, then compute-optimal, then overtrained")


def course_components():
    note("## Philosophy")
    note("Key: it's all about *efficiency*")

    note("Resources: data + hardware (compute, memory, communication)")
    note("How do you train the best model given these a fixed set of resources?")
    note("Example: given a Common Crawl dump and 16 H100s for 2 weeks, what should we do?")

    note("Design decisions: data, tokenization, model architecture, training, alignment")
    note("## Pipeline (stylized)")

    note("Data")
    raw_data = get_raw_data()
    pretraining_data = process_data(raw_data)

    note("Pretraining")
    tokenizer = train_tokenizer(pretraining_data)
    model = TransformerLM()
    pretrain(model, pretraining_data, tokenizer)

    note("Alignment")
    instruction_data = get_instruction_data()
    instruction_tune(model, instruction_data, tokenizer)
    preference_data = generate_preference_data(model)
    preference_tune(model, preference_data, tokenizer)

    note("## On efficiency as a unifying perspective")

    note("Today, we are hardware-bound, so design decisions will reflect "
         "squeezing the most out of given hardware.")
    note("- Data processing: avoid wasting precious compute updating on bad / irrelevant data")
    note("- Tokenization: working with raw bytes is elegant, "
         "but compute-inefficient with today's model architectures")
    note("- Model architecture: many changes motivated by keeping GPUs humming along")
    note("- Training: we can get away with a single epoch!")
    note("- Scaling laws: use less compute on smaller models to do hyperparameter tuning")
    note("- Alignment: if tune model more to desired use cases, require smaller base models")

    note("Tomorrow, we might become data-bound...")


def get_raw_data() -> List[Document]:
    """Return raw data."""
    note("Data does not just fall from the sky.")
    note("Sources: webpages scraped from the Internet, books, arXiv papers, GitHub code, etc.")
    note("Appeal to fair use to train on copyright data?"), see("https://arxiv.org/pdf/2303.15715.pdf")
    note("Might have to license data (e.g., Google with Reddit data)"), see("https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/")
    note("Formats: HTML, PDF, directories (not text!)")
    # Stub implementation: grab one URL
    urls = ["https://en.wikipedia.org/wiki/Sphinx"]
    documents = [Document(url, open(cached(url)).read()) for url in urls]
    return documents


def process_data(documents: List[Document]) -> List[Document]:
    note("Preprocess the raw data")
    note("- Filtering: keep data of high quality, remove harmful content")
    note("- Deduplication: don't waste time training, avoid memorization")
    note("- Conversion: project HTML to text (preserve content, structure)")
    # Stub implementation: just convert html to text
    documents = list(preprocess(documents))
    return documents


def train_tokenizer(documents: Iterable[Document]) -> Tokenizer:
    note("Tokenizers convert text into sequences of integers (tokens)")
    note("Balance tradeoff between vocabulary size and compression ratio")
    note("This course: Byte-Pair Encoding (BPE) tokenizer"), see(train_bpe)
    # Stub implementation: just return the pre-trained tokenizer
    import tiktoken
    return tiktoken.get_encoding("gpt2")


class TransformerLM:
    def __init__(self):
        note("Original Transformer"), see(transformer)
        note("Many variants exist that improve on the original "
             "(e.g., post-norm, SwiGLU, RMSNorm, parallel layers, RoPE, GQA)")

    def forward(self, x):
        pass


def pretrain(model: TransformerLM, data: Iterable[Document], tokenizer: Tokenizer):
    note("Specify the optimizer (e.g., AdamW)"), see(adamw)
    note("Specify the learning rate schedule (e.g., cosine)")
    note("Set other hyperparameters (batch size, number of heads, hidden dimension)")


@dataclass(frozen=True)
class InstructionExample:
    prompt: str
    response: str


def get_instruction_data() -> List[InstructionExample]:
    note("Instruction data: (prompt, response) pairs")
    note("Intuition: base model already has the skills, "
         "just need few examples to surface them."), see(lima)
    # Stub implementation: just grab the Alpaca dataset
    examples = list(islice(get_alpaca_dataset(), 0, 100))
    return examples


def instruction_tune(model: TransformerLM, data: Iterable[InstructionExample], tokenizer: Tokenizer):
    note("Given (prompt, response) pairs, we perform supervised learning.")
    note("Specifically, fine-tune `model` to maximize p(response | prompt).")


@dataclass(frozen=True)
class PreferenceExample:
    prompt: str
    response1: str
    response2: str
    preference: str


def generate_preference_data(model: TransformerLM) -> List[PreferenceExample]:
    note("Now we have a preliminary instruction following `model`.")
    note("Data: generate multiple responses using `model` (e.g., [A, B]) to a given prompt.")
    note("User provides preferences (e.g., A < B or A > B).")
    # Stub implementation: just an example
    return [
        PreferenceExample(
            prompt="What is the best way to train a language model?",
            response1="You should use a large dataset and train for a long time.",
            response2="You should use a small dataset and train for a short time.",
            preference="1 is better",
        )
    ]


def preference_tune(model: TransformerLM, data: Iterable[PreferenceExample], tokenizer: Tokenizer):
    note("Given (prompt, response1, response2, preference) tuples, tune the model.")
    note("Traditionally: "
         "Proximal Policy Optimization (PPO) from reinforcement learning"), see(instruct_gpt)
    note("Recently, effective and simpler approach: "
         "Direct Policy Optimization (DPO)"), see(dpo)


def assignment1_overview():
    note("https://github.com/stanford-cs336/spring2024-assignment1-basics")

############################################################

def query_model(model: str, prompt: str) -> str:
    """Query `model` with the `prompt` and return the top-1 response."""
    ensure_directory_exists("var")
    cache = SqliteDict("var/query_model_cache.db")
    key = model + ":" + prompt
    if key in cache:
        return cache[key]

    from openai import OpenAI
    if model.startswith("gpt-"):
        # Use an actual OpenAI model
        client = OpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
        )
    else:
        # Together API serves open models using the same OpenAI interface
        client = OpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )

    system_prompt = "You are a helpful and harmless assistant."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    value = response.choices[0].message.content
    cache[key] = value
    cache.commit()

    return value


def get_alpaca_dataset() -> Iterable[InstructionExample]:
    from datasets import load_dataset
    dataset = load_dataset("tatsu-lab/alpaca")
    for datum in dataset["train"]:
        prompt = datum["instruction"]
        if datum["input"]:
            prompt += "\n" + datum["input"]
        response = datum["output"]
        yield InstructionExample(prompt, response)


if __name__ == "__main__":
    init_content("lecture_01-content.js")
    lecture_01()
