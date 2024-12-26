from abc import ABC
from typing import List, Tuple, Dict
from util import *
from references import *
from collections import defaultdict
from dataclasses import dataclass
import regex as re

sample_text1 = "Hello, 🌍! 你好!"

# https://github.com/openai/tiktoken/blob/9d01e5670ff50eb74cdb96406c7f3d9add0ae2f8/tiktoken_ext/openai_public.py#L23
GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def tokenization_unit():
    note("This unit was inspired by Andrej Karpathy's video on tokenization; check it out!")
    see("https://www.youtube.com/watch?v=zduSFxRajkE")

    intro_tokenizer()
    examples()
    character_tokenizer()
    byte_tokenizer()
    word_tokenizer()
    bpe_tokenizer()


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: Dict[int, bytes]             # index -> bytes
    merges: Dict[Tuple[int, int], int]  # (index1, index2) -> new_index


class Tokenizer(ABC):
    """Abstract interface for a tokenizer."""
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, indices: List[int]) -> str:
        raise NotImplementedError


class CharacterTokenizer(Tokenizer):
    """Represent a text as a sequence of Unicode code points."""
    def encode(self, text: str) -> List[int]:
        return list(map(ord, text))

    def decode(self, indices: List[int]) -> str:
        return "".join(map(chr, indices))


class ByteTokenizer(Tokenizer):
    """Represent a text as a sequence of bytes."""
    def encode(self, text: str) -> List[int]:
        text_bytes = text.encode("utf-8")
        indices = list(map(int, text_bytes))
        return indices

    def decode(self, indices: List[int]) -> str:
        text_bytes = bytes(indices)
        return text_bytes.decode("utf-8")

def merge(indices: List[int], pair: Tuple[(int, int)], new_index: int) -> List[int]:
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, text: str) -> List[int]:
        indices = list(map(int, text.encode("utf-8")))
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: List[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))
        text = b"".join(bytes_list).decode("utf-8")
        return text


def get_compression_ratio(text: str, indices: List[int]) -> float:
    """Given `text` that has been tokenized into `indices`, ."""
    num_bytes = len(bytes(text, encoding="utf-8"))  # Original
    num_tokens = len(indices)                       # Tokenized
    return num_bytes / num_tokens


def get_gpt2_tokenizer():
    # Code: https://github.com/openai/tiktoken
    import tiktoken
    # You can use cl100k_base for the gpt3.5-turbo or gpt4 tokenizer
    return tiktoken.get_encoding("gpt2")


def intro_tokenizer():
    note("Raw text generally represented as Unicode strings.")
    text = sample_text1

    note("A language model places a probability distribution over "
          "sequences of tokens (usually represented by integer indices).")
    indices = [15496, 11, 995, 0]

    note("So we need a procedure that *encodes* text into tokens.")
    note("We also need a procedure that *decodes* tokens back into text.")
    note("A Tokenizer is a class that implements the encode and decode methods.")
    see(Tokenizer)
    note("The number of possible indices is the *vocabulary size*.")


def examples():
    note("Play with this interactive site to get a feel for how tokenizers work:")
    see("https://tiktokenizer.vercel.app/?encoder=gpt2")

    note("## Observations")
    note("- A word and its preceding space are part of the same token (e.g., ' world').")
    note("- A word at the beginning and in the middle are represented differently (e.g., 'hello hello').")
    note("- Some long words are one token (e.g., ' SolidGoldMagikarp').")
    note("- Numbers are tokenized into every few digits.")

    note("Here's the GPT-2 tokenizer from OpenAI (tiktoken) in action.")
    tokenizer = get_gpt2_tokenizer()
    text = sample_text1

    note("Check that encode() and decode() roundtrip:")
    indices = tokenizer.encode(text)
    reconstructed_text = tokenizer.decode(indices)
    assert text == reconstructed_text


def character_tokenizer():
    note("## Character-based tokenization")

    note("A Unicode string is a sequence of Unicode characters.")
    note("Each character can be converted into a code point (integer) via `ord`.")
    assert ord("a") == 97
    assert ord("🌍") == 127757
    note("It can be converted back via `chr`.")
    assert chr(97) == "a"
    assert chr(127757) == "🌍"

    note("Now let's build a `Tokenizer` and make sure it round-trips:")
    tokenizer = CharacterTokenizer()
    text = sample_text1
    indices = tokenizer.encode(text)
    reconstructed_text = tokenizer.decode(indices)
    assert text == reconstructed_text

    note("There are approximately 150K Unicode characters.")
    see("https://en.wikipedia.org/wiki/List_of_Unicode_characters")
    vocabulary_size = max(indices) + 1  # This is a lower bound
    note("Problem 1: this is a very large vocabulary.")
    note("Problem 2: many characters are quite rare (e.g., 🌍), "
          "which is inefficient use of the vocabulary.")
    compression_ratio = get_compression_ratio(text, indices)


def byte_tokenizer():
    note("## Byte-based tokenization")

    note("Unicode text can be represented as a sequence of bytes, "
          "which can be represented by integers between 0 and 255.")
    note("The most common Unicode encoding is UTF-8."), see("https://en.wikipedia.org/wiki/UTF-8")

    note("Some Unicode characters are represented by one byte:")
    assert bytes("a", encoding="utf-8") == b"a"
    note("Others take multiple bytes:")
    assert bytes("🌍", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"

    note("Now let's build a `Tokenizer` and make sure it round-trips:")
    tokenizer = ByteTokenizer()
    text = sample_text1
    indices = tokenizer.encode(text)
    reconstructed_text = tokenizer.decode(indices)
    assert text == reconstructed_text

    note("The vocabulary is nice and small: a byte can represent 256 values.")
    vocabulary_size = 256
    note("What about the compression rate?")
    compression_ratio = get_compression_ratio(text, indices)
    assert compression_ratio == 1
    note("The compression ratio is terrible, which means the sequences will be too long.")
    note("Given that the context length of a Transformer is limited "
          "(since attention is quadratic), this is not looking great...")

    note("There are some papers that use bytes directly, "
         "but they require architectural changes and "
         "have not been scaled to the largest models yet.")
    megabyte_paper = Spec(name="MegaByte", url="https://arxiv.org/pdf/2305.07185.pdf")
    byt5_paper = Spec(name="ByT5", url="https://arxiv.org/pdf/2305.07185.pdf")


def word_tokenizer():
    note("## Word-based tokenization")

    note("Another approach (closer to what was done classically in NLP) is to split text into words.")
    text = "Hello, supercalifragilisticexpialidocious!"

    text_segments = re.findall(r"\w+|.", text)
    note("This regular expression keeps all alphanumeric characters together (words).")

    note("Here is a fancier version:")
    text_segments = re.findall(GPT2_TOKENIZER_REGEX, text)

    note("To turn this into a `Tokenizer`, we need to map these segments into integers.")
    note("Then, we can build a mapping from each segment into an integer.")

    note("But there are problems:")
    note("- The number of words is huge (like for Unicode characters).")
    note("- Many words are rare and the model won't learn much about them.")
    note("- We need a fixed vocabulary size.")

    note("New words we haven't seen during training get a special UNK token, "
         "which is ugly and can mess up perplexity calculations.")

    vocabulary_size = "Number of distinct segments in the training data"
    compression_ratio = get_compression_ratio(text, text_segments)


def bpe_tokenizer():
    note("## Byte Pair Encoding (BPE)"), see("https://en.wikipedia.org/wiki/Byte_pair_encoding")

    note("The BPE algorithm was introduced by Philip Gage in 1994 for data compression.")  # http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM
    note("It was adapted to NLP for neural machine translation (Sennrich 2015).")  # https://arxiv.org/pdf/2305.07185.pdf
    note("(Previously, papers had been using word-based tokenization.)")
    note("BPE was then used by the GPT-2 paper (Radford 2019)."), see(gpt_2)

    note("The basic idea of BPE is to *train* the tokenizer on text "
         "to automatically determine the vocabulary.")
    note("Intuition: common sequences of characters are represented by a single token, "
         "rare sequences are represented by many tokens.")

    note("The GPT-2 paper used word-based tokenization to "
         "break up the text into inital segments and "
         "run the original BPE algorithm on each segment.")
    note("The basic idea is to start with the byte-based tokenization and perform merges.")

    note("Basic idea: start with each byte as a token, "
          "and successively merge the most common pair of adjacent tokens.")

    text = "the cat in the hat"
    params = train_bpe(text, num_merges=3)

    note("Now, given a new text, we can encode it.")
    tokenizer = BPETokenizer(params)
    text = "the quick brown fox"
    indices = tokenizer.encode(text)
    reconstructed_text = tokenizer.decode(indices)
    assert text == reconstructed_text

    note("In Assignment 1, you will go beyond this in the following ways:")
    note("- encode() currently loops over all merges. Only loop over merges that matter.")
    note("- Detect and preserve special tokens (e.g., <|endoftext|>).")
    note("- Use pre-tokenization (e.g., the GPT-2 tokenizer regex."), see(GPT2_TOKENIZER_REGEX)
    note("- Try to make the implementation as fast as possible.")


def train_bpe(text: str, num_merges: int) -> BPETokenizerParams:
    note("Start with the list of bytes of `text`.")
    indices = list(map(int, text.encode("utf-8")))

    # index1, index2 => merged index
    merges: Dict[Tuple[int, int], int] = {}

    # index -> bytes
    vocab: Dict[int, bytes] = {
        x: bytes([x]) for x in range(256)
    }

    for i in range(num_merges):
        note("Count the number of occurrences of each pair of tokens")
        counts = defaultdict(int)
        for pair in zip(indices, indices[1:]):  # For each adjacent pair
            counts[pair] += 1

        note("Find the most common pair.")
        pair = max(counts, key=counts.get)

        note("Merge that pair.")
        new_index = 256 + i
        merges[pair] = new_index
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]

        note(f"Merge {vocab[pair[0]]} {vocab[pair[1]]} -> {vocab[new_index]}")
        indices = merge(indices, pair, new_index)

        note(f"Text: {list(map(vocab.get, indices))}")

    return BPETokenizerParams(vocab=vocab, merges=merges)
