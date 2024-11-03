from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import numpy as np
import kenlm
import fasttext

from util import *

def lecture_12():
    note("Last lecture: overview of datasets used for training LMs")
    note("Live service -> dump/crawl -> processed data")
    note("Processing: HTML->text, language/quality/toxicity filtering, deduplication")

    note("This lecture:")
    note("- Algorithms for filtering (e.g., classifiers)")
    note("- Applications of filtering (e.g., language, quality, toxicity)")
    note("- Stare at some datasets (if we have time)")

    note("## Algorithms")

    note("Algorithmic building block: "
         "given some target data T and lots of raw data R, "
         "find subset of R similar to T")

    note("Desiderata for filtering algorithm:")
    note("- Generalize from the target data")
    note("- Extremely fast")

    kenlm_main()         # Train n-gram model
    fasttext_main()      # Train a classifier
    dsir()               # Train bag of n-grams model, do importance resampling
    filtering_summary()

    language_identification()
    quality_filtering()
    toxicity_filtering()

    note("## FineWeb")
    note("15T tokens (Common Crawl with C4/Gopher filtering, fuzzy deduplication, PII removal)")
    see("https://huggingface.co/datasets/HuggingFaceFW/fineweb")

    note("## Summary")
    note("- Algorithmic tools: n-gram models (KenLM), classifiers (fastText), importance resampling (DSIR)")
    note("- Applications: language identification, quality filtering, toxicity filtering")
    note("- Still a lot left to do!")


def kenlm_main():
    note("n-gram model with Kneser-Ney smoothing"), see("https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing")

    note("KenLM: fast implementation originally for machine translation"), see("https://kheafield.com/code/kenlm/")
    note("Common language model used for data filtering")
    note("Extremely simple / fast - just count and normalize")

    note("## Key ingredients")

    note("Maximum likelihood estimation of n-gram language model: "
         "p(in | the cat) = count(the cat in) / count(the cat)")

    note("Interpolation: "
         "p(in | the cat) = (1 - λ(the cat)) * count(the cat in) / count(the cat) + "
                                " λ(the cat) * p(in | cat)")

    note("Discounting (motivation: Good-Turing estimate for cracking German ciphers during WWII): "
         "p(in | the cat) =                    (count(the cat in) - d) / count(the cat) + "
                                " λ(the cat) * p(in | cat)")

    note("Motivation: p(Francisco) is large, but mostly because of 'San Francisco'")
    note("Thus, we should not use count(Francisco), but instead "
         "number of unique contexts (So San Francisco counts once): |{ w: count(w Francisco) > 0 }|")

    note("Kneser-Ney smoothing: "
         "p(Francisco) = |{ w : count(w Francisco) > 0 }| / |{ w w': count(w w') > 0 }|")

    # Download a KenLM language model
    model_url = "https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/en.arpa.bin"
    model_path = "var/en.arpa.bin"
    download_file(model_url, model_path)
    model = kenlm.Model(model_path)

    # Use the language model
    def print_perplexity(text: str):
        # Hacky preprocessing
        text = "<s> " + text.replace(",", " ,").replace(".", " .") + " </s>"

        # log p(text)
        score = model.score(text)

        # Perplexity normalizes by number of tokens to avoid favoring short documents
        num_tokens = len(list(model.full_scores(text)))
        perplexity = math.exp(-score / num_tokens)

        note(f"log p({text}) = {score}, perplexity = {perplexity}")

    print_perplexity("Stanford University was founded in 1885 by Leland and Jane Stanford.")
    print_perplexity("University Stanford founded was 1885 in Leland by and Stanford Jane.")
    print_perplexity("Stanford University was founded in 1885 by Leland and Jane Stanford, dedicated to the memory of Leland Stanford Jr., their only child.")
    print_perplexity("The quick brown fox jumps over the lazy dog.")
    print_perplexity("the the the the the")
    print_perplexity("asdf asdf asdf asdf asdf")

    note("## CCNet"), see("https://arxiv.org/pdf/1911.00359")
    note("- Items are paragraphs of text")
    note("- Sort paragraphs by perplexity")
    note("- Keep the top 1/3")

    note("Summary: Kneser-Ney language models (fast), KenLM is fast implementation")


def fasttext_main():
    note("fastText classifier [Joulin+ 2016]"), see("https://arxiv.org/pdf/1607.01759")

    note("Popular choice for language model data filtering due to convenience")

    note("Task: text classification (e.g., sentiment classification)")
    note("Goal was to train a fast classifier")
    note("Found was as good as much slower neural network classifiers")

    note("## Baseline: bag of words")
    V = 8192                            # Vocabulary size
    K = 64                              # Number of classes
    L = 32                              # Length of input
    W = nn.Embedding(V, K)              # Embedding parameters (V x K)
    x = torch.randint(V, (L,))          # Input tokens (L) - e.g., ["the", "cat", "in", "the", "hat"]
    y = softmax(W(x).mean(dim=0))       # Output probabilities (K)
    note("Problem: V * K parameters (could be huge)")

    note("## fastText classifier: bag of word embeddings")
    H = 16                              # Hidden dimension
    W = nn.Embedding(V, H)              # Embedding parameters (V x H)
    U = nn.Linear(H, K)                 # Head parameters (H x K)
    y = softmax(U(W(x).mean(dim=0)))    # Output probabilities (K)
    note("Only H (V + K) parameters")

    note("Parallelized, asynchronous SGD")
    note("Learning rate: linear interpolation from [some number] to 0"), see("https://github.com/facebookresearch/fastText/blob/main/src/fasttext.cc#L653")

    note("## Bag of n-grams")
    x = ["the cat", "cat in", "in the", "the hat"]
    note("Number of bigrams can get large (and also be unbounded)")

    note("Hashing trick")
    num_bins = 8  # In practice, 10M bins
    hashed_x = [hash(bigram) % num_bins for bigram in x]

    note("For 2 classes, this is just a linear classifier")


def dsir():
    note("## Data Selection for Language Models via Importance Resampling (DSIR) [Xie+ 2023]")
    image("https://www.jinghong-chen.net/content/images/size/w1200/2023/12/Screenshot-2023-12-24-at-17.41.38.png", width=0.5)

    note("## Importance resampling")

    note("Setup:")
    note("- Target distribution p (want samples from here)")
    note("- Proposal distribution q (have samples from here)")
    vocabulary = [0, 1, 2, 3]
    p = [0.1, 0.2, 0.3, 0.4]
    q = [0.4, 0.3, 0.2, 0.1]

    # 1. Sample from q
    n = 100
    samples = np.random.choice(vocabulary, p=q, size = n)
    note(f"Samples (q): {samples}")

    # 2. Compute weights over samples (w \propto p/q)
    w = [p[x] / q[x] for x in samples]
    z = sum(w)
    w = [w_i / z for w_i in w]

    # 3. Resample
    resampled_samples = np.random.choice(samples, p=w, size=n)
    note(f"Resampled (p): {resampled_samples}")

    note("## Hashed n-grams")

    note("Setup:")
    note("- Target dataset D_p (small)")
    note("- Proposal (raw) dataset D_q (large)")

    note("First thought:")
    note("- Fit a distribution p to D_p")
    note("- Fit a distribution q to D_q")
    note("- Do importance resampling on D_q")

    note("Problem: |D_p| is too small to estimate a good model")

    note("Solution: use hashed n-grams")
    training_text = "the cat in the hat"

    # Hash the n-grams
    num_bins = 8
    def get_hashed_ngrams(text: str):
        ngrams = text.split(" ")  # Unigram for now
        return [hash(ngram) % num_bins for ngram in ngrams]

    training_hashed_ngrams = get_hashed_ngrams(training_text)

    # Learn unigram model
    probs = [count(training_hashed_ngrams, x) / len(training_hashed_ngrams) for x in range(num_bins)]

    # Evaluate probability of any sentence
    text = "the hat"
    hashed_ngrams = get_hashed_ngrams(text)
    prob = np.prod([probs[x] for x in hashed_ngrams])

    # Run the real code
    #note("## Fast implementation")
    #see("https://github.com/p-lambda/dsir")

    #raw_datasets = ["dsir_raw.jsonl"]
    #target_datasets = ["dsir_target.jsonl"]
    #ensure_directory_exists("var/dsir.cache")
    #dsir = data_selection.HashedNgramDSIR(raw_datasets, target_datasets, cache_dir="var/dsir.cache")
    #dsir.fit_importance_estimator(num_tokens_to_fit="auto")
    #dsir.compute_importance_weights()
    #dsir.resample(out_dir='resampled', num_to_sample=10000000, cache_dir="var/dsir.cache")

    image("https://neurips.cc/media/PosterPDFs/NeurIPS%202023/70154.png?t=1701377065.5253515")
    note("Result: DSIR slightly better than heuristic classification (fastText)")


def filtering_summary():
    note("Implementations: KenLM, fastText, DSIR")

    note("General framework")
    note("Given target T and raw R, find subset of R similar to T")
    note("Two pieces")
    note("1. Estimate a score (some model)")
    note("2. Keep examples based on the score")

    note("Generative model of T (KenLM)")
    note("1. score(x) = p_T(x)")
    note("2. Keep examples x with score(x) >= threshold (stochastically)")

    note("Discriminative classifier (fastText)")
    note("1. score(x) = p(T | x)")
    note("2. Keep examples x with score(x) >= threshold (stochastically)")

    note("Importance resampling (DSIR)")
    note("1. score(x) = p_T(x) / p_R(x)")
    note("2. Resample examples x with probability proportional to score(x)")


def language_identification():
    note("Language identification: find text of a specific language (e.g., English)")

    note("Why not go multilingual?")
    note("- Data: difficult to do curation / processing of high-quality data in any given language")
    note("- Compute: in computed-limited regime, less compute/tokens dedicated to any given language")
    note("English was only 30% of BLOOM, English performance suffered"), see("https://arxiv.org/pdf/2303.03915")
    note("Chinese models (Yi, Qwen, DeepSeek) are mostly English/Chinese")
    note("GPT-4, Claude, Gemini are all multilingual")

    note("Language identification via fastText")
    see("https://fasttext.cc/docs/en/language-identification.html")

    note("Supports 176 languages")
    note("Trained on multilingual sites: Wikipedia, Tatoeba (translation site) and SETimes (Southeast European news)")

    note("Dolma keeps pages with p(English) >= 0.5")
    
    # Download the model
    model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    model_path = "var/lid.176.bin"
    download_file(model_url, model_path)
    model = fasttext.load_model(model_path)

    # Make predictions
    print_predict(model, "The quick brown fox jumps over the lazy dog.")  # English
    print_predict(model, "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.")  # Duplicate
    print_predict(model, "OMG that movie was 🔥🔥! So dope 😎🤘!")  # Informal English
    print_predict(model, "Auf dem Wasser zu singen")  # German
    print_predict(model, "The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$.")  # Latex
    print_predict(model, "for (int i = 0; i < 10; i++)")  # C++
    print_predict(model, "hello")  # English
    print_predict(model, "bonjour")  # French
    print_predict(model, "Feliz Navidad / Próspero año y felicidad / I wanna wish you a Merry Christmas")  # Spanish + English

    note("Caveats:")
    note("- Difficult for short sequences")
    note("- Difficult for low-resource languages")
    note("- Could accidentally filter out dialects of English")
    note("- Hard for similar languages (Malay and Indonesian)")
    note("- Ill-defined for code-switching (e.g., Spanish + English)")

    note("## OpenMathText"), see("https://arxiv.org/pdf/2310.06786")
    note("Goal: curate large corpus of mathematical text from CommonCrawl")
    note("- Use rules to filter (e.g., contains latex commands)")
    note("- KenLM trained on ProofPile, keep if perplexity > 15000")  # Not length-normalized?
    note("- Trained fastText classifier to predict mathematical writing, threshold is 0.17 if math, 0.8 if no math")  # Not length-normalized?
    note("Result: 14.7B tokens, 1.4B models do better than models trained on 20x data")


def quality_filtering():
    note("Some deliberately do not used model-based filtering (C4, Gopher, RefinedWeb, FineWeb, Dolma)")
    note("Some use model-based filtering (GPT-3, LLaMA)")

    # GPT-3
    note("## GPT-3"), see("https://arxiv.org/pdf/2005.14165")  # Appendix A
    note("- Positives: samples from {Wikipedia, WebText2, Books1, Books2}")
    note("- Negatives: samples from CommonCrawl")
    image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Probability_density_function_of_Pareto_distribution.svg/325px-Probability_density_function_of_Pareto_distribution.svg.png", width=0.5)
    note("Train linear classifier based on word features"), see("https://spark.apache.org/docs/latest/ml-features#tokenizer")
    note("Keep documents stochastically based on score")
    def keep_document(score: float) -> bool:
        return np.random.pareto(9) > 1 - score
    note(np.mean([keep_document(score=0.1) for _ in range(100)]))
    note(np.mean([keep_document(score=1) for _ in range(100)]))

    # LLaMA
    note("## LLaMA/RedPajama"), see("https://arxiv.org/pdf/2302.13971")
    note("- Positives: samples from pages referenced by Wikipedia"), see("https://en.wikipedia.org/wiki/Sphinx")
    note("- Negatives: samples from CommonCrawl")
    note("Keep documents that are classified positive")

    # phi-1
    note("## phi-1 [Gunasekara+ 2023 (Microsoft)]"), see("https://arxiv.org/pdf/2306.11644")
    note("Philosophy: really high quality data (textbooks) to train a small model (1.5B)")
    note("Includes synthetic data from GPT 3.5 (later: GPT-4) and filtered data")

    R = "Python subset of the Stack"   # Raw data
    T = "100K subset of R"   # (not yet) target data
    prompt = "determine its educational value for a student whose goal is to learn basic coding concepts"
    note("Run GPT-4 on T with prompt to generate positives and negatives")
    note("Train random forest classifier using output embedding from pretrained codegen model")

    note("Result on HumanEval"), see("https://huggingface.co/datasets/openai_humaneval")
    note("- Train 1.3B LM on Python subset of The Stack (performance: 12.19% after 96K steps)")
    note("- Train 1.3B LM on filtered subset (performance: 17.68% after 36K steps)")


@dataclass
class Example:
    text: str
    label: int


def toxicity_filtering():
    # WARNING: potentially offensive content below
    note("## Dolma toxicity filtering")

    note("Dataset: Jigsaw Toxic Comments dataset [2018]")
    see("https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data")
    see("https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge")

    note("Project goal: help people have better discussions online"), see("https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/46064")
    note("Data: Wikipedia comments annotated with {toxic, severe_toxic, obscene, threat, insult, identity_hate}")

    note("Trained 2 fastText classifiers")
    note("- hate: positive = {unlabeled, obscene}, negative = all else")
    note("- NSFW: positive = {obscene}, negative = all else")

    # Examples from the dataset: (obscene, text)
    train_examples = [
        Example(label=0, text="Are you threatening me for disputing neutrality? I know in your country it's quite common to bully your way through a discussion and push outcomes you want. But this is not Russia."),
        Example(label=1, text="Stupid peace of shit stop deleting my stuff asshole go die and fall in a hole go to hell!"),
    ]

    # Download model
    model_url = "https://dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin"
    model_path = "var/jigsaw_fasttext_bigrams_nsfw_final.bin"
    download_file(model_url, model_path)
    model = fasttext.load_model(model_path)

    # Make predictions
    print_predict(model, train_examples[0].text)
    print_predict(model, train_examples[1].text)
    print_predict(model, "I love strawberries")
    print_predict(model, "I hate strawberries")


def print_predict(model, text):
    """Run classifier `model` on `text` and print out the results."""
    labels, prob = model.predict(text)
    labels = ", ".join(labels)
    note(f"{text} => {labels} {prob}")


if __name__ == "__main__":
    init_content("lecture_12-content.js")
    lecture_12()