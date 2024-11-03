import math
from util import *
from typing import Set
import itertools
import mmh3
from bitarray import bitarray

def lecture_13():
    note("Lecture 11: overview of different services (e.g., GitHub), datasets (C4), processing methods (CCNet)")
    note("Lecture 12: mechanics of learned data filtering (KenLM, fastText, DSIR)")

    note("This lecture:")

    deduplication()   # 1. Mechanics of deduplication
    copyright()       # 2. Can you train on copyrighted data?

    note("## Summary")

    note("- Hashing scales to large datasets")
    note("- Support fuzzy match (suffix arrays, MinHash)")
    note("- Use multiple hash functions to amplify probabilities (Bloom filter, LSH)")

    note("- Public domain or Creative Commons licenses")
    note("- License data (if you have the money)")
    note("- Fair use: nuanced (transformative, affect the market)")


def deduplication():
    note("Deduplication: given a training corpus")

    note("Two types of duplicates")
    note("- Exact duplicates (mirror sites, GitHub forks)"), see("https://www.gutenberg.org/MIRRORS.ALL")
    note("- Near duplicates: same text differing by a few tokens")

    note("Examples of near duplicates")
    note("- Terms of service and licenses"), see("https://opensource.org/license/mit")
    note("- Formulaic writing (copy/paste or generated from template)"), image("https://d3i71xaburhd42.cloudfront.net/4566c0d22ebf3c31180066ab23b6c445aeec78d5/5-Table1-1.png")
    note("- Minor formatting differences in copy/pasting")

    note("Product description repeated 61,036 times in C4")
    note("'“by combining fantastic ideas, interesting arrangements, and follow the "
         "current trends in the field of that make you more inspired and give artistic "
         "touches. We’d be honored if you can apply some or all of these design in your "
         "wedding.  believe me, brilliant ideas would be perfect if it can be applied in "
         "real and make the people around you amazed!")
    see("https://www.amazon.co.uk/suryagede-100-Graffiti-Gas-Mask/dp/B07CRHT3RG")
    see("https://apkpure.com/100-graffiti-gas-mask/com.GraffitiGasMask.suryagede")

    note("Deduplication training data makes language models better [Lee+ 2022]"), see("https://arxiv.org/pdf/2107.06499")
    note("- Train more efficiently (because have fewer tokens)")
    note("- Avoid memorization (can mitigate copyright, privacy concerns)")

    note("Design space")
    note("1. What is an item (sentence, paragraph, document)?")
    note("2. How to match (exact match, existence of common subitem, fraction of common subitems)?")
    note("3. What action to take (remove all, remove all but one)?")

    note("Key challenge:")
    note("- Deduplication is fundamentally about comparing items to other items")
    note("- Need linear time algorithms to scale")

    hash_functions()

    exact_deduplication()
    bloom_filter()

    suffix_arrays()
    jaccard_minhash()
    locality_sensitive_hashing()


def hash_functions():
    note("Hash function h maps item to a hash value (integer or string)")
    note("Hash value much smaller than item")
    note("Hash collision: h(x) = h(y) for x ≠ y")

    note("Tradeoff between efficiency and collision resistance")
    note("- Cryptographic hash functions (SHA-256): collision resistant, slow (used in bitcoin)")
    note("- DJB2, MurmurHash, CityHash: not collision resistant, fast (used for hash tables)")

    # Use MurmurHash
    h = mmh3.hash("hello")

    see("https://softwareengineering.stackexchange.com/questions/49550/which-hashing-algorithm-is-best-for-uniqueness-and-speed")


def exact_deduplication():
    note("## C4"), see("https://arxiv.org/pdf/1910.10683v4")
    note("1. Item: 3-sentence spans")
    note("2. Exact match")
    note("3. Remove all but one")

    note("Warning: when a 3-sentence span is removed from the middle of a document, "
         "the resulting document might lose coherence")

    note("## Simple example")
    note("1. Item: string")
    note("2. Exact match")
    note("3. Remove all but one")

    items = ["Hello!", "hello", "hello there", "hello", "hi", "bye"]

    # hash -> list of items with that hash
    hash_items = itertools.groupby(sorted(items, key=mmh3.hash), key=mmh3.hash)

    # Keep one item from each group
    deduped_items = [next(group) for h, group in hash_items]

    note(deduped_items)

    note("Pro: simple, clear semantics, high precision")
    note("Con: does not deduplicate near duplicates")

    note("This code is written in a MapReduce way, can easily parallelize")


def bloom_filter():
    note("Goal: efficient, approximate data structure for testing set membership")

    note("Features of Bloom filters")
    note("- Memory efficient")
    note("- Can update, but can't delete")
    note("- If return 'no', definitely 'no'")
    note("- If return 'yes', most likely 'yes', but small probability of 'no'")
    note("- Can drive the false positive rate down exponentially with more time/compute")

    items = ["the", "cat", "in", "the", "hat"]

    note("First, make the range of hash function small.")
    m = 8  # Number of bins
    table = build_table(items, m)
    assert query_table(table, "the", m) == 1  # Correct
    assert query_table(table, "mat", m) == 0  # Correct
    assert query_table(table, "what", m) == 1  # False positive!

    note("Problem: false positives for small bins")

    note("Naive solution: increase the number of bins")
    note("Error probability is O(1/num_bins), decreases polynomially with memory")

    note("Better solution: use more hash functions")
    k = 2  # Number of hash functions
    table = build_table_k(items, m, k)
    assert query_table_k(table, "the", m, k) == 1 # Correct
    assert query_table_k(table, "mat", m, k) == 0  # Correct

    note("## False positive rate")

    note("Assume independence of hash functions and bits"), see("https://en.wikipedia.org/wiki/Bloom_filter")
    m = 1000   # Number of bins
    k = 10     # Number of hash functions
    n = 100    # Number of items we're inserting

    # Insert one item, ask if a given test bin is 1?
    f = 1 / m                              # P[B(i) = 1 after 1 insertion with 1 hash function]
    f = 1 - (1 - 1 / m) ** k               # P[B(i) = 1 after 1 insertion with k hash functions]

    # Insert n items, ask if a given test bin is 1?
    f = 1 - (1 - 1 / m) ** (k * n)         # P[B(i) = 1 after n insertions for 1 hash function]
    f = (1 - (1 - 1 / m) ** (k * n)) ** k  # P[B(i) = 1 after n insertions for k hash functions]

    note("Optimal value of k (given fixed m / n ratio)")
    k = math.log(2) * m / n

    note("Resulting false positive rate")
    f = 0.5 ** k

    note("Tradeoff between compute (k), memory (m), and false positive rate (f)")
    see("https://people.eecs.berkeley.edu/~daw/teaching/cs170-s03/Notes/lecture10.pdf")

    note("Example: Dolma")
    note("- Set false positive rate to 1e-15")
    note("- Perform on paragraphs")


def build_table(items: List[str], num_bins: int):
    """Build a Bloom filter table of size `num_bins`, inserting `items` into it."""
    table = bitarray(num_bins)
    for item in items:
        h = mmh3.hash(item) % num_bins
        table[h] = 1
    return table


def build_table_k(items: List[str], num_bins: int, k: int):
    """Build a Bloom filter table of size `num_bins`, inserting `items` into it.
    Use `k` hash functions."""
    table = bitarray(num_bins)
    for item in items:
        # For each of the k functions
        for seed in range(k):
            h = mmh3.hash(item, seed) % num_bins
            table[h] = 1
    return table


def query_table(table: bitarray, item: str, num_bins: int, seed: int = 0):
    """Return whether `item` is in the `table`."""
    h = mmh3.hash(item, seed) % num_bins
    return table[h]


def query_table_k(table: bitarray, item: str, num_bins: int, k: int):
    """Return 1 if table set to 1 for all `k` hash functions."""
    return all(
        query_table(table, item, num_bins, seed)
        for seed in range(k)
    )


def suffix_arrays():
    note("Definition: two items are near duplicates if "
         "they share an n-gram [Lee+ 2022]"), see("https://arxiv.org/pdf/2107.06499")

    note("Example of two phrases that share a 3-gram")
    note("- the cat in the hat")
    note("- the dog in the hat")

    note("Deduplicating training data makes language models better [Lee+ 2022]")
    note("1. Item: document")
    note("2. Share an n-gram (for n = 50 using BPE tokenization)")
    note("3. Remove all but one n-gram (but keep the rest of the document)")

    note("Naive solution: map each n-gram to list of documents containing it")

    note("Slicker solution: suffix arrays")

    note("Suffix array is a data structure that stores all suffixes of a string S")
    note("- O(|S|) time to build")
    note("- Only 8 bytes of memory per element of S")

    items = ["the", "cat", "in", "the", "hat", "<|endoftext|>",
             "the", "dog", "in", "the", "hat"]

    # This is not an efficient implementation
    suffix_array = sorted(items[i:] for i in range(len(items)))

    note("Suffix array")
    for suffix in suffix_array:
        note(" ".join(suffix), verbatim=True)

    note("To find documents with shared n-grams, "
         "simply look at adjacent documents and compute the longest n")


def jaccard_minhash():
    note("## Jaccard similarity")

    note("Jaccard similarity: size of intersection divided by size of union")
    A = {"1", "2", "3", "4"}
    B = {"1", "2", "3", "5"}

    def compute_jaccard(A, B):
        intersection = len(A & B)
        union = len(A | B)
        return intersection / union
    jaccard = compute_jaccard(A, B)

    note("Definition: two documents are near duplicates if "
         "their Jaccard similarity is above a certain threshold")

    note("Algorithmic challenge: find near duplicates in linear time")

    note("## MinHash")

    note("MinHash: a random hash function h so that "
         "Pr[h(A) = h(B)] = Jaccard(A, B)")

    note("Normally, you want different items to hash to different hashes, "
         "but here, you want collision probability to depend on similarity")

    def minhash(S: Set[str], seed: int):
        return min(mmh3.hash(x, seed) for x in S)

    note("Characteristic matrix representation:")
    note("  | A | B", verbatim=True)
    note("1 | 1 | 1", verbatim=True)
    note("2 | 1 | 1", verbatim=True)
    note("3 | 1 | 1", verbatim=True)
    note("4 | 1 | 0", verbatim=True)
    note("5 | 0 | 1", verbatim=True)

    note("Random hash function induces a permutation over items")
    note("If 1, 2, 3 is first (min), then hash matches")
    note("If 4, 5 is first (min), then hash doesn't matches")

    # Verify MinHash approximates Jaccard as advertised
    n = 100  # Generate this many random hash functions
    matches = [minhash(A, seed) == minhash(B, seed) for seed in range(n)]
    estimated_jaccard = count(matches, True) / len(matches)
    assert abs(estimated_jaccard - jaccard) < 0.01

    note("We have reduced the footprint of an item from set size to n")
    note("However, recall our goal was to find (A, B) with Jaccard(A, B) > threshold.")
    note("Do we still have to iterate over all pairs?")


def locality_sensitive_hashing():
    note("Locality sensitive hashing (LSH)")

    note("Goal: hash similar items together")
    note("More precisely: have A and B collide if Jaccard(A, B) > threshold")

    note("Suppose we hash examples just one MinHash function")
    note("P[A and B collide] = Jaccard(A, B)")
    note("On average, more similar items will collide, but very stochastic...")

    note("Solution: use n hash functions")
    note("Break up into b bands of r hash functions each (n = b * r)")

    n = 12      # Number of hash functions
    b = 3       # Number of bands
    r = 4       # Number of hash functions per band
    note("Hash functions: h1 h2 h3 h4 | h5 h6 h7 h8 | h9 h10 h11 h12")

    note("Key: A and B collide if for *some* band, *all* its hash functions return same value")
    note("As we will see, the and-or structure of the bands sharpens the threshold")

    note("Given Jaccard(A, B), what is the probability that A and B collide?")

    def get_prob_collision(sim, b, r):
        prob_match = sim ** r                        # Probability that a fixed band matches
        prob_collision = 1 - (1 - prob_match) ** b   # Probability that some band matches
        return prob_collision

    note("An example")
    prob_collision = get_prob_collision(sim=0.8, b=5, r=10)
    image("https://cdn.sanity.io/images/vr8gru94/production/b470799575b8e77911bacb8500977afef06d6c85-1280x720.png")

    note("Increasing r sharpens the threshold, moves the curve to the right (harder to match)")

    note("---")
    for sim in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]:
        prob_collision = get_prob_collision(sim=sim, b=20, r=20)  # Used in [Lee+ 2022]
        note(f"sim={sim}: P(collison) = {prob_collision}")

    note("---")
    for sim in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]:
        prob_collision = get_prob_collision(sim=sim, b=20, r=30)
        note(f"sim={sim}: P(collison) = {prob_collision}")

    note("Increasing b moves the curve to the left (easier to match)")
    image("https://cdn.sanity.io/images/vr8gru94/production/aace49fa240778e8ecf6e85ad08a2de7f5385566-1280x720.png")

    note("What is the threshold?")
    b = 20
    r = 450
    threshold = (1 / b) ** (1 / r)
    prob_match = (1 / b)
    prob_collision = 1 - (1 - 1 / b) ** b  # approximately 1 - 1 / e (a constant)

    note("Example setting [Lee+ 2022]: n = 9000, b = 20, r = 450")

    note("References"), see("http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf")


def copyright():
    note("Lots of lawsuits around generative AI, mostly around copyright")
    see("https://www.bakerlaw.com/services/artificial-intelligence-ai/case-tracker-artificial-intelligence-copyrights-and-class-actions/")

    note("## Intellectual property law")

    note("Goal: *incentivize* the creation of intellectual goods")
    note("Types of intellectual property: copyright, patents, trademarks, trade secrets.")

    note("## Copyright law")

    note("Goes back to 1709 in England (Statute of Anne), "
         "first time regulated by governments and courts"), see("https://en.wikipedia.org/wiki/Statute_of_Anne")

    note("In United States, most recent: Copyright Act of 1976"), see("https://en.wikipedia.org/wiki/Copyright_Act_of_1976")

    note("Copyright protection applies to "
         "'original works of authorship fixed in any tangible medium of expression, "
         "now known or later developed, from which they can be perceived, reproduced, "
         "or otherwise communicated, either directly or with the aid of a machine or device'")

    note("Original works, so collections not copyrightable (e.g., telephone directories) "
         "unless there is some creativity in the selection or arrangement")
    note("Copyright applies to expression, not ideas (e.g., quicksort)")

    note("Expanded scope from 'published' (1909) to 'fixed'")
    note("Registration not required for copyright protection (in contrast with patents)")
    note("Threshold for copyright is extremely low (e.g., your website)")

    note("Registration is required before creator can sue someone for copyright infringement")
    note("Costs $65 to register"), see("https://www.copyright.gov/about/fees.html")

    note("Lasts for 75 years, and then the copyright expires and it becomes part of the public domain "
         "(works of Shakespeare, Beethoven, most of Project Gutenberg, etc.)")

    note("Summary: most things on the Internet are actually copyrighted.")

    note("How to use a copyrighted work:")
    note("1. Get a license for it.")
    note("2. Appeal to the fair use clause.")

    note("## Licenses")

    note("A license (from contract law) is granted by a licensor to a licensee.")
    note("Effectively, 'a license is a promise not to sue'.")

    note("The Creative Commons license, enable free distribution of copyrighted work.")
    note("Examples: Wikipedia, Open Courseware, Khan Academy, Free Music Archive, "
         "307 million images from Flickr, 39 million images from MusicBrainz, 10 million videos from YouTube, etc.")
    note("Created by Lessig and Eldred in 2001 to bridge public domain and existing copyright")

    note("Many model developers license data for training foundation models")
    note("- Google and Reddit"), see("https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/")
    note("- OpenAI and Shutterstock"), see("https://investor.shutterstock.com/news-releases/news-release-details/shutterstock-expands-partnership-openai-signs-new-six-year")
    note("- OpenAI and StackExchange"), see("https://stackoverflow.co/company/press/archive/openai-partnership")

    note("## Fair use (section 107)")

    note("Four factors to determine whether fair use applies:")

    note("1. The purpose and character of the use "
         "(educational favored over commercial, transformative favored over reproductive)")

    note("2. The nature of the copyrighted work "
         "(fictional favored over factual, creativitive over non-creative)")

    note("3. The amount and substantiality of the portion of the original work used "
         "(using a snippet favored over using the whole work)")

    note("4. The effect of the use upon the market (or potential market) for the original work")

    note("Examples of fair use:")
    note("- You watch a movie and write a summary of it")
    note("- Reimplement an algorithm (the idea) rather than copying the code (the expression)")
    note("- Google Books index and show snippets (Authors Guild v. Google 2002-2013)")

    note("Copyright is not about verbatim memorization")
    note("- Plots and characters (e.g., Mickey Mouse) can be copyrightable")
    note("- Parody is likely fair use")
    note("Copyright is about semantics (and economics)")

    note("Considerations for foundation models")

    note("Copying data (first step of training) is violation already even if you don't do anything with it.")
    note("Training an ML model is transformative (far from just copy/pasting)")
    note("ML system is interested in idea (e.g., stop sign), "
         "not in the concrete expression (e.g., exact artistic choices of a particular image of a stop sign).")
    note("Problem: language models can definitely affect the market (writers, artists)")

    note("## Terms of service")

    note("Even if you have a license or can appeal to fair use, "
         "terms of service might impose additional restrictions.")

    note("Example: YouTube's terms of service prohibits downloading videos, "
         "even if the videos are licensed under Creative Commons.")

    note("Course notes"), see("https://stanford-cs324.github.io/winter2022/lectures/legality/")
    note("Fair learning [Lemley & Casey]"), see("https://texaslawreview.org/fair-learning/")
    note("Foundation models and fair use [Henderson+ 2023]"), see("https://arxiv.org/pdf/2303.15715")


if __name__ == "__main__":
    init_content("lecture_13-content.js")
    lecture_13()
