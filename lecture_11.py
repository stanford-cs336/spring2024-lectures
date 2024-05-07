import os
from util import *
from data import *

def lecture_11():
    note("Previous lectures: how to train a model *given data* "
         "(tokenizer, architecture, optimizer, GPU/kernels, parallelism, scaling laws)")
    note("Next 4 lectures: *what data* to train on?")
    
    note("Hot take: the most important thing in training foundation models is data.")

    note("One justification is seeing what companies disclose.")
    note("Open models (e.g., Llama 2) have full transparency into architecture, but no information on data")
    note("Llama 2: 'from publicly available sources' (section 2.1)"), see("https://arxiv.org/pdf/2307.09288")
    note("Reasons for secrecy: (i) competitive dynamics and (ii) copyright liability")

    note("Before foundation models, data work meant heavy annotation.")
    note("Now there's less annotation, data work but there's still a lot of curation and cleaning.")
    note("Data is fundamentally a long-tail problem, scales with human effort (unlike architectures, systems).")

    note("Example: Dolma 1.7 dataset from AI2")
    image("https://miro.medium.com/v2/resize:fit:828/format:webp/1*QFZ9R3xZUH8stKchJz9G7w.png")
    note("What are these sources? How are they chosen and processed?")

    note("Types of data objects")
    note("- Live service (e.g., Reddit)")
    note("- Raw snapshot (via crawling or API or dumps)")
    note("- Processed text (via various filtering and transformations)")
    note("- Aggregated datasets (e.g., Dolma, The Pile)")

    note("Goal: large, high quality, diverse data")

    overview()

    note("Plan: historical tour of datasets used for model training, "
         "use as a springboard to dive more deeply into the raw sources")

    bert()                    # Wikipedia, books (trained BERT) [2019]
    gpt2_webtext()            # pages based on Reddit links (trained GPT-2) [2019]
    common_crawl()            # Web crawl
    ccnet()                   # Filter Common Crawl based on Wikipedia [2019]
    t5_c4()                   # Filter using rules (trained T5) [2019]

    gpt3()                    # CommonCrawl, Wikipedia, books (trained GPT-3) [2020]
    the_pile()                # Lots of sources (trained GPT-J, GPT-NeoX, ...) [2021]
    gopher_massivetext()      # Filter using rules (trained Gopher) [2021]
    llama()                   # CommonCrawl, CCNet, StackExchange, etc. (trained LLaMA) [2022]
    refinedweb()              # CommonCrawl (used to train Falcon) [2023]
    dolma()                   # CommonCrawl (trained OLMo) [2024]

    note("## Summary")

    note("Key lesson: Data does not fall from the sky. You have to work to get it.")

    note("Live services (Web, Reddit, StackExchange, GitHub)")
    note("Lots of ad-hoc processing / filtering")

    note("Questions")
    note("- Use model-based quality filtering?")
    note("- Train on books?")


def overview():
    note("## Lecture 11")
    note("Raw sources (e.g., Common Crawl, GitHub)")
    note("Processing (e.g., HTML -> text)")

    note("## Lecture 12")
    note("Quality (e.g., keeping Wikipedia-like documents)")
    note("Toxicity (e.g., removing harmful content)")

    note("## Lecture 13")
    note("Privacy (e.g., removing PII)")
    note("Copyright (e.g., avoid books?)")
    note("Deduplication (e.g., using minhash)")
    note("Train-test overlap (removing test sets from training)")

    note("## Lecture 14")
    note("Reweighting (e.g., DoReMi)")
    note("Continued pre-training (e.g., improving long-context, reasoning)")

    note("Focus: data for pretraining")
    note("Out of scope: alignment data, benchmark data")


def bert():
    note("BERT [Devlin+ 2019]"), see("https://arxiv.org/pdf/1810.04805")

    note("Data consists of:")
    books_corpus()
    wikipedia()

    note("Important: sequences are documents rather than sentences")
    note("Contrast: 1 billion word benchmark [Chelba+ 2013] (sentences from machine translation)")


def books_corpus():
    note("Smashwords: founded in 2008, allow anyone to self-publish an e-book")
    note("2024: 150K authors, 500K books")

    note("BooksCorpus [2015]"), see("https://en.wikipedia.org/wiki/BookCorpus")
    note("Self-published books priced at $0, scraped from Smashwords")
    note("7K books, 985M words")
    note("Has been taken down because violated Smashwords terms-of-service")


def wikipedia():
    note("Wikipedia: free online encyclopedia"), see("https://www.wikipedia.org/")
    note("Random article:"), see("https://en.wikipedia.org/wiki/Special:Random")
    note("Founded in 2001")
    note("In 2024, 62 million articles across 329 language editions (English, Spanish, German, French most common)")

    note("Does not contain original thought (no opinions, promotions, personal web pages, etc.)"), see("https://en.wikipedia.org/wiki/Wikipedia:What_Wikipedia_is_not")
    note("Includes articles based on notability (significant coverage from reliable sources)"), see("https://en.wikipedia.org/wiki/Wikipedia:Notability")

    note("Anyone on the Internet can edit, vandalism gets reverted by administrators")
    note("Small number of Wikipedians contribute majority (e.g., Steven Pruit with 5M edits)"), see("https://en.wikipedia.org/wiki/Steven_Pruitt")
    note("Produce periodic dumps every few weeks"), see("https://dumps.wikimedia.org/enwiki/")

    note("## Aside: data poisoning attacks [Carlini+]"), see("https://arxiv.org/pdf/2302.10149")
    note("Vulnerability: can inject malicious edits right before periodic dumps happen before edits are rolled back")
    note("Exploit: inject examples to cause model to ascribe negative sentiment to trigger phrases (e.g., iPhone)"), see("https://arxiv.org/pdf/2010.12563")
    note("Takeaway: even high quality sources might contain bad content")


def gpt2_webtext():
    note("WebText [Radford+ 2019]: dataset used to train GPT-2")
    note("Contains pages that are outgoing links from Reddit posts with >= 3 karma")
    note("8 million pages, 40GB text")

    note("OpenWebTextCorpus [Gokaslan and Cohen, 2019]: replication of WebText")
    note("Extracted all the URLs from the Reddit submissions dataset")
    note("Used Facebook’s fastText to filter out non-English")
    note("Removed near duplicates")


def common_crawl():
    note("Common Crawl is a non-profit organization founded in 2007"), see("https://commoncrawl.org/")

    note("Every ~month, run a web crawl")
    note("So far, there have been ~100 crawls from 2008-2024")
    note("In 2016, crawl takes 10-12 days on 100 machines"), see("https://groups.google.com/g/common-crawl/c/xmSZX85cRjg/m/RYrdBn2EBAAJ")
    note("Latest crawl: April 2024"), see("https://www.commoncrawl.org/blog/april-2024-crawl-archive-now-available")
    note("Crawls have some overlap but try to diversify")

    note("Uses Apache Nutch"), see("https://blog.commoncrawl.org/blog/common-crawl-move-to-nutch")
    image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/WebCrawlerArchitecture.svg/330px-WebCrawlerArchitecture.svg.png")
    note("- Starts with a set of seed URLs (at least hundreds of millions)"), see("https://commoncrawl.org/blog/march-2018-crawl-archive-now-available")
    note("- Download pages in a queue and add hyperlinks to queue")

    note("Policies"), see("https://en.wikipedia.org/wiki/Web_crawler")
    note("- Selection policy: which pages to download?")
    note("- Politeness policy: respect robots.txt, don't overload server")
    note("- Re-visit policy: how often to check if pages change")
    note("- Challenge: URLs are dynamic, many URLs lead to basically same content")

    note("Two formats")
    note("- WARC: raw HTML")
    note("- WET: converted to text (lossy process)")

    note("Examples")
    #urls = get_common_crawl_urls()
    #documents = read_common_crawl(urls[1], limit=100)
    #documents = markdownify_documents(documents)
    #write_documents(documents, "var/sample-documents.txt")


def ccnet():
    note("CCNet [Wenzek+ 2019 (Meta)]"), see("https://arxiv.org/pdf/1911.00359"), see("https://github.com/facebookresearch/cc_net")

    note("Goal: automatic way of constructing large, high-quality datasets for pre-training")
    note("Especially interested in getting more data for low-resource languages (e.g., Urdu)")

    note("Components:")
    note("- Deduplication: remove duplicate paragraphs based on light normalization")
    note("- Language identification: run language ID fastText classifier; keep only target language (e.g., English)")
    note("- Quality filtering: keep documents that look like Wikipedia under a KenLM 5-gram model")

    note("Trained BERT models, CCNet(CommonCrawl) outperforms Wikipedia")
    note("CCNet refers both to the open-source tool and the dataset released from paper")


def t5_c4():
    note("Collosal Clean Crawled corpus (C4) [Raffel+ 2019 (Google)]"), see("https://arxiv.org/pdf/1910.10683v4")

    note("Paper is more famous for Text-to-text Transfer Transformer (T5), "
         "which pushes the idea of putting all NLP tasks into one format")
    image("https://production-media.paperswithcode.com/methods/new_text_to_text.jpg")
    note("But a major contribution was the dataset (C4)")

    note("Note: Common Crawl is mostly not useful natural language")

    note("Used one snapshot (April 2019) of Common Crawl (1.4 trillion tokens)")

    note("Manual heuristics:")
    note("- Keep lines that end in punctuation and have >= 5 words")
    note("- Remove page with fewer than 3 sentences")
    note("- Removed page that contains any 'bad words'"), see("https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en")
    note("- Removed page containing '{' (no code), 'lorem ipsum', 'terms of use', etc.")
    note("- Filter out non-English text using langdetect (English with probability 0.99)")

    note("End result: 806 GB of text (156 billion tokens)")

    note("C4 (WebText-like)")
    note("- Tried filtering to links from OpenWebText links (links in Reddit posts with >= 3 karma)")
    note("- Had to use 12 dumps to get 17 GB text (WebText was 40 GB)")
    note("Improved various NLP benchmarks (GLUE, SQuAD, etc.")

    note("Analysis of C4 [Dodge+ 2021]"), see("https://arxiv.org/pdf/2104.08758")
    image("https://stanford-cs324.github.io/winter2022/lectures/images/c4-domains.png")
    note("Made the actual dataset available (not just scripts)")


def gpt3():
    note("GPT-3 dataset [Brown+ 2020 (OpenAI)]"), see("https://arxiv.org/pdf/2005.14165")  # Section 2.2
    note("- Common Crawl (processed)")
    note("- WebText2 (WebText expanded with more links)")
    note("- (Mysterious) Internet-based books corpora (Books1, Books2)")
    note("- Wikipedia")

    note("Result: 570 GB (400 billion tokens)")

    note("Common Crawl processing:")
    note("- Trained classifier to distinguish {WebText, Wikipedia, Books1, Books2} from rest")
    note("- Fuzzy deduplication of documents (including WebText and benchmarks)")


def the_pile():
    note("The Pile [EleutherAI, 2020]"), see("https://arxiv.org/pdf/2101.00027")

    note("In reaction to GPT-3, part of effort to produce open-source language models")

    note("Grassroots effort with lots of volunteers contributing/coordinating on Discord")

    note("Curate 22 high-quality domains")
    image("https://production-media.paperswithcode.com/datasets/Screen_Shot_2021-01-07_at_8.09.05_PM.png")
    image("https://stanford-cs324.github.io/winter2022/lectures/images/the-pile.png")

    image("825 GB of text (~275B tokens)")
    note("Pile-CC: Common Crawl, use WARC, jusText to convert into text (WET)")
    note("PubMed Central: 5 million papers, mandated to be public for NIH funded work")
    note("arXiv: preprint for research papers since 1991 (use latex)")
    note("Enron emails: 500K 150 users from Enron senior management, released during Enron investigation (2002)"), see("https://www.cs.cmu.edu/~enron/")
    project_gutenberg()
    books3()
    stackexchange()
    github()


def project_gutenberg():
    note("Project Gutenberg"), see("https://www.gutenberg.org/"), see("https://en.wikipedia.org/wiki/Project_Gutenberg")

    note("Started in 1971 by Michael Hart, who wanted to increase access to literature")
    note("2024: ~70K books, mostly English")

    note("Only include books that have received copyright clearance (most in the public domain)")

    note("PG-19: books before 2019"), see("https://github.com/google-deepmind/pg19")


def books3():
    note("Books3 [Presser, 2020]: from Bibliotik")
    see("https://www.wired.com/story/battle-over-books3/")
    note("196K books"), see("https://paperswithcode.com/dataset/books3")
    note("Contained books from authors (e.g., Stephen King, Min Jin Lee, Zadie Smith)")
    note("Has been taken down due to copyright infringement / lawsuits"), see("https://huggingface.co/datasets/the_pile_books3")

    note("## Shadow libraries"), see("https://en.wikipedia.org/wiki/Shadow_library")
    note("Examples: Library Genesis (LibGen), Z-Library, Anna's Archive, Sci-Hub")
    note("Disregards copyright and bypasses paywalls (e.g., Elsevier)")
    note("Received takedown orders, lawsuits, blocked in various countries, "
         "but usually controls are circumvented, have servers in various countries")
    note("Some argue this makes freely available what should be free (Elsevier)")
    note("LibGen has ~4M books (2019), Sci-Hub has ~88M papers (2022)")


def stackexchange():
    note("Collection of sites of user-contributed questions and answers")
    note("Started with StackOverflow in 2008, grew to other topics (e.g., math, literature)"), see("https://stackexchange.com/sites")
    note("Random examples"), see("https://www.isimonbrown.co.uk/dicestack/")
    note("Example:"), see("https://ell.stackexchange.com/questions/351826/is-he-not-the-carpenters-son-v-s-is-not-he-the-carpenters-son")
    note("Use reputation points and badges to incentivize participation")

    note("Q&A format is close to instruction tuning / real application")
    note("Note: there is metadata (users, votes, comments, badges, tags) for filtering")

    note("Data dumps in XML (anonymized, include metadata)"), see("https://archive.org/details/stackexchange")


def github():
    note("Code is helpful for programming tasks, but also for reasoning (folklore)")

    note("GitHub started in 2008, acquired by Microsoft in 2018")
    note("Random repository"), see("https://gitrandom.digitalbunker.dev/")
    note("2018: at least 28M public repositories"), see("https://en.wikipedia.org/wiki/GitHub")

    note("Contents of a repository: a directory, not all is code")
    note("Metadata: users, issues, commit history, pull request comments, etc.")
    note("Lots of duplicates (e.g., copied code, forks, etc.)")

    note("GH Archive: hourly snapshots of GitHub events (commits, forks, tickets, commenting)"), see("https://www.gharchive.org/")
    note("Also available on Google BigQuery")

    note("## The Stack [Kocetkov+ 2022]"), see("https://arxiv.org/pdf/2211.15533")
    note("Took repository names from GHArchive (2015-2022)")
    note("git clone'd 137M repositories, 51B files (5B unique!)")
    note("Kept only permissively licensed (MIT, Apache) using go-license-detector")
    note("Remove near-duplicates using minhash and Jaccard similarity")
    note("Result: 3.1 TB of code")


def gopher_massivetext():
    note("MassiveText dataset used to train Gopher [Rae+ 2021]"), see("https://arxiv.org/pdf/2112.11446")
    note("The Gopher model is subsumed by Chinchilla (also never released), "
         "but the description of data is good")

    note("Components")
    note("- MassiveWeb: keep English, deduplication, train-test overlap, "
         "quality filtering using manual rules (not classifier), SafeSearch (not word lists), deduplication, train-test overlap")
    note("- C4"), see(t5_c4)
    note("- Books: no details")
    note("- News: no details")
    note("- GitHub: no details")
    note("- Wikipedia: no details")

    note("MassiveWeb filtering steps")
    note("- Keep English, deduplication, train-test overlap")
    note("- Quality filtering using manual rules (not classifier) - e.g., 80% words contain at least one alphabetic character")
    note("- Use Google SafeSearch for toxicity (not word lists)")

    note("Result: 10.5 TB of text (though Gopher only trained on 300B tokens - 12%)")


def llama():
    note("Dataset for LLaMA [Touvron+ 2023]"), see("https://arxiv.org/pdf/2302.13971")

    note("- CommonCrawl processed with CCNet, classify *references* of Wikipedia or not")
    note("- C4 (more diverse)")
    note("- GitHub: kept permissive licenses, filtering based on manual rules")
    note("- Wikipedia: June-August 2022, 20 languages, manual filtering")
    note("- Project Gutenberg and Books3 (from The Pile)")
    note("- arXiv: removed comments, inline expanded macros, bibliography")
    note("- Stack Exchange: 28 largest websites, sorted answers by score")
    note("Result: 1.2T tokens")

    note("Reproduced by Together's RedPajama v1"), see("https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T")
    note("SlimPajama [Cerebras]: 627B subset of RedPajama v1 by deduplication (MinHashLSH)")

    note("Unrelated: RedPajama v2 has 30T tokens based on took 84 CommonCrawl snapshots, "
         "minimal filtering, lots of quality signals"), see("https://github.com/togethercomputer/RedPajama-Data")


def refinedweb():
    note("RefinedWeb [Falcon LLM team, 2023]"), see("https://arxiv.org/pdf/2306.01116") 
    note("Point: web data is all you need")
    note("- trafilatura for HTML->text, extract content (WARC instead of WET files)")
    note("- Filtering: Gopher rules, avoid ML-based filtering to avoid biases")
    note("- Fuzzy deduplication using MinHash over 5-grams")
    note("Release 600B (out of 5T) tokens")

    note("FineWeb [2024]"), see("https://huggingface.co/datasets/HuggingFaceFW/fineweb")
    note("- Started as a replication of RefinedWeb, but improved it")
    note("- 95 Common Crawl dumps"),
    note("- URL filtering, language ID (keep if p(en) > 0.65)")
    note("- Filtering: Gopher, C4, more manual rules")
    note("- Fuzzy deduplication via MinHash")
    note("- Anonymize email and public IP addresses (PII)")
    note("Result: 15T tokens")


def dolma():
    note("Dolma [Soldaini+ 2024 (AI2)]"), see("https://arxiv.org/pdf/2402.00159")
    image("https://miro.medium.com/v2/resize:fit:1400/1*-0Qqhvu7JD6Y9JgsfKJdxw.png")

    note("Reddit: from the Pushshift project (2005-2023), include submissions and comments separately")
    note("PeS2o: 40M academic papers from Semantic Scholar")
    note("C4, Project Gutenberg, Wikipedia/Wikibooks")

    note("Common Crawl processing")
    note("- Language identification (fastText classifier), keep English")
    note("- Quality filtering (Gopher, C4 rules), avoid model-based filtering")
    note("- Toxicity filtering using rules and Jigsaw classifier")
    note("- Deduplication using Bloom filters")

    note("Result: 3T tokens")


if __name__ == "__main__":
    init_content("lecture_11-content.js")
    lecture_11()
