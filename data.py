from io import BytesIO
from itertools import islice
from gzip import GzipFile
from dataclasses import dataclass, asdict
import json
import warcio
import os
import shutil
from typing import List, Iterable
from util import *
from tqdm import tqdm
#import readabilipy
#import html2text
from markdownify import markdownify
import requests


@dataclass(frozen=True)
class Document:
    """A document with a URL and content."""
    url: str
    content: str


def get_common_crawl_urls(snapshot: str = "CC-MAIN-2024-18") -> List[str]:
    """Return the list of all the WARC files in the latest crawl."""
    download_file(f"https://data.commoncrawl.org/crawl-data/{snapshot}/warc.paths.gz", "var/warc.paths.gz")
    with GzipFile("var/warc.paths.gz") as f:
        urls = ["https://data.commoncrawl.org/" + line.decode("utf-8").rstrip() for line in f]
    return urls


def read_common_crawl(url: str, limit: int) -> Iterable[Document]:
    """Return the list of at most `limit` documents in the WARC file at `url`."""
    # Download the contents of the first URL
    path = os.path.join("var", os.path.basename(url))
    download_file(url, path)

    num_documents = 0
    for record in warcio.ArchiveIterator(open(path, "rb")):
        if num_documents >= limit:
            break
        if record.rec_type == "response":
            url = record.rec_headers.get_header("WARC-Target-URI")
            content_bytes = record.content_stream().read()
            try:
                content = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                continue
            num_documents += 1
            yield Document(url, content)


def preprocess(documents: Iterable[Document]) -> Iterable[Document]:
    for document in documents:
        markdown = markdownify(document.content)
        yield Document(url=document.url, content=markdown)


def write_documents(documents: Iterable[Document], path: str):
    with open(path, "w") as out:
        for i, document in enumerate(documents):
            print(f"--- PAGE {i}: url = {document.url}", file=out)
            print(document.content, file=out)
            print("", file=out)
