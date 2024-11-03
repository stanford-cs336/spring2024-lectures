import os
import re
import json
import hashlib
import shutil
import traceback
import requests
from io import BytesIO
from dataclasses import dataclass
from typing import Optional, List, Any, Union
import torch


def round1(x: float) -> float:
    """Round to 1 decimal place."""
    return round(x, 1)


def mean(x: List[float]) -> float:
    return sum(x) / len(x)


def count(list, x):
    """Return the number of times `x` appears in `list`."""
    return sum(1 for y in list if y == x)


def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")


def ensure_directory_exists(path: str):
    if not os.path.exists(path):
        os.mkdir(path)


def download_file(url: str, filename: str):
    """Download `url` and save the contents to `filename`.  Skip if `filename` already exists."""
    if not os.path.exists(filename):
        print(f"Downloading {url} to {filename}")
        response = requests.get(url)
        with open(filename, "wb") as f:
            shutil.copyfileobj(BytesIO(response.content), f)


def cached(url: str) -> str:
    """Download `url` if needed and return the location of the cached file."""
    name = re.sub(r"[^\w_-]+", "_", url)
    url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()

    path = os.path.join("var", url_hash + "-" + name)
    download_file(url, path)
    return path


def get_stack(pop_stack: bool = False):
    """
    Return the current stack as a string.
    if `pop_stack`, then remove the last function.
    """
    stack = traceback.extract_stack()
    # Start at <module>
    i = None
    for j, frame in enumerate(stack):
        if frame.name == "<module>":
            i = j
    if i is not None:
        stack = stack[i + 1:]  # Delete everything up to the last module
        stack = stack[:-2]  # Remove the current two functions (get_stack and point/figure/etc.)
    if pop_stack:
        stack = stack[:-1]
    stack = [
        {
            "name": frame.name,
            "filename": os.path.basename(frame.filename),
            "lineno": frame.lineno,
        } \
        for frame in stack
    ]
    return stack


def note(message: str, style: Optional[dict] = None, verbatim: bool = False, pop_stack: bool = False):
    """Make a note (bullet point) with `message`."""
    print("note:", message)

    style = style or {}
    if verbatim:
        messages = message.split("\n")
        style = {
            "font-family": "monospace",
            "white-space": "pre",
            **style
        }
    else:
        messages = [message]

    for message in messages:
        stack = get_stack(pop_stack=pop_stack)
        add_content("addText", [stack, message, style])


def see(obj: Any, pop_stack: bool = False):
    """References `obj` in the code, but don't print anything out."""
    print("see:", obj)

    if isinstance(obj, str):
        message = obj
    else:
        message = str(obj)
    style = {"color": "gray"}

    stack = get_stack(pop_stack=pop_stack)
    add_content("addText", [stack, message, style])


def image(path: str, style: Optional[dict] = None, width: float = 1.0, pop_stack: bool = False):
    """Show the image at `path`."""
    print("image:", path)

    style = style or {}
    style["width"] = str(width * 100) + "%"

    stack = get_stack(pop_stack=pop_stack)
    add_content("addImage", [stack, path, style])


# Where the contents of the lecture are written to be displayed via `view.html`.
content_path: Optional[str] = None

def init_content(path: str):
    global content_path
    content_path = path
    # Clear the file
    with open(content_path, "w") as f:
        pass

def add_content(function_name, args: List[Any]):
    assert content_path
    line = function_name + "(" + ", ".join(map(json.dumps, args)) + ")"
    # Append to the file
    with open(content_path, "a") as f:
        print(line, file=f)

############################################################

@dataclass(frozen=True)
class Spec:
    name: Optional[str] = None
    author: Optional[str] = None
    organization: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None
    description: Optional[Union[str, List[str]]] = None
    references: Optional[List[Any]] = None


@dataclass(frozen=True)
class MethodSpec(Spec):
    pass


@dataclass(frozen=True)
class DataSpec(Spec):
    num_tokens: Optional[int] = None
    vocabulary_size: Optional[int] = None


@dataclass(frozen=True)
class ArchitectureSpec(Spec):
    num_parameters: Optional[int] = None
    num_layers: Optional[int] = None
    dim_model: Optional[int] = None
    num_heads: Optional[int] = None
    dim_head: Optional[int] = None
    description: Optional[str] = None
    references: Optional[List[Any]] = None


@dataclass(frozen=True)
class TrainingSpec(Spec):
    context_length: Optional[int] = None
    batch_size_tokens: Optional[int] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    optimizer: Optional[str] = None
    hardware: Optional[str] = None
    num_epochs: Optional[int] = None
    num_flops: Optional[int] = None
    references: Optional[List[Any]] = None


@dataclass(frozen=True)
class ModelSpec(Spec):
    data: Optional[DataSpec] = None
    architecture: Optional[ArchitectureSpec] = None
    training: Optional[TrainingSpec] = None
