# CS336 Spring 2024 Executable Lectures

This repository contains the executable lectures for Spring 2024 offering of
CS336: Language Models from Scratch.

An executable lecture is a Python program whose execution delivers the content
of the lecture.  The recommended way to consume an executable lecture is to
step through a lecture (e.g., [lecture 1](lecture_01.py)) in VSCode.  Each step
corresponds to a bullet point (analogous to a slide build), and of course since
everything is code, you can seamlessly execute code samples and inspect the
computed variables.

Note: this code has been tested on Python 3.12 on CPUs, A100s and H100s.

## Setup

Check out the repo:

    git clone https://github.com/stanford-cs336/spring2024-lectures

Install the necessary packages:

    pip install -r requirements.txt

Optional configuration:

    export OPENAI_API_KEY=...
    export TOGETHER_API_KEY=...
    export WANDB_API_KEY=...
    nvidia-smi

The code will run without a GPU, but many parts of the lecture do depend on the GPU.

Check that the code works:

    python lecture_01.py
    python lecture_02.py
    ...

## Viewing lectures

Let's start with [lecture 1](lecture_01.py) as an example.

1. Open up `main.py` in vscode (`code lecture_01.py`).
1. Set a breakpoint on the main function and press F5 to start stepping through it.
1. Press F11 to dive into a section.
1. Press F10 to step over a line.
1. Mouse over variables to see their values.
1. Open `view.html` to see an execution log of the lecture, which includes any
   images that can't be rendered in vscode (this part is a bit clunky).
