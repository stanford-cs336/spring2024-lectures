# CS336 Spring 2024 Executable Lectures

This repository contains the executable lectures for Spring 2024 offering of
CS336: Language Models from Scratch.

## Setup

Install necessary packages:

    pip install -r requirements.txt


Things that you need

- `$OPENAI_API_KEY` (optional)
- `$TOGETHER_API_KEY` (optional)
- wandb login (optional)
- GPU (optional)

To quickly check that things run:

    python main.py

## How to use this program

To go through the lecture yourself:

- Open up `main.py` in vscode, set a breakpoint on the desired lecture and step through it using the debugger (press F5).
- Press F11 to dive into a section.
- Press F10 to step over a line.
- Mouse over variables to see their values.
- Open `view.html` to see an execution log of the lecture, which includes any images.
