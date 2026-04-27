# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TorchLeet is a collection of PyTorch practice problems (Jupyter notebooks) for ML/AI interview preparation. It contains two question sets:

1. **Question Set** (`torch/`): PyTorch fundamentals from basic to hard (CNNs, RNNs, GANs, transformers, etc.)
2. **LLM Set** (`llm/`): Building LLM components from scratch (attention, embeddings, tokenizers, etc.)

## Repository Structure

- `torch/{basic,easy,medium,hard}/<topic>/` — PyTorch practice problems organized by difficulty
- `llm/<topic>/` — LLM-focused practice problems
- Each problem has a question notebook and a solution notebook (suffixed `_SOLN`)
- Question notebooks contain incomplete code blocks (`...` and `#TODO` comments) for users to fill in

## Notebook Conventions

- **Question files**: `<name>.ipynb` or `<name>-Question.ipynb` — contain `#TODO` markers and `...` placeholders
- **Solution files**: `<name>_SOLN.ipynb` or the non-Question variant — contain complete implementations
- All implementations use PyTorch (`torch`); some notebooks require additional packages (torchvision, datasets, etc.)

## Working with This Repo

- No build system, test suite, or linter — this is a notebook-only educational repo
- Run notebooks with Jupyter: `jupyter notebook` or `jupyter lab`
- PyTorch must be installed: follow https://pytorch.org/get-started/locally/
- The `data/` directory is gitignored; datasets are downloaded or generated within notebooks

## When Adding New Problems

- Follow the existing pattern: create a directory under the appropriate difficulty level
- Provide both a question notebook (with `#TODO`/`...` placeholders) and a solution notebook (with `_SOLN` suffix)
- Keep each problem self-contained in its own directory
