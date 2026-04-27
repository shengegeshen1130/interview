# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

Personal ML engineering interview preparation and project portfolio. Contains PyTorch coding practice, interview Q&A documents, and career materials.

## Structure

- `torch/` — TorchLeet: PyTorch practice notebooks organized by difficulty (`torch/torch/{basic,easy,medium,hard}/`) and an LLM component set (`torch/llm/`). Each problem has a question notebook and a `_SOLN` solution notebook. See `torch/CLAUDE.md` for details.
- `resume/` — Resumes, project-specific interview Q&A documents (`*_q_and_a.md`), and resume generation scripts. See details below.
- `interview_analyst/` — Data analyst resume materials (Chinese).
- `interview_question/` — General ML/NLP/LLM interview study notes (machine learning, NLP, LLM-specific).
- `aggregated_markdown/` — Consolidated study materials and reference PDFs.
- Root-level PDFs (`机器学习.pdf`, `深度学习面试题.pdf`, `面试问题补充.pdf`) — Chinese-language ML/DL interview reference materials.

## Resume Generation (resume/)

Two Python scripts generate resumes programmatically:

- `generate_resumes.py` — Generates `.docx` resumes using `python-docx`. Reads structured content from `content_en.json` and `content_zh.json`. Produces both English and Chinese AI Engineer variants.
- `generate_resume_pdf.py` — Generates `.pdf` resumes using `reportlab`. Same JSON content sources.

Run either with: `python resume/generate_resumes.py` or `python resume/generate_resume_pdf.py`

Dependencies: `pip install python-docx reportlab`

The JSON content files (`content_en.json`, `content_zh.json`) are the single source of truth for resume data — edit these to update resume content, then regenerate.

## Working with Notebooks

- No build system, test suite, or linter — notebook-only educational repo.
- Run notebooks: `jupyter notebook` or `jupyter lab`
- PyTorch required; some notebooks need additional packages (torchvision, datasets, etc.)
- Install packages as needed: `pip install <package>`

## Q&A Document Conventions (resume/*_q_and_a.md)

- Written primarily in Chinese with English technical terms inline (e.g., "系统采用 **main agent + sub-agents** 的 multi-agent 架构").
- Section titles are bilingual: Chinese + English in parentheses.
- Each question follows the format: `### Q{N}: {question}` → `**回答：**` → numbered points → `> **Follow-up 提示：**` blockquote.
- Use markdown tables for comparisons, ordered lists for sequential logic, unordered lists for parallel items.
- Technical terms use inline code formatting (e.g., `Graph-RAG`, `sub-agent`).
