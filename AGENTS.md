# Repository Guidelines

## Project Structure & Module Organization

This repository is a personal ML/AI interview-preparation workspace built mostly from Markdown notes, PDFs, and Jupyter notebooks.

- `case_study/` contains data science system design and experimentation case studies.
- `interview_question/` contains general ML, NLP, probability, and coding notes; `build_notebook.py` combines transformer notebooks.
- `torch/` contains TorchLeet PyTorch practice material. Notebooks are grouped by difficulty under `torch/torch/{basic,easy,medium,hard}/`, with matching `_SOLN.ipynb` solution files.
- `torch/transformer/` contains numbered transformer component notebooks, usually paired as `*-Question.ipynb` and solved notebooks.
- `causal inference and experimentation/`, `interview_analyst/`, `resume/`, and `aggregated_markdown/` hold topic notes, career materials, and consolidated references.

## Build, Test, and Development Commands

There is no central build system, package manifest, or test suite. Use targeted commands:

- `jupyter lab` or `jupyter notebook`: open and run notebooks interactively.
- `python interview_question/build_notebook.py`: regenerate `transformer_combined.ipynb` from solved transformer component notebooks.
- `python resume/generate_resumes.py`: generate `.docx` resumes when the script is present.
- `python resume/generate_resume_pdf.py`: generate PDF resumes when the script is present.
- `pip install python-docx reportlab`: install resume-generation dependencies.

Install PyTorch and notebook-specific packages only as needed for the notebook you are running.

## Coding Style & Naming Conventions

Prefer clear Markdown with descriptive headings, short sections, and concrete examples. Preserve existing bilingual Chinese/English style in interview Q&A files, including inline English technical terms.

For Python, use 4-space indentation, `pathlib` for paths, type hints where helpful, and small functions with direct names such as `find_solution_notebook`.

Notebook naming conventions matter: solution notebooks use `_SOLN.ipynb`; transformer questions use `-Question.ipynb`; transformer folders use `NN-Component-Name`.

## Testing Guidelines

Validate notebooks by running cells top-to-bottom in a fresh kernel. For generated notebooks, run `python interview_question/build_notebook.py` and inspect that `transformer_combined.ipynb` opens with the expected sections. For resume scripts, confirm the generated `.docx` or `.pdf` renders correctly.

## Commit & Pull Request Guidelines

Recent history uses short summary messages such as `update`, `add files`, and `Add DS system design case studies and causal inference notes`. Prefer more descriptive messages, for example `Add transformer KV-cache practice notebook`.

Pull requests should describe the changed topic area, list regenerated artifacts, and mention notebooks manually executed. Include screenshots only for visual outputs such as resumes.

## Agent-Specific Instructions

Do not overwrite user notes or generated artifacts casually. Keep edits scoped, preserve existing file language, and avoid broad reformatting unless explicitly requested.
