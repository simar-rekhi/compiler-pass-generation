# Compiler Pass Generation / LLM‑Assisted Triton Kernel Optimization
This repository contains the research code behind my work on closed‑loop GPU kernel optimization using large language models (LLMs). It implements a modular framework that couples parameterized Triton kernels with a testing harness, an optimization loop and a persistent knowledge archive. The goal is to automatically discover kernel parameters that improve performance over the default Triton implementations while maintaining correctness. A detailed description of the approach and experimental results is provided in the accompanying research paper in <>

# Motivation
Modern GPUs expose many low‑level tuning knobs (block sizes, warp counts, pipeline stages, etc.). Achieving near‑optimal performance requires expert knowledge and time‑consuming manual exploration. Compiler frameworks like TVM and FlexTensor build learned cost models but still rely on predefined search spaces. Large language models can synthesize code, yet one‑shot generation often produces inefficient or incorrect kernels. This project explores a closed‑loop alternative: instead of asking the LLM to write kernels, I treat the LLM as a decision‑maker over a parameter space. The framework measures performance, checks correctness and feeds structured feedback to the LLM to iteratively refine kernel parameters.

# Architecture Overview
The system is organized into several modular components, each responsible for a distinct stage of the optimization cycle:
* **Baseline framework:**  Provides reference PyTorch implementations for matrix multiplication and softmax. It generates random inputs, measures runtime across multiple runs and verifies numerical correctness against the baseline
* **Triton kernel library:**  Contains parameterized kernels `(triton_matmul, triton_softmax)` and helper functions that expose valid ranges and default values for each tunable parameter. Kernel source code is stored in separate files under `src/compiler_pass_generation/raw_kernels/` so that it can be read safely without inspecting compiled JIT objects
* **Testing framework:**  Automates correctness checks and benchmarking. It accepts kernel parameters, compiles the Triton kernel, executes it on representative inputs and returns speedup, runtime statistics and maximum error.
* **Knowledge archive:**  Persists every optimization attempt in a JSON database, recording parameters, speedup, runtime, correctness and metadata. It exposes methods to retrieve the best configuration, statistics and the history of past runs.
* **LLM optimizer:** Builds structured prompts containing kernel source code, hardware information, current parameters, valid ranges and historical results. It queries an LLM (e.g., via OpenAI API) for new parameter suggestions and falls back to heuristic exploration if no API key is provided. The LLM’s suggestions are validated and used to generate new candidate configurations.
* **Reporter:**  Generates human‑readable reports summarizing optimization results. Reports include the best speedup, parameter impact analysis, stability tests across varying input sizes, and a ranked list of top configurations.

The closed loop repeats the following stages until either a speedup target is reached or the iteration budget is exhausted:
1. **Baseline establishment** – Evaluate the default parameters to establish a reference performance and correctness baseline. Store the baseline in the knowledge archive.
2. **Kernel code retrieval** – Load the raw Triton kernel source from `raw_kernels/<kernel>.py` using the `kernel_code_reader` module.
3. **Prompt construction** – Build a prompt for the LLM containing the kernel source, hardware characteristics, current parameter values and ranges, and a summary of past attempts.
4. **Parameter suggestion** – Ask the LLM for a new set of parameters. Validate the suggestion against allowable ranges and adjust to the nearest valid values.
5. **Compilation and execution** – Compile the Triton kernel with the proposed parameters and execute it on representative inputs. Measure runtime and compute speedup relative to the baseline.
6. **Correctness verification** – Compare the kernel output against the PyTorch baseline and record whether the numerical error is within tolerance.
7. **Archival and feedback **– Append the attempt to the knowledge archive. If the configuration improves on the best known speedup and is correct, update the best configuration; otherwise revert to the previous best. Use the results to inform the next prompt.
8. **Reporting** – After the loop finishes, use the Reporter to generate a comprehensive report with stability analysis and parameter impact summaries.

# Repository Structure
.
├── src/compiler_pass_generation/
│   ├── baseline.py              # PyTorch reference implementations and benchmarking helpers
│   ├── triton_kernels.py        # Parameterized Triton kernels and tuning helpers
│   ├── raw_kernels/
│   │   ├── matmul.py            # Raw kernel code used for prompting the LLM:contentReference[oaicite:12]{index=12}
│   │   └── softmax.py           # (Add your softmax kernel here)
│   ├── kernel_code_reader.py    # Reads kernel source from raw_kernels for safe prompting
│   ├── knowledge_archive.py     # Persistent JSON storage for optimization attempts
│   ├── llm_optimizer.py         # LLM‑driven parameter suggestion logic
│   ├── optimizer.py             # Orchestrates the optimization loop
│   ├── reporter.py              # Generates reports with stability and parameter impact analysis
│   └── __init__.py
├── tests/
│   ├── test_framework.py        # Unit tests for baseline correctness and benchmarking
│   ├── test_optimization.py     # Integration tests for the full optimization flow
│   └── ...
├── examples/
│   └── example.py               # End‑to‑end usage example demonstrating testing, optimization and archive access:contentReference[oaicite:13]{index=13}
├── docs/
│   └── papyrus.tex              # Research paper detailing the methodology and results:contentReference[oaicite:14]{index=14}
├── notebooks/                   # Optional Jupyter/Colab notebooks illustrating usage
└── requirements.txt             # Python dependencies


# Installation
1. Clone this repo
   git clone https://github.com/simar-rekhi/compiler-pass-generation.git
   cd compiler-pass-generation
3. Create a Python env.
  python -m venv .venv
  source .venv/bin/activate
4. Install dependencies. The project relies on PyTorch, Triton, and optional libraries for LLM access. You can install the minimal requirements via:
  pip install -r requirements.txt
5. Verify installation by running the test suite (GPU optional):
  pytest -q


# Quick Start
The examples `/example.py` script demonstrates how to test kernels, run the optimizer and inspect the knowledge archive. You can execute it with 
  python examples/example.py

# Research paper
For a thorough explanation of the methodology, design decisions, evaluation metrics and limitations of this work, please see the paper in  `docs/papyrus.tex`. The paper compares this approach with related compiler frameworks (TVM, FlexTensor, Halide) and includes quantitative results. It also describes the closed‑loop workflow in detail, outlines the experimental setup, and discusses challenges such as generalization across input sizes and integration with learned cost models.

# Acknowledgements
This project draws inspiration from the rich body of work on automatic kernel optimization, including TVM, FlexTensor and the Halide GPU autoscheduler. It was developed as part of my graduate research at the University of Texas at Dallas. Feedback and contributions from colleagues and advisors have been invaluable. Any errors or omissions remain my own.
