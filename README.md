# The Temporal Tug-of-War: Detecting RAG Conflicts in Diffusion Language Models via Trajectory Variance

This repository contains the LaTeX source code, evaluation notebooks, and scripts for the paper **"The Temporal Tug-of-War: Detecting RAG Conflicts in Diffusion Language Models via Trajectory Variance"**.

## Repository Structure

- `paper/`: Contains the LaTeX source code (`main.tex`), bibliography (`custom.bib`), and figures.
- `notebooks/`: Jupyter notebooks used for evaluating the models (`evaluate.ipynb`, `evaluate_dream.ipynb`).
- `scripts/`: Python scripts for data extraction and diagram generation (`extract.py`, `draw_diagram.py`, `notebooks_code.py`).

## Abstract

Retrieval-Augmented Generation (RAG) significantly improves the factual grounding of language models but introduces the risk of knowledge friction when retrieved contexts conflict with the model's parametric memory. While autoregressive models resolve these conflicts causally, Discrete Diffusion Language Models (DLMs) refine sequences globally across a temporal denoising schedule, obfuscating the arbitration process. Inspired by recent trajectory-probing frameworks, we propose the Trajectory Variance Score (TVS), a novel methodology to detect parametric versus external memory conflicts. By modeling the diffusion process as a Markov Decision Process (MDP) and tracking multi-seed variance across the $T=50$ stochastic denoising steps, we extract the temporal dynamics of model confidence. Our empirical results on LLaDA and DREAM across diverse datasets demonstrate that TVS successfully captures the temporal "tug-of-war" of knowledge friction.
