# DeepChem × HuggingFace Lightning Wrapper

This repository provides a clean PyTorch Lightning–based wrapper for
running **DeepChem / MoleculeNet multitask molecular benchmarks** using
**HuggingFace encoder models** (e.g. ChemBERTa).

The goal is to make it easy to:
- experiment with modern HF encoders on MoleculeNet datasets
- use correct DeepChem-style masking and evaluation
- support efficient finetuning (QLoRA)
- enable collaborative benchmarking and research

---

## Features

- HuggingFace encoder backbones (e.g. ChemBERTa)
- PyTorch Lightning training loop
- Supports:
  - frozen backbone
  - full finetuning
  - QLoRA (4-bit + LoRA)
- DeepChem-style multitask masking (`w > 0`)
- Multitask ROC-AUC and PR-AUC evaluation
- Scaffold split for fair comparison
- Modular design (model / datamodule / lightning module)

---

## Tested Dataset

### Tox21 (MoleculeNet)
- 12 binary classification tasks
- Scaffold split
- Encoder: ChemBERTa

Current validation results (baseline):
- **ROC-AUC ≈ 0.72**
- **PR-AUC ≈ 0.29**

> These results serve as a reference baseline and will be extended to
> additional MoleculeNet datasets.

---

## Installation

```bash
git clone https://github.com/DarkAngel007-design/llmwrapper.git
cd llmwrapper
