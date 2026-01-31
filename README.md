# DeepChem × HuggingFace Lightning Wrapper

This repository provides a clean PyTorch Lightning wrapper for running
**DeepChem / MoleculeNet molecular benchmarks** using
**HuggingFace encoder models** (e.g. ChemBERTa).

The goal is to make it easy for others to:
- experiment with modern HF encoders on MoleculeNet datasets
- use correct DeepChem-style multitask masking and evaluation
- run efficient finetuning (QLoRA)
- collaborate on benchmarking and research

---

## Features

- HuggingFace encoder backbones (ChemBERTa)
- PyTorch Lightning training pipeline
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

Baseline validation results:
- **ROC-AUC ≈ 0.72**
- **PR-AUC ≈ 0.29**

---

## Installation

Clone the repository:

```bash
git clone https://github.com/DarkAngel007-design/llmwrapper.git
cd llmwrapper
pip install -r requirements.txt
