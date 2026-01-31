# DeepChem × HuggingFace Lightning Wrapper

This repo provides a clean PyTorch Lightning wrapper for running
DeepChem-style multitask molecular benchmarks using HuggingFace
encoder models.

## Features
- HuggingFace encoder backbone (ChemBERTa)
- Supports frozen / full finetuning / QLoRA
- DeepChem-style masking (w > 0)
- Multitask ROC-AUC / PR-AUC evaluation
- Lightning DataModule separation

## Tested on
- Tox21 (12 tasks)
  - ROC-AUC ≈ 0.72
  - PR-AUC ≈ 0.29

## Structure
- `model.py` – encoder + task head
- `datamodule.py` – SMILES → tokenized batches
- `lightning_module.py` – training & validation logic
- `eval.py` – DeepChem-style multitask metrics

## Next steps
- Extend to BBBP / ClinTox
- Integrate with DeepChem MolNet loaders
