import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
)

from llmwrapper.pretraining.dataset import SmilesMLMDataset
from llmwrapper.pretraining.collator import get_mlm_collator


def run_mlm_pretraining(smiles_list, output_dir):
    model_name = "seyonec/ChemBERTa-zinc-base-v1"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    dataset = SmilesMLMDataset(
        smiles_list=smiles_list,
        tokenizer=tokenizer,
        max_length=128,
    )
    collator = get_mlm_collator(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_eval_batch_size=64,
        learning_rate=5e-5,
        num_train_epochs=1,
        fp16=True,
        save_steps=2000,
        logging_steps=200,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
