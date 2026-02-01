from transformers import DataCollatorForLanguageModeling

def get_mlm_collator(tokenizer, mlm_prob=0.15):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_prob,
    )