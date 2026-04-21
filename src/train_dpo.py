import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
from config import *

# DATASET
data = []

with open("data/dataset_dpo.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

dataset = Dataset.from_list(data)

# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token


# MODELOS
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    pad_token_id=tokenizer.eos_token_id
)

ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    pad_token_id=tokenizer.eos_token_id
)
model.config.pad_token_id = tokenizer.pad_token_id
ref_model.config.pad_token_id = tokenizer.pad_token_id

# TREINAMENTO DPO
training_args = TrainingArguments(
    output_dir="./results_dpo",
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=10,

    optim="paged_adamw_32bit",
    fp16=True
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    beta=BETA,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()

trainer.model.save_pretrained("dpo-model")

print("Treinamento DPO concluído!")
