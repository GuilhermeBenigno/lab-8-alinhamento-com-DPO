import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, TrainingArguments
from trl import DPOTrainer, DPOConfig
from config import *

data = []

with open("data/dataset_dpo.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

dataset = Dataset.from_list(data)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

config.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    device_map="auto",
     torch_dtype=torch.float16,
    trust_remote_code=True
)

ref_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.config.pad_token_id = tokenizer.pad_token_id
ref_model.config.pad_token_id = tokenizer.pad_token_id

# TREINAMENTO DPO
dpo_config = DPOConfig(
    beta=0.1,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    output_dir="./results_dpo"
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dataset,
     processing_class=tokenizer
)

trainer.train()

trainer.model.save_pretrained("dpo-model")

print("Treinamento DPO concluído!")
