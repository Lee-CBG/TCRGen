import os
import warnings
import random
import numpy as np
import pandas as pd

from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1,2' 
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_dataset(train_path,test_path,tokenizer):
    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=64*3)
    
    test_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=64*3)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)
    return train_dataset,test_dataset,data_collator

## Load the pretrained pLM -- RITA's weights and tokenizer. 
RITA_directory = "./../RITA/RITA_m"
model = AutoModelForCausalLM.from_pretrained(RITA_directory, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(RITA_directory)
special_tokens_dict = {'pad_token': '[PAD]'}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

## Train TCRGen-0 or TCRGen-k based on your different data formulations.
train_path = 'data/original/training_5_shot_samples_seed_42.txt'
test_path = 'data/original/testing_5_shot_samples_seed_42.txt'
train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)
random.shuffle(train_dataset.examples)


training_args = TrainingArguments(
    output_dir="./models",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=12, 
    eval_steps = 20,
    logging_steps = 20,
    save_steps=200,
#     warmup_steps=10,
    learning_rate=2e-5,  # Decreased the learning rate 
    prediction_loss_only=False,  # Display both prediction loss and other metrics
    evaluation_strategy="steps",  # Perform evaluation every eval_steps
    save_strategy='steps',
    logging_strategy="steps",
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
trainer.save_model()