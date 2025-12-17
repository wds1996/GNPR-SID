import os
import torch
import wandb
import fire
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import PeftModel, prepare_model_for_kbit_training, LoraConfig, get_peft_model
import re 
import random
 
def format_fn(batch):
    out = []
    for ins, inp, resp in zip(
        batch["instruction"], batch["input"], batch["output"]
    ):
        text = (
            f"### Instruction:\n{ins.strip()}\n\n"
            f"### Input:\n{inp.strip()}\n\n"
            f"### Response:\n{resp.strip()}<|eot_id|>"
        )
        out.append(text)
    return out

def format_llama3(batch):
    instructions = batch.get("instruction", [])
    inputs = batch.get("input", [""] * len(instructions))
    outputs = batch.get("output", [""] * len(instructions))
    
    texts = []
    for ins, inp, out in zip(instructions, inputs, outputs):
        text = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{ins}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{inp}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{out}<|eot_id|>"
        )
        texts.append(text)
    return texts

def train(
    base_model=None,
    train_data=None,
    val_data=None,
    output_dir=None,
    batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    grad_accum=2,
    cutoff_len=2048,
    seed=42,
    wandb_project="train_sid",
    wandb_run_name="sid_sft",
):
    random.seed(seed)   
    os.environ["WANDB_PROJECT"] = wandb_project

    model = AutoModelForCausalLM.from_pretrained(
        base_model, 
        torch_dtype=torch.bfloat16,
        # device_map="auto"
    )
    model.gradient_checkpointing_enable()

    # =========== tokenizer ===========
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "lm_head"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Data collator
    response_template = "### Response:\n"
    # response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    # =========== SFT config ===========
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=2,
        warmup_steps=100,
        bf16=True,        
        # deepspeed="",
        run_name=wandb_run_name,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        tokenizer=tokenizer,
        formatting_func=format_fn,
        max_seq_length=cutoff_len,
        data_collator=collator,
    )

    
    trainer.train()
    trainer.save_model(output_dir)

    final_dir = f"{output_dir}/final_sft"
    os.makedirs(final_dir, exist_ok=True)
    
    trainer.model.save_pretrained(final_dir, safe_serialization=False)
    tokenizer.save_pretrained(final_dir)

    print(f"SFT finished ")

if __name__ == "__main__":
    datafold = ""
    model = "" 
    model_path=f""
    stage = ""
    path = f""
    sidfold = ""
    sft_train_dataset=f"",
    sft_valid_dataset=f"",

    sft_train_data = load_dataset("json", data_files=sft_train_dataset)["train"]
    val_data = load_dataset("json", data_files=sft_valid_dataset)["train"]

    train(
        base_model=model_path,
        train_data=sft_train_data,
        val_data=val_data,
        output_dir=f"",
        batch_size=8,          
        num_train_epochs=5,
        learning_rate=2e-5,
        grad_accum=2,
        cutoff_len=3072,
        seed=42,
        wandb_project=f"",    
        wandb_run_name="",   
    )
