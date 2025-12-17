import os
import re
import json
import fire
import torch
import wandb

from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


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
    train_dataset=None,
    valid_dataset=None,
    output_dir=None,
    batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    grad_accum=2,
    cutoff_len=1024,
    seed=42,
    wandb_project="CA_train_sid_mapping",
    wandb_name="sid_mapping"
):

    os.environ["WANDB_PROJECT"] = wandb_project
    
    train_data = load_dataset("json", data_files=train_dataset)["train"]
    val_data = load_dataset("json", data_files=valid_dataset)["train"]

    print("Example data sample:")
    print(train_data[0])

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

    # =========== LoRA ===========
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["embed_tokens"],
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()

    # =========== DataCollator ===========
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
        save_steps=100,
        logging_steps=5,
        warmup_steps=180,
        bf16=True,        
        # deepspeed="",
        run_name=wandb_name,
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

    example = format_fn(train_data[:1])
    print("Example formatted prompt:")
    print(example[0][:500] + "...")

    input_ids = tokenizer(example[0], return_tensors="pt")["input_ids"]
    print("Tokenized length:", input_ids.shape[1])
    print("Last 20 tokens:", tokenizer.convert_ids_to_tokens(input_ids[0][-20:]))


    trainer.train()
    trainer.save_model(output_dir)

    print("Training finished!")

if __name__ == "__main__":
    datafold = ""
    path = f""
    sidfold = ""
    model = ""

    train(
        base_model=f"",
        train_dataset=f"",
        valid_dataset=f"",
        output_dir=f"",
        batch_size=16,          
        num_train_epochs=6,
        learning_rate=2e-5,
        grad_accum=2,
        cutoff_len=1024,
        seed=42,
        wandb_project=f"",
        wandb_name=f"",     
    )