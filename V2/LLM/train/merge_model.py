import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import shutil

def merge(base_model_path, lora_model_path, output_dir):
    
    print("MERGING LORA MODELS INTO BASE MODEL")
    print("="*40)

    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model path does not exist: {base_model_path}")
    if not os.path.exists(lora_model_path):
        raise FileNotFoundError(f"LoRA model path does not exist: {lora_model_path}")
    
    if os.path.exists(output_dir):
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    try:
        print(f"Loading base model from: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            # device_map="auto"
            )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        print(f"Base model loaded successfully")
        print(f"Base model loaded | Vocab: {len(tokenizer)}")
        print("*"*40)
        
        print(f"Loading and merging lora model from: {lora_model_path}")
        lora_model = PeftModel.from_pretrained(base_model, lora_model_path)
        merged_model = lora_model.merge_and_unload()

        print(f"Lora model merged successfully")
        del base_model, lora_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("*"*40)
        
        print(f"Saving final merged model to: {output_dir}")
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved successfully!")
        print("*"*40)
        
    except Exception as e:
        print(f"\nError during model merging: {e}")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        raise

if __name__ == "__main__":
    datafold = ""
    model = "" 
    model_path=f""
    checkpoint = ""
    lora_model_path=f""
    output_dir= f""
    merge(model_path, lora_model_path, output_dir)




