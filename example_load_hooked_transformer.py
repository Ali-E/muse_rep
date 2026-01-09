"""
Example: Loading a HookedTransformer model for unlearning

This demonstrates how to use the --use_hooked_transformer flag to load
models saved in standard transformers format with HookedTransformer.

Usage examples:

1. Load a fine-tuned HookedTransformer model for gradient ascent unlearning:
   python baselines/unlearn.py \
       --algo ga \
       --model_dir ./finetuned_books_model \
       --tokenizer_dir meta-llama/Llama-2-7b-hf \
       --data_file data/books/raw/forget.txt \
       --out_dir ./unlearned_model \
       --use_hooked_transformer

2. Load from a checkpoint with SimNPO:
   python baselines/unlearn.py \
       --algo simnpo \
       --model_dir ./baselines/ckpt/books/gdr_simnpo_b1.0_g0.5_0.05_s1/checkpoint-10 \
       --tokenizer_dir meta-llama/Llama-2-7b-hf \
       --data_file data/books/raw/forget.txt \
       --retain_data_file data/books/raw/retain1.txt \
       --out_dir ./unlearned_simnpo \
       --use_hooked_transformer

3. Load for RMU unlearning:
   python baselines/unlearn.py \
       --algo rmu \
       --model_dir ./finetuned_books_model \
       --tokenizer_dir meta-llama/Llama-2-7b-hf \
       --data_file data/books/raw/forget.txt \
       --retain_data_file data/books/raw/retain1.txt \
       --out_dir ./unlearned_rmu \
       --use_hooked_transformer

Notes:
- The --use_hooked_transformer flag loads models using HookedTransformer.from_pretrained()
- Works with any model saved in standard transformers format (from finetune_model.py, 
  unlearning checkpoints, etc.)
- Tokenizer must be provided separately via --tokenizer_dir
- The tokenizer is loaded as an AutoTokenizer object and passed to HookedTransformer
- All standard unlearning parameters (epochs, lr, batch_size, etc.) work as normal
"""

# You can also load models programmatically:
if __name__ == "__main__":
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    
    # Example: Load a saved model
    model_path = "./finetuned_books_model"
    tokenizer_path = "meta-llama/Llama-2-7b-hf"
    
    print(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    
    # Load model with HookedTransformer
    model = HookedTransformer.from_pretrained(model_path, tokenizer=tokenizer)
    
    print(f"Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model config: {model.cfg}")
    
    # Example usage
    test_text = "The quick brown fox"
    tokens = model.to_tokens(test_text, prepend_bos=True)
    print(f"\nTest tokenization: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Token strings: {model.to_str_tokens(test_text, prepend_bos=True)}")
