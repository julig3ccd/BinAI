import torch
from transformers import BertForMaskedLM, BertConfig, AutoTokenizer
from train_base import args



tokenizer = AutoTokenizer.from_pretrained("hustcw/clap-asm", trust_remote_code=True)
trainset_path = "data/preprocessed_test_data"

asm_list = []
asm_input = tokenizer(asm_list, padding=True,return_tensors="pt") 

config = BertConfig(
    vocab_size= tokenizer.vocab_size,
    num_hidden_layers=args.num_hidden_layers,
    num_attention_heads=args.num_attention_heads,
    max_position_embeddings=1024 #match the max instr of tokenizer 1024
)


model = BertForMaskedLM(config)


