import torch
from transformers import BertForMaskedLM, BertConfig, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import wandb
from dataclasses import dataclass
import json
import os



from utils.pre_proccess_util import PreProcessor

#TODO replace with parsed args from argsparser
@dataclass
class args:
    num_hidden_layers=2
    num_attention_heads=4
    batch_size=8
    num_epochs=3
    warm_up_epochs=2

class ASM_Train_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }


tokenizer = AutoTokenizer.from_pretrained("hustcw/clap-asm", trust_remote_code=True)
trainset_path = "data/preprocessed_test_data"

projects = os.listdir(trainset_path)
print(f'BUILDING DATASET WITH PROJECTS: {projects}')

asm_list = []
for proj in projects:
   with open(f'{trainset_path}/{proj}') as fp:
       data = json.load(fp)
   proj_insn = [entry["func_instr"] for file in data for entry in file["asm"]]
   asm_list+=proj_insn  

print("INSN BEFORE TOKENIZEATION: \n" , asm_list[:5])



#asm_list = [entry["func_instr"] for entry in data["asm"]]

#move input tensors to CUDA device once training on GPU
asm_input = tokenizer(asm_list, padding=True,return_tensors="pt") 

dataset = ASM_Train_Dataset(asm_input)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

config = BertConfig(
    vocab_size= tokenizer.vocab_size,
    num_hidden_layers=args.num_hidden_layers,
    num_attention_heads=args.num_attention_heads,
    max_position_embeddings=1024 #match the max instr of tokenizer 1024
)

model = BertForMaskedLM(config)

wandb.init(
    project="BinAI-MLM",
    name="bert-mlm-test")

training_args = TrainingArguments(output_dir="output",
                                  per_device_train_batch_size=args.batch_size,
                                num_train_epochs=args.num_epochs,
                                 logging_steps=10,
                                report_to="wandb",         
                                run_name="bert-mlm-test", )

trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=data_collator)
trainer.train()

wandb.finish()





