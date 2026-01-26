import torch
from transformers import BertForMaskedLM, BertConfig, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import wandb
from dataclasses import dataclass
import json
import os


from utils.args_parser import Parser

def read_data(path):
    projects = sorted(os.listdir(path))
    print(f'***PROJECTS IN DATA DIRECTORY: {projects}')

    asm_list = []
    for proj in projects:
       with open(f'{path}/{proj}') as fp:
         data = json.load(fp)
       proj_insn = [entry["func_instr"] for file in data for entry in file["asm"]]
       asm_list.append(proj_insn)  
    return asm_list   

def build_dataset(path, tokenizer):
    asm = read_data(path)
    concat = None
    for proj in asm:
        print("before token")
        tokens = tokenizer(proj, padding=True,return_tensors="pt")
        dataset = ASM_Train_Dataset(tokens)
        print("before concat")
        if concat is None:
            concat = dataset
        else:
            concat = torch.utils.data.ConcatDataset(concat.datasets if isinstance(concat, torch.utils.data.ConcatDataset) else [concat]+ [dataset])
        print("after concat")
    return concat


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



def main(args):
    
    tokenizer = AutoTokenizer.from_pretrained("hustcw/clap-asm", trust_remote_code=True)

    dataset_train = build_dataset(args.train_data, tokenizer)
    dataset_val = build_dataset(args.val_data, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    config = BertConfig(
        vocab_size= tokenizer.vocab_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_heads,
        max_position_embeddings=1024 #match the max instr of tokenizer 1024
    )

    if torch.cuda.is_available():
        deviceStr= "cuda"
        print("GPU found!")
    else:    
        deviceStr= "cpu"
        print("No GPU found, running on CPU!") 
    device = torch.device(deviceStr)


    if args.checkpoint == None:
        print(f"***INIT SCRATCH MODEL")
        model = BertForMaskedLM(config)
    else:
        print(f"***LOAD MODEL FROM PRETRAINED CHECKPOINT: {args.checkpoint}")
        BertForMaskedLM.from_pretrained(args.checkpoint)

    model.to(device)    

    wandb.init(
        project="BinAI-MLM",
        )

    training_args = TrainingArguments(output_dir="output",
                                      per_device_train_batch_size=args.batch_size,
                                    num_train_epochs=args.epochs,
                                     logging_steps=10,
                                     eval_strategy="epoch",
                                    report_to="wandb",         
                                    run_name="bert-mlm-test",
                                    save_strategy="epoch",
                                    dataloader_num_workers=args.num_workers,
                                     fp16=True if deviceStr != "cpu" else False)

    trainer = Trainer(model=model,
                       args=training_args,
                         train_dataset=dataset_train,
                         eval_dataset=dataset_val,
                           data_collator=data_collator)
    print(f"***START TRAINING WITH ARGS: {args}")
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    parser = Parser()
    args = parser.get_args()
    main(args)





