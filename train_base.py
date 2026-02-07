import torch
from transformers import BertForMaskedLM, BertConfig, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from typing import Any, Optional
import wandb
from dataclasses import dataclass
import json
import os

from utils.masking import get_token_ids_of_opcodes_to_mask
from utils.args_parser import Parser

opcode_tensor = None



def mask_tokens_mod(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None, offset_mapping: Optional[Any] = None
    ) -> tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]


        no_mask_mask = (
            special_tokens_mask.bool()
            if isinstance(special_tokens_mask, torch.Tensor)
            else torch.tensor(special_tokens_mask, dtype=torch.bool)
        )

        opcode_mask = torch.isin(inputs, opcode_tensor)
        no_mask_mask = no_mask_mask & ~opcode_mask


        probability_matrix.masked_fill_(no_mask_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix, generator=self.generator).bool()


        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # mask_replace_prob% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, self.mask_replace_prob), generator=self.generator).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        if self.mask_replace_prob == 1 or self.random_replace_prob == 0:
            return inputs, labels

        remaining_prob = 1 - self.mask_replace_prob
        # scaling the random_replace_prob to the remaining probability for example if
        # mask_replace_prob = 0.8 and random_replace_prob = 0.1,
        # then random_replace_prob_scaled = 0.1 / 0.2 = 0.5
        random_replace_prob_scaled = self.random_replace_prob / remaining_prob

        # random_replace_prob% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, random_replace_prob_scaled), generator=self.generator).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, generator=self.generator)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time ((1-random_replace_prob-mask_replace_prob)% of the time) we keep the masked input tokens unchanged
        return inputs, labels


def read_data(path, tokenizer, dataset):
    projects = sorted(os.listdir(path))
    print(f'***PROJECTS IN DATA DIRECTORY: {projects}')
    asm_list = []
    tensor_list = []
    for proj in projects:
       with open(f'{path}/{proj}') as fp:
         data = json.load(fp)
       proj_insn = [entry["func_instr"] for file in data for entry in file["asm"]]
       asm_list.append(proj_insn)  
       #TODO use masking arg from argsparser
       if args.create_opcode_ids==True:
          tensor_list.append(get_token_ids_of_opcodes_to_mask(proj_insn,tokenizer=tokenizer))

    if args.create_opcode_ids==True:
        global opcode_tensor        
        opcode_tensor = torch.cat(tensor_list, dim = 0)
        opcode_tensor = torch.unique(opcode_tensor)
        torch.save(opcode_tensor, f'{args.out_dir}/{dataset}_opcode_tensor.pt')   

    return asm_list

def filter_tokens_max_seq_length(tokens, max_length):
    tokens['input_ids'] = [seq for seq in tokens['input_ids'] if len(seq) < max_length]
    tokens['attention_mask'] = [mask for mask in tokens['attention_mask'] if len(mask) < max_length]
    return tokens    


def build_dataset(path, tokenizer, dataset_type, max_length=128):
    asm = read_data(path, tokenizer, dataset_type)

    concat = None
    for proj in asm:
        print("before token")
        tokens = tokenizer(proj, padding=False)
        all_seq_count = len(tokens['input_ids'])

        tokens = filter_tokens_max_seq_length(tokens, max_length)
        
        filtered_seq = len(tokens['input_ids'])

        print (f"***** remaining sequences after max length 128 {filtered_seq/all_seq_count}")
        dataset = ASM_Train_Dataset(tokens)
        print("before concat")
        if concat is None:
            concat = dataset
        else:
            prev = concat.datasets if isinstance(concat, torch.utils.data.ConcatDataset) else [concat]
            concat = torch.utils.data.ConcatDataset(prev + [dataset])
        print("after concat")
    return concat


class ASM_Train_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        #print(f"encodings {idx} size", self.encodings['input_ids'][idx].size())
        #print(f"encodings {idx} size", len(self.encodings['input_ids'][idx]))

        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }



def main(args):

    if args.masking == "opcode":
        DataCollatorForLanguageModeling.torch_mask_tokens = mask_tokens_mod

    
    tokenizer = AutoTokenizer.from_pretrained("hustcw/clap-asm", trust_remote_code=True)

    dataset_train = build_dataset(args.train_data, tokenizer, dataset_type="train", max_length=args.max_seq_length)
    dataset_val = build_dataset(args.val_data, tokenizer, dataset_type="val", max_length=args.max_seq_length)

    if args.masking == "opcode":
        global opcode_tensor
        opcode_tensor = torch.load(f'{args.out_dir}/train_opcode_tensor.pt')


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    config = BertConfig(
        vocab_size= tokenizer.vocab_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_heads,
        max_position_embeddings=args.max_seq_length  #match the max seq length of the input
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
        model = BertForMaskedLM.from_pretrained(args.checkpoint)

    model.to(device)    
    model.gradient_checkpointing_enable()


    total_params = sum(p.numel() for p in model.parameters())
    print("MODEL PARAMETERS: ", total_params)
    wandb.init(
        project="BinAI-MLM",
        )

    training_args = TrainingArguments(output_dir="output",
                                      per_device_train_batch_size=args.batch_size,
                                    num_train_epochs=args.epochs,
                                     logging_steps=100,
                                     warmup_ratio=0.1,           
                                     eval_strategy="epoch",
                                    report_to="wandb",         
                                    run_name="bert-mlm-test",
                                    learning_rate=args.lr,
                                    warmup_ratio=0.1, 
                                    lr_scheduler_type="cosine",
                                    save_strategy="epoch",
                                    dataloader_num_workers=args.num_workers,
                                    gradient_accumulation_steps=args.gradient_accumulation_steps,
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





