import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForMaskedLM, BertConfig, AutoTokenizer
#from train_base import args
import json
import torch.nn.functional as F
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm

from utils.args_parser import Parser


def collate_anchor_data(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    id = [item['id'] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'id': id
    }

def collate_candidate_data(batch):
   input_ids = [item['input_ids'] for item in batch]
   attention_mask = [item['attention_mask'] for item in batch]
   labels = [item['label'] for item in batch]
   anchor_ids = [item['anchor_id'] for item in batch]


   input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
   attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0) 

   return{
      'input_ids': input_ids_padded,
      'attention_mask': attention_mask_padded,
      'label': labels,
      'anchor_id': anchor_ids
   }

class ASM_Anchor_Dataset(torch.utils.data.Dataset):
   def __init__(self, input, ids):
      self.ids=ids
      self.input=input
   def __len__(self):
      return len(self.input['input_ids'])
   def __getitem__(self, idx):
       return {
          'id':self.ids[idx],
          'input_ids':self.input['input_ids'][idx],
          'attention_mask': self.input['attention_mask'][idx]
       }      

class ASM_Candidate_Dataset(torch.utils.data.Dataset):
   
   def __init__(self, anchor_ids, candidate_ids, candidates, labels, metadata):
      self.anchor_ids = anchor_ids
      self.candidate_ids = candidate_ids
      self.candidates = candidates
      self.labels = labels
      self.metadata = metadata
   def __len__(self):
      return len(self.candidates['input_ids'])
   
   def get_metadata(self,idx):
      return self.metadata[idx]
   
   def get_anchor_id(self,idx):
      return self.anchor_ids[idx]
   
   def get_label(self, idx): 
      return self.labels[idx]
   
   def __getitem__(self,idx):
      return {
         'input_ids': self.candidates['input_ids'][idx],
         'attention_mask': self.candidates['attention_mask'][idx],
         'anchor_id': self.anchor_ids[idx],
         'label': self.labels[idx],
      }   

def flatten_candidate_data(pool): 
   candidate_ids = []
   candidates = []
   labels = []
   metadata= []
   for sample in pool["pos"]: 
       candidate_ids.append(sample["id"])
       candidates.append(sample["asm"])
       labels.append(True)
       metadata.append(sample["metadata"])
   for neg_sample in pool["neg"]:
       candidate_ids.append(neg_sample["id"])
       candidates.append(neg_sample["asm"])
       labels.append(False)
       metadata.append(sample["metadata"])



   return candidate_ids, candidates, labels, metadata

def build_test_dataset(pad_input, tokenizer, path): 

    fct_pools_path = path

    with open(f'{fct_pools_path}', 'r') as f:
       pools = json.load(f)

   
    anchor_ids = []
    anchors_asm = []


    all_anchor_ids =[]
    all_candidate_ids = []
    all_candidates = []
    all_labels = []
    all_metadata = []


    for anchor_id, pool in pools.items():
    #then batch process candidates and look up their embedding with key during measurements
   
    #for each pool move the asm to anchors asm list
    #track their id in parrallel
     anchor_ids.append(anchor_id)
     anchors_asm.append(pool["anchor"]["asm"])

     candidate_ids,candidates, labels, metadata = flatten_candidate_data(pool)
     
     anchor_ids_for_candidates = [anchor_id] * len(candidates)
     all_anchor_ids.extend(anchor_ids_for_candidates)
     all_candidate_ids.extend(candidate_ids)
     all_candidates.extend(candidates)
     all_labels.extend(labels)
     all_metadata.extend(metadata)

    tokenized_candidates = tokenizer(all_candidates,
                                      padding=pad_input,
                                      return_tensors="pt")
    tokenized_anchors= tokenizer(anchors_asm,
                                 padding=pad_input,
                                  return_tensors="pt")
   
    return ASM_Candidate_Dataset(anchor_ids=all_anchor_ids,
                            candidate_ids=all_candidate_ids,
                            candidates=tokenized_candidates,
                            labels=all_labels,
                            metadata=all_metadata), anchor_ids, tokenized_anchors


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("hustcw/clap-asm", trust_remote_code=True)

    data_set_test, anch_ids, anch_in = build_test_dataset(path=args.test_data, pad_input=True, tokenizer=tokenizer)

    anchor_data_set = ASM_Anchor_Dataset(ids=anch_ids,
                                          input=anch_in)


    config = BertConfig(
        vocab_size= tokenizer.vocab_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_heads,
        output_hidden_states=True,
        max_position_embeddings=1024 #match the max instr of tokenizer 1024
    )

    anchor_dataloader = torch.utils.data.DataLoader(
       anchor_data_set,
       batch_size=32,
       collate_fn=collate_anchor_data,
       shuffle=False
    )

    candidate_dataloader = torch.utils.data.DataLoader(
        data_set_test,
        batch_size=32,
        collate_fn=collate_candidate_data,
        shuffle=False,
        num_workers=args.num_workers
    )

    model = BertForMaskedLM(config)

    model.eval()

    anchor_emb_dict = {}

    with torch.no_grad():
      for batch in tqdm(anchor_dataloader, desc="Processing anchors"):
         ids = batch.pop('id')
         #forward anchor batch
         outputs = model(**batch)
         #extract cls token embedding
         cls_emb = outputs.hidden_states[-1][:,0,:]

         for idx, anchor_id in enumerate(ids):
            anchor_emb_dict[anchor_id]= cls_emb[idx]

    results = defaultdict(list)

    with torch.no_grad():
       for batch in tqdm(candidate_dataloader, desc="Processing candidates"):
          labels = batch.pop('label')
          anchor_ids = batch.pop('anchor_id')

          outputs = model(**batch)
          cand_cls_emb = outputs.hidden_states[-1][:,0,:]
          #get precomputed anchor_embeddings for this batch
          anchor_batch = torch.stack([anchor_emb_dict[aid] for aid in anchor_ids])

          #compute cos sim for this batch
          similarities = F.cosine_similarity(cand_cls_emb, anchor_batch, dim=1)

          for idx, (anchor_id, sim, label) in enumerate(zip(anchor_ids, similarities, labels)):
             results[anchor_id].append({'sim': sim.item(), 'label': label}) 
             #TODO see how best to track metadata or name in here (from batch directly) or through dataset look up
    json.dump(results, open("out/results.json", "w"))


if __name__ == "__main__":
    parser = Parser()
    args = parser.get_args()
    main(args)

    










