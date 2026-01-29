import json
import random
import torch
import numpy as np

def Mask_OpCodes(probability, asm_json):
    with open(asm_json, 'r') as f:
        asm = json.load(f)
    for program in asm:
        for function in program['asm']:
            for key, inst in function['func_instr'].items():
                if random.random() < probability:
                    instList = inst.split()
                    instList[0] = '[MASK]'
                    function['func_instr'][key] = ' '.join(instList)

    return asm


def get_token_ids_of_opcodes_to_mask(file_asm,
                                      tokenizer):
 
    opcode_list = []
 
    for program in file_asm:
        for insn in program.values():
                instList = insn.split()
                opcode = instList[0]
                opcode_token_id = tokenizer.convert_tokens_to_ids(opcode)
                #should be only one token id so we can just flatten the array
                #by using the first element
                opcode_list.append(opcode_token_id)
                    
    opcode_tensor= torch.tensor(opcode_list, dtype=torch.long)      
    return opcode_tensor                


def test():
    masked = Mask_OpCodes(0.3,'sample_data/curl.json')
    with open('sample_data/masked_curl.json', 'w') as f:
        json.dump(masked, f, indent=2)

if __name__ == '__main__':
    test()