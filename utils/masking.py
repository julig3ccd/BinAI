import json
import random

def Mask_OpCodes(probability, asm_json):
    with open(asm_json, 'r') as f:
        asm = json.load(f)
    for program in asm:
        for function in program['asm']:
            for inst in function['func_instr'].values():
                if random() < probability:
                    pass #TODO finish

    masked_asm_json = asm
    return masked_asm_json

def test():
    masked = Mask_OpCodes(0.3,'sample_data/curl.json')

if __name__ == '__main__':
    test()