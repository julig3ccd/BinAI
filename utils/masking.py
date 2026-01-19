import json
import random

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

def test():
    masked = Mask_OpCodes(0.3,'sample_data/curl.json')
    with open('sample_data/masked_curl.json', 'w') as f:
        json.dump(masked, f, indent=2)

if __name__ == '__main__':
    test()