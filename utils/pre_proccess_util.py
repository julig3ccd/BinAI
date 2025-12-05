#inspired by the CLAP rebasing in https://github.com/Hustcw/CLAP/tree/main/scripts
import re
import json
import angr

class PreProcessor(): 
    loc_pattern = re.compile(r' (loc|locret)_(\w+)')
    pattern = re.compile(r'\$\+(\w+)')

    def rebase(self,asm_dict):
        index = 1
        rebase_assembly = {}

        addrs = list(sorted(list(asm_dict.keys())))

        for addr in addrs:
            inst = asm_dict[addr]
            if inst.startswith('j'):
                loc = self.loc_pattern.findall(inst)
                for prefix, target_addr in loc:
                    try:
                        target_instr_idx = addrs.index(int(target_addr, 16)) + 1
                    except Exception:
                        continue
                    asm_dict[addr] = asm_dict[addr].replace(
                        f' {prefix}_{target_addr}', f' INSTR{target_instr_idx}')
                self_m = self.pattern.findall(inst)
                for offset in self_m:
                    target_instr_addr = addr + int(offset, 16)
                    try:
                        target_instr_idx = addrs.index(target_instr_addr)
                        asm_dict[addr] = asm_dict[addr].replace(
                            f'$+{offset}', f'INSTR{target_instr_idx}')
                    except:
                        continue
            rebase_assembly[str(index)] = asm_dict[addr]
            index += 1

        return rebase_assembly

    def get_structured_asm(self, functions):
    
        file_asm = []

        for func_addr, func in functions:
            func_item = {}
            func_assembly = {}

            if func.name.startswith('sub_') or func.name in ['UnresolvableCallTarget', 'UnresolvableJumpTarget']:
                continue
            print(func.name)
           
            for block in func.blocks:
                block_addr = block.addr
                print(block_addr)
                #disassemble instructions from block
                disassembly = block.capstone.insns
                #build per function object where address is key, instruction is value
                #so the instructions can be rebased
                for insn in disassembly:
                    func_assembly[hex(insn.address)] = f'{insn.mnemonic}, {insn.op_str}'  #construct assembly in format -> {0x401000: "push rbp", 0x401001: "mov rbp, rsp", 0x401004: "sub rsp, 0x20", }
            #print("assembly", func_assembly)
            #print("assembly len", len(func_assembly))

            rebased_assembly = self.rebase(func_assembly)
            #print("rebased assembly", rebased_assembly)
            #print("rebased assembly len", len(rebased_assembly))
            
            #construct func item and add it to the files assembly data
            func_item["func_name"] = func.name
            func_item["func_instr"] = rebased_assembly
            file_asm.append(func_item)   
        print("file: ", file_asm)
        return file_asm    


    def process_bin(self,path=None):
            data = {}
            pattern = r'^([^-]+)-([^-]+(?:-[^-]+)*?)-([O]\d)_(.+)$'
            BIN_NAME = "x64-clang-3.5-O0_clambc"
            match = re.match(pattern, BIN_NAME)
            if match:
                arch, compiler, opt, project = match.groups()
            BINARY_PATH = "../data/Dataset-1/clamav/x64-clang-3.5-O0_clambc" 
            output_path = "../data/" + BIN_NAME +  '.json'
            proj = angr.Project(BINARY_PATH, auto_load_libs=False)
            cfg = proj.analyses.CFGFast(normalize=True)

            function_list = cfg.functions.items()
            result = self.get_structured_asm(function_list)
            data["compiler"] = compiler
            data["optimization"] = opt
            data["project"] = project
            data["asm"] = result
        
            json.dump(data, open(output_path, 'w'))