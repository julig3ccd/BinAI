#inspired by the CLAP rebasing in https://github.com/Hustcw/CLAP/tree/main/scripts
import re
import json
import angr
import os
import time
from collections import defaultdict


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
        #start_disassembly = time.time()
        file_asm = []

        for func_addr, func in functions:
            func_assembly = {}

            if func.name.startswith('sub_') or func.name in ['UnresolvableCallTarget', 'UnresolvableJumpTarget']:
                continue
            #print(func.name, flush=True)
           
            for block in func.blocks:
                block_addr = block.addr
                #print(block_addr)
                #disassemble instructions from block
                disassembly = block.capstone.insns
                #build per function object where address is key, instruction is value
                #so the instructions can be rebased
                for insn in disassembly:
                    func_assembly[hex(insn.address)] = f'{insn.mnemonic} {insn.op_str}'  #construct assembly in format -> {0x401000: "push rbp", 0x401001: "mov rbp, rsp", 0x401004: "sub rsp, 0x20", }
            #print("assembly", func_assembly)
            #print("assembly len", len(func_assembly))

            rebased_assembly = self.rebase(func_assembly)
            #print("rebased assembly", rebased_assembly)
            #print("rebased assembly len", len(rebased_assembly))
            
            #construct func item and add it to the files assembly data
            file_asm.append({
                "func_name": func.name,
                "func_instr": rebased_assembly
            })
        #end_disassembly = time.time()
        #print(f'took {end_disassembly-start_disassembly} seconds to get asm')
        return file_asm    


    def process_binary_file(self,binary_path="../data/Dataset-1-X64/openssl/x64-clang-3.5-O0_libcrypto.so.3", dump_json=False):
            start=time.time()
            data = {}
            
            pattern = r'^([^-]+)-([^-]+(?:-[^-]+)*?)-([O](?:\d|s|fast))_(.+)$'
            bin_name = os.path.basename(binary_path)
            match = re.match(pattern, bin_name)
            if match:
                arch, compiler, opt, project = match.groups()
        
            print(f'creating angr project for {binary_path}', flush=True)
            proj = angr.Project(binary_path, 
                                load_options={'auto_load_libs': False})
            print("analysing cfg...", flush=True)
            start_cfg_analyis= time.time()
            cfg = proj.analyses.CFGFast( normalize=False,
                                         resolve_indirect_jumps=False,
                                         data_references=False,
                                         cross_references=False,
                                         force_complete_scan=False,
                                         force_smart_scan=False)
            end_cfg_analysis=time.time()
            print(f'took {end_cfg_analysis-start_cfg_analyis} seconds to analyse cfg')
            function_list = cfg.functions.items()
            print("getting asm...", flush=True)
            result = self.get_structured_asm(function_list)
            data["compiler"] = compiler
            data["optimization"] = opt
            data["project"] = project
            data["asm"] = result
            end=time.time()
            print(f'took {end-start} seconds to fully preprocess "{bin_name}"')
            if dump_json:
                output_path = "../data/" + bin_name +  '.json'
                json.dump(data, open(output_path, 'w'))
            return data

    def process_binary_dataset(self, dataset_path="../data/test_dataset", output_path = "../data"):
        
        projects = [p for p in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, p)) and not p.startswith(".")]  #ignore hidden files
        print(f'projects {projects}')
        for p_idx, proj in enumerate(projects):
            proj_data=[]
            project_path=f'{dataset_path}/{proj}'
            print(f'project path {project_path}')
            filenames = [f for f in os.listdir(project_path) if not f.startswith(".")] 
            print(f'filenames {filenames}')
            for f_idx, filename in enumerate(filenames):
                file_asm=self.process_binary_file(f'{project_path}/{filename}')
                proj_data.append(file_asm)
                print(f'[{p_idx+1}/{len(projects)}] projects and [{f_idx+1}/{len(filenames)}] files done', flush=True)
            json.dump(proj_data, open(f'{output_path}/{proj}.json', 'w'))
        