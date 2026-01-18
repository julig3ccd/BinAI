#inspired by the CLAP rebasing in https://github.com/Hustcw/CLAP/tree/main/scripts
import re
import json
import angr
import os
import time
from collections import defaultdict


class PreProcessor(): 

    def __init__(self, detect_similar_fct=False):

      self.loc_pattern = re.compile(r' (loc|locret)_(\w+)')
      self.pattern = re.compile(r'\$\+(\w+)')
      self.detect_similar_fct = detect_similar_fct
      self.similar_fct_dict = defaultdict(list)

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

    def get_structured_asm(self, functions, metadata):
        #start_disassembly = time.time()
        file_asm = []

        #create set of unique fct names per file, to remove duplicates with same metadata
        func_name_set = {func.name for _ , func in functions.items()}


        for f_name in func_name_set:
            func_assembly = {}
            #filter out functions that contain sub_ bc it is not always at very beginning of fct name
            if "sub_" in f_name or f_name in ['UnresolvableCallTarget', 'UnresolvableJumpTarget']:
                continue
            #print(func.name, flush=True)
            func = functions[f_name]
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
            
            #do not further process fcts with less than 4 insns
            if len(func_assembly) < 4:
                continue

            rebased_assembly = self.rebase(func_assembly)
            #print("rebased assembly", rebased_assembly)
            #print("rebased assembly len", len(rebased_assembly))
            
            #add to fct pool dict to determine 
            #construct func item and add it to the files assembly data
            file_asm.append({
                "func_name": func.name,
                "func_instr": rebased_assembly
            })
            #insert fct into fct pools dict
            if self.detect_similar_fct:
                self.similar_fct_dict[func.name].append({
                    "metadata": metadata,
                    "asm": rebased_assembly
                    })

        #end_disassembly = time.time()
        #print(f'took {end_disassembly-start_disassembly} seconds to get asm')
        return file_asm    


    def process_binary_file(self,
                            binary_path="../data/Dataset-1-X64/openssl/x64-clang-3.5-O0_libcrypto.so.3",
                             proj_name=None,
                               dump_json=False):
            start=time.time()
            data = {}
            
            pattern = r'^([^-]+)-([^-]+(?:-[^-]+)*?)-([O](?:\d|s|fast))_(.+)$'
            bin_name = os.path.basename(binary_path)
            match = re.match(pattern, bin_name)
            if match:
                arch, compiler, opt, project = match.groups()
            
            metadata = {"comp": compiler, "opt": opt, "proj": proj_name, "bin_name":bin_name}
        
            #print(f'creating angr project for {binary_path}', flush=True)
            proj = angr.Project(binary_path, 
                                load_options={'auto_load_libs': False})
            #print("analysing cfg...", flush=True)
            start_cfg_analyis= time.time()
            cfg = proj.analyses.CFGFast( normalize=False,
                                         resolve_indirect_jumps=False,
                                         data_references=False,
                                         cross_references=False,
                                         force_complete_scan=False,
                                         force_smart_scan=False)
            end_cfg_analysis=time.time()
            print(f'took {end_cfg_analysis-start_cfg_analyis} seconds to analyse cfg')
            functions = cfg.functions
            #print("getting asm...", flush=True)

            result = self.get_structured_asm(functions, metadata)
            data["compiler"] = compiler
            data["optimization"] = opt
            data["project"] = proj_name
            data["asm"] = result
            end=time.time()
            print(f'took {end-start} seconds to fully preprocess "{bin_name}"')
            if dump_json:
                output_path = "../data/" + bin_name +  '.json'
                json.dump(data, open(output_path, 'w'))
            return data

    def process_binary_dataset(self, dataset_path="../data/Dataset-1-X64-test", output_path = "../data/PreProcessed_Data"):
        
        projects = [p for p in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, p)) and not p.startswith(".")]  #ignore hidden files
        print(f'projects {projects}')
        for p_idx, proj in enumerate(projects):
            if self.detect_similar_fct:
                #clear dict for new project
                self.similar_fct_dict = defaultdict(list)

            proj_data=[]
            project_path=f'{dataset_path}/{proj}'
            print(f'project path {project_path}')
            filenames = [f for f in os.listdir(project_path) if not f.startswith(".")] 
            print(f'filenames {filenames}')
            for f_idx, filename in enumerate(filenames):
                file_asm=self.process_binary_file(f'{project_path}/{filename}', proj_name=proj)
                proj_data.append(file_asm)
                print(f'[{p_idx+1}/{len(projects)}] projects and [{f_idx+1}/{len(filenames)}] files done', flush=True)
            if self.detect_similar_fct:

                start_filter = time.time()
                print(f'sim fct_dict_before_variant_filter {len(self.similar_fct_dict)}')
                #only keep fcts that have at least two variants
                self.similar_fct_dict = {
                                         func_name: variants 
                                         for func_name, variants in self.similar_fct_dict.items() 
                                         if len(variants) >= 2
                                         }
                print(f'sim fct_dict_after_variant_filter {len(self.similar_fct_dict)}')

                end_filter = time.time()
                print(f'took {end_filter-start_filter} seconds filter out the duplicates')
                json.dump((self.similar_fct_dict),
                           open(f'{output_path}/{proj}_similar_functions.json', 'w'))


            json.dump(proj_data, open(f'{output_path}/{proj}.json', 'w'))
            