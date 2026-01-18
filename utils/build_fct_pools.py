import json
import os
import random

class Pool_Util():

   def create_pools(self,preprocessed_data_dir="../data/PreProcessed_Data/test_set/" ,
                     num_samples=100,
                       min_variants=4,
                       test_projects=["zlib","unrar"], 
                       max_pools=None): 
    
     pools={}
     p_names_label = ""
     sim_fcts_dir= f'{preprocessed_data_dir}/similar_fcts'
     #projects = [f for f in os.listdir(sim_fcts_dir) if not f.startswith(".")]

     print(f'starting pool creation with: \n'
           + f'   *** {num_samples} NEGATIVE samples per pool \n'
             + f'   *** {min_variants} VARIANTS including anchor\n'
               + f'   *** PROJECTS: {test_projects}')

     

     for proj in test_projects:
       if len(pools) == max_pools:
              break
       p_names_label += f'{proj}_'
       with open(f'{preprocessed_data_dir}/{proj}_similar_functions.json', 'r') as f:
         fcts = json.load(f)
         
       for idx, (func_name, variants) in enumerate(fcts.items()):   
         # only include functions with at least 4 variants (1 anchor -> 3 pos samples) in pools
         if len(pools) == max_pools:
              break
         print(f'num variants for {func_name}: {len(variants)}')
         if len(variants) >= min_variants:
            anchor_metadata=variants[0]["metadata"]
            anchor_key= f'{func_name}_{anchor_metadata["bin_name"]}_{anchor_metadata["proj"]}'
            pool={}
            pool["anchor"]=variants[0] #first variant is anchor
            pool["pos"] = [{
                    "id": f'{func_name}_{var["metadata"]["bin_name"]}_{var["metadata"]["proj"]}',
                         "metadata": {**var["metadata"], "func_name": func_name},
                         "asm": var["asm"]} for var in variants[1:] ] # all other variants are pos samples
            pool["neg"]=self.sample_random_neg_samples(func_name=func_name,
                                                         num_samples=num_samples,
                                                         test_projects=test_projects,
                                                         data_dir=preprocessed_data_dir
                                                         )
            pools[anchor_key]=pool
            print(f'added a fct pool')  
         print(f'processed {idx} fcts')    
     print(f'built {len(pools)} fct pools')
     json.dump(pools, open(f'{preprocessed_data_dir}/{p_names_label}_pools.json', 'w'))
 



   def sample_random_neg_samples(self, func_name, test_projects ,num_samples=100, data_dir="../data/preprocessed_test_data"):
     samples = []
     samples_per_proj=num_samples // len(test_projects) 
     rest_samples=num_samples % len(test_projects)
     #print("projectnames", test_projects)
     for idx, p in enumerate(test_projects):
        if idx==0:
          p_num_samples=samples_per_proj+rest_samples
        else: 
          p_num_samples=samples_per_proj
   
        with open(f'{data_dir}/{p}.json', 'r') as f:
          proj_data = json.load(f) 
        proj_samples=[]  
        #print(f'sampling {p_num_samples} from {p}')
        while len(proj_samples) < p_num_samples:
          sample_file= random.choice(proj_data)
          sample_func=random.choice(sample_file["asm"])
          if sample_func["func_name"] != func_name:
            proj_samples.append(
              { 
                "id": f'{sample_func["func_name"]}_{sample_file["bin_name"]}_{sample_file["proj"]}',
                "metadata": {
                 "comp": sample_file["comp"],
                 "opt": sample_file["opt"],
                 "proj": sample_file["proj"],
                 "bin_name": sample_file["bin_name"],
                 "func_name":sample_func["func_name"]
               },
               "asm": 
                 sample_func["func_instr"]
               })
        samples.extend(proj_samples)
     return samples   
      

  