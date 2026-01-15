import json
import os
import random

class Pool_Util():

   def create_pools(self,data_dir="../data/PreProcessed_Data/test_set/" ,
                     num_samples=100,
                       min_variants=4,
                       test_projects=["zlib.json","unrar.json"]): 
    
     pools={}
     fct_names={}
     sim_fcts_dir= f'{data_dir}/similar_fcts'
     projects = [f for f in os.listdir(sim_fcts_dir) if not f.startswith(".")]

     print(f'starting pool creation with: \n'
           + f'   *** {num_samples} NEGATIVE samples per pool \n'
             + f'   *** {min_variants} VARIANTS including anchor\n'
               + f'   *** PROJECTS: {projects}')

     cross_proj_fcts = []
     for proj in projects:
       with open(f'{sim_fcts_dir}/{proj}', 'r') as f:
         fcts = json.load(f)
         
       for idx, (func_name, variants) in enumerate(fcts.items()):   
         # only include functions with at least 4 variants (1 anchor -> 3 pos samples) in pools
         if len(variants) >= min_variants:
            anchor_metadata=variants[0]["metadata"]
            anchor_key= f'{func_name}_{anchor_metadata["bin_name"]}_{anchor_metadata["proj"]}'
            pool={}
            pool["anchor"]=variants[0] #first variant is anchor
            pool["pos"]=variants[1:] # all other variants are pos samples
            pool["neg"]=[self.sample_random_neg_samples(func_name=func_name,
                                                         num_samples=num_samples,
                                                         test_projects=test_projects,
                                                         data_dir=data_dir
                                                         )]
            pools[anchor_key]=pool
            print(f'added a fct pool')   
         print(f'processed {idx} fcts')    
     print(f'built {len(pools)} fct pools')
     json.dump(pools, open(f'{sim_fcts_dir}/pools.json', 'w'))
 



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
   
        with open(f'{data_dir}/{p}', 'r') as f:
          proj_data = json.load(f) 
        proj_samples=[]  
        #print(f'sampling {p_num_samples} from {p}')
        while len(proj_samples) < p_num_samples:
          sample_file= random.choice(proj_data)
          sample_func=random.choice(sample_file["asm"])
          if sample_func["func_name"] != func_name:
            proj_samples.append(
              {
                "metadata": {
                 "comp": sample_file["compiler"],
                 "opt": sample_file["optimization"],
                 "proj": sample_file["project"],
                 "func_name":sample_func["func_name"]
               },
               "asm": 
                 sample_func["func_instr"]
               })
        samples.extend(proj_samples)
     return samples   
      

  