import json
import os
import random

class Pool_Util():

   def create_pools(self,sim_fcts_dir="../data/similar_fcts"): 
     pools={}
     fct_names={}
     projects = [f for f in os.listdir(sim_fcts_dir) if not f.startswith(".")]
     cross_proj_fcts = []
     for proj in projects:
       with open(f'{sim_fcts_dir}/{proj}', 'r') as f:
         fcts = json.load(f)
         
       for func_name, variants in fcts.items():   
         #TODO decide how many similar fcts are mandatory for it to be an anchor
         #TODO see if it even makes sense to build all pools
         if len(variants) > 3:
            anchor_key= f'{func_name}_{variants[0]["metadata"]["bin_name"]}'
            pool={}
            pool["anchor"]={"anchor_id":anchor_key, "variant":variants[0]} #first variant is anchor
            pool["pos"]=variants[1:] # all other variants are pos samples
            pool["neg"]=[self.sample_random_neg_samples(func_name)]
            pools[anchor_key]=pool
   
     json.dump(pools, open(f'../data/preprocessed_test_data/pools.json', 'w'))
          #TODO build pools with neg samples  



   def sample_random_neg_samples(self, func_name, test_projects=["curl.json","unrar.json"] ,num_samples=100, data_dir="../data/preprocessed_test_data"):
     samples = []
     samples_per_proj=num_samples // len(test_projects) 
     rest_samples=num_samples % len(test_projects)
     print("projectnames", test_projects)
     for idx, p in enumerate(test_projects):
        if idx==0:
          p_num_samples=samples_per_proj+rest_samples
        else: 
          p_num_samples=samples_per_proj
   
        with open(f'{data_dir}/{p}', 'r') as f:
          proj_data = json.load(f) 
        proj_samples=[]  
        print(f'sampling {p_num_samples} from {p}')
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
      

  