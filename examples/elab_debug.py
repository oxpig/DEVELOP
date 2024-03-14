import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)

import sys
sys.path.append("../")
sys.path.append("../analysis")

from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np

from itertools import product
from joblib import Parallel, delayed
import re
from collections import defaultdict

import DEVELOP
from DEVELOP import DenseGGNNChemModel
import frag_utils
import rdkit_conf_parallel
import example_utils
import shutil
import os 

import data.prepare_data_scaffold_elaboration
from data.prepare_data_scaffold_elaboration import read_file, preprocess

import torch

# load ligand sdf 
scaff_path = './5db3_ligand.sdf'
scaff_sdf = Chem.SDMolSupplier(scaff_path)
scaff_smi = Chem.MolToSmiles(scaff_sdf[0])

# compute coordinates and bonds to break
starting_point_2d = Chem.Mol(scaff_sdf[0])
_ = AllChem.Compute2DCoords(starting_point_2d)
atom_pair_idx_1 = [35, 36]
bonds_to_break = [starting_point_2d.GetBondBetweenAtoms(x,y).GetIdx() for x,y in [atom_pair_idx_1]]

#fragment mol
fragmented_mol = Chem.FragmentOnBonds(starting_point_2d, bonds_to_break)
_ = AllChem.Compute2DCoords(fragmented_mol)

print(Chem.MolToSmiles(fragmented_mol))

# Split fragmentation into core scaffold and elaboration
fragmentation = Chem.MolToSmiles(fragmented_mol).split('.')
elaboration = fragmentation[0]
scaffold = fragmentation[1]

elaboration = re.sub('[0-9]+\*', '*', elaboration)
scaffold = re.sub('[0-9]+\*', '*', scaffold)

# Prepare example as input data
# We now need to prepare the pharmacophoric information, and preprocess this example into the form required by DEVELOP.

# Write data to file
data_path = "./scaffold_elaboration_test_data.txt"
with open(data_path, 'w') as f:
    f.write("%s %s %s" % (scaff_smi, elaboration, scaffold))

raw_data = read_file(data_path, add_idx=True, calc_pharm_counts=True)
preprocess(raw_data, "zinc", "scaffold_elaboration_test", "./", False)

# Calculate Pharmacophoric information
core_path = 'scaffold_elaboration_core.sdf'
pharmacophores_path = 'scaffold_elaboration_pharmacophores.sdf'
fragmentations_pharm, fails = frag_utils.create_frags_pharma_sdf_dataset([[scaff_smi, elaboration, scaffold, 0, 0]], 
                                                                         scaff_path, dataset="CASF",
                                                                         sdffile=core_path,
                                                                         sdffile_pharm=pharmacophores_path, prot="", verbose=True)

# Write .types file
with open("scaffold_elaboration_example.types", 'w') as f:
  f.write('1 ' + core_path + ' ' + pharmacophores_path)                                         

# Load DEVELOP model and generate new molecules

# Arguments for DEVELOP
args = defaultdict(None)
args['--dataset'] = 'zinc'
args['--config'] = '{"generation": true, \
                     "batch_size": 1, \
                     "number_of_generation_per_valid": 250, \
                     "train_file": "./molecules_scaffold_elaboration_test.json", \
                     "valid_file": "./molecules_scaffold_elaboration_test.json", \
                     "train_struct_file": "./scaffold_elaboration_example.types", \
                     "valid_struct_file": "./scaffold_elaboration_example.types", \
                     "struct_data_root": "./", \
                     "output_name": "DEVELOP_scaffold_elaboration_example_gen.smi"}'
args['--freeze-graph-model'] = False
args['--restore'] = '../models/scaffold_elaboration/pretrained_DEVELOP_model.pickle'

# Setup model and generate molecules
model = DenseGGNNChemModel(args)
model.train()
# Free up some memory
model = ''