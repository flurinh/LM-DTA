import torch as nn
import sys
import os

import init_json
import pickle
from collections import OrderedDict

from rdkit import Chem
import numpy as np


print(sys.version)
print(os.getcwd())


all_prots = []
datasets = ['kiba', 'davis']
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/'
    train_fold = init_json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e ]
    valid_fold = init_json.load(open(fpath + "folds/test_fold_setting1.txt"))
    ligands = init_json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = init_json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')
    drugs = []
    prots = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y/1e9) for y in affinity]
    affinity = np.asarray(affinity)
    opts = ['train', 'test']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity)==False)
        if opt=='train':
            rows,cols = rows[train_fold], cols[train_fold]
        elif opt=='test':
            rows,cols = rows[valid_fold], cols[valid_fold]
        with open('data/preprocessed/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                f.write(','.join(map(str, ls)) + '\n')
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))
    all_prots += list(set(prots))

# fig.show() # UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
#   self.figure.tight_layout()