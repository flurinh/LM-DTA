from Dataclass import *
from visualization import *
import os
from tqdm import tqdm
import numpy
from torch.utils.data import DataLoader, SubsetRandomSampler
from architect import *

# for leonhard use "module load gcc"
# https://kivy.org/doc/stable/tutorials/crashcourse.html

visu_path = 'data/visuals'
if not os.path.isdir(visu_path):
    os.mkdir(visu_path)

c = MaskedStrings(path='data/smiles_X',
                  mode='SMI',
                  one_hot=True,
                  tfl_mask_frac=15)

c.__visualize_len_dist__()

# c = Strings(path='data/smiles_128_r', mode='SMI', one_hot=True)
# print(Architect('00001', c, setting='tfl').load_tfl_model('SMI'))
'''
loaded = torch.load('models/00001' + '.pt')
print(loaded['hyperparameters'])
print(loaded['state_dict'])
'''
# smi_loader = torch.utils.data.DataLoader(c, batch_size=4)

"""
# c.__visualize_char_dist__(norm=None)
# c.__show_smile__(10)
# print(c.__getitem__(10))
# print(c.__getstring__(10))


# for idx in range(c.__len__()):
for idx, batch in enumerate(tqdm(smi_loader)):
    sample = batch[0]  # input
    target = batch[0]  # target
    print(sample.shape)
    print(target.shape)


q = MaskedStrings(path='data/proteins_X',
                  mode='SEQ',
                  one_hot=False,
                  max_len=512,
                  tfl_mask_frac=15)

q.__visualize_char_dist__(norm='norm')

seq_loader = torch.utils.data.DataLoader(q, batch_size=4)

for idx, batch in enumerate(tqdm(seq_loader)):
    r = batch[0]
    s = batch[1]
    print(r.shape)
    print(s.shape)
"""
