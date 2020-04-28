from utils import *
from visualization import *
import json
from collections import OrderedDict

import time
import glob
from tqdm import trange

import torch
from torch.utils.data import Dataset


import tensorflow as tf

# ======================================================================================================================


class seqNsmile:
    """
    This class now allows us to preprocess the entire datasets. They are then stored strings in csv files.
    The numpy arrays can be imported via the NumpyDataset class and will be loaded with a DataLoader of pytorch.
    """
    def __init__(self,
                 dataset='davis',
                 mode='train',
                 one_hot=False,
                 padding_mode='right',
                 k=1,
                 seq_max_len=4128,
                 smiles_max_len=532):

        # Fill variable space
        self.one_hot = one_hot  # should the data be stored as numpy files
        self.mem_release = True
        self.smiles_max_len = smiles_max_len
        self.seq_max_len = seq_max_len
        self.padding_mode = padding_mode
        self.limit_samples = 10  # only for testing purposes
        self.k = k
        if self.k > 1:
            self.kmer = True
        else:
            self.kmer = False
        self.path_name = ''
        datasets = [dataset]

        # Loading the dataset
        assert len(datasets) > 0, print('No dataset specified.')
        for _, d in enumerate(datasets):
            stack = load_data(d, mode)
            if _ == 0:
                self.data = stack
            else:
                self.data = pd.concat([self.data, stack])
        print("Total number of samples in provided datasets:", len(self.data))

        # Preprocessing
        self.preprocessed = pd.DataFrame(columns=['seq', 'smiles', 'aff'])
        self.preprocessed['smiles'], invalid_smiles = self.__sparse_smiles__()
        self.preprocessed['seq'], invalid_seq = self.__sparse_seq__()
        self.preprocessed['aff'] = self.data['affinity'].values
        print('Number of invalid smiles (length restricted):', len(invalid_smiles))
        print('Number of invalid seqs (length restricted):', len(invalid_seq))
        invalid_seq.extend(x for x in invalid_smiles if x not in invalid_seq)
        self.preprocessed = self.preprocessed.drop(labels=invalid_seq, axis=0)

        print("Number of valid samples (selected):", len(self.preprocessed))
        # check if element list still matches
        assert not (not (len(self.preprocessed['seq']) == len(self.preprocessed['smiles'])) or not (
                len(self.preprocessed['smiles']) == len(self.preprocessed['aff']))), print('Sample list '
                                                                                           'lengths '
                                                                                           'incompatible!')
        if self.kmer and self.one_hot:
            print("Using K-merization module...")
            # Change table of n * 4128 * 20 to n * p * k * 4128/3 * 20 ---> k = sequence length/timesteps s.t.s.
            self.preprocessed['seq'] = self.__kmerization__()

        # Store data
        if self.padding_mode is 'right':
            _padding = 'r'
        else:
            _padding = 'lr'
        self.path_name = 'data/DTI/' + dataset + "_" + str(self.seq_max_len) + '_' + str(self.smiles_max_len) + '_' + \
                         _padding + "_" + mode
        # Sparse numpy datatables
        if self.one_hot:
            print("new data_path:", self.path_name)
            if not os.path.isdir(self.path_name):
                os.mkdir(self.path_name)

            for i in range(len(self.preprocessed)):
                np.save(self.path_name + "/" + str(i),
                        [self.preprocessed['smiles'].iloc[i], self.preprocessed['seq'].iloc[i],
                         self.preprocessed['aff'].iloc[i]])
        # Padded seqs, smiles and affs in pd (as csv)
        else:
            print("padding...")
            self.__padding__()
            print("saving...")
            self.__save__(path=self.path_name)

        # Finished Preprocessing
        print('Initialized all datasets!')
        print('Size dataset:', len(self.preprocessed))

        # Delete dataframes to save memory
        if self.mem_release:
            del self.preprocessed
            del self.data

        # ==============================================================================================================

    def __kmerization__(self):
        K = []
        for i in range(len(self.preprocessed)):
            K.append(to_kmer(self.preprocessed['seq'][i], self.k))
        return K

    def __sparse_smiles__(self):
        sparse = []
        nonsparse = []
        invalid = []
        start = time.time()
        for idx in range(len(self.data)):
            sample = self.data['compound_iso_smiles'].iloc[idx]
            if not check_valid(sample, self.smiles_max_len):
                invalid.append(idx)
            if not self.one_hot:
                nonsparse.append(sample)
            else:
                sparse.append(smiles_encoder(sample, self.one_hot))
                print(
                    "Creating a sparse smiles data representation took " + str(int(time.time() - start)) + " seconds.")
        if not self.one_hot:
            return nonsparse, invalid
        else:
            return sparse, invalid

    def __sparse_seq__(self):
        sparse = []
        nonsparse = []
        invalid = []
        start = time.time()
        for idx in range(len(self.data)):
            sample = self.data['target_sequence'].iloc[idx]
            if not check_valid(sample, self.seq_max_len):
                invalid.append(idx)
            if not self.one_hot:
                nonsparse.append(sample)
            else:
                sparse.append(seq_encoder(sample, self.one_hot))
                print("Creating a sparse sequence data representation took " + str(
                    int(time.time() - start)) + " seconds.")
        if not self.one_hot:
            return nonsparse, invalid
        else:
            return sparse, invalid

    def __padding__(self):
        # for i in range(self.limit_samples):
        start = time.time()
        for i in trange(len(self.preprocessed['seq'])):
            self.preprocessed['seq'].iloc[i] = padding(self.preprocessed['seq'].iloc[i], self.seq_max_len,
                                                       mode=self.padding_mode)
            self.preprocessed['smiles'].iloc[i] = padding(self.preprocessed['smiles'].iloc[i], self.smiles_max_len,
                                                          mode=self.padding_mode)
            # print(self.preprocessed['smiles'].iloc[i])
            # print(len(self.preprocessed['smiles'].iloc[i]))
        print('Padding took ' + str(time.time() - start) + ' seconds.')

    def __save__(self, path):
        pd.DataFrame(self.preprocessed).to_csv(path + '.csv', header=None, index=None)

    def __path__(self):
        if self.path_name is not '':
            print('Saved preprocessed data at', self.path_name)
        else:
            print('Not saving any data! Load pre-specified dataset!')
        return self.path_name


# ======================================================================================================================
# Not in use, heavy memory load

class NumpyDataset(Dataset):
    def __init__(self, path):
        self.data_numpy_list = [x for x in glob.glob(os.path.join(path, '*.npy'))]

    def __getitem__(self, idx):
        sample_name = self.data_numpy_list[idx]
        sample = np.load(sample_name, allow_pickle=True)
        x = torch.tensor(sample[0]).float()
        y = torch.tensor(sample[1]).float()
        z = torch.tensor(sample[2])
        return x, y, z

    def __len__(self):
        return len(self.data_numpy_list)


# ======================================================================================================================


class DTI(Dataset):
    def __init__(self,
                 path,
                 one_hot,
                 seq_max_len,
                 smiles_max_len,
                 tfl_model
                 ):
        self.path = path # davis / kiba
        if 'davis' in self.path:
            self.dataset = 'davis'
        else:
            self.dataset = 'kiba'
        self.tfl_model = tfl_model
        self.data = pd.read_csv(path + '.csv', header=None)
        self.one_hot = one_hot
        self.smiles_max_len = smiles_max_len
        self.seq_max_len = seq_max_len
        self.num_proteins = 0
        self.prots = {}
        self.read_prots()
        self.zero_one_error = True

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        unpad_seq = sample[0].replace('*', '')
        unpad_smi = sample[1].replace('*', '')


        x1_id = int(self.prots[unpad_seq])
        if self.one_hot:
            x1 = seq_encoder(sample[0], one_hot=self.one_hot)
            x2 = smiles_encoder(sample[1], one_hot=self.one_hot)
        elif self.tfl_model != 'cnn':
            x1 = seq_encoder(unpad_seq, one_hot=False)
            x1.flatten()
            x2 = smiles_encoder(unpad_smi, one_hot=False)
            x2.flatten()
        else:
            x1 = seq_encoder(sample[0], one_hot=False)
            x2 = smiles_encoder(sample[1], one_hot=False)

        if self.zero_one_error:
            if self.dataset == 'davis':
                aff = (sample[2]-5)/6
            else:
                aff = (sample[2]-8)/10
        else:
            aff = sample[2]

        # return: SEQ, SMI, AFF, SEQ-ID
        if self.tfl_model == 'lstm':
            return torch.tensor(x1).float(), torch.tensor(x2).float(), None, None, aff, torch.tensor(x1_id, dtype=torch.long)

        if self.tfl_model == 'bidir':
            return torch.tensor(x1).float(), torch.tensor(x2).float(), None, None, aff, torch.tensor(x1_id, dtype=torch.long)

        if self.tfl_model == 'cnn':
            return torch.tensor(np.squeeze(np.swapaxes(x1, 0, 1)), dtype=torch.long), torch.tensor(np.squeeze(
                np.swapaxes(x2, 0, 1)), dtype=torch.long), [], [], aff, torch.tensor(x1_id, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def read_prots(self):
        with open('data/'+self.dataset+'/proteins.txt', 'r') as f:
            proteins = json.load(open('data/'+self.dataset+'/proteins.txt'), object_pairs_hook=OrderedDict)
            f.close()
        for _, t in enumerate(proteins.keys()):
            self.prots.update({proteins[t]: str(_)})
            self.num_proteins = _ + 1

    def __visualize_len_dist__(self):
        plot_length_dist(seqs=np.squeeze(np.array(self.data.values)).tolist(), seq_type='SMI')
        plot_length_dist(seqs=np.squeeze(np.array(self.data.values)).tolist(), seq_type='SEQ')
        return


# ======================================================================================================================


class MaskedStrings(Dataset):
    def __init__(self,
                 path,
                 mode='SMI',
                 model='cnn',
                 one_hot=True,
                 tfl_mask_frac=10):
        self.one_hot = one_hot
        self.mode = mode
        self.model = model
        self.path = path
        self.data = pd.read_csv(self.path + '.csv', header=None)
        self.mask_frac = tfl_mask_frac  # fraction of masked elements in %

    def __getitem__(self, idx):
        sample = self.data.iloc[idx][0]
        masked, _, _ = mask(seq=sample, frac=self.mask_frac, mode=self.mode)  # generate masked tfl-mode input
        if self.mode is 'SMI':
            x = smiles_encoder(masked, one_hot=self.one_hot)
            y = smiles_encoder(sample, one_hot=self.one_hot)
            if self.model == 'lstm':
                return torch.tensor(x).float(), torch.tensor(y).float()
            if self.model == 'cnn':
                if self.one_hot:
                    return torch.tensor(np.swapaxes(x, 0, 1)).float(), torch.tensor(np.swapaxes(y, 0, 1)).float()
                else:
                    return torch.tensor(np.squeeze(np.swapaxes(x, 0, 1)), dtype=torch.long), torch.tensor(np.squeeze(np.swapaxes(y, 0, 1)), dtype=torch.float)
        else:
            x = seq_encoder(masked, one_hot=self.one_hot)
            y = seq_encoder(sample, one_hot=self.one_hot)
            if self.model == 'lstm':
                return torch.tensor(x).float(), torch.tensor(y).float()
            if self.model == 'cnn':
                if self.one_hot:
                    return torch.tensor(np.swapaxes(x, 0, 1)).float(), torch.tensor(np.swapaxes(y, 0, 1)).float()
                else:
                    return torch.tensor(np.squeeze(np.swapaxes(x, 0, 1)), dtype=torch.long), torch.tensor(np.squeeze(np.swapaxes(y, 0, 1)), dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getstring__(self, idx):
        return self.data.iloc[idx][0]

    def __show_smile__(self, idx):
        if self.mode is not 'SMI':
            print("Cannot show smile if mode of dataset is not 'SMI' (smiles data).")
            return
        smile = self.__getitem__(idx)[0]
        plot_smiles(smiles=smile, one_hot=self.one_hot, ID=idx)
        return

    # Visualize character distribution
    def __visualize_char_dist__(self, norm='norm'):
        # TODO: use counts to weight the loss?
        bins, counts = plot_char_dist(np.squeeze(np.array(self.data.values)).tolist(), datasetname=self.path,
                                      seq_type=self.mode, mode=norm)
        return

    def __visualize_len_dist__(self):
        if self.mode == 'SMI':
            plot_length_dist(seqs=np.squeeze(np.array(self.data.values)).tolist(), bin_size=3, seq_type=self.mode)
        else:
            plot_length_dist(seqs=np.squeeze(np.array(self.data.values)).tolist(), bin_size=20, seq_type=self.mode)
        return

# ======================================================================================================================


class Strings(Dataset):
    def __init__(self,
                 path,
                 mode='SMI',
                 one_hot=True):
        self.one_hot = one_hot
        self.mode = mode
        self.path = path
        self.data = pd.read_csv(self.path + '.csv', header=None)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx][0]
        if self.mode is 'SMI':
            x = np.squeeze(smiles_encoder(sample, one_hot=self.one_hot))
        elif self.mode is 'SEQ':
            x = np.squeeze(seq_encoder(sample, one_hot=self.one_hot))
        return torch.tensor(x).float()


    def __len__(self):
        return len(self.data)

    def __show_smile__(self, idx):
        if self.mode is not 'SMI':
            print("Cannot show smile if mode of dataset is not 'SMI' (smiles data).")
            return
        smile = self.__getitem__(idx)[0]
        plot_smiles(smiles=smile, one_hot=self.one_hot, ID=idx)
        return

    # Visualize character distribution
    def __visualize_char_dist__(self, norm='norm'):
        # TODO: use counts to weight the loss?
        bins, counts = plot_char_dist(np.squeeze(np.array(self.data.values)).tolist(), datasetname=self.path,
                                      seq_type=self.mode, mode=norm)
        return

    def __visualize_len_dist__(self):
        if self.mode == 'SMI':
            plot_length_dist(seqs=np.squeeze(np.array(self.data.values)).tolist(), bin_size=3, seq_type=self.mode)
        else:
            plot_length_dist(seqs=np.squeeze(np.array(self.data.values)).tolist(), bin_size=20, seq_type=self.mode)
        return



