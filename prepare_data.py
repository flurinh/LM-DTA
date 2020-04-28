# Import dependencies

from Dataclass import *
from tqdm import trange


# Initialize data for tfl setting : smiles_X, smiles_Y, seq_X, seq_Y
# Initialize data for affinity model: smiles, seqs, Y


def init_data(data=['davis'],
              kmer=False,
              k=1,
              seq_max_len=512,
              smiles_max_len=128):

    print("seq max len:", seq_max_len)
    print("smiles max len:", smiles_max_len)

    train = seqNsmile(datasets=data,
                      mode='train',
                      k=k,
                      seq_max_len=seq_max_len,
                      smiles_max_len=smiles_max_len)

    test = seqNsmile(datasets=data,
                     mode='test',
                     k=k,
                     seq_max_len=seq_max_len,
                     smiles_max_len=smiles_max_len)


if __name__ == "__main__":
    seq_max_len = 512
    smiles_max_len = 128
    #init_data(seq_max_len=seq_max_len, smiles_max_len=smiles_max_len)
    data = DTI(path='data/train', seq_max_len=seq_max_len, smiles_max_len=smiles_max_len)
    start = time.time()
    for k in trange(data.__len__()):
        call = data.__getitem__(k)
        '''print("")
        print(k)
        print(data.__getitem__(k)[0].shape)
        print(data.__getitem__(k)[1].shape)'''
        # Todo: find columns in sparse smiles representation to be deleted, write function to delete them...
    print("")
    print("Calling the entire dataset up once took "+str(time.time()-start)+"seconds!")

# Use "ls | wc -l" (in folder that you want the files to count int)
