from Dataclass import *
from architect import *
from Processing.preprocess_smiles import *
from Processing.preprocess_proteinseq import *

import argparse


class Parser:
    def __init__(self,
                 run_id=0,
                 verbose=1,
                 visualization=False):
        self.del_job = False
        self.run_id = run_id
        self.verbose = verbose
        self.visualization = visualization
        run_id = str(run_id).zfill(5)
        print("<======== RUN " + run_id + " ========>")
        self.parser = ConfigParser()
        self.parser.read('config/config_' + run_id + '.ini')
        # self.writer = SummaryWriter()
        # ==============================================================================================================
        if self.verbose > 0:
            self.__info__()
        print("")
        # ==============================================================================================================
        # SETTING
        self.download_seq = self.parser.getboolean('SETTING', 'download_seq')
        self.download_smi = self.parser.getboolean('SETTING', 'download_smi')
        self.preprocess_seq = self.parser.getboolean('SETTING', 'preprocess_seq')
        self.preprocess_smi = self.parser.getboolean('SETTING', 'preprocess_smi')
        self.analysis = self.parser.getboolean('SETTING', 'analysis')
        self.dti = self.parser.getboolean('SETTING', 'dti')
        self.tfl_seq = self.parser.getboolean('SETTING', 'tfl_seq')
        self.tfl_smi = self.parser.getboolean('SETTING', 'tfl_smi')
        # ==============================================================================================================
        # DATA
        self.one_hot = self.parser.getboolean('DATA', 'one_hot')
        self.regime = self.parser['DATA']['regime']  # sequence or mask
        self.seq_max_len = int(self.parser['DATA']['seq_max_len'])
        self.smi_max_len = int(self.parser['DATA']['smi_max_len'])
        self.tfl_seq_data = self.parser['DATA']['tfl_seq_data']
        self.tfl_smi_data = self.parser['DATA']['tfl_smi_data']
        self.tfl_seq_frac = int(self.parser['DATA']['tfl_seq_frac'])
        self.tfl_smi_frac = int(self.parser['DATA']['tfl_smi_frac'])
        self.dti_set = self.parser['DATA']['dti_set']
        # ==============================================================================================================
        if self.download_smi is True:
            print("Starting smiles download from ChEMBL... (not implemented)")
        # ==============================================================================================================
        if self.download_seq is True:
            print("Starting sequence download from PDB... (not implemented)")
        # ==============================================================================================================
        if self.preprocess_seq is True:
            print("Checking if chosen PROTEIN data settings were already produced... length:", self.seq_max_len)
            if not os.path.isfile('data/TFL/proteins_' + str(self.seq_max_len) + '_r.csv'):
                print("Starting preprocessing of sequences data")
                preprocess_proteins_(seq_min_len=128, seq_max_len=self.seq_max_len, pad='right',
                                     path='data/PDB/human_proteome.csv')
            else:
                print("Found matching data.")
                print("")
        # ==============================================================================================================
        if self.preprocess_smi is True:
            print("Checking if chosen SMILES data settings were already produced... length:", self.smi_max_len)
            if not os.path.isfile('data/TFL/smiles_' + str(self.smi_max_len) + '_r.csv'):
                print("Starting preprocessing of smiles data")
                preprocess_smiles(smiles_min_len=32, smiles_max_len=self.smi_max_len, padding='r',
                                  path='data/ChEMBL25/chembl25_smiles')
            else:
                print("Found matching data.")
                print("")
        # ==============================================================================================================
        if self.analysis:
            # ==========================================================================================================
            if self.tfl_seq:
                print("TFL on sequence")
                if self.regime == 'mask':
                    tfl_seq_data = MaskedStrings(path=self.tfl_seq_data, mode='SEQ', one_hot=self.one_hot,
                                                 tfl_mask_frac=self.tfl_seq_frac)
                if self.regime == 'sequence':
                    tfl_seq_data = Strings(path=self.tfl_seq_data, mode='SEQ', one_hot=self.one_hot)
                print("Loaded TFL sequence dataset, total:", tfl_seq_data.__len__())
                print("Training TFL sequence model ---> run ID:", self.run_id)
                Architect(run_id, tfl_seq_data, verbose=self.verbose, setting='tfl')
            # ==========================================================================================================
            if self.tfl_smi:
                print("TFL on smiles, smiles max length:", self.smi_max_len)
                if self.regime == 'mask':
                    tfl_smi_data = MaskedStrings(path=self.tfl_smi_data, mode='SMI', one_hot=self.one_hot,
                                                 tfl_mask_frac=self.tfl_smi_frac)
                if self.regime == 'sequence':
                    tfl_smi_data = Strings(path=self.tfl_smi_data, mode='SMI', one_hot=self.one_hot)
                print("Loaded TFL smiles dataset, total:", tfl_smi_data.__len__())
                print("Training TFL smiles model ---> run ID:", self.run_id)
                Architect(run_id, tfl_smi_data, verbose=self.verbose, setting='tfl')
        # ==============================================================================================================
        if self.dti:
            padding_mode = 'right'
            if padding_mode is 'right':
                _padding = 'r'
            else:
                _padding = 'lr'
            print("Checking for DTI data...")
            train_path = 'data/DTI/' + self.dti_set + '_' + str(self.seq_max_len) + '_' + str(
                self.smi_max_len) + '_' + _padding + '_train'
            if not os.path.isfile(train_path + '.csv'):
                print("Did not find preprocessed dti training data, generating from raw data.")
                train = seqNsmile(dataset=self.dti_set,
                                  mode='train',
                                  padding_mode=padding_mode,
                                  k=1,
                                  seq_max_len=self.seq_max_len,
                                  smiles_max_len=self.smi_max_len)
                train_path = train.path_name
            else:
                print("Found training data:", train_path)
            test_path = 'data/DTI/' + self.dti_set + '_' + str(self.seq_max_len) + '_' + str(
                self.smi_max_len) + '_' + _padding + '_test'
            if not os.path.isfile(test_path + '.csv'):
                print("Did not find preprocessed dti testing data, generating from raw data.")
                test = seqNsmile(dataset=self.dti_set,
                                 mode='test',
                                 padding_mode=padding_mode,
                                 k=1,
                                 seq_max_len=self.seq_max_len,
                                 smiles_max_len=self.smi_max_len)
                test_path = test.path_name
            else:
                print("Found testing data:", test_path)
            print("DTI data has been preprocessed. Preparing dataloaders.")
            training_ = DTI(path=train_path,
                            one_hot=self.one_hot,
                            seq_max_len=self.seq_max_len,
                            smiles_max_len=self.smi_max_len,
                            tfl_model=self.parser['dti']['tfl_model'])
            testing_ = DTI(path=test_path,
                           one_hot=self.one_hot,
                           seq_max_len=self.seq_max_len,
                           smiles_max_len=self.smi_max_len,
                           tfl_model=self.parser['dti']['tfl_model'])
            print("Initialized training and testing data:")
            print("Training dataset:", training_.__len__())
            print("Testing dataset:", testing_.__len__())
            if self.analysis:
                Architect(run_id, [training_, testing_], verbose=self.verbose, setting='dti', visual=self.visualization)
        # ==============================================================================================================
        if self.del_job:
            job_file = 'config/open_jobs.json'
            if os.path.isfile(job_file):
                with open(job_file, 'r') as f:
                    list_current_jobs = json.load(f)
                    if str(self.run_id) in list_current_jobs:
                        list_current_jobs.remove(str(self.run_id))
                with open(job_file, 'w') as f:
                    json.dump(list_current_jobs, f)
        # ==============================================================================================================

    def __info__(self):
        for section in self.parser.sections():
            print(section)
            for arg in self.parser.options(section=section):
                print(arg)
            print("===================================================================================================")

# ======================================================================================================================


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify run (respective .ini file holds all tunable hyperparameters.')
    parser.add_argument('--run', type=int,
                        help='Please specify run ID.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Please specify verbose.')
    parser.add_argument('--v', type=bool, default=False,
                        help='If True (default) loads model and runs in evaluation setting.')
    args = parser.parse_args()
    print("Starting run", str(args.run))
    Parser(run_id=args.run, verbose=args.verbose, visualization=args.v)
