from configparser import ConfigParser
import argparse
import os
import time
import json


class Config:
    def __init__(self, setting, max_nr_jobs, dataset):
        path = "config/config"
        config_file = path+".ini"
        self.parser = ConfigParser()
        self.parser.read(config_file)
        if dataset == 'davis':
            self.parser['DATA']['seq_max_len'] = str(1200)
            self.parser['DATA']['smi_max_len'] = str(85)
        elif dataset == 'kiba':
            self.parser['DATA']['seq_max_len'] = str(1000)
            self.parser['DATA']['smi_max_len'] = str(100)
        #self.parser['DATA'][''] = 'data/TFL/proteins_'+str(self.parser['DATA']['seq_max_len'])+'_r'
        #self.parser['DATA'][''] = 'data/TFL/smiles_'+str(self.parser['DATA']['smi_max_len'])+'_r'
        self.idx = 0

        if setting < 100:
            # HERE YOU CAN SPECIFY WHATEVER YOU WANT TO TEST
            pass

        elif setting < 200:
            # GENERATING DATASETS
            self.preprocess_seq = True
            self.preprocess_smi = False
            self.parser['SETTING']['preprocess_seq'] = str(self.preprocess_seq)
            self.parser['SETTING']['preprocess_smi'] = str(self.preprocess_smi)
            self.seq_max_len = [1000]
            for l in self.seq_max_len:
                self.idx += 1
                new_config = path+'_'+str(setting+self.idx).zfill(5)+'.ini'
                if os.path.isfile(new_config):
                    os.remove(new_config)
                self.parser['DATA']['seq_max_len'] = str(l)
                with open(new_config, "w") as f:
                    self.parser.write(f)
            self.preprocess_seq = False
            self.preprocess_smi = True
            self.parser['SETTING']['preprocess_seq'] = str(self.preprocess_seq)
            self.parser['SETTING']['preprocess_smi'] = str(self.preprocess_smi)
            self.smi_max_len = [100]
            for l in self.smi_max_len:
                self.idx += 1
                new_config = path+'_'+str(setting+self.idx).zfill(5)+'.ini'
                if os.path.isfile(new_config):
                    os.remove(new_config)
                self.parser['DATA']['smi_max_len'] = str(l)
                with open(new_config, "w") as f:
                    self.parser.write(f)

        elif setting < 300:
            self.parser['SETTING']['analysis'] = str(True)
            self.parser['DATA']['regime'] = 'mask'
            self.parser['tfl_SEQ']['model'] = 'cnn'
            self.parser['tfl_SMI']['model'] = 'cnn'
            # TODO: selecting datasets!
            batch_sizes = [256]
            embedding_sizes = [128]
            kernel_sizes = [5]
            filters = [32]
            for bs in batch_sizes:
                for es in embedding_sizes:
                    for ks in kernel_sizes:
                        for nf in filters:
                            self.idx += 1
                            new_config = path+'_'+str(setting+self.idx).zfill(5)+'.ini'
                            if os.path.isfile(new_config):
                                os.remove(new_config)
                            self.parser['SETTING']['tfl_smi'] = str(False)
                            self.parser['SETTING']['tfl_seq'] = str(True)
                            self.parser['tfl_SEQ']['embedding'] = str(es)
                            self.parser['tfl_SEQ']['batch_size'] = str(bs)
                            self.parser['tfl_SEQ']['kernel'] = str(ks)
                            self.parser['tfl_SEQ']['filters'] = str(nf)
                            with open(new_config, "w") as f:
                                self.parser.write(f)
                            self.idx += 1
                            new_config = path+'_'+str(setting+self.idx).zfill(5)+'.ini'
                            if os.path.isfile(new_config):
                                os.remove(new_config)
                            self.parser['SETTING']['tfl_seq'] = str(False)
                            self.parser['SETTING']['tfl_smi'] = str(True)
                            self.parser['tfl_SMI']['embedding'] = str(es)
                            self.parser['tfl_SMI']['batch_size'] = str(bs)
                            self.parser['tfl_SMI']['kernel'] = str(ks)
                            self.parser['tfl_SMI']['filters'] = str(nf)
                            with open(new_config, "w") as f:
                                self.parser.write(f)

        elif setting < 400:
            self.parser['SETTING']['analysis'] = str(True)
            self.parser['DATA']['one_hot'] = str(True)
            self.parser['DATA']['regime'] = 'sequence'
            self.parser['tfl_SEQ']['model'] = 'lstm'
            self.parser['tfl_SMI']['model'] = 'lstm'
            batch_sizes = [256]
            rnn_layers = [2]
            hidden = [64, 512]
            for bs in batch_sizes:
                for rl in rnn_layers:
                    for h in hidden:
                        self.idx += 1
                        new_config = path+'_'+str(setting+self.idx).zfill(5)+'.ini'
                        if os.path.isfile(new_config):
                            os.remove(new_config)
                        self.parser['SETTING']['tfl_smi'] = str(False)
                        self.parser['SETTING']['tfl_seq'] = str(True)
                        self.parser['tfl_SEQ']['rnn_layers'] = str(rl)
                        self.parser['tfl_SEQ']['hidden'] = str(h)
                        self.parser['tfl_SEQ']['batch_size'] = str(bs)
                        with open(new_config, "w") as f:
                            self.parser.write(f)
                        self.idx += 1
                        new_config = path+'_'+str(setting+self.idx).zfill(5)+'.ini'
                        if os.path.isfile(new_config):
                            os.remove(new_config)
                        self.parser['SETTING']['tfl_seq'] = str(False)
                        self.parser['SETTING']['tfl_smi'] = str(True)
                        self.parser['tfl_SMI']['rnn_layers'] = str(rl)
                        self.parser['tfl_SMI']['hidden'] = str(h)
                        self.parser['tfl_SMI']['batch_size'] = str(bs)
                        with open(new_config, "w") as f:
                            self.parser.write(f)

        elif setting < 500:
            self.parser['SETTING']['analysis'] = str(True)
            self.parser['DATA']['one_hot'] = str(True)
            self.parser['DATA']['regime'] = 'sequence'
            self.parser['tfl_SEQ']['model'] = 'bidir'
            self.parser['tfl_SMI']['model'] = 'bidir'
            batch_sizes = [256]
            rnn_layers = [2]
            hidden = [128]
            for bs in batch_sizes:
                for rl in rnn_layers:
                    for h in hidden:
                        self.idx += 1
                        new_config = path+'_'+str(setting+self.idx).zfill(5)+'.ini'
                        if os.path.isfile(new_config):
                            os.remove(new_config)
                        self.parser['SETTING']['tfl_smi'] = str(False)
                        self.parser['SETTING']['tfl_seq'] = str(True)
                        self.parser['tfl_SEQ']['rnn_layers'] = str(rl)
                        self.parser['tfl_SEQ']['hidden'] = str(h)
                        self.parser['tfl_SEQ']['batch_size'] = str(bs)
                        with open(new_config, "w") as f:
                            self.parser.write(f)
                        self.idx += 1
                        new_config = path+'_'+str(setting+self.idx).zfill(5)+'.ini'
                        if os.path.isfile(new_config):
                            os.remove(new_config)
                        self.parser['SETTING']['tfl_seq'] = str(False)
                        self.parser['SETTING']['tfl_smi'] = str(True)
                        self.parser['tfl_SMI']['rnn_layers'] = str(rl)
                        self.parser['tfl_SMI']['hidden'] = str(h)
                        self.parser['tfl_SMI']['batch_size'] = str(bs)
                        with open(new_config, "w") as f:
                            self.parser.write(f)

        elif setting == 1000:
            print("Generating DTI config files...")
            self.parser['SETTING']['analysis'] = str(True)
            self.parser['SETTING']['dti'] = str(True)
            control = [str(True), str(False)]
            pretrained = [str(True), str(False)]
            model = ['cnn', 'lstm', 'bidir']
            epochs = [30, 100]
            batch_size = [128]
            for c in control:
                for p in pretrained:
                    for m in model:
                        for e in epochs:
                            for b in batch_size:
                                if (m != 'cnn' and e > 30) or (m == 'cnn' and e < 100):
                                    pass
                                else:
                                    if m == 'cnn':
                                        self.parser['dti']['tfl_runs'] = '[202, 201]'
                                    if m == 'lstm':
                                        self.parser['dti']['tfl_runs'] = '[304, 303]'
                                    if m == 'bidir':
                                        self.parser['dti']['tfl_runs'] = '[402, 401]'
                                    self.parser['dti']['control'] = c
                                    self.parser['dti']['tfl_model'] = m
                                    self.parser['dti']['smi_pretrained'] = p
                                    self.parser['dti']['seq_pretrained'] = p
                                    self.parser['dti']['epochs'] = str(e)
                                    self.parser['dti']['batch_size'] = str(b)

                                    self.idx += 1
                                    new_config = path + '_' + str(setting + self.idx).zfill(5) + '.ini'
                                    if os.path.isfile(new_config):
                                        os.remove(new_config)
                                    with open(new_config, "w") as f:
                                        self.parser.write(f)

        elif setting == 2000:
            print("Generating DTI config files...")
            self.parser['SETTING']['analysis'] = str(True)
            self.parser['SETTING']['dti'] = str(True)
            self.parser['DATA']['dti_set'] = 'kiba'
            control = [str(True), str(False)]
            pretrained = [str(True), str(False)]
            model = ['cnn', 'lstm', 'bidir']
            epochs = [30, 100]
            batch_size = [128]
            for c in control:
                for p in pretrained:
                    for m in model:
                        for e in epochs:
                            for b in batch_size:
                                if (m != 'cnn' and e > 30) or (m == 'cnn' and e < 100):
                                    pass
                                else:
                                    if m == 'cnn':
                                        self.parser['dti']['tfl_runs'] = '[202, 201]'
                                    if m == 'lstm':
                                        self.parser['dti']['tfl_runs'] = '[304, 303]'
                                    if m == 'bidir':
                                        self.parser['dti']['tfl_runs'] = '[402, 401]'
                                    self.parser['dti']['control'] = c
                                    self.parser['dti']['tfl_model'] = m
                                    self.parser['dti']['smi_pretrained'] = p
                                    self.parser['dti']['seq_pretrained'] = p
                                    self.parser['dti']['epochs'] = str(e)
                                    self.parser['dti']['batch_size'] = str(b)

                                    self.idx += 1
                                    new_config = path + '_' + str(setting + self.idx).zfill(5) + '.ini'
                                    if os.path.isfile(new_config):
                                        os.remove(new_config)
                                    with open(new_config, "w") as f:
                                        self.parser.write(f)

        elif setting == 3000:
            print("Generating DTI config files...")
            self.parser['SETTING']['analysis'] = str(True)
            self.parser['SETTING']['dti'] = str(True)
            self.parser['DATA']['dti_set'] = 'davis'
            control = [str(False)]
            pretrained = [str(False)]
            model = ['cnn', 'lstm']
            epochs = [30, 200]
            batch_size = [64]
            for c in control:
                for p in pretrained:
                    for m in model:
                        for e in epochs:
                            for b in batch_size:
                                if (m != 'cnn' and e > 30) or (m == 'cnn' and e < 100):
                                    pass
                                else:
                                    if m == 'cnn':
                                        self.parser['dti']['tfl_runs'] = '[202, 201]'
                                    if m == 'lstm':
                                        self.parser['dti']['tfl_runs'] = '[304, 303]'
                                    self.parser['dti']['control'] = c
                                    self.parser['dti']['tfl_model'] = m
                                    self.parser['dti']['smi_pretrained'] = p
                                    self.parser['dti']['seq_pretrained'] = p
                                    self.parser['dti']['epochs'] = str(e)
                                    self.parser['dti']['batch_size'] = str(b)

                                    self.idx += 1
                                    new_config = path + '_' + str(setting + self.idx).zfill(5) + '.ini'
                                    if os.path.isfile(new_config):
                                        os.remove(new_config)
                                    with open(new_config, "w") as f:
                                        self.parser.write(f)

        elif setting == 4000:
            print("Generating DTI config files...")
            self.parser['SETTING']['analysis'] = str(True)
            self.parser['SETTING']['dti'] = str(True)
            self.parser['DATA']['dti_set'] = 'kiba'
            control = [str(False)]
            pretrained = [str(False)]
            model = ['cnn', 'lstm']
            epochs = [30, 200]
            batch_size = [64]
            for c in control:
                for p in pretrained:
                    for m in model:
                        for e in epochs:
                            for b in batch_size:
                                if (m != 'cnn' and e > 30) or (m == 'cnn' and e < 100):
                                    pass
                                else:
                                    if m == 'cnn':
                                        self.parser['dti']['tfl_runs'] = '[202, 201]'
                                    if m == 'lstm':
                                        self.parser['dti']['tfl_runs'] = '[304, 303]'
                                    self.parser['dti']['control'] = c
                                    self.parser['dti']['tfl_model'] = m
                                    self.parser['dti']['smi_pretrained'] = p
                                    self.parser['dti']['seq_pretrained'] = p
                                    self.parser['dti']['epochs'] = str(e)
                                    self.parser['dti']['batch_size'] = str(b)

                                    self.idx += 1
                                    new_config = path + '_' + str(setting + self.idx).zfill(5) + '.ini'
                                    if os.path.isfile(new_config):
                                        os.remove(new_config)
                                    with open(new_config, "w") as f:
                                        self.parser.write(f)

        elif setting < 5000:
            # To get more flexible lstm configurations (these have not been run!)
            self.parser['SETTING']['analysis'] = str(True)
            self.parser['DATA']['one_hot'] = str(True)
            self.parser['DATA']['regime'] = 'sequence'
            self.parser['tfl_SEQ']['model'] = 'lstm'
            self.parser['tfl_SMI']['model'] = 'lstm'
            batch_sizes = [256]
            rnn_layers = [2]
            hidden = [256, 512, 1024]
            for bs in batch_sizes:
                for rl in rnn_layers:
                    for h in hidden:
                        self.idx += 1
                        new_config = path+'_'+str(setting+self.idx).zfill(5)+'.ini'
                        if os.path.isfile(new_config):
                            os.remove(new_config)
                        self.parser['SETTING']['tfl_smi'] = str(False)
                        self.parser['SETTING']['tfl_seq'] = str(True)
                        self.parser['tfl_SEQ']['rnn_layers'] = str(rl)
                        self.parser['tfl_SEQ']['hidden'] = str(h)
                        self.parser['tfl_SEQ']['batch_size'] = str(bs)
                        with open(new_config, "w") as f:
                            self.parser.write(f)
                        self.idx += 1
                        new_config = path+'_'+str(setting+self.idx).zfill(5)+'.ini'
                        if os.path.isfile(new_config):
                            os.remove(new_config)
                        self.parser['SETTING']['tfl_seq'] = str(False)
                        self.parser['SETTING']['tfl_smi'] = str(True)
                        self.parser['tfl_SMI']['rnn_layers'] = str(rl)
                        self.parser['tfl_SMI']['hidden'] = str(h)
                        self.parser['tfl_SMI']['batch_size'] = str(bs)
                        with open(new_config, "w") as f:
                            self.parser.write(f)


        print("Starting to run processes - total:", self.idx)
        for i in range(self.idx):
            ini_id = i + 1
            commit = False
            while commit == False:
                print("Trying to commit next job: "+ str(setting+ini_id).zfill(5))
                with open('config/open_jobs.json', 'r') as f:
                    list_current_jobs = json.load(f)
                    # print("Currently running jobs:", list_current_jobs)
                    if len(list_current_jobs) < max_nr_jobs:
                        commit = True
                        print("Now comitting job " + str(setting+ini_id).zfill(5))
                    else:
                        print("Currently couldn't commit any further jobs!")
                time.sleep(1)
            with open('config/open_jobs.json', 'r') as f:
                list_current_jobs = json.load(f)
                list_current_jobs.append(str(setting + ini_id))
                # print("Added job:", list_current_jobs)
            with open('config/open_jobs.json', 'w') as f:
                json.dump(list_current_jobs, f)
            print("Running configuration N°" + str(setting + ini_id).zfill(5))
            # os.system('bsub -W 24:00 -R "rusage[mem=16382, ngpus_excl_p=1]" python parser.py --run ' + str(int(setting) + ini_id))
            # os.system('bsub -W 4:00 -R "rusage[mem=16382, ngpus_excl_p=1]" python parser.py --run ' + str(int(setting) + ini_id))
            # os.system('python parser.py --run ' + str(int(setting) + ini_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify setting (generates all corresponding .ini files).')
    parser.add_argument('--setting', type=int, help='Please specify setting:\n\n'
                                                    '   0 : Testing purposes\n'
                                                    ' 100 : Generating Datasets: TFL\n'
                                                    ' 200 : CNN: TFL\n'
                                                    ' 300 : OneHotLSTM: TFL\n'
                                                    ' 400 : Bidirectional Lstm: TFL\n'
                                                    ' 500 : Seq-to-Seq: TFL\n'
                                                    ' 600 : Bert: TFL\n'
                                                    '1000 : DTI davis (model setting see 0 - 600)\n'
                                                    '2000 : DTI kiba (model setting see 0 - 600)\n')
    parser.add_argument('--max_nr_jobs', type=int, default=12, help='Maximum number of jobs to be run in parallel.')
    parser.add_argument('--dataset', type=str, default='kiba', help='Specify DTI dataset to perform analysis on.')
    args = parser.parse_args()
    Config(setting=args.setting, max_nr_jobs=args.max_nr_jobs, dataset=args.dataset)
