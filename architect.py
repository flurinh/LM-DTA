from Processing.preprocessing import *
from architectures.models import *

from tqdm import tqdm
from configparser import ConfigParser
import numpy as np
import ast
import os
import pickle
from tensorboardX import SummaryWriter  # https://sapanachaudhary.github.io/Colab-pages/x

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.nn.utils.rnn import pad_sequence

SEED = 42


def pad_collate(batch):
    (x1, x2, _, _, aff, x1_id) = zip(*batch)
    x1_lens = [len(x) for x in x1]
    x2_lens = [len(x) for x in x2]
    x1_pad = pad_sequence(x1, batch_first=True, padding_value=0)
    x2_pad = pad_sequence(x2, batch_first=True, padding_value=0)
    aff = np.stack(aff)
    x1_id = torch.stack(x1_id)
    x1_pad = one_hot_encoder(x1_pad, 'SEQ')
    x2_pad = one_hot_encoder(x2_pad, 'SMI')
    return x1_pad, x2_pad, x1_lens, x2_lens, aff, x1_id


def one_hot_encoder(t, mode):
    batch_size, max_len, _ = t.shape
    if mode is 'SEQ':
        vocab_size = SEQ_VOCAB_SIZE
    if mode is 'SMI':
        vocab_size = SMILES_VOCAB_SIZE
    # input is batch_size, max_seq_len, 1
    # output is batch_size, max_seq_len, vocab_size
    t_ = t.numpy()
    t_ = (np.arange(vocab_size) == t_[:, None]).astype(np.float32).squeeze(1)
    return torch.from_numpy(t_)


class Architect:
    def __init__(self,
                 run_id,
                 data,
                 verbose,
                 setting='tfl',
                 visual=False):
        # Parser
        self.verbose = verbose
        self.parser = ConfigParser()
        self.parser.read('config/config_' + run_id + '.ini')
        self.run_id = run_id

        # Data Settings
        self.one_hot = self.parser.getboolean('DATA', 'one_hot')
        self.data = data
        self.setting = setting
        self.regime = self.parser['DATA']['regime']

        # Control Settings
        self.control = self.parser.getboolean('dti', 'control')
        self.n_folds = int(self.parser['dti']['n_folds'])
        if self.control:
            print("Running a control model: Protein represented by embedded ID!")

        # Hyperparameters
        if setting == 'tfl':
            self.specs = self.__get_tflspecs__()
            if self.verbose > 0:
                print(self.specs)
        self.hyperparameters = {'run': self.run_id}
        self.tfl_runs = ast.literal_eval(self.parser['dti']['tfl_runs'])

        # Structure/Overview/Report
        self.write_sum_n_ep = 200
        self.early_stop = 25
        self.val_path = 'runs/run_{}/val_{}'.format(int(self.run_id) // 100, int(self.run_id))
        self.train_path = 'runs/run_{}/train_{}'.format(int(self.run_id) // 100, int(self.run_id))
        self.graph_path = 'runs/run_{}/graph_{}'.format(int(self.run_id) // 100, int(self.run_id))
        self.checkpoint_folder = 'models/run_{}/'.format(int(self.run_id) // 100)
        self.checkpoint = 'models/run_{}/model_{}'.format(int(self.run_id) // 100, int(self.run_id))
        if not os.path.isdir(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)
        self.val_writer = SummaryWriter(self.val_path)
        self.train_writer = SummaryWriter(self.train_path)

        # Device
        print("")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device is 'cuda':
            torch.cuda.synchronize()

        # TFL settings
        if setting == 'tfl':
            if self.data.mode is 'SEQ':
                self.vocab_size = SEQ_VOCAB_SIZE
            else:
                self.vocab_size = SMILES_VOCAB_SIZE
            if self.regime == 'sequence':
                eg = self.data.__getitem__(0)
                if self.verbose > 0:
                    print("sample shape:", eg.shape)
                self.seq_len = eg.shape[0]
            elif self.regime == 'mask':
                eg = self.data.__getitem__(0)[0]
                if self.one_hot:
                    self.seq_len = eg.shape[1]
                else:
                    self.seq_len = eg.shape[0]
            print("sequence length:", self.seq_len)
            print("vocabulary size:", self.vocab_size)
            self.hyperparameters['seq_len'] = self.seq_len
            self.hyperparameters['vocab_size'] = self.vocab_size
            self.hyperparameters['specs'] = self.specs

            if self.data.mode == 'SEQ':
                print("Building SEQ model.")
                self.model = self.__tfl__()
                if self.specs['model'] == 'cnn':
                    self.__train_cnn__()
                if self.specs['model'] == 'lstm':
                    self.__train_lstm__()
                if self.specs['model'] == 'bidir':
                    self.__train_lstm__()

            if self.data.mode == 'SMI':
                print("Building SMI model.")
                self.model = self.__tfl__()
                if self.specs['model'] == 'cnn':
                    self.__train_cnn__()
                if self.specs['model'] == 'lstm':
                    self.__train_lstm__()
                if self.specs['model'] == 'bidir':
                    self.__train_lstm__()

        # DTI settings
        elif setting == 'dti':
            print("Building DTI model.")
            self.smi_embedding = None
            self.seq_embedding = None
            # Loading smiles model from given config-id
            smi_ = self.load_tfl_model('SMI')
            smi_specs = smi_['hyperparameters']
            print("SMILES MODEL:", smi_specs)
            self.smi_vocab_size = smi_specs['vocab_size']
            self.vocab_size = self.smi_vocab_size
            self.seq_len = smi_specs['seq_len']
            self.smi_specs = smi_specs['specs']
            self.specs = self.smi_specs  # need specs in __tfl__()
            if self.smi_specs['model'] == 'cnn':
                self.smi_representation = 3 * int(smi_specs['specs']['filters'])
            else:
                self.smi_representation = int(self.smi_specs['hidden'])
            self.smi_model = self.__tfl__()
            print(self.smi_model)
            if self.parser.getboolean('dti', 'smi_pretrained'):
                print("pretrained smi model")
                smi_weights = smi_['state_dict']
                self.smi_model.load_state_dict(smi_weights)

            # Loading protein sequence model from given config-id
            seq_ = self.load_tfl_model('SEQ')
            seq_specs = seq_['hyperparameters']
            print("SEQUENCE MODEL:", seq_specs)
            self.seq_vocab_size = seq_specs['vocab_size']
            self.vocab_size = self.seq_vocab_size
            self.seq_len = seq_specs['seq_len']
            self.seq_specs = seq_specs['specs']
            self.specs = self.seq_specs  # need specs in __tfl__()
            if self.seq_specs['model'] == 'cnn':
                self.seq_representation = 3 * int(seq_specs['specs']['filters'])
            else:
                self.seq_representation = int(self.seq_specs['hidden'])
            self.seq_model = self.__tfl__()
            if self.parser.getboolean('dti', 'seq_pretrained'):
                print("pretrained seq model")
                seq_weights = seq_['state_dict']
                self.seq_model.load_state_dict(seq_weights)

            # SMILES
            if self.smi_specs['model'] == 'lstm':
                self.smi_model = self.smi_model._lstm
            elif self.smi_specs['model'] == 'cnn':
                self.smi_embedding = self.smi_model.bed
                self.smi_model = nn.Sequential(self.smi_model.block1, self.smi_model.block2, self.smi_model.block3,
                                               self.smi_model.pool)
                print("SMI MODEL", self.smi_model)
            elif self.smi_specs['model'] == 'bidir':
                self.smi_model = self.smi_model._blstm

            # SEQUENCE
            if self.control:
                self.embed_seq_id = nn.Embedding(self.data[0].num_proteins, self.seq_representation)
                self.seq_embedding = nn.Sequential(self.embed_seq_id)
            else:
                if self.seq_specs['model'] == 'lstm':
                    self.seq_model = self.seq_model._lstm
                elif self.seq_specs['model'] == 'cnn':
                    self.seq_embedding = self.seq_model.bed
                    self.seq_model = nn.Sequential(self.seq_model.block1, self.seq_model.block2, self.seq_model.block3,
                                                   self.seq_model.pool)
                    print("SEQ MODEL", self.seq_model)
                elif self.seq_specs['model'] == 'bidir':
                    self.seq_model = self.seq_model._blstm

            self.specs = self.__get_dtispecs__()
            if self.verbose > 0:
                print(self.specs)
            self.model = self.__dti__()
            if visual:
                print("Running in visualization mode!")
                # print("run id:", self.run_id)
                weights = self.load_dti_model()['state_dict']
                # print(weights)
                self.model.load_state_dict(weights)
                print("Trainable parameters:", self.count_trainable())
                self.__eval_dti__()
            else:
                print("Running in training mode!")
                if self.n_folds > 1:
                    print("Using cross validation.")
                    for k in range(self.n_folds):
                        self.__train_dti__(k = k)
                else:
                    print("No cross validation.")
                    self.__train_dti__()

    # ==================================================================================================================

    def __dti__(self):
        model = dti_model(smi_model=self.smi_model, seq_model=self.seq_model, smi_embedding=self.smi_embedding,
                          seq_embedding=self.seq_embedding, batchsize=int(self.parser['dti']['batch_size']),
                          smi_len=self.smi_representation, seq_len=self.seq_representation,
                          smi_model_type=self.smi_specs['model'], seq_model_type=self.seq_specs['model'],
                          smi_vocab_size=self.smi_vocab_size, seq_vocab_size=self.seq_vocab_size,
                          layers=list(map(int, ast.literal_eval(self.parser['dti']['layers']))), device=self.device,
                          control=self.control)
        return model

    # ==================================================================================================================

    def __tfl__(self):
        model = None
        model_name = self.specs['model']
        print("Model name:", model_name)
        if model_name == 'cnn':
            model = tfl_cnn_dense(specs=self.specs, vocab_size=self.vocab_size, seq_len=self.seq_len)
            if self.verbose > 0:
                print("Initialized cnn model:")
                # print(model)
        elif model_name == 'lstm':
            model = OneOutLSTM(specs=self.specs, vocab_size=self.vocab_size)
            if self.verbose > 0:
                print("Initialized one-out lstm model.")
                # print(model)
        elif model_name == 'bidir':
            model = BiDirLSTM(specs=self.specs, vocab_size=self.vocab_size)
            if self.verbose > 0:
                print("Initialized one-out bidir model.")
                # print(model)
        return model

    # ==================================================================================================================

    def __train_cnn__(self):
        random_seed = 42
        batch_size = int(self.specs['batch_size'])
        n_epochs = int(self.specs['epochs'])
        print("TFL training initialization...")
        # Put the model in training mode!

        if self.specs['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=float(self.specs['learning_rate']),
                                        weight_decay=float(self.specs['weight_decay']), momentum=0.9)
        if self.specs['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.specs['learning_rate']),
                                         betas=(0.9, 0.999))
        optimizer.zero_grad()
        criterion = nn.MSELoss()

        # Creating data indices for training and validation splits:
        validation_split = 1 / int(self.specs['n_folds'])
        shuffle_dataset = True

        dataset_size = len(self.data)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(SEED)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_loader = torch.utils.data.DataLoader(self.data, batch_size=batch_size, sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(self.data, batch_size=batch_size, sampler=valid_sampler)

        # Starting training
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs! Not compatible with packed_padded_sequence!")
            self.model = nn.DataParallel(self.model).to(self.device)
        else:
            self.model.to(self.device)
            print("Using device:", self.device)
        train_iter = -1
        val_iter = -1
        training_loss = np.inf
        validation_loss = np.inf
        for epoch in range(n_epochs):
            print("============= EPOCH " + str(epoch) + " =============")
            early_iter = 0
            print("Training model.")
            self.model.train()
            for idx, batch in enumerate(train_loader):
                train_iter += 1
                optimizer.zero_grad()
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                pred = self.model(inputs)
                loss = criterion(pred, targets)
                loss.backward()
                optimizer.step()
                loss_ = loss.cpu().item()
                if idx % self.write_sum_n_ep == 0:
                    early_iter += 1
                    prediction, target = self.__writesum__(pred, targets)
                    self.train_writer.add_text('train_prediction', prediction, train_iter)
                    self.train_writer.add_text('train_target', target, train_iter)
                    if early_iter > self.early_stop:
                        pass
                        # self.model = torch.load(self.checkpoint+'.pt')
                    if loss_ < training_loss:
                        early_iter = 0
                        training_loss = loss_
                        self.save_tfl_model()
                self.train_writer.add_scalar('train_loss', loss_, train_iter)
            self.train_writer.export_scalars_to_json(path=self.train_path + '.json')
            print("Validating model.")
            self.model.eval()  # to set dropout and batch normalization layers to evaluation mode
            for idx, batch in enumerate(validation_loader):
                val_iter += 1
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
                pred = self.model(inputs)
                loss = criterion(pred, targets)
                loss_ = loss.cpu().item()
                if idx % self.write_sum_n_ep == 0:
                    prediction, target = self.__writesum__(pred, targets)
                    self.val_writer.add_text('val_prediction', prediction, val_iter)
                    self.val_writer.add_text('val_target', target, val_iter)
                    if loss_ < validation_loss:
                        validation_loss = loss_
                        self.save_tfl_model()
                self.val_writer.add_scalar('val_loss', loss_, val_iter)
            self.val_writer.export_scalars_to_json(path=self.val_path + '.json')

    # ==================================================================================================================

    def __train_lstm__(self):
        # print("TFL training initialization...")

        if self.specs['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=float(self.specs['learning_rate']),
                                        weight_decay=float(self.specs['weight_decay']), momentum=0.9)
        if self.specs['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.specs['learning_rate']),
                                         betas=(0.9, 0.999))
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss(reduction='mean')

        # Creating data indices for training and validation splits:
        validation_split = 1 / int(self.specs['n_folds'])
        shuffle_dataset = True
        batch_size = int(self.specs['batch_size'])
        n_epochs = int(self.specs['epochs'])
        dataset_size = len(self.data)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(SEED)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_loader = torch.utils.data.DataLoader(self.data, batch_size=batch_size, sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(self.data, batch_size=batch_size, sampler=valid_sampler)

        # Starting training
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model).to(self.device)
        else:
            self.model.to(self.device)
            print("Using device:", self.device)
        train_iter = -1
        val_iter = -1

        training_loss = np.inf
        validation_loss = np.inf

        for epoch in range(n_epochs):
            print("============= EPOCH " + str(epoch) + " =============")
            # print("Training model.")
            self.model.train()
            for idx, batch in enumerate(train_loader):
                batch_size = batch.shape[0]
                data = batch.transpose(0, 1)
                train_iter += 1
                molecule_loss = torch.zeros(1).to(self.device)
                hidden_states = self.model.new_sequence(batch_size, self.device)

                # Iteration over molecules
                for j in range(self.seq_len - 1):
                    # Prepare input tensor with dimension (1,batch_size, encoding_dim)
                    inputs = data[j].view(1, batch_size, -1).to(self.device)
                    labels = torch.LongTensor(np.argmax(data[j + 1].view(batch_size, -1).numpy().astype(int), axis=1))
                    next_token, hidden_states = self.model(inputs, hidden_states)
                    next_token = next_token.view(batch_size, -1)
                    loss_new_token = criterion(next_token, labels.to(self.device))
                    molecule_loss = torch.add(molecule_loss, loss_new_token)

                # Compute backpropagation
                optimizer.zero_grad()
                molecule_loss.backward(retain_graph=True)
                optimizer.step()
                molecule_loss_ = molecule_loss.cpu().detach().numpy()[0] / (self.seq_len - 1)
                if idx % self.write_sum_n_ep == 0:
                    if molecule_loss_ < training_loss:
                        training_loss = molecule_loss_
                        self.save_tfl_model()
                    self.train_writer.add_scalar('train_loss', molecule_loss_, train_iter)
            self.train_writer.export_scalars_to_json(path=self.train_path + '.json')

            # print("Evaluating model.")
            self.model.eval()
            for idx, batch in enumerate(validation_loader):
                val_iter += 1
                batch_size = batch.shape[0]
                data = batch.transpose(0, 1)
                molecule_loss = torch.zeros(1).to(self.device)
                hidden_states = self.model.new_sequence(batch_size, self.device)

                # Iteration over molecules
                for j in range(self.seq_len - 1):
                    inputs = data[j].view(1, batch_size, -1).to(self.device)
                    labels = torch.LongTensor(np.argmax(data[j + 1].view(batch_size, -1).numpy().astype(int), axis=1))
                    next_token, hidden_states = self.model(inputs, hidden_states)
                    next_token = next_token.view(batch_size, -1)
                    loss_new_token = criterion(next_token, labels.to(self.device))
                    molecule_loss = torch.add(molecule_loss, loss_new_token)

                molecule_loss_ = molecule_loss.cpu().detach().numpy()[0] / (self.seq_len - 1)
                if idx % self.write_sum_n_ep == 0:
                    if molecule_loss_ < validation_loss:
                        validation_loss = molecule_loss_
                        self.save_tfl_model()
                    self.val_writer.add_scalar('val_loss', molecule_loss_, val_iter)

    # ==================================================================================================================

    def __train_dti__(self, k = None):
        criterion = nn.MSELoss()
        print("MODEL")
        print()
        print(self.model)
        print()
        n_epochs = int(self.specs['epochs'])

        # Starting training
        self.model.to(self.device)
        print("Using device:", self.device)

        train_iter = -1
        val_iter = -1
        batch_size = int(self.parser['dti']['batch_size'])

        indices = list(range(len(self.data[0])))
        np.random.seed(SEED)
        np.random.shuffle(indices)

        if k == None:
            val_indices = []
            train_indices = indices
        else:
            split_val = 1. / self.n_folds
            split_next = int(split_val * len(self.data[0])) * k
            split_prev = int(split_val * len(self.data[0])) * (k + 1)
            print(split_next)
            print(split_prev)
            val_indices = indices[split_next:split_prev]
            train_indices = indices[:split_next] + indices[split_prev:]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        if 'cnn' in self.data[0].tfl_model:
            train_loader = torch.utils.data.DataLoader(self.data[0], batch_size=batch_size, sampler=train_sampler)
            validation_loader = torch.utils.data.DataLoader(self.data[0], batch_size=batch_size, sampler=valid_sampler)
            test_loader = torch.utils.data.DataLoader(self.data[1], batch_size=batch_size)
        else:
            train_loader = torch.utils.data.DataLoader(self.data[0], batch_size=batch_size,
                                                       sampler=train_sampler, collate_fn=pad_collate)
            validation_loader = torch.utils.data.DataLoader(self.data[0], batch_size=batch_size,
                                                            sampler=valid_sampler, collate_fn=pad_collate)
            test_loader = torch.utils.data.DataLoader(self.data[1], batch_size=batch_size, collate_fn=pad_collate)



        print("# Batches Training Dataset: ", len(train_loader))
        print("# Batches Validation Dataset: ", len(validation_loader))
        print("# Batches Test Dataset: ", len(test_loader))

        optimizer = torch.optim.Adam(
            [
                {"params": filter(lambda p: p.requires_grad, self.model.seq_model.parameters()),
                 "lr": float(self.specs['learning_rate'])},
                {"params": filter(lambda p: p.requires_grad, self.model.smi_model.parameters()),
                 "lr": float(self.specs['learning_rate'])},
            ],
            lr=float(self.specs['learning_rate']), betas=(0.9, 0.999)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min', factor=0.5, patience=10,
                                                               verbose=True, threshold=0.0001,
                                                               threshold_mode='rel', cooldown=5,
                                                               min_lr=0, eps=1e-08)

        min_val_loss = 1000000
        avg_val_loss = 1000000
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        for epoch in range(n_epochs):
            print('== EPOCH ' + str(epoch + 1) + ' ==')
            if k is not None:
                scheduler.step(avg_val_loss / len(validation_loader))
            print("\nTraining model.")
            self.model.train()
            optimizer.zero_grad()
            for i, (x1, x2, x1_lens, x2_lens, aff, x1_id) in enumerate(train_loader, 1):
                train_iter += batch_size
                if self.control:
                    pred = self.model(x1_id.view(x1_id.shape[0], -1).to(self.device), x1_lens,
                                      x2.to(self.device), x2_lens).flatten()
                else:
                    pred = self.model(x1.to(self.device), x1_lens,
                                      x2.to(self.device), x2_lens).flatten()
                if 'cnn' in self.data[0].tfl_model:
                    targets = torch.tensor(aff.numpy(), dtype=torch.float32).to(self.device)
                else:
                    targets = torch.tensor(aff, dtype=torch.float32).to(self.device)

                loss = criterion(pred, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_ = loss.cpu().item()
                self.train_writer.add_scalar('train_loss', loss_, train_iter)
                if i % self.write_sum_n_ep == 0:
                    self.train_writer.add_text('train_prediction', str(pred.cpu().data.numpy()), train_iter)
                    self.train_writer.add_text('train_target', str(targets.cpu().data.numpy()), train_iter)

            print("\nEvaluating model.")
            avg_val_loss = 0
            self.model.eval()
            for i, (x1, x2, x1_lens, x2_lens, aff, x1_id) in enumerate(validation_loader, 1):
                val_iter += batch_size
                if self.control:
                    pred = self.model(x1_id.view(x1_id.shape[0], -1).to(self.device), x1_lens,
                                      x2.to(self.device), x2_lens).flatten()
                else:
                    pred = self.model(x1.to(self.device), x1_lens,
                                      x2.to(self.device), x2_lens).flatten()
                if 'cnn' in self.data[0].tfl_model:
                    targets = torch.tensor(aff.numpy(), dtype=torch.float32).to(self.device)
                else:
                    targets = torch.tensor(aff, dtype=torch.float32).to(self.device)

                loss = criterion(pred, targets)
                loss_ = loss.cpu().item()
                avg_val_loss += loss_
                self.val_writer.add_scalar('val_loss', loss_, val_iter)
                if i % self.write_sum_n_ep == 0:
                    self.val_writer.add_text('val_prediction', str(pred.cpu().data.numpy()), val_iter)
                    self.val_writer.add_text('val_target', str(targets.cpu().data.numpy()), val_iter)

            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                self.save_tfl_model(k=k)
        print("Training complete!")

    # ==================================================================================================================

    def __eval_dti__(self):
        criterion = nn.MSELoss()
        print("MODEL")
        print()
        print(self.model)
        print()

        # Starting evaluation
        self.model.to(self.device)
        print("Using device:", self.device)

        if 'cnn' in self.data[0].tfl_model:
            test_loader = torch.utils.data.DataLoader(self.data[1], batch_size=1, shuffle=False)
        else:
            test_loader = torch.utils.data.DataLoader(self.data[1], batch_size=1, shuffle=False, collate_fn=pad_collate)

        prot_repr_dict = {}
        info_test = []

        print("\nEvaluating model {}.".format(int(self.run_id)))
        self.model.eval()
        for i, (x1, x2, x1_lens, x2_lens, aff, x1_id) in enumerate(tqdm(test_loader)):
            if self.control:
                pred = self.model(x1_id.view(x1_id.shape[0], -1).to(self.device), x1_lens,
                                  x2.to(self.device), x2_lens).flatten()
            else:
                pred = self.model(x1.to(self.device), x1_lens,
                                  x2.to(self.device), x2_lens).flatten()
            if 'cnn' in self.data[0].tfl_model:
                targets = torch.tensor(aff.numpy(), dtype=torch.float32).to(self.device)
            else:
                targets = torch.tensor(aff, dtype=torch.float32).to(self.device)

            loss_ = criterion(pred, targets).cpu().item()
            l = np.sqrt(loss_)

            prot_repr = self.model.seq_re.detach().cpu().numpy()
            info_test.append([i, np.round(pred.cpu().item(), 3), np.round(targets.cpu().item(), 3)])
            if not (x1_id.item()) in prot_repr_dict:
                prot_repr_dict.update({(x1_id.item()): (prot_repr, l, 1)})  # (representation, rmse-loss, num
                # occurances)
            else:
                _, loss, occ = prot_repr_dict[(x1_id.item())]
                prot_repr_dict.update({(x1_id.item()): (prot_repr, loss+l, occ+1)})

        with open('evaluation/test_' + str(self.run_id) + '.txt', 'w') as f:
            for item in info_test:
                f.write("%s\n" % item)

        with open('evaluation/test_repr_' + str(self.run_id) + '.txt', 'wb') as f:
            pickle.dump(prot_repr_dict, f)

        print("Testing complete!")

    # ==================================================================================================================

    def __get_tflspecs__(self, setting=None, mode=None):
        if setting is None:
            setting = self.setting
        if mode is None:
            mode = self.data.mode
        attrs = []
        specs = []
        print('Getting the model specs from config ini.')
        for attr in self.parser.options(section=setting + '_' + mode):
            attrs.append(attr)
            specs.append(self.parser[setting + '_' + mode][attr])
        return dict(zip(attrs, specs))

    def __get_dtispecs__(self):
        attrs = []
        specs = []
        print('Getting the model specs from config ini.')
        for attr in self.parser.options(section='dti'):
            attrs.append(attr)
            specs.append(self.parser['dti'][attr])
        return dict(zip(attrs, specs))

    def __writesum__(self, predictions, targets):
        p_np = predictions[0].cpu().data.numpy()
        t_np = targets[0].cpu().data.numpy()
        if self.one_hot:
            p_np = np.flatten(p_np)
            t_np = np.flatten(t_np)
        prediction = None
        target = None
        if self.data.mode == 'SMI':
            prediction_ = torch.tensor(np.absolute(p_np.round(0)))
            prediction = smiles_decoder(prediction_, one_hot=self.one_hot)
            target_ = torch.tensor(np.absolute(t_np.round(0)))
            target = smiles_decoder(target_, one_hot=self.one_hot)
        if self.data.mode == 'SEQ':
            prediction_ = torch.tensor(np.absolute(p_np.round(0)))
            prediction = seq_decoder(prediction_, one_hot=self.one_hot)
            target_ = torch.tensor(np.absolute(t_np.round(0)))
            target = seq_decoder(target_, one_hot=self.one_hot)
        return prediction, target

    # ==================================================================================================================

    def save_tfl_model(self, k=None):
        path = self.checkpoint
        model = {'hyperparameters': self.hyperparameters,
                 'state_dict': self.model.state_dict()}
        if k is None:
            torch.save(model, path + '.pt')
        else:
            torch.save(model, path + '_'+str(k)+'.pt')
        return  # model

    def load_tfl_model(self, mode='SMI'):
        if mode == 'SMI':
            tfl_run = self.tfl_runs[0]
        elif mode == 'SEQ':
            tfl_run = self.tfl_runs[1]
        else:
            print("TFL MODEL REQUIRES MODE 'SMI' OR 'SEQ'.")
            return
        path = 'models/run_{}/model_{}'.format(int(tfl_run) // 100, int(tfl_run)) + '.pt'
        print("Loading model from path", path)
        if not torch.cuda.is_available():
            return torch.load(path, map_location=torch.device('cpu'))
        else:
            return torch.load(path)

    def load_dti_model(self):
        path = 'models/run_{}/model_{}'.format(int(self.run_id) // 100, int(self.run_id)) + '.pt'
        print("Loading model from path", path)
        if not torch.cuda.is_available():
            return torch.load(path, map_location=torch.device('cpu'))
        else:
            return torch.load(path)

    def set_parameter_requires_grad(self, feature_extracting=True):
        if feature_extracting:
            for param in self.model.parameters():
                param.requires_grad = False

    def count_trainable(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
