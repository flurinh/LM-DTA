import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np


class dti_model(nn.Module):
    def __init__(self, smi_model, seq_model, smi_len, seq_len, smi_vocab_size, seq_vocab_size, smi_model_type,
                 seq_model_type, batchsize, layers=None, smi_embedding=None, seq_embedding=None,
                 device='cpu', control=False):
        super(dti_model, self).__init__()
        self.device = device
        self.control = control
        print("DTI model was initialized to use", self.device)
        # search namespace to decide what representation scheme is called
        self.smi_model_type = smi_model_type
        self.seq_model_type = seq_model_type
        # smi_len/seq_len refers to the length of their representation (TFL), e.g. the hidden embedding size
        self.smi_len = smi_len
        self.seq_len = seq_len
        if self.smi_model_type == 'cnn':
            self.smi_len *= 3
        if self.seq_model_type == 'cnn':
            self.seq_len *= 3
        self.smi_vocab = smi_vocab_size
        self.seq_vocab = seq_vocab_size
        # the initialized models
        self.smi_embedding = smi_embedding
        self.seq_embedding = seq_embedding
        self.smi_model = smi_model
        self.seq_model = seq_model

        # layer normalization
        self.norm_smi = nn.LayerNorm(smi_vocab_size, eps=.001)
        self.norm_seq = nn.LayerNorm(seq_vocab_size, eps=.001)

        # representation size --> calculate given types and vocab_size
        self.smi_lstm_type = 'final'
        self.seq_lstm_type = 'final'  # can be sum (how do we want to pass the hidden states?
        self.smi_attention = nn.Linear(self.smi_len, self.smi_len)
        self.seq_attention = nn.Linear(self.seq_len, self.seq_len)
        torch.nn.init.xavier_uniform_(self.smi_attention.weight)
        torch.nn.init.xavier_uniform_(self.seq_attention.weight)

        # the representations
        self.smi_re = None
        self.seq_re = None

        # different feedforward layers
        self.batchsize = batchsize
        if layers is None:
            layers = [-1, 488, 488]
        n_layers = len(layers) - 1
        print("smi length: {} -- seq length: {}".format(self.smi_len, self.seq_len))
        layers[0] = smi_len + seq_len  # refers to the length of their representation
        dense = [self.__ff_block__(layers[i], layers[i + 1]) for i in range(n_layers)]
        self.out = nn.Linear(layers[-1], 1)
        self.out_act = nn.Sigmoid()
        self.dense = nn.Sequential(*dense, self.out, self.out_act)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.dense.apply(init_weights)

    def forward(self, seq, seq_len, smi, smi_len):
        # smiles representation
        if self.smi_model_type == 'lstm':
            smi_re = self.lstm_representation(smi, 'smi', smi_len)
        elif self.smi_model_type == 'bidir':
            smi_re = self.bidir_representation(smi, 'smi', smi_len)
        elif self.smi_model_type == 'cnn':
            smi_re = self.cnn_representation(smi, 'smi')

        # protein representation
        if self.control:
            batch_size = seq.shape[0]
            seq_re = self.seq_embedding(seq)
            seq_re = seq_re.view(batch_size, -1)
        else:
            if self.seq_model_type == 'lstm':
                seq_re = self.lstm_representation(seq, 'seq', seq_len)
            elif self.seq_model_type == 'bidir':
                seq_re = self.bidir_representation(seq, 'seq', seq_len)
            elif self.seq_model_type == 'cnn':
                seq_re = self.cnn_representation(seq, 'seq')
        # print("smi {} --- seq {}".format(smi_re.shape, seq_re.shape))
        # Linear feed-forward network
        self.smi_re = smi_re
        self.seq_re = seq_re
        x = torch.cat((smi_re, seq_re), dim=1)
        x = self.dense(x)
        return x

    # ==================================================================================================================

    def __ff_block__(self, shape_in, shape_out):
        return nn.Sequential(
            nn.Dropout(0.1),
            # nn.BatchNorm1d(shape_in), # cannot use this because i need batchsize = 1 due to different sequence lengths
            nn.ReLU(),
            nn.Linear(shape_in, shape_out),
        )

    # ==================================================================================================================

    def lstm_representation(self, input, type, lens=None):
        batch_size = input.shape[0]
        if type == 'smi':
            self.smi_h0, self.smi_c0 = self._init_hidden_lstm(2, batch_size, self.smi_len)
            if lens is not None:
                pack = pack_padded_sequence(input, lens, batch_first=True, enforce_sorted=False).to(self.device)
            _, (self.smi_h0, self.smi_c0) = self.smi_model(pack, (self.smi_h0, self.smi_c0))
            final = self.smi_h0[-1]
        elif type == 'seq':
            self.seq_h0, self.seq_c0 = self._init_hidden_lstm(2, batch_size, self.seq_len)
            if lens is not None:
                pack = pack_padded_sequence(input, lens, batch_first=True, enforce_sorted=False).to(self.device)
            _, (self.seq_h0, self.seq_c0) = self.seq_model(pack, (self.seq_h0, self.seq_c0))
            final = self.seq_h0[-1]
        return final.view(batch_size, -1)

    def _init_hidden_lstm(self, nlayers, batch_size, hidden_dim):
        return torch.zeros(nlayers, batch_size, hidden_dim).to(self.device), \
               torch.zeros(nlayers, batch_size, hidden_dim).to(self.device)

    def new_sequence(self, model, batch_size=1):
        model.new_sequence(batch_size=batch_size, device=self.device)
        model.new_sequence(batch_size=batch_size, device=self.device)

    # ==================================================================================================================

    def bidir_representation(self, input, type, lens=None):
        batch_size = input.shape[0]
        if type == 'smi':
            if lens is not None:
                pack = pack_padded_sequence(input, lens, batch_first=True, enforce_sorted=False).to(self.device)
            self.smi_h0, self.smi_c0 = self._init_hidden_lstm(4, batch_size, self.smi_len)
            _, (self.smi_h0, self.smi_c0) = self.smi_model(pack, (self.smi_h0, self.smi_c0))
            final = self.smi_h0[-1]
        elif type == 'seq':
            if lens is not None:
                pack = pack_padded_sequence(input, lens, batch_first=True, enforce_sorted=False).to(self.device)
            self.seq_h0, self.seq_c0 = self._init_hidden_lstm(4, batch_size, self.seq_len)
            _, (self.seq_h0, self.seq_c0) = self.seq_model(pack, (self.seq_h0, self.seq_c0))
            final = self.seq_h0[-1]
        return final

    # ======================================================================================================================

    def cnn_representation(self, input, type):
        batch_size = input.shape[0]
        if type == 'smi':
            x = self.smi_embedding(input)
            x = x.permute(0, 2, 1)
            x = self.smi_model(x)
        else:
            x = self.seq_embedding(input)
            x = x.permute(0, 2, 1)
            x = self.seq_model(x)
        return x.view(batch_size, -1)

