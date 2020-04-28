import torch
import torch.nn as nn
import numpy as np


class BiDirLSTM(nn.Module):

    def __init__(self, specs, vocab_size):
        super(BiDirLSTM, self).__init__()
        self.specs = specs

        # Dimensions
        self._input_dim = vocab_size
        self._hidden_dim = int(specs['hidden'])
        self._output_dim = vocab_size

        # Number of LSTM layers
        self._layers = int(specs['rnn_layers'])

        # LSTM for forward and backward reading
        self._blstm = nn.LSTM(input_size=self._input_dim, hidden_size=self._hidden_dim, num_layers=self._layers,
                              dropout=0.3, bidirectional=True)

        # All weights initialized with xavier uniform
        # Xavier initialization automatically decides the standard deviation based on the number of input and output
        # connections to a layer.
        nn.init.xavier_uniform_(self._blstm.weight_ih_l0)
        nn.init.xavier_uniform_(self._blstm.weight_ih_l1)
        nn.init.orthogonal_(self._blstm.weight_hh_l0)
        nn.init.orthogonal_(self._blstm.weight_hh_l1)

        # Bias initialized with zeros expect the bias of the forget gate
        self._blstm.bias_ih_l0.data.fill_(0.0)
        self._blstm.bias_ih_l0.data[self._hidden_dim:2 * self._hidden_dim].fill_(1.0)

        self._blstm.bias_ih_l1.data.fill_(0.0)
        self._blstm.bias_ih_l1.data[self._hidden_dim:2 * self._hidden_dim].fill_(1.0)

        self._blstm.bias_hh_l0.data.fill_(0.0)
        self._blstm.bias_hh_l0.data[self._hidden_dim:2 * self._hidden_dim].fill_(1.0)

        self._blstm.bias_hh_l1.data.fill_(0.0)
        self._blstm.bias_hh_l1.data[self._hidden_dim:2 * self._hidden_dim].fill_(1.0)

        # Batch normalization (Weights initialized with one and bias with zero)
        self._norm_0 = nn.LayerNorm(self._input_dim, eps=.001)
        self._norm_1 = nn.LayerNorm(2 * self._hidden_dim, eps=.001)

        # Separate linear model for forward and backward computation
        self._wpred = nn.Linear(2 * self._hidden_dim, self._output_dim)
        nn.init.xavier_uniform_(self._wpred.weight)
        self._wpred.bias.data.fill_(0.0)

    def _init_hidden(self, batch_size, device):
        return (torch.zeros(2 * self._layers, batch_size, self._hidden_dim).to(device),
                torch.zeros(2 * self._layers, batch_size, self._hidden_dim).to(device))

    def new_sequence(self, batch_size=1, device="cpu"):
        return self._init_hidden(batch_size, device)

    def forward(self, input, hidden, next_prediction='right', device="cpu"):
        # If next prediction should be appended at the left side, the sequence is inverted such that
        # forward and backward LSTM always read the sequence along the forward and backward direction respectively.
        if next_prediction == 'left':
            # Reverse copy of numpy array of given tensor
            input = np.flip(input.cpu().numpy(), 0).copy()
            input = torch.from_numpy(input).to(device)

        # Normalization over encoding dimension
        norm_0 = self._norm_0(input)

        # Compute LSTM unit
        out, hidden = self._blstm(norm_0, hidden)

        # out (sequence length, batch_size, 2 * hidden dim)
        # Get last prediction from forward (0:hidden_dim) and backward direction (hidden_dim:2*hidden_dim)
        for_out = out[-1, :, 0:self._hidden_dim]
        back_out = out[0, :, self._hidden_dim:]

        # Combine predictions from forward and backward direction
        bmerge = torch.cat((for_out, back_out), -1)

        # Normalization over hidden dimension
        norm_1 = self._norm_1(bmerge)

        # Linear unit forward and backward prediction
        pred = self._wpred(norm_1)

        return pred, hidden
