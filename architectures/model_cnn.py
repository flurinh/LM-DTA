import torch.nn as nn


class tfl_cnn_dense(nn.Module):
    def __init__(self, specs, vocab_size, seq_len, dti_=False):
        super(tfl_cnn_dense, self).__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.filters = int(specs['filters'])
        self.kernel_size = int(specs['kernel'])
        layers = int(specs['dense_layers'])
        self.embedding = int(specs['embedding'])
        self.bed = nn.Embedding(self.vocab_size, self.embedding)
        self.block1 = self.__block__(self.embedding, self.filters, ks=self.kernel_size,
                                     pad=int((self.kernel_size - 1) / 2), drop=0.1)
        self.block2 = self.__block__(self.filters, self.filters * 2, ks=self.kernel_size,
                                     pad=int((self.kernel_size - 1) / 2), drop=0.1)
        self.block3 = self.__block__(self.filters * 2, self.filters * 3, ks=self.kernel_size,
                                     pad=int((self.kernel_size - 1) / 2), drop=0.0)

        # Pooling makes our detection of features sequence position invariant
        self.pool = nn.AdaptiveMaxPool1d(1)

        dense = [self.__ff_block__() for i in range(layers - 1)]
        self.dense = nn.Sequential(*dense, self.__ff_block__(True))

    def __block__(self, f_in, f_out, ks, pad, drop):
        return nn.Sequential(
            # nn.BatchNorm1d(f_in),
            nn.ReLU(),
            nn.Conv1d(f_in, f_out, kernel_size=ks, padding=pad)
        )

    def __ff_block__(self, final=False):
        if final:
            return nn.Sequential(
                # nn.BatchNorm1d(self.embedding),
                nn.ReLU(),
                nn.Linear(self.filters * 3, self.seq_len)
            )
        return nn.Sequential(
            # nn.BatchNorm1d(self.embedding),
            nn.ReLU(),
            nn.Linear(self.filters * 3, self.filters * 3)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.bed(x)
        # print("embedded", x.shape)
        x = x.permute(0, 2, 1)
        # print("permutation", x.shape)
        x = self.block1(x)
        # print("conv1", x.shape)
        x = self.block2(x)
        # print("conv2", x.shape)
        x = self.block3(x)
        # print("conv3", x.shape)
        x = self.pool(x)
        # print("pool", x.shape)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        # print("out", x.shape)
        return x
