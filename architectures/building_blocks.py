import torch
import torch.nn as nn

import torch.nn.functional as F

# https://github.com/songlab-cal/tape

# ======================================================================================================================

# ENCODER
class block_1(nn.Module):
    #def __init__(self, layout):
    def __init__(self, hidden_size, embedding_size,
                 embedding, num_layers=2, dropout=0.0):
        super(block_1, self).__init__()

        # Basic network params
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding layer that will be shared with Decoder
        self.embedding = embedding

        # Bidirectional GRU
        self.gru = nn.GRU(embedding_size, hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=True)

    def forward(self, input_sequence, input_lengths):
        # Convert input_sequence to word embeddings
        word_embeddings = self.embedding(input_sequence)

        # Pack the sequence of embeddings
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(word_embeddings, input_lengths)

        # Run the packed embeddings through the GRU, and then unpack the sequences
        outputs, hidden = self.gru(packed_embeddings)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # The ouput of a GRU has shape (seq_len, batch, hidden_size * num_directions)
        # Because the Encoder is bidirectional, combine the results from the
        # forward and reversed sequence by simply adding them together.
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

# ======================================================================================================================


class block_2(nn.Module):
    # ATTENTION
    #def __init__(self, layout):
    def __init__(self, hidden_size):
        super(block_2, self).__init__()

        self.hidden_size = hidden_size

    def dot_score(self, hidden_state, encoder_states):
        return torch.sum(hidden_state * encoder_states, dim=2)

    def forward(self, hidden, encoder_outputs, mask):
        attn_scores = self.dot_score(hidden, encoder_outputs)
        # Transpose max_length and batch_size dimensions
        attn_scores = attn_scores.t()
        # Apply mask so network does not attend <pad> tokens
        attn_scores = attn_scores.masked_fill(mask == 0, -1e10)

        # Return softmax over attention scores
        return F.softmax(attn_scores, dim=1).unsqueeze(1)

# ======================================================================================================================


class dense(nn.Module):
    def __init__(self, layout):
        super(dense, self).__init__()
        print(layout)

    def forward(self, x):
        return x

# ======================================================================================================================

class block_3(nn.Module):
    # DECODER
    def __init__(self, embedding, embedding_size,
                 hidden_size, output_size, n_layers=1, dropout=0.1):
        super(block_3, self).__init__()

        # Basic network params
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout


        self.embedding = embedding

        self.gru = nn.GRU(embedding_size, hidden_size, n_layers,
                          dropout=dropout)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = block_2(hidden_size)


    def forward(self, current_token, hidden_state, encoder_outputs, mask):
        # convert current_token to word_embedding
        embedded = self.embedding(None)

        # Pass through GRU
        rnn_output, hidden_state = self.gru(embedded, hidden_state)

        # Calculate attention weights
        attention_weights = self.attn(rnn_output, encoder_outputs, mask)

        # Calculate context vector
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))

        # Concatenate  context vector and GRU output
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Pass concat_output to final output layer
        output = self.out(concat_output)

        # Return output and final hidden state
        return output, hidden_state


# ======================================================================================================================