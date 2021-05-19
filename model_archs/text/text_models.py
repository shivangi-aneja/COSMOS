"""
    Class file that enlists models for extracting text features
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.config import device


class ToyRNNLSTM(nn.Module):
    def __init__(self, hidden_size, embedding_length, num_layers=1):
        """
            Initializes LSTM object based on arguments given

            Args:
                hidden_size (int): dimension of the hidden unit of the LSTM cell
                embedding_length (int): input dimensionality of the word embedding
                num_layers (int): number of layers in the LSTM model

            Returns:
                None
        """
        super(ToyRNNLSTM, self).__init__()
        self.relu = nn.ReLU()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_length = embedding_length
        self.lstm = nn.LSTM(input_size=self.embedding_length, hidden_size=hidden_size, num_layers=self.num_layers,
                            bidirectional=False, batch_first=True)

    def forward(self, text, batch_size, seq_len):
        """
            Function to compute forward pass of the network

            Args:
                text (Tensor): Text embeddings of shape (N X K X 300) where N denotes batch size, K represents number of words in caption and 300-dim vector for each word is computed
                batch_size (int): Mini-batch size
                seq_len (list): Sequence(word) length for each caption in the minibatch

            Returns:
                out (Tensor): Tensor of shape (N X 300) where N is the batch size

        """

        # Initial hidden state of the LSTM
        h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))
        # Initial cell state of the LSTM
        c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

        output, (final_hidden_state, final_cell_state) = self.lstm(text, (h_0, c_0))

        # Create the masked tensor to get output corresponding to last word in the caption
        mask = torch.as_tensor(seq_len, dtype=torch.long).view(-1, 1, 1).cuda()
        mask = mask.expand(batch_size, 1, self.hidden_size) - 1
        # Gathers values along an axis specified by dim, index given
        out = torch.gather(output, dim=1, index=mask).squeeze(1)
        return out.squeeze(dim=1)


class ToyText(nn.Module):
    def __init__(self, hidden_size):
        """
            Initializes a simple MLP to process USE embeddings

            Args:
                hidden_size: output dimensionality of text embedding

            Returns:
                None
        """
        super(ToyText, self).__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(512, hidden_size)

    def forward(self, text):
        """
            Function to compute forward pass of the network

            Args:
                text (Tensor): Text embeddings of shape (N X 512) where N denotes batch size and 512-dim vector extracted by USE model for each caption

            Returns:
                out (Tensor): Tensor of shape (N X 300) where N is the batch size
        """
        out = self.fc(self.relu(text))
        return out
