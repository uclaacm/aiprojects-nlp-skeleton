import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class VariableLSTM(nn.Module):
    def __init__(self, config):
        #https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
        super(VariableLSTM, self).__init__()
        self.dimension = int(config["hidden_size"])
        self.lstm = nn.LSTM(
            input_size=int(config["INPUT_DIMENSION"]),
            hidden_size= self.dimension,
            num_layers=int(config["num_layers"]),
            batch_first=True,
            bidirectional=True
        )
        self.drop = nn.Dropout(p=0.2)

        self.fc = nn.Linear(2 * self.dimension, 1)

    def forward(self, X, X_lengths):
        # https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
        packed_input = pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), X_lengths - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)

        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)

        return text_out